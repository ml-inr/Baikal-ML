"""
Convert MC reconstruction ROOT files to HDF5.

Based on root2h5.py (MC truth pipeline). Adds:
  - reco_prty: per-event reconstruction scalars (BRecoMuon fields)
  - reco_vectors: fXYZRec and fDirectionRec unpacked as 6 extra columns in reco_prty

Keeps all MC truth data from root2h5.py:
  - raw/labels: MCEventMask flags (ground truth hit labels)
  - prime_prty: primary particle properties
  - muons_prty: individual and aggregate muon track properties
  - raw/t_res: time residuals (eval_tres)

Cluster centers are computed per-file (coords_are_same: false).

Usage:
    cd data_manager/root2h5
    python root2h5_mc_reco.py
"""

from multiprocessing import Process, Queue, Event
import numpy as np
import awkward as ak
import uproot as ur
import h5py as h5
import os
import yaml
import logging
import traceback
from contextlib import contextmanager

from eval_tres import eval_tres


# --- Configuration ---

def read_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")

try:
    config = read_config('root2h5_config_mc_reco.yaml')
except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"Error loading configuration: {e}")
    exit(1)

take_single_cluster      = config['general']['take_single_cluster']
take_clust_num           = config['general']['take_clust_num']
split_multi              = config['general']['split_multi']
shift_coords_to_cl_center = config['general']['shift_coords_to_cl_center']
center_times             = config['general']['center_times']
exclude_big_ts           = config['general']['exclude_big_ts']
t_threshold              = float(config['general']['t_threshold'])
coords_are_same          = config['general']['coords_are_same']

h5_name       = config['output']['h5_name']
h5_prefix     = config['output']['h5_prefix']
particle      = config['input']['particle']
root_dir_path = config['input']['root_dir_path']

pathes_data        = config['root_paths']['data']
pathes_primary     = config['root_paths']['primary']
pathes_resp_muons  = config['root_paths']['resp_muons']
path_mu_scalar     = config['root_paths']['mu_scalar']
path_geometry      = config['root_paths']['geometry']
pathes_reco_ev     = config['root_paths']['reco_ev']
pathes_reco_vectors = config['root_paths'].get('reco_vectors', [])
path_reco_mask     = config['root_paths'].get('reco_mask', None)

MAX_QUEUE_SIZE = config['multiprocessing']['MAX_QUEUE_SIZE']
NUM_WORKERS    = config['multiprocessing']['NUM_WORKERS']


# --- Helper functions ---

def check_file(rf_path):
    try:
        with ur.open(rf_path) as rf:
            ev_num = rf['Events/BEvent./BEvent.fPulseN'].num_entries
            return ev_num > 1 or (
                ev_num == 1
                and rf['Events/BEvent./BEvent.fPulseN'].array(library="np", entry_start=0, entry_stop=1)[0] != 0
            )
    except Exception:
        return False


def get_single_mask(active_clusters, num_un_clusters, take_clust_num):
    mask = num_un_clusters == 1
    active_cluster_id = np.array([ac[0] for ac in active_clusters])
    if take_clust_num is not None:
        mask &= active_cluster_id == take_clust_num
    return mask, active_cluster_id[mask]


def get_multi_mask(active_clusters, num_un_clusters):
    mask = num_un_clusters != 1
    return mask, [ac for ac, m in zip(active_clusters, mask) if m]


def _make_object_array(lst):
    """Build a 1-D object array from a list of arrays, always ragged-safe."""
    arr = np.empty(len(lst), dtype=object)
    for i, x in enumerate(lst):
        arr[i] = x
    return arr


def flatten_multi(datas, channels, cluster_ids_multi):
    ress = []
    for data in datas:
        res = []
        for (d, cls_ids, chs) in zip(data, cluster_ids_multi, channels):
            for cl_id in cls_ids:
                cl_mask = (chs // 288) == cl_id
                res.append(d[cl_mask])
        ress.append(res)
    return ress


def cast_to_single(data, mask_single, mask_multi,
                   nums_multi, take_single_cluster,
                   split_multi):
    result = []
    if take_single_cluster:
        result.append(data[mask_single])
    if split_multi:
        result.append(np.repeat(data[mask_multi], nums_multi, axis=0))
    return np.concatenate(result) if result else data


def get_start_index(rf):
    test = rf['Events/BEvent./BEvent.fPulseN'].array(library="np", entry_start=0, entry_stop=1)[0]
    return 1 if test == 0 else 0


def get_cl_centers(rf_path):
    with ur.open(rf_path) as rf:
        st = get_start_index(rf)
        coordinates = np.array(ak.unzip(rf[path_geometry].array()))[:, st:]
    num_clusters = coordinates.shape[-1] // 288
    coordinates = coordinates.reshape(*coordinates.shape[:-1], num_clusters, 288)
    cl_centers = np.mean(coordinates, axis=(1, -1))
    return cl_centers.T


# --- Queue helper ---

def safe_put(q, item, do_quit, timeout=1):
    """Put item into queue, checking do_quit so workers never block forever."""
    while not do_quit.is_set():
        try:
            q.put(item, timeout=timeout)
            return
        except Exception:
            continue


# --- Listener (HDF5 writer) ---

@contextmanager
def error_handling(do_quit):
    try:
        yield
    except Exception as e:
        logging.error(f"Error in listener: {e}")
        do_quit.set()
        raise


def listener(result_q, do_quit, n_workers):
    file_path = os.path.join(h5_prefix, h5_name)
    with error_handling(do_quit), h5.File(file_path, 'a') as hf:
        while n_workers > 0:
            res = result_q.get()
            if res is None:
                n_workers -= 1
                logging.info(f"Worker finished. Still workers: {n_workers}")
            elif res[0]:
                write_data(hf, res[1], res[2])
                logging.info(f"Written data for file part: {res[2]}")

    logging.info("All workers finished. Listener exiting.")
    do_quit.set()


def get_dtype(key, value):
    if any(s in key for s in ['data', 't_res', 'prime', 'reco_prty']) or (
        'muons' in key and 'starts' not in key
    ):
        return np.float32
    elif 'ev_ids' in key:
        return value.dtype
    else:
        return np.int32


def write_data(hf, res, file_num):
    for key, value in res.items():
        dataset_path = f"{particle}/{key}/part_{file_num}/data"
        kwarg = {
            'dtype': get_dtype(key, value),
            'compression': 'gzip' if any(s in key for s in ['raw', 'reco', 'muons']) and 'starts' not in key else None,
        }
        if dataset_path in hf:
            del hf[dataset_path]
        dtype = kwarg['dtype']
        arr = value
        if np.issubdtype(dtype, np.integer) and np.issubdtype(np.asarray(value).dtype, np.floating):
            arr = np.nan_to_num(value, nan=0)
        hf.create_dataset(dataset_path, data=arr.astype(dtype), **kwarg)


# --- Main processing ---

def process_file(args_q, result_q, do_quit):
    while not do_quit.is_set():
        args = args_q.get()
        if args is None:
            break

        rf_path, id_prefix, cl_centers = args

        try:
            if not check_file(rf_path):
                logging.info(f"File {rf_path} has no data.")
                continue
            logging.info(f"Processing file: {os.path.basename(rf_path)}")

            with ur.open(rf_path) as rf:
                res_dict = {}
                st = get_start_index(rf)

                # --- Read pulse data ---
                num_channels = rf['Events/BEvent./BEvent.fPulseN'].array(library="np")[st:]
                channels = rf['Events/BEvent./BEvent.fPulses/BEvent.fPulses.fChannelID'].array(library='np')[st:]
                active_clusters = [np.unique(ch // 288) for ch in channels]
                num_un_clusters = np.array([len(cl) for cl in active_clusters])

                num_resp_mu = rf['Events/BMCEvent./BMCEvent.fResponseMuonsN'].array(library="np").astype('int')[st:]

                # --- Read reco scalars (per-event) ---
                reco_ev_arrays = [rf[p].array(library="np")[st:] for p in pathes_reco_ev]

                # --- Read reco vectors (per-event, 3-element each) ---
                # fXYZRec and fDirectionRec: shape (3, n_events) after ak.unzip → (n_events, 3)
                for p in pathes_reco_vectors:
                    vec = np.array(ak.unzip(rf[p].array()))[: , st:]  # (3, n_events)
                    reco_ev_arrays.extend([vec[i] for i in range(vec.shape[0])])

                reco_ev_matrix = np.column_stack(reco_ev_arrays)  # (n_events, n_fields)

                # --- Read reco mask (per-hit, variable-length per event) ---
                reco_mask_raw = rf[path_reco_mask].array(library='np')[st:] if path_reco_mask else None

                # --- Cluster masks ---
                if take_single_cluster:
                    mask_single, cluster_ids = get_single_mask(active_clusters, num_un_clusters, take_clust_num)
                    mask_multi, cluster_ids_multi = get_multi_mask(active_clusters, num_un_clusters)
                    nums_multi = num_un_clusters[mask_multi]
                    if split_multi:
                        cluster_ids_multi_flat = np.array([cl for clss in cluster_ids_multi for cl in clss])
                        cluster_ids = np.concatenate((cluster_ids, cluster_ids_multi_flat), axis=0)

                # --- Read hit data ---
                data = [rf[p].array(library="np")[st:] for p in pathes_data]

                # --- Split multi-cluster events ---
                if split_multi:
                    data_multi = [d[mask_multi] for d in data]
                    data_multi = flatten_multi(data_multi, channels[mask_multi], cluster_ids_multi)
                    data_multi = [_make_object_array(d) for d in data_multi]
                    num_channels_multi = np.array([len(chs) for chs in data_multi[-1]])
                    num_resp_mu_multi = cast_to_single(
                        num_resp_mu, mask_single, mask_multi, nums_multi,
                        take_single_cluster, split_multi
                    )

                    if reco_mask_raw is not None:
                        reco_mask_multi = flatten_multi(
                            [reco_mask_raw[mask_multi]],
                            channels[mask_multi],
                            cluster_ids_multi
                        )[0]
                        reco_mask_multi = _make_object_array(reco_mask_multi)

                # --- Extract single-cluster events ---
                if take_single_cluster:
                    data = [d[mask_single] for d in data]
                    num_channels = num_channels[mask_single]
                    num_resp_mu = num_resp_mu[mask_single]
                    if reco_mask_raw is not None:
                        reco_mask_events = reco_mask_raw[mask_single]
                    if split_multi:
                        data = [np.concatenate((ds, dm), axis=0) for ds, dm in zip(data, data_multi)]
                        num_channels = np.concatenate((num_channels, num_channels_multi), axis=0)
                        num_resp_mu = num_resp_mu_multi
                        if reco_mask_raw is not None:
                            reco_mask_events = np.concatenate((reco_mask_events, reco_mask_multi), axis=0)

                # --- Apply cluster splitting to reco scalars ---
                reco_prty = cast_to_single(
                    reco_ev_matrix, mask_single, mask_multi,
                    nums_multi, take_single_cluster, split_multi
                )

                # --- Processing hit data ---
                ev_starts = np.concatenate(([0], np.cumsum(num_channels))).astype(np.int64)
                mu_starts = np.concatenate(([0], np.cumsum(num_resp_mu))).astype(np.int64)

                # Flatten and sort by time
                r_data = np.array([np.concatenate(d) for d in data])
                sort_idxs = np.concatenate([
                    np.argsort(r_data[1, ev_starts[i]:ev_starts[i + 1]]) + ev_starts[i]
                    for i in range(len(ev_starts) - 1)
                ])
                r_data = r_data[:, sort_idxs]
                if reco_mask_raw is not None:
                    reco_mask_flat = np.concatenate(reco_mask_events)[sort_idxs]

                # --- Read and process coordinates ---
                coordinates = np.array(ak.unzip(rf[path_geometry].array()))[:, st:]
                if coords_are_same:
                    coordinates = np.repeat(coordinates, channels.shape[0], axis=1)
                coordinates = np.transpose(
                    cast_to_single(
                        np.transpose(coordinates, (1, 0, 2)),
                        mask_single, mask_multi, nums_multi,
                        take_single_cluster, split_multi
                    ),
                    (1, 0, 2)
                )

                # Look up triggered detector coordinates
                channels_int = r_data[-1].astype(np.int32)
                tr_coord = [
                    [coordinates[j, i, channels_int[ev_starts[i]:ev_starts[i + 1]]]
                     for i in range(len(ev_starts) - 1)]
                    for j in range(3)
                ]
                tr_coord = np.array([np.concatenate(tr_coord[i], axis=0) for i in range(len(tr_coord))])
                r_data = np.concatenate((r_data, tr_coord), axis=0)

                # --- Exclude hits with extreme times ---
                if exclude_big_ts:
                    mask = np.logical_and(np.abs(r_data[1]) <= t_threshold, ~np.isnan(r_data[1, :]))
                    r_data = r_data[:, mask]
                    if reco_mask_raw is not None:
                        reco_mask_flat = reco_mask_flat[mask]
                    shifts = np.array([
                        np.sum(~mask[ev_starts[i]:ev_starts[i + 1]])
                        for i in range(len(ev_starts) - 1)
                    ])
                    shifts = np.cumsum(shifts)
                    ev_starts[1:] = ev_starts[1:] - shifts

                # --- Calculate time residuals ---
                resp_muons_prty = np.array(
                    [rf[p].array(library="np")[st:] for p in pathes_resp_muons], dtype=object
                )
                mu_scalar = np.array(
                    [rf[p].array(library="np")[st:].astype(np.float64) for p in path_mu_scalar]
                ).T
                resp_muons_prty[-2] += mu_scalar[:, 0]
                resp_muons_prty = cast_to_single(
                    resp_muons_prty.T, mask_single, mask_multi, nums_multi,
                    take_single_cluster, split_multi
                ).T
                resp_muons_prty[:2] = resp_muons_prty[:2] / 180 * np.pi
                r_resp_muons_prty = np.array(
                    [np.concatenate(rmp, axis=0) for rmp in resp_muons_prty]
                ).astype(np.float64)
                om_coords = np.concatenate([
                    np.tile(r_data[-3:, ev_starts[i]:ev_starts[i + 1]], (1, num_resp_mu[i]))
                    for i in range(len(ev_starts) - 1)
                ], axis=1)
                reps_mu = np.repeat(ev_starts[1:] - ev_starts[:-1], num_resp_mu, axis=0)
                mus = np.repeat(r_resp_muons_prty, reps_mu, axis=1)
                t_det = np.concatenate([
                    np.tile(r_data[1, ev_starts[i]:ev_starts[i + 1]], (num_resp_mu[i]))
                    for i in range(len(ev_starts) - 1)
                ], axis=0)
                t_res = eval_tres(mus.T, om_coords.T, t_det, ev_starts, mu_starts)

                # --- Center times ---
                if center_times:
                    ts_sum = np.array([
                        np.sum(r_data[1, ev_starts[i]:ev_starts[i + 1]])
                        for i in range(len(ev_starts) - 1)
                    ])
                    ts_avg = ts_sum / (ev_starts[1:] - ev_starts[:-1])
                    r_data[1] = np.concatenate([
                        r_data[1, ev_starts[i]:ev_starts[i + 1]] - ts_avg[i]
                        for i in range(len(ev_starts) - 1)
                    ])

                # --- Shift coordinates to cluster center ---
                if shift_coords_to_cl_center:
                    cl_centers_broad = np.expand_dims(
                        np.repeat(np.transpose(cl_centers, axes=(1, 0)), 288, axis=1), axis=1
                    )
                    coordinates += -cl_centers_broad
                    channels_int2 = r_data[3].astype('int')
                    tr_coord = [
                        [coordinates[j, i, channels_int2[ev_starts[i]:ev_starts[i + 1]]]
                         for i in range(len(ev_starts) - 1)]
                        for j in range(3)
                    ]
                    tr_coord = np.array([np.concatenate(tr_coord[i], axis=0) for i in range(len(tr_coord))])
                    r_data[-3:] = tr_coord

                # --- Build results ---
                sig_strings = r_data[3] // 36
                un_strings = [
                    set(sig_strings[ev_starts[i]:ev_starts[i + 1]])
                    for i in range(len(ev_starts) - 1)
                ]
                res_dict['raw/num_un_strings'] = np.array([len(s) for s in un_strings]).astype('int')

                # Event IDs
                ev_ids = np.arange(rf['Events/BEvent./BEvent.fPulseN'].num_entries - st)
                if split_multi:
                    ids_multi = ev_ids[mask_multi]
                    ids_multi = np.repeat(ids_multi, nums_multi)
                if take_single_cluster:
                    ev_ids = ev_ids[mask_single]
                    if split_multi:
                        ev_ids = np.concatenate((ev_ids, ids_multi))
                ev_ids = np.array([
                    f"{particle}_{id_prefix}_{str(int(ev_id))}" for ev_id in ev_ids
                ]).astype('bytes')
                res_dict['ev_ids'] = ev_ids

                # Primary particle properties
                prime_prty = np.array([rf[p].array(library="np")[st:] for p in pathes_primary])
                res_dict['prime_prty'] = cast_to_single(
                    np.transpose(prime_prty, (1, 0)),
                    mask_single, mask_multi, nums_multi,
                    take_single_cluster, split_multi
                )

                # Raw hit data (amplitude, time, x, y, z) + MC labels + channels + ev_starts
                res_dict['raw/cluster_ids'] = cluster_ids
                res_dict['raw/data'] = np.transpose(r_data[[0, 1, 4, 5, 6]], (1, 0))
                res_dict['raw/labels'] = r_data[2]      # MCEventMask flags (ground truth)
                res_dict['raw/channels'] = r_data[3]
                res_dict['raw/ev_starts'] = ev_starts
                res_dict['raw/t_res'] = t_res

                # Reco hit mask (ScanfitMask flags) + per-event reco signal hit count
                if reco_mask_raw is not None:
                    starts = ev_starts[:-1].astype(np.intp)
                    reco_n_sig_hits = np.add.reduceat(
                        (reco_mask_flat != 0).astype(np.int32), starts
                    )
                    res_dict['raw/reco_mask'] = reco_mask_flat.astype(np.int32)
                    res_dict['raw/reco_n_sig_hits'] = reco_n_sig_hits

                # Muon track data
                res_dict['muons_prty/aggregate'] = cast_to_single(
                    mu_scalar, mask_single, mask_multi, nums_multi,
                    take_single_cluster, split_multi
                )
                res_dict['muons_prty/individ'] = np.transpose(r_resp_muons_prty, (1, 0))
                res_dict['muons_prty/mu_starts'] = mu_starts

                # Reconstruction per-event properties
                res_dict['reco_prty'] = reco_prty

                logging.info(
                    f"Successfully processed file: {os.path.basename(rf_path)}, "
                    f"({len(ev_ids)} events, reco_prty shape: {reco_prty.shape})"
                )
                safe_put(result_q, (True, res_dict, id_prefix), do_quit)

        except Exception as e:
            print(traceback.format_exc())
            print(f"Error processing file {rf_path}: {str(e)}")
            safe_put(result_q, (False, None, id_prefix), do_quit)

    safe_put(result_q, None, do_quit)


# --- Main ---

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("MC reco data processing started")

    args_q = Queue()
    result_q = Queue(maxsize=MAX_QUEUE_SIZE)
    do_quit = Event()

    listener_process = Process(target=listener, args=(result_q, do_quit, NUM_WORKERS))
    listener_process.start()

    workers = []
    for _ in range(NUM_WORKERS):
        worker = Process(target=process_file, args=(args_q, result_q, do_quit))
        worker.start()
        workers.append(worker)

    logging.info(f"Initiated {NUM_WORKERS} workers with {MAX_QUEUE_SIZE} maximal queue size")

    filenames = []
    cl_centers_all = []
    root_files = sorted(f for f in os.listdir(root_dir_path) if f.endswith('.root'))

    for i, root_file in enumerate(root_files):
        rf_path = os.path.join(root_dir_path, root_file)
        logging.info(f"Queuing file {i + 1}/{len(root_files)}: {root_file}")

        if not check_file(rf_path):
            logging.warning(f"Skipping empty/unreadable file: {root_file}")
            continue

        cl_center = get_cl_centers(rf_path)
        id_prefix = root_file.split(".")[0]

        args_q.put((rf_path, id_prefix, cl_center))
        filenames.append(id_prefix)
        cl_centers_all.append(cl_center)

    for _ in range(NUM_WORKERS):
        args_q.put(None)

    for worker in workers:
        worker.join()

    do_quit.wait()
    listener_process.join()

    # Write per-file cluster centers and metadata
    with h5.File(os.path.join(h5_prefix, h5_name), 'a') as hf:
        for name, center in zip(filenames, cl_centers_all):
            ds_path = f"{particle}/clusters_centers/part_{name}/data"
            if ds_path in hf:
                del hf[ds_path]
            hf.create_dataset(ds_path, data=center)
        meta_path = f"{particle}/coords_are_cluster_centered/data"
        if meta_path in hf:
            del hf[meta_path]
        hf.create_dataset(meta_path, data=shift_coords_to_cl_center)

    logging.info("Cluster information written to file")
    logging.info("MC reco data processing completed")


if __name__ == "__main__":
    file_path = os.path.join(h5_prefix, h5_name)
    if not os.path.exists(file_path):
        with h5.File(file_path, 'w') as hf:
            pass
    main()
