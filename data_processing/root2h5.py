from multiprocessing import Process, Queue, Event
import numpy as np
import awkward as ak
import uproot as ur
import h5py as h5
import os
import yaml
import logging
from contextlib import contextmanager

import traceback

from eval_tres import eval_tres

# Reading configuration file
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
    config = read_config('root2h5_config.yaml')
except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"Error loading configuration: {e}")
    exit(1)

# Use config values
take_single_cluster = config['general']['take_single_cluster']
take_clust_num = config['general']['take_clust_num']
split_multi = config['general']['split_multi']

shift_coords_to_cl_center = config['general']['shift_coords_to_cl_center']
center_times = config['general']['center_times']
exclude_big_ts = config['general']['exclude_big_ts']
t_threshold = float(config['general']['t_threshold'])

coords_are_same = config['general']['coords_are_same']

h5_name = config['output']['h5_name']
h5_prefix = config['output']['h5_prefix']

particle = config['input']['particle']
MC_dir_path = config['input']['MC_dir_path']

pathes_data = config['root_paths']['data']
pathes_primary = config['root_paths']['primary']
pathes_resp_muons = config['root_paths']['resp_muons']
path_mu_scalar = config['root_paths']['mu_scalar']
path_geometry = config['root_paths']['geometry']

MAX_QUEUE_SIZE = config['multiprocessing']['MAX_QUEUE_SIZE']
NUM_WORKERS = config['multiprocessing']['NUM_WORKERS']

# Check if file is valid
def check_file(rf_path):
    try:
        with ur.open(rf_path) as rf:
            ev_num = rf['Events/BEvent./BEvent.fPulseN'].num_entries
            return ev_num>1 or (ev_num==1 and rf['Events/BEvent./BEvent.fPulseN'].array(library="np", entry_start=0, entry_stop=1)[0]!=0)
    except Exception:
        return False

# Get mask for single cluster events
def get_single_mask(active_clusters, num_un_clusters, take_clust_num):
    mask = num_un_clusters == 1
    active_cluster_id = np.array([ac[0] for ac in active_clusters])
    
    if take_clust_num is not None:
        mask &= active_cluster_id == take_clust_num
    
    return mask, active_cluster_id[mask]

# Get mask for multi cluster events
def get_multi_mask(active_clusters, num_un_clusters):
    mask = num_un_clusters != 1
    return mask, [ac for ac, m in zip(active_clusters, mask) if m]

# Splits data for multi cluster events into single clusters
def flatten_multi(datas, channels, cluster_ids_multi):
    ress = []
    for data in datas:
        res = []
        # iterate over events
        for (d,cls_ids,chs) in zip(data,cluster_ids_multi,channels):
            for cl_id in cls_ids:
                cl_mask = (chs//288)==cl_id
                res.append(d[cl_mask])
        ress.append(res)
    return ress

# Transforms input array so that data for individual clusters can be extracted. Required for spliting.
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

# Calculate clusters centers
def get_cl_centers(rf_path):
    with ur.open(rf_path) as rf:
        st = get_start_index(rf)
        # shape (3, num_evs, num_oms)
        coordinates = np.array(ak.unzip(rf[path_geometry].array()))[:,st:]
    
    # Reshape to group OMs into clusters
    num_clusters = coordinates.shape[-1] // 288
    coordinates = coordinates.reshape(*coordinates.shape[:-1], num_clusters, 288)
    
    # Calculate mean for each cluster
    cl_centers = np.mean(coordinates, axis=(1, -1))
    
    return cl_centers.T

# Listener - write processed data to file
@contextmanager
def error_handling(do_quit):
    try:
        yield
    except Exception as e:
        logging.error(f"Error in listener: {e}")
        do_quit.set()
        raise

def listener(result_q, do_quit, n_workers):
    
    with error_handling(do_quit):
        while n_workers > 0:
            res = result_q.get()
            if res is None:
                n_workers -= 1
                logging.debug(f"Worker finished. Remaining workers: {n_workers}")
            elif res[0]:
                write_data(res[1], res[2])
                logging.debug(f"Data written for {res[2]}")
    
    logging.info("All workers finished. Listener exiting.")
    do_quit.set()

def get_dtype(key, value):
    if any(substring in key for substring in ['data', 't_res', 'prime']) or ('muons' in key and 'starts' not in key):
        return np.float32
    elif 'ev_ids' in key:
        return value.dtype
    else:
        return np.int32

def write_data(res, file_num):
    file_path = os.path.join(h5_prefix, h5_name)
    with h5.File(file_path, 'a') as hf:
        for key, value in res.items():
            dataset_path = f"{particle}/{key}/part_{file_num}/data"
            
            kwarg = {
                'dtype': get_dtype(key, value),
                'compression': 'gzip' if any(substring in key for substring in ['raw', 'reco', 'muons']) and 'starts' not in key else None
            }
            
            hf.create_dataset(dataset_path, data=value.astype(kwarg['dtype']), **kwarg)

# Main processing routine; takes 1 file for processing
def process_file(args_q, result_q, do_quit):
    while not do_quit.is_set():
        args = args_q.get()
        if args is None:
            break
        
        rf_path, id_prefix, cl_centers = args
        
        try:
            # Check if file has data
            if not check_file(rf_path):
                logging.info(f"File {rf} has no data.")
                continue  # Skip empty files
            with ur.open(rf_path) as rf:
                # init dict for out data
                res_dict = {}
                # get first index
                st = get_start_index(rf)
                # Read channels and identify clusters
                num_channels = rf['Events/BEvent./BEvent.fPulseN'].array(library="np")[st:]
                channels = rf['Events/BEvent./BEvent.fPulses/BEvent.fPulses.fChannelID'].array(library='np')[st:]
                active_clusters = [np.unique(ch // 288) for ch in channels]
                num_un_clusters = np.array([len(cl) for cl in active_clusters])
                             
                # First, get masks and identify clusters
                if take_single_cluster:
                    mask_single, cluster_ids = get_single_mask(active_clusters, num_un_clusters, take_clust_num)
                    mask_multi, cluster_ids_multi = get_multi_mask(active_clusters, num_un_clusters)
                    nums_multi = num_un_clusters[mask_multi]
                    if split_multi:
                        cluster_ids_multi_flat = np.array([ cl for clss in cluster_ids_multi for cl in clss  ])
                        cluster_ids = np.concatenate( (cluster_ids,cluster_ids_multi_flat), axis=0 )
                # Read data
                data = [rf[p].array(library="np")[st:] for p in pathes_data]
                num_resp_mu = rf['Events/BMCEvent./BMCEvent.fResponseMuonsN'].array(library="np").astype('int')[st:]
                
                # Get data for multicluster events before modifying variables
                if split_multi:    
                    data_multi = [d[mask_multi] for d in data]
                    data_multi = flatten_multi(data_multi, channels[mask_multi], cluster_ids_multi)
                    data_multi = [np.array(d, dtype=object) for d in data_multi]
                    num_channels_multi = np.array([ len(chs) for chs in data_multi[-1] ])
                    # for muons, we need to keep tham all for multicluster events 
                    num_resp_mu_multi = cast_to_single( num_resp_mu, mask_single, mask_multi, nums_multi, take_single_cluster, split_multi )
                # Now extract data for single events
                if take_single_cluster:
                    data = [ d[mask_single] for d in data ]
                    num_channels = num_channels[mask_single]
                    num_resp_mu = num_resp_mu[mask_single]
                    # And extend it for multi-cluster events
                    if split_multi:
                        data = [ np.concatenate( (ds,dm), axis=0 ) for ds,dm in zip(data,data_multi) ]
                        num_channels = np.concatenate( (num_channels,num_channels_multi), axis=0 )
                        # revert to reqired data
                        num_resp_mu = num_resp_mu_multi

                # Processing data
                ev_starts = np.concatenate(([0], np.cumsum(num_channels)))
                mu_starts = np.concatenate(([0], np.cumsum(num_resp_mu)))
                
                # Falttern and sort
                r_data = np.array([np.concatenate(d) for d in data])
                sort_idxs = np.concatenate([np.argsort(r_data[1, ev_starts[i]:ev_starts[i+1]]) + ev_starts[i] for i in range(len(ev_starts)-1)])
                r_data = r_data[:, sort_idxs]
                
                # Read coordinates; braodcast, if needed, for unniform processing
                coordinates = np.array(ak.unzip(rf[path_geometry].array()))[:,st:]
                if coords_are_same:
                    coordinates = np.repeat( coordinates, channels.shape[0], axis=1 )
                coordinates = np.transpose( cast_to_single( np.transpose(coordinates, (1,0,2)), mask_single, mask_multi, nums_multi, take_single_cluster, split_multi),
                                       (1,0,2) )

                # Read out coordinates of trigered detectors
                channels = r_data[-1].astype(np.int32)
                tr_coord = [ [coordinates[j,i,channels[ev_starts[i]:ev_starts[i+1]]] for i in range(len(ev_starts)-1)] for j in range(3) ]
                tr_coord = np.array( [ np.concatenate( tr_coord[i], axis=0 ) for i in range(len(tr_coord)) ]  )
                r_data = np.concatenate( (r_data,tr_coord), axis=0 )

                # Exclude hits with big time residuals - known bug
                if exclude_big_ts:
                    mask = np.logical_and(np.abs(r_data[1]) <= t_threshold, ~np.isnan(r_data[1,:]))
                    r_data = r_data[:, mask]
                    shifts = np.array([ np.sum( ~mask[ev_starts[i]:ev_starts[i+1]] ) for i in range(len(ev_starts)-1) ])
                    shifts = np.cumsum(shifts)
                    ev_starts[1:] = ev_starts[1:]-shifts
                
                # Calculate t_res
                resp_muons_prty = np.array([rf[p].array(library="np")[st:] for p in pathes_resp_muons], dtype=object)
                mu_scalar = np.array([rf[p].array(library="np")[st:].astype(np.float64) for p in path_mu_scalar]).T
                resp_muons_prty[-2] += mu_scalar[:, 0]
                resp_muons_prty = cast_to_single(resp_muons_prty.T, mask_single, mask_multi, nums_multi, take_single_cluster, split_multi).T
                # Convert to radians
                resp_muons_prty[:2] = resp_muons_prty[:2] / 180 * np.pi
                # Prepare required arrays
                r_resp_muons_prty = np.array([np.concatenate(rmp, axis=0) for rmp in resp_muons_prty]).astype(np.float64)
                om_coords = np.concatenate([ np.tile( r_data[-3:,ev_starts[i]:ev_starts[i+1]], (1,num_resp_mu[i]) ) for i in range(len(ev_starts)-1) ], axis=1)
                reps_mu = np.repeat(ev_starts[1:]-ev_starts[:-1], num_resp_mu, axis=0)
                mus = np.repeat(r_resp_muons_prty, reps_mu, axis=1)
                t_det = np.concatenate([ np.tile( r_data[1,ev_starts[i]:ev_starts[i+1]], (num_resp_mu[i]) ) for i in range(len(ev_starts)-1) ], axis=0)
                t_res = eval_tres(mus.T, om_coords.T, t_det, ev_starts, mu_starts)
                
                # Make so that avg time is zero
                if center_times:
                    ts_sum = np.array([ np.sum( r_data[1,ev_starts[i]:ev_starts[i+1]] ) for i in range(len(ev_starts)-1) ])
                    ts_avg = ts_sum/(ev_starts[1:]-ev_starts[:-1])
                    r_data[1] = np.concatenate([r_data[1, ev_starts[i]:ev_starts[i+1]] - ts_avg[i] for i in range(len(ev_starts)-1)])
                
                # Now shift coordinates to center. Not earlier - important for tres cal
                if shift_coords_to_cl_center:
                    cl_centers_broad = np.expand_dims( np.repeat( np.transpose(cl_centers, axes=(1,0)), 288, axis=1), axis=1 )
                    coordinates += -cl_centers_broad
                    # recalculate coordinates
                    channels = r_data[3].astype('int')
                    tr_coord = [ [coordinates[j,i,channels[ev_starts[i]:ev_starts[i+1]]] for i in range(len(ev_starts)-1)] for j in range(3) ]
                    tr_coord = np.array( [ np.concatenate( tr_coord[i], axis=0 ) for i in range(len(tr_coord)) ]  )
                    r_data[-3:] = tr_coord

                # Define results
                # Get number of unique string
                sig_strings = r_data[3] // 36
                un_strings = [ set(sig_strings[ev_starts[i]:ev_starts[i+1]]) for i in range(len(ev_starts)-1)  ]
                res_dict['raw/num_un_strings'] = np.array([ len(s) for s in un_strings ]).astype('int')
                # Make event ids
                ev_ids = np.arange(rf['Events/BEvent./BEvent.fPulseN'].num_entries-st)
                if split_multi:
                    ids_multi = ev_ids[mask_multi]
                    ids_multi = np.repeat( ids_multi, nums_multi )
                if take_single_cluster:
                    ev_ids = ev_ids[mask_single]
                    if split_multi:
                        ev_ids = np.concatenate( (ev_ids,ids_multi) )
                ev_ids = np.array([ f"{particle}_{id_prefix}_{str(int(ev_id))}" for ev_id in ev_ids ]).astype(np.string_)
                res_dict['ev_ids'] = ev_ids
                # Primary particle properties
                prime_prty = np.array([ rf[p].array(library="np")[st:] for p in pathes_primary ])
                res_dict['prime_prty'] = cast_to_single(np.transpose( prime_prty, (1,0) ), 
                                                 mask_single, mask_multi, nums_multi, take_single_cluster, split_multi)
                # Data
                res_dict['raw/cluster_ids'] = cluster_ids
                res_dict['raw/data'] = np.transpose( r_data[[0,1,4,5,6]], (1,0) )
                res_dict['raw/labels'] = r_data[2]
                res_dict['raw/channels'] = r_data[3]
                res_dict['raw/ev_starts'] = ev_starts
                res_dict['raw/t_res'] = t_res
                # Muons
                res_dict['muons_prty/aggregate'] = cast_to_single(mu_scalar, mask_single, mask_multi, nums_multi, take_single_cluster, split_multi)
                res_dict['muons_prty/individ'] = np.transpose( r_resp_muons_prty, (1,0) )
                res_dict['muons_prty/mu_starts'] = mu_starts
                
                result_q.put((True, res_dict, id_prefix))
        
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error processing file {rf_path}: {str(e)}")
            result_q.put((False, None, id_prefix))
    
    result_q.put(None)

# run files processing
def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Data processing started")

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

    cl_centers = None
    root_files = [f for f in os.listdir(MC_dir_path) if f.endswith('.root')]

    for root_file in root_files:
        rf_path = os.path.join(MC_dir_path, root_file)
        if cl_centers is None:
            cl_centers = get_cl_centers(rf_path)
            logging.info("Cluster centers calculated")
        id_prefix = root_file.split(".")[0]
        args_q.put((rf_path, id_prefix, cl_centers))

    # Signal workers to terminate
    for _ in range(NUM_WORKERS):
        args_q.put(None)

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    # Wait for listener to finish
    do_quit.wait()
    listener_process.join()

    # Write clusters coordinates to file
    with h5.File(os.path.join(h5_prefix, h5_name), 'a') as hf:
        hf.create_dataset(f"{particle}/clusters_centers/data", data=cl_centers)
        hf.create_dataset(f"{particle}/coords_are_cluster_centered/data", data=shift_coords_to_cl_center)

    logging.info("Cluster information written to file")

    logging.info("Data processing completed")

if __name__ == "__main__":
    main()
