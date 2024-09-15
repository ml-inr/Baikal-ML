import numpy as np
import h5py as h5
import random as rd
import logging
from typing import Tuple, List, Dict

file_prefix = '/home3/ivkhar/Baikal/data/'

# configuration
config = {
    # files
    "h5_in_name": 'baikal_2020_flat.h5',
    "h5_out_postfix": '',
    # cuts and filters
    "take_signal_hits": False,
    "take_reco_hits": False,
    "take_reco_adds": False,
    "impose_reco_cuts": False,
    "cut_on_hits": 0,
    "cut_on_strings": 0,
    "apply_low_Q_filter": False,
    "Q_low_limit": 0,
    "apply_hi_Q_filter": False,
    "Q_hi_limit": 100,
    # NN filtering
    "take_NN_filt_hits": False,
    "h5_nn_preds": '/home/ivkhar/Baikal/data/baikal_preds_mu-nu-sep_small-test.h5',
    "nn_h5_preds_folder": 'trespen',
    "nn_class_threshold": 0.9,
    "take_parts_from_preds": False,
    # particles, events_per_file * num_files
    # muatm: 33 000 * 20004, nuatm: 40 000 * 2000, nue2: 24 000 * 400 
    "particles": ['muatm', 'nuatm', 'nue2'],
    "limit_num_parts": {'muatm': 400, 'nuatm': 260, 'nue2': 100},
    "sample_parts": True,
    # which data to pull
    "other_keys": ['ev_ids', 'prime_prty'], # full: ['ev_ids', 'prime_prty', 'reco/global']
    "data_keys": ['data', 'labels', 't_res', 'channels', 'num_un_strings'], # full: ['data','labels','channels','num_un_strings','t_res']   
    "mus_keys": [], # full ['muons_prty/' + w for w in ['individ', 'aggregate']]
    # exclude parts
    "exclude_parts": {
        'muatm': [],
        'nuatm': [],
        'nue2': []
        }
}

assert ( not (config['take_signal_hits'] and config['take_NN_filt_hits']) ), 'Choose only one method to take signal hits.'

config['prefix'] = 'reco/' if config['take_reco_hits'] else 'raw/'
config['data_keys'] = [config['prefix'] + k for k in config['data_keys']]
config['keys_to_pull'] = config['data_keys'] + ['raw/ev_starts'] + config['mus_keys'] + config['other_keys']
if 'muons_prty/individ' in config['mus_keys']:
    config['mus_keys'] += ['muons_prty/mu_starts']
config['h5_in'] = f"{file_prefix}h5s/{config['h5_in_name']}"
config['h5_out'] = f"{file_prefix}filtered/{config['h5_in_name'][:-3]}{config['h5_out_postfix']}.h5"
with open(f"{config['h5_out'][:-3]}.config",'w') as f:
    for key, value in config.items():
        f.write('%s:%s\n' % (key, value))

# constants
CHANNELS_PER_STRING = 36

# aplly reconstruction-based cuts
def reco_cut(gl_reco: np.ndarray) -> np.ndarray:
    return (gl_reco[:, 19] == 3) & (gl_reco[:, 20] < 20) & (gl_reco[:, 21] / (gl_reco[:, 17] - 5) < 30)

# creating mask for reading the required data
def make_masks(hf: h5.File, part: str, particle: str, ev_starts: np.ndarray, ev_lens: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ev_lens_init = ev_lens
    # make events and hits mask, init as True
    mask_evs = np.full(ev_starts.shape[0] - 1, True)
    mask_hits = np.full(hf[f"{particle}/{config['prefix']}/data/{part}/data"].shape[0], True)
    # init num un strings
    num_un_strings = hf[f"{particle}/{config['prefix']}/num_un_strings/{part}/data"][()]
    ## hit-wise cuts
    # cut on signal or filtered hits
    if config['take_signal_hits'] or config['take_NN_filt_hits'] or config['apply_low_Q_filter']:
        pass_hits = np.full(hf[f"{particle}/raw/labels/{part}/data"].shape[0:1], True)
        if config['take_signal_hits']:
            pass_hits &= hf[f"{particle}/raw/labels/{part}/data"][()] != 0
        if config['take_NN_filt_hits']:
            with h5.File(config['h5_nn_preds'], 'r') as hp:
                nn_preds = hp[f"{config['h5_nn_preds_folder']}/{particle}/{part}/preds"][..., 0]
                pass_hits &= nn_preds >= config['nn_class_threshold']
        if config['apply_low_Q_filter']:
            pass_hits &= hf[f"{particle}/raw/data/{part}/data"][..., 0] >= config['Q_low_limit']
        mask_hits &= pass_hits
        # recalculate num_un_strings
        tr_chs = hf[f"{particle}/{config['prefix']}/channels/{part}/data"][()]
        tr_chs = np.where(mask_hits, tr_chs, -1)
        sig_strings = tr_chs // CHANNELS_PER_STRING
        sig_strings = [ sig_strings[ev_starts[i]:ev_starts[i+1]] for i in range(ev_starts.shape[0] - 1) ]
        un_strings = [np.unique(s) for s in sig_strings]
        num_un_strings = np.array([np.sum(s > 0) for s in un_strings])
        # recalculate ev lens
        ev_lens = np.array([np.sum(mask_hits[ev_starts[i]:ev_starts[i+1]]) for i in range(ev_starts.shape[0] - 1)])
    ## event-wise cuts
    # cut on global reco params
    if config['impose_reco_cuts']:
        gl_reco = hf[f"{particle}/reco/global/{part}/data"][()]
        mask_evs &= reco_cut(gl_reco)
    # cut on number of hits
    if config['cut_on_hits'] != 0:
        mask_evs &= ev_lens >= config['cut_on_hits']
    # cut on number of strings
    if config['cut_on_strings'] >= 1:
        mask_evs &= num_un_strings >= config['cut_on_strings']
    # mask hits
    mask_evs_to_hits = np.repeat(mask_evs, ev_lens_init)
    mask_hits &= mask_evs_to_hits
    # recalculate ev lens
    ev_lens = np.array([np.sum(mask_hits[ev_starts[i]:ev_starts[i+1]]) for i in range(ev_starts.shape[0] - 1) if mask_evs[i]])

    return mask_evs, mask_hits, num_un_strings, ev_lens

# process dataset
def process_dataset(hf: h5.File, ho: h5.File, particle: str, part: str, ds: str, mask_evs: np.ndarray, mask_hits: np.ndarray, num_un_strings: np.ndarray, ev_starts: np.ndarray) -> None:
    kwargs = {}
    if 'raw' in ds or 'reco' in ds:
        kwargs['dtype'] = np.int32 if any(key in ds for key in ['starts', 'num_un_strings', 'cluster_ids', 'channels', 'labels']) else np.float32
        if any(key in ds for key in ['channels','labels','data']):
            kwargs['compression'] = 'gzip'
        ## prepare data to write
        # special cases
        if 'num_un_strings' in ds:
            n_data = num_un_strings[mask_evs]
        elif 'ev_starts' in ds:
            n_data = ev_starts
        # datasets pulled by event mask
        elif 'global' in ds or 'cluster_ids' in ds:
            n_data = hf[f"{particle}/{ds}/{part}/data"][()][mask_evs]
        # datasets pulled by hits mask
        else:
            # concat with reco adds if requested
            # !!! RECO ADDS NOT TESTED AT ALL !!!
            if config['take_reco_adds'] and ('reco/data' in ds):
                o_data = np.concatenate(
                    (hf[f"{particle}/{ds}/{part}/data"][()],
                    hf[f"{particle}/reco/add/{part}/data"][()]),
                    axis=-1)
            else:
                o_data = hf[f"{particle}/{ds}/{part}/data"][()].astype(kwargs['dtype'])
            if 'data' in ds and config['apply_hi_Q_filter']:
                o_data[:, 0] = np.minimum(o_data[:, 0], config['Q_hi_limit'])
            n_data = o_data[mask_hits]
    # other keys
    elif 'ev_ids' in ds:
        kwargs['dtype'] = hf[f"{particle}/{ds}/{part}/data"][()].dtype
        n_data = hf[f"{particle}/{ds}/{part}/data"][()][mask_evs]
    elif 'prime_prty' in ds:
        kwargs['dtype'] = np.float32
        n_data = hf[f"{particle}/{ds}/{part}/data"][()][mask_evs]
    # muons
    elif 'muons' in ds:
        kwargs['dtype'] = np.int32 if 'starts' in ds else np.float32
        if 'starts' not in ds:
            kwargs['compression'] = 'gzip'
        mu_starts = hf[f"{particle}/muons_prty/mu_starts/{part}/data"][()]
        mu_lens_init = np.diff(mu_starts)
        mu_lens = mu_lens_init[mask_evs]
        mu_starts = np.concatenate(([0], np.cumsum(mu_lens)))
        if 'starts' in ds:
            n_data = mu_starts
        elif 'individ' in ds:
            mask_evs_to_ind_mus = np.repeat(mask_evs, mu_lens_init)
            n_data = hf[f"{particle}/{ds}/{part}/data"][()].astype(kwargs['dtype'])[mask_evs_to_ind_mus]
        elif 'aggregate' in ds:
            n_data = hf[f"{particle}/{ds}/{part}/data"][()].astype(kwargs['dtype'])[mask_evs]
        else:
            raise ValueError(f"Undefined dataset {ds}")
    else:
        raise ValueError(f"Undefined dataset {ds}")

    rs = ds if 'muons' in ds else 'reco_global' if 'global' in ds else ds.split('/')[-1]
    ho.create_dataset(f"{particle}/{rs}/{part}/data", data=n_data, **kwargs)

def main() -> None:
    """Main function to process particle physics data."""
    logging.info("Data filtering started")
    
    with h5.File(config['h5_in'], 'r') as hf, h5.File(config['h5_out'], 'w') as ho:
        for particle in config['particles']:
            logging.info(f"Processing particle: {particle}")
            # read subfolders
            if config['take_parts_from_preds']:
                with h5.File(config['h5_nn_preds'], 'r') as hpr:
                    parts = [p for p in list(hpr[f"{config['nn_h5_preds_folder']}/{particle}"].keys()) if p.startswith('p')]
            else:
                parts = list(hf[f"{particle}/{config['data_keys'][0]}"].keys())
            # exclude bad parts
            parts = [p for p in parts if p.split('_')[1] not in config["exclude_parts"][particle] ]
            # take only a subset
            if config['limit_num_parts'][particle] is not None:
                parts = rd.sample(parts, k=config['limit_num_parts'][particle]) if config['sample_parts'] else parts[:config['limit_num_parts'][particle]]
            # loop over parts
            for part in parts:
                ev_starts = hf[f"{particle}/{config['prefix']}/ev_starts/{part}/data"][()].astype(np.int64)
                ev_lens = np.diff(ev_starts)
                mask_evs, mask_hits, num_un_strings, ev_lens = make_masks(hf, part, particle, ev_starts, ev_lens)
                ev_starts = np.concatenate(([0], np.cumsum(ev_lens)))

                for ds in config['keys_to_pull']:
                    process_dataset(hf, ho, particle, part, ds, mask_evs, mask_hits, num_un_strings, ev_starts)

    logging.info("Data processing completed")

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    main()