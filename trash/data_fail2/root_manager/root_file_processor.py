import logging
import traceback
from typing import Tuple, List, Dict, Optional

import numpy as np
import uproot as ur
import awkward as ak

from data_fail2.root_manager.constants import Constants as Cnst
from data_fail2.root_manager.utils import eval_tres, get_start_index
#from config import Config


class RootFileProcessor:
    def __init__(self, config: Dict):
        self.coords_are_same = config['general']['coords_are_same']
        self.take_single_cluster = config['general']['take_single_cluster']
        self.take_clust_num = config['general']['take_clust_num']
        self.split_multi = config['general']['split_multi']
        self.shift_coords_to_cl_center = config['general']['shift_coords_to_cl_center']
        self.center_times = config['general']['center_times']
        self.exclude_big_ts = config['general']['exclude_big_ts']
        self.t_threshold = float(config['general']['t_threshold'])
        
        self.particle = config['input']['particle']
        self.MC_dir_path = config['input']['MC_dir_path']
        
        self.pathes_data = config['root_paths']['data']
        self.pathes_primary = config['root_paths']['primary']
        self.pathes_resp_muons = config['root_paths']['resp_muons']
        self.path_mu_scalar = config['root_paths']['mu_scalar']
        self.path_geometry = config['root_paths']['geometry']
        

    @staticmethod
    def _check_file(rf_path: str) -> bool:
        try:
            with ur.open(rf_path) as rf:
                ev_num = rf['Events/BEvent./BEvent.fPulseN'].num_entries
                return ev_num > 1 or (ev_num == 1 and rf['Events/BEvent./BEvent.fPulseN'].array(library="np", entry_start=0, entry_stop=1)[0] != 0)
        except Exception:
            return False

    # Get mask for single cluster events
    @staticmethod
    def _get_single_mask(active_clusters, num_un_clusters, take_clust_num):
        mask = num_un_clusters == 1
        active_cluster_id = np.array([ac[0] for ac in active_clusters])

        if take_clust_num is not None:
            mask &= active_cluster_id == take_clust_num

        return mask, active_cluster_id[mask]


    # Get mask for multi cluster events
    @staticmethod
    def _get_multi_mask(active_clusters, num_un_clusters):
        mask = num_un_clusters != 1
        return mask, [ac for ac, m in zip(active_clusters, mask) if m]
    
    # Splits data for multi cluster events into single clusters
    @staticmethod
    def _flatten_multi(datas, channels, cluster_ids_multi):
        ress = []
        for data in datas:
            res = []
            # iterate over events
            for (d,cls_ids,chs) in zip(data,cluster_ids_multi,channels):
                for cl_id in cls_ids:
                    cl_mask = (chs//Cnst.CHANNEL_DIVISOR)==cl_id
                    res.append(d[cl_mask])
            ress.append(res)
        return ress

    # Transforms input array so that data for individual clusters can be extracted. Required for spliting.
    @staticmethod
    def _cast_to_single(data, mask_single, mask_multi, 
                    nums_multi, take_single_cluster, 
                    split_multi):
        
        result = []
        if take_single_cluster:
            result.append(data[mask_single])
        
        if split_multi:
            result.append(np.repeat(data[mask_multi], nums_multi, axis=0))
        
        return np.concatenate(result) if result else data

    def process_root_file(self, rf_path: str, id_prefix: str, cl_centers: Optional[np.ndarray] = None) -> Dict:
        try:
            # Check if file has data
            if not self._check_file(rf_path):
                logging.info(f"File {rf} has no data.")
                return  # Skip empty files
            with ur.open(rf_path) as rf:
                # init dict for out data
                res_dict = {}
                
                # get first index
                st = get_start_index(rf)
                
                # Read channels and identify clusters
                num_channels = rf['Events/BEvent./BEvent.fPulseN'].array(library="np")[st:]
                channels = rf['Events/BEvent./BEvent.fPulses/BEvent.fPulses.fChannelID'].array(library='np')[st:]
                active_clusters = [np.unique(ch // Cnst.CHANNEL_DIVISOR) for ch in channels]
                num_un_clusters = np.array([len(cl) for cl in active_clusters])
                             
                # First, get masks and identify clusters
                if self.take_single_cluster:
                    mask_single, cluster_ids = self._get_single_mask(active_clusters, num_un_clusters, self.take_clust_num)
                    mask_multi, cluster_ids_multi = self._get_multi_mask(active_clusters, num_un_clusters)
                    nums_multi = num_un_clusters[mask_multi]
                    if self.split_multi:
                        cluster_ids_multi_flat = np.array([ cl for clss in cluster_ids_multi for cl in clss  ])
                        cluster_ids = np.concatenate( (cluster_ids,cluster_ids_multi_flat), axis=0 )
                
                # Read data
                data = [rf[p].array(library="np")[st:] for p in self.pathes_data]
                num_resp_mu = rf['Events/BMCEvent./BMCEvent.fResponseMuonsN'].array(library="np").astype('int')[st:]
                
                # Get data for multicluster events before modifying variables
                if self.split_multi:    
                    data_multi = [d[mask_multi] for d in data]
                    data_multi = self._flatten_multi(data_multi, channels[mask_multi], cluster_ids_multi)
                    data_multi = [np.array(d, dtype=object) for d in data_multi]
                    num_channels_multi = np.array([ len(chs) for chs in data_multi[-1] ])
                    # for muons, we need to keep tham all for multicluster events 
                    num_resp_mu_multi = self._cast_to_single(num_resp_mu, mask_single, mask_multi, nums_multi, self.take_single_cluster, self.split_multi)
                
                # Now extract data for single events
                if self.take_single_cluster:
                    data = [ d[mask_single] for d in data ]
                    num_channels = num_channels[mask_single]
                    num_resp_mu = num_resp_mu[mask_single]
                    # And extend it for multi-cluster events
                    if self.split_multi:
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
                coordinates = np.array(ak.unzip(rf[self.path_geometry].array()))[:,st:]
                if self.coords_are_same:
                    coordinates = np.repeat( coordinates, channels.shape[0], axis=1 )
                coordinates = np.transpose( self._cast_to_single( np.transpose(coordinates, (1,0,2)), mask_single, mask_multi, nums_multi, self.take_single_cluster, self.split_multi),
                                       (1,0,2) )

                # Read out coordinates of trigered detectors
                channels = r_data[-1].astype(np.int32)
                tr_coord = [ [coordinates[j,i,channels[ev_starts[i]:ev_starts[i+1]]] for i in range(len(ev_starts)-1)] for j in range(3) ]
                tr_coord = np.array( [ np.concatenate( tr_coord[i], axis=0 ) for i in range(len(tr_coord)) ]  )
                r_data = np.concatenate( (r_data,tr_coord), axis=0 )

                # Exclude hits with big time residuals - known bug
                if self.exclude_big_ts:
                    mask = np.logical_and(np.abs(r_data[1]) <= self.t_threshold, ~np.isnan(r_data[1,:]))
                    r_data = r_data[:, mask]
                    shifts = np.array([ np.sum( ~mask[ev_starts[i]:ev_starts[i+1]] ) for i in range(len(ev_starts)-1) ])
                    shifts = np.cumsum(shifts)
                    ev_starts[1:] = ev_starts[1:]-shifts
                
                # Calculate t_res
                resp_muons_prty = np.array([rf[p].array(library="np")[st:] for p in self.pathes_resp_muons], dtype=object)
                mu_scalar = np.array([rf[p].array(library="np")[st:].astype(np.float64) for p in self.path_mu_scalar]).T
                resp_muons_prty[-2] += mu_scalar[:, 0]
                resp_muons_prty = self._cast_to_single(resp_muons_prty.T, mask_single, mask_multi, nums_multi, self.take_single_cluster, self.split_multi).T
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
                if self.center_times:
                    ts_sum = np.array([ np.sum( r_data[1,ev_starts[i]:ev_starts[i+1]] ) for i in range(len(ev_starts)-1) ])
                    ts_avg = ts_sum/(ev_starts[1:]-ev_starts[:-1])
                    r_data[1] = np.concatenate([r_data[1, ev_starts[i]:ev_starts[i+1]] - ts_avg[i] for i in range(len(ev_starts)-1)])
                
                # Now shift coordinates to center. Not earlier - important for tres cal
                if self.shift_coords_to_cl_center:
                    cl_centers_broad = np.expand_dims( np.repeat( np.transpose(cl_centers, axes=(1,0)), Cnst.CHANNEL_DIVISOR, axis=1), axis=1 )
                    coordinates += -cl_centers_broad
                    # recalculate coordinates
                    channels = r_data[3].astype('int')
                    tr_coord = [ [coordinates[j,i,channels[ev_starts[i]:ev_starts[i+1]]] for i in range(len(ev_starts)-1)] for j in range(3) ]
                    tr_coord = np.array( [ np.concatenate( tr_coord[i], axis=0 ) for i in range(len(tr_coord)) ]  )
                    r_data[-3:] = tr_coord

                # Define results
                # Get number of unique string
                sig_strings = r_data[3] // Cnst.STRING_DIVISOR
                un_strings = [ set(sig_strings[ev_starts[i]:ev_starts[i+1]]) for i in range(len(ev_starts)-1)  ]
                res_dict['raw/num_un_strings'] = np.array([ len(s) for s in un_strings ]).astype('int')
                # Make event ids
                ev_ids = np.arange(rf['Events/BEvent./BEvent.fPulseN'].num_entries-st)
                if self.split_multi:
                    ids_multi = ev_ids[mask_multi]
                    ids_multi = np.repeat( ids_multi, nums_multi )
                if self.take_single_cluster:
                    ev_ids = ev_ids[mask_single]
                    if self.split_multi:
                        ev_ids = np.concatenate( (ev_ids,ids_multi) )
                ev_ids = np.array([ f"{self.particle}_{id_prefix}_{str(int(ev_id))}" for ev_id in ev_ids ]).astype(np.string_)
                
                
                res_dict['ev_ids'] = ev_ids
                # Primary particle properties
                prime_prty = np.array([ rf[p].array(library="np")[st:] for p in self.pathes_primary ])
                res_dict['prime_prty'] = self._cast_to_single(np.transpose( prime_prty, (1,0) ), 
                                                 mask_single, mask_multi, nums_multi, self.take_single_cluster, self.split_multi)
                # Data
                res_dict['raw/cluster_ids'] = cluster_ids
                res_dict['raw/data'] = np.transpose( r_data[[0,1,4,5,6]], (1,0) )
                res_dict['raw/labels'] = r_data[2]
                res_dict['raw/channels'] = r_data[3]
                res_dict['raw/ev_starts'] = ev_starts
                res_dict['raw/t_res'] = t_res
                # Muons
                res_dict['muons_prty/aggregate'] = self._cast_to_single(mu_scalar, mask_single, mask_multi, nums_multi, self.take_single_cluster, self.split_multi)
                res_dict['muons_prty/individ'] = np.transpose( r_resp_muons_prty, (1,0) )
                res_dict['muons_prty/mu_starts'] = mu_starts
                
                return (True, res_dict, id_prefix)
                #result_q.put((True, res_dict, id_prefix))
        
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error processing file {rf_path}: {str(e)}")
            return (False, None, id_prefix)