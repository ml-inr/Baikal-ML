import numpy as np
import h5py as h5
import random as rd

file_prefix = '/home3/ivkhar/Baikal/data/'

# configuration
config = {
    "h5_in_name": 'baikal_2020_flat_mid-eq.h5',
    "h5_out_name": 'baikal_2020_sig-noise_mid-eq_normed.h5',
    "keys_to_pull": ['ev_starts','data','labels','t_res','channels','ev_ids','num_un_strings','prime_prty'],
    "do_pull_mus": False,
    "mus_to_pull": ['muons_prty/mu_starts', 'muons_prty/aggregate', 'muons_prty/individ'],
    "particles": ['muatm','nuatm','nue2'],
    "dsets": ['train','test','val'],
    # number of files to use for test and val sets
    # events_per_file * num_files:
    # muatm: 33 000 * 20004, nuatm: 40 000 * 2000, nue2: 24 000 * 400 
    "num_files_to_test": {'muatm':5, 'nuatm': 0, 'nue2':0},
    "num_files_to_val": {'muatm':5, 'nuatm': 0,'nue2':0},
    "num_files_to_train": {'muatm':200, 'nuatm': 0,'nue2':0},
    "norm_name": '',
}

# needed to shuffle train
# 100 <--> ~50% mem load
num_files_in_group = 30

# constants
keys_evs = ['global','cluster_ids','ev_ids','num_un_strings','prime_prty','muons_prty/aggregate']
keys_hits = ['data', 'labels' ,'channels' ,'t_res' ,'prime_prty']
keys_mus = ['muons_prty/individ']

if config["do_pull_mus"]:
    config["keys_to_pull"] += config["mus_to_pull"]
with open(f"{file_prefix}normed/{config['h5_out_name'][:-3]}.config",'w') as f:
    for key, value in config.items():
        f.write('%s:%s\n' % (key, value))

with h5.File(f"{file_prefix}filtered/{config['h5_in_name']}",'r') as hf:
    ### define train, test, val sets
    # and get numbers of events
    parts_dict = {'train':[], 'test':[], 'val':[]} # {dset:files in}
    nums_dict = {'train':{}, 'test':{}, 'val':{}} # {dset:{part:num}}
    group_files = {'train':[], 'test':[], 'val':[]} # {dset:num}
    total_all = {'train':{}, 'test':{}, 'val':{}} # {dset:num}
    for key_dict in total_all.keys():
        for key_data in config['keys_to_pull']:
            total_all[key_dict]['evs'] = 0
            total_all[key_dict]['hits'] = 0
            if config['do_pull_mus']:
                total_all[key_dict]['mus'] = 0
    for particle in config['particles']:
        parts = list(hf[particle+'/data'].keys())
        parts = rd.sample( parts, k=len(parts) )
        # assert that we have enough data
        assert len(parts)>=(config['num_files_to_test'][particle]+config['num_files_to_val'][particle]+config['num_files_to_train'][particle])
        # fill dics
        if config['num_files_to_val'][particle]!=0:
            parts_dict['val'] += [(particle,p) for p in parts[-config['num_files_to_val'][particle]:]]
        if config['num_files_to_test'][particle]!=0:
            parts_dict['test'] += [(particle,p) for p in parts[-config['num_files_to_val'][particle]-config['num_files_to_test'][particle]:-config['num_files_to_val'][particle]]]           
        if config['num_files_to_train'][particle]!=0:
            parts_dict['train'] += [(particle,p) for p in parts[:config['num_files_to_train'][particle]]]       
    # shuffle train to get arbitrary order of mus and nus
    parts_dict['train'] = rd.sample(parts_dict['train'], k=len(parts_dict['train']))
    # get number of events in files
    for ds in config['dsets']:
        for part in parts_dict[ds]:
            nums_dict[ds][part] = {}
            num_evs = hf[part[0]+'/ev_starts/'+part[1]+'/data'].shape[0]-1
            num_flat_hits = hf[part[0]+'/data/'+part[1]+'/data'].shape[0]
            nums_dict[ds][part]['evs'] = num_evs
            nums_dict[ds][part]['hits'] = num_flat_hits
            total_all[ds]['evs'] += num_evs
            total_all[ds]['hits'] += num_flat_hits
            if config['do_pull_mus']:
                num_flat_mus = hf[part[0]+'/muons_prty/individ/'+part[1]+'/data'].shape[0]
                total_all[ds]['mus'] += num_flat_mus
    # split parts in groups for convinience
    for ds in config['dsets']:
        le = (len(parts_dict[ds])%num_files_in_group)
        if le==0:
            num_groups = len(parts_dict[ds])//num_files_in_group
        else:
            num_groups = len(parts_dict[ds])//num_files_in_group +1
        for j in range(num_groups):
            parts = parts_dict[ds][j*num_files_in_group:(j+1)*num_files_in_group]
            group_files[ds].append(parts)
    ### normalization
    with h5.File(f"{file_prefix}normed/{config['h5_out_name']}",'w') as ho:
        # get mean, std
        mean = hf['norm_param_global/mean'+config['norm_name']][()].astype(np.float32)
        std = hf['norm_param_global/std'+config['norm_name']][()].astype(np.float32)
        # loop over train/test/val
        for ds in config['dsets']: 
            # declare data datasets 
            h_dsets = {}
            for key in config['keys_to_pull']:
                part_t =  list(hf[config['particles'][0]+'/data'].keys())[0]
                # print(config['particles'][0]+'/'+key+'/'+part_t+'/data')
                shape = hf[config['particles'][0]+'/'+key+'/'+part_t+'/data'].shape[1:]
                if 'start' in key:
                    d_dtype = np.int64
                else:
                    d_dtype = hf[config['particles'][0]+'/'+key+'/'+part_t+'/data'].dtype
                # dsets: shape[0] = evs num
                if any([k in key for k in keys_evs]):
                    h_dsets[key] = ho.create_dataset(ds+'/'+key+'/data', shape=np.concatenate(([total_all[ds]['evs']],shape), axis=0), dtype=d_dtype, chunks=True)
                elif any([k in key for k in keys_hits]):
                    h_dsets[key] = ho.create_dataset(ds+'/'+key+'/data', shape=np.concatenate(([total_all[ds]['hits']],shape), axis=0), dtype=d_dtype, chunks=True)
                elif any([k in key for k in keys_mus]):
                    h_dsets[key] = ho.create_dataset(ds+'/'+key+'/data', shape=np.concatenate(([total_all[ds]['mus']],shape), axis=0), dtype=d_dtype, chunks=True)
                elif 'ev_starts' in key: 
                    h_dsets[key] = ho.create_dataset(ds+'/'+key+'/data', shape=np.concatenate(([total_all[ds]['evs']+1],shape), axis=0), dtype=d_dtype, chunks=True)
                    h_dsets[key][0] = 0
                elif 'mu_starts' in key: 
                    h_dsets[key] = ho.create_dataset(ds+'/'+key+'/data', shape=np.concatenate(([total_all[ds]['evs']+1],shape), axis=0), dtype=d_dtype, chunks=True)
                    h_dsets[key][0] = 0
            # read data by groups
            # shuffle, if train
            n = {}
            last_hit_idx = 0
            if config['do_pull_mus']:
                last_mu_idx = 0
            for key in config['keys_to_pull']:
                if 'starts' in key:
                    n[key] = 1
                else:
                    n[key] = 0
            for parts in group_files[ds]:
                # read and concat
                # init temp data dict
                t_data = {}
                for key in config['keys_to_pull']:
                    # ev starts will be ev lens
                    t_data[key] = []
                for part in parts:
                    # data
                    for key in config['keys_to_pull']:
                        if key=='data':
                            norm_data = (hf[part[0]+'/'+key+'/'+part[1]+'/data'][()]-mean)/std
                            t_data[key].append( norm_data )
                        elif key=='ev_starts' or key=='muons_prty/mu_starts':
                            t_data[key].append( np.diff( hf[part[0]+'/'+key+'/'+part[1]+'/data'][()] ) )
                        else:
                            t_data[key].append( hf[part[0]+'/'+key+'/'+part[1]+'/data'][()] )
                # write data
                # shuffle
                num_ev_group = np.sum([ t_d.shape[0] for t_d in t_data['ev_starts'] ])
                # get evs starts
                ev_lens = np.concatenate( (t_data['ev_starts']), axis=0 )
                part_ev_starts = np.concatenate( ([0],np.cumsum(ev_lens)) )
                if config['do_pull_mus']:
                    mu_lens = np.concatenate( (t_data['muons_prty/mu_starts']), axis=0 )
                    part_mu_starts = np.concatenate( ([0],np.cumsum(mu_lens)) )
                if ds=='train':
                    idxs_shuffled = rd.sample(range(num_ev_group), k=num_ev_group)
                    ev_lens = ev_lens[idxs_shuffled]
                    if config['do_pull_mus']:
                        mu_lens = mu_lens[idxs_shuffled]
                ev_starts = np.cumsum(ev_lens.astype(np.int64)) + last_hit_idx
                last_hit_idx = ev_starts[-1]
                if config['do_pull_mus']:
                    mu_starts = np.cumsum(mu_lens) + last_mu_idx
                    last_mu_idx = mu_starts[-1]
                # read data
                for key in config['keys_to_pull']:
                    data = np.concatenate(t_data[key], axis=0)
                    if key=='ev_starts':
                        data = ev_starts
                    if key=='muons_prty/mu_starts':
                        data = mu_starts
                    if ds=='train':
                        if any([k in key for k in keys_evs]):
                            # print(key)
                            data = data[idxs_shuffled]
                        elif any([k in key for k in keys_hits]):
                            data = np.concatenate( [ data[part_ev_starts[i]:part_ev_starts[i+1]] for i in idxs_shuffled ], axis=0 )
                        elif any([k in key for k in keys_mus]):
                            data = np.concatenate( [ data[part_mu_starts[i]:part_mu_starts[i+1]] for i in idxs_shuffled ], axis=0 )
                    h_dsets[key][n[key]:n[key]+data.shape[0]]=data
                    n[key] = n[key]+data.shape[0]
        # write normilizing parameters
        ho.create_dataset('norm_param/mean', data=mean, dtype='float32')
        ho.create_dataset('norm_param/std', data=std, dtype='float32') 