import tensorflow as tf
import numpy as np
import h5py as h5

import matplotlib.pyplot as plt

# GPU memory growth setting
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from sig_noise_generators import make_val_dataset

# Config
h5f = '/home3/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5'
particles = ['muatm','nuatm','nue2']
save_prefix = ''
num_hit_cut = 0
num_string_cut = 0

nns = ['unet_revised_focal','unet_revised_add','encoder_check_standard','encoder_check_focal','unet_62re_focal_allms',
        'unet_62re_entr_allms','encoder_63re3_entr_allms','unet_62_w25_entr_allms','encoder_62_w25_focal', 'encoder_63_focal3']
nns = ['/home/ivkhar/Baikal/models/MC2020/'+nn for nn in nns]
path_save = '/home/ivkhar/Baikal/sig_noise_plots/'

val_batch_size = 1024
set_up_Q_lim = True
up_Q_lim = 100
set_low_Q_lim = False
low_Q_lim = 0

thresholds = np.arange(0.5,1.,0.025)

mean_mode = 'hits' # hits or events

# Get idxs for elements
# Particles must go in slices
def get_particle_slices(h5f, particles):
    particle_evt_idxs = {}
    particle_hit_idxs = {}
    particle_nums = {}
    with h5.File(h5f,'r') as hf:
        ev_ids = hf['val/ev_ids/data'][:]
        for part in particles:
            part_enc = part.encode()
            num_symb = len(part_enc)
            rel_bytes = np.frompyfunc(lambda x: x[:num_symb], 1, 1)(ev_ids)
            coinc = np.nonzero(rel_bytes==part_enc)
            num_ps = len(coinc[0])
            p_start = coinc[0][0]
            p_evt_idxs = np.s_[ p_start:(p_start+num_ps) ]
            p_hit_idxs = np.s_[ hf['val/ev_starts/data'][p_start]:hf['val/ev_starts/data'][p_start+num_ps] ]
            particle_evt_idxs[part] = p_evt_idxs
            particle_hit_idxs[part] = p_hit_idxs
            particle_nums[part] = num_ps
    return particle_evt_idxs, particle_hit_idxs, particle_nums

# Make predictions
def make_preds(nn, h5f, val_dataset, val_bs, lim_num_steps=None):
    model = tf.keras.models.load_model(nn, compile=False)
    with h5.File(h5f,'r') as hf:
        num_steps = (hf['val/ev_starts/data'].shape[0]-1) // val_bs
    if lim_num_steps is not None:
        num_steps = np.minimum( num_steps, lim_num_steps )
    # iterate iver dset
    proc = 0
    flat_preds = []
    flat_y_true = []
    flat_tres = []
    for (data, labels, ws) in val_dataset.__iter__():
        preds = model.predict( data, verbose=0 )[:,:,0]
        mask = tf.cast( data[:,:,-1], bool )
        flat_preds.append( preds[mask] )
        y_true = labels[mask]
        flat_tres.append( y_true[:,-1] )
        flat_y_true.append( y_true[:,0] )
        proc += 1
        if proc >= num_steps:
            break
    flat_preds = np.concatenate(flat_preds)
    flat_tres = np.concatenate(flat_tres)
    flat_y_true = np.concatenate(flat_y_true)
    with h5.File(h5f,'r') as hf:
        event_starts = hf['val/ev_starts/data'][:num_steps*val_bs+1]
    return flat_preds, flat_y_true, flat_tres, event_starts

# get metrics
def metrics_over_thresholds(predictions, labels, tres, ev_starts, thresholds, particles, particle_evt_idxs, particle_hit_idxs, mean_mode,
                                chs_ids, num_hit_cut, num_string_cut):

    assert mean_mode in ['hits','events']
    metric_names = ['accuracy','precision','recall','tres']

    ev_lens = np.diff( ev_starts )
    metrics = {mn: {part: np.zeros(len(thresholds)) for part in particles} for mn in metric_names}
    
    for part in particles:
        
        # restrsict to particle
        hit_idxs = particle_hit_idxs[part]
        ev_idxs = particle_evt_idxs[part]
        p_preds = predictions[hit_idxs]
        p_labels = labels[hit_idxs]
        p_tres = np.abs(tres[hit_idxs])
        p_ev_lens = ev_lens[ev_idxs]
        p_ev_sts = np.concatenate(([0],np.cumsum(p_ev_lens)))
        if chs_ids is not None:
            p_chs_ids = chs_ids[hit_idxs]
        
        if mean_mode=='hits':
            red_array = np.array([0])
        else:
            red_array = p_ev_sts[:-1]
            
        mask_true_signal = p_labels!=0.
        mask_true_noise = ~mask_true_signal
        for i,threshold in enumerate(thresholds):

            # get mask
            ident_sigs = p_preds>=threshold
            num_ident_sigs = np.add.reduceat(ident_sigs, p_ev_sts[:-1])
            pass_evs_nhit = num_ident_sigs>=num_hit_cut
            pass_nsig = np.repeat( pass_evs_nhit, p_ev_lens )
            
            if chs_ids is not None:
                tr_chs = np.where( ident_sigs, p_chs_ids, -1 )
                strings = [ tr_chs[p_ev_sts[i]:p_ev_sts[i+1]] // 36 for i in range(len(p_ev_sts)-1) ]
                #strings = np.array_split(tr_chs // 36, p_ev_sts[1:])
                un_strings = [ np.unique(s) for s in strings]
                num_un_strings = np.array([ np.sum(s>0) for s in un_strings ])
                pass_evs_nstr = num_un_strings>=num_string_cut
                pass_nstr = np.repeat( pass_evs_nstr, p_ev_lens )
                
                idxs_evs = pass_evs_nhit*pass_evs_nstr
                idxs_hits = pass_nsig*pass_nstr

                # reduce, faster
                p_preds = p_preds[idxs_hits]
                p_labels = p_labels[idxs_hits]
                p_ev_lens = p_ev_lens[idxs_evs]
                p_ev_sts = np.concatenate(([0],np.cumsum(p_ev_lens)))
                p_chs_ids = p_chs_ids[idxs_hits]
                p_tres = p_tres[idxs_hits]
                mask_true_signal = mask_true_signal[idxs_hits]
                mask_true_noise = mask_true_noise[idxs_hits]

            if mean_mode=='hits':
                red_array = np.array([0])
            else:
                red_array = p_ev_sts[:-1]
            
            # standard metrics
            mask_reco_signal = p_preds>=threshold
            mask_reco_noise = ~mask_reco_signal
            # false signal
            f_s = mask_true_noise & mask_reco_signal
            f_s = np.add.reduceat(f_s, red_array)
            # false noise
            f_n = mask_true_signal & mask_reco_noise
            f_n = np.add.reduceat(f_n, red_array)
            # true noise
            t_n = mask_true_noise & mask_reco_noise
            t_n = np.add.reduceat(t_n, red_array)
            # true signal
            t_s = mask_true_signal & mask_reco_signal
            t_s = np.add.reduceat(t_s, red_array)

            # need to avg over events
            metrics['accuracy'][part][i] = np.mean((t_s+t_n)/(t_s+t_n+f_s+f_n+1e-6))
            metrics['precision'][part][i] = np.mean(t_s/(t_s+f_s+1e-6))
            metrics['recall'][part][i] = np.mean(t_s/(t_s+f_n+1e-6))

            # time residuals
            mask_pass = p_preds>=threshold
            nums_reco_signal = np.add.reduceat(mask_pass, red_array)
            tres_pass = np.where(mask_pass, p_tres, 0.)
            metrics['tres'][part][i] = np.mean(np.add.reduceat(tres_pass, red_array)/(nums_reco_signal+1e-6))
            
    return metrics

# plot metrics as function of threshod
def plot_graphics_thresholds(metrics, thresholds, particle_nums=None, plot_for_all=False, ylims=None):

    n_rows = 4 if plot_for_all else 3
    
    fig, axs = plt.subplots(4, n_rows, tight_layout=True, figsize=(4*n_rows, 16) )
    for c,(mn,pt_dict) in enumerate(metrics.items()):
        for i,(pt,mv) in enumerate(pt_dict.items()):
            axs[i,c].grid()
            axs[i,c].plot(thresholds, mv)
            axs[i,c].set(xlabel='threshold', ylabel=mn)
            axs[i,c].set_title(mn+' on '+pt)
            if ylims is not None:
                axs[i,c].set_ylim(ylims[mn])
        
    if plot_for_all:
        tot_evs = np.sum([ particle_nums[p] for p in particles ])
        factor = {}
        for p in particles:
            factor[p] = particle_nums[p]/tot_evs

        for i,(mn,pt_dict) in enumerate(metrics.items()):
            metr = np.zeros( thresholds.shape[0] )
            for pt,mv in pt_dict.items():
                metr += mv*factor[pt]
            axs[3,i].grid()
            axs[3,i].plot( thresholds, metr)
            axs[3,i].set(xlabel='threshold', ylabel=mn)
            axs[3,i].set_title(mn+' on all')
            if ylims is not None:
                axs[3,i].set_ylim(ylims[m_n])
    plt.savefig(path_save+save_prefix+'_thresholds.png')
    plt.close(fig)

# cross dependence of metrics
def plot_graphics_versus(metrics, particles, particle_nums=None, plot_for_all=False):

    m_names = [['precision','recall'], ['precision','tres'], ['recall', 'tres']]
    n_rows = len(m_names)
    n_rows = n_rows+1 if plot_for_all else n_rows
    n_cols = len(m_names)

    fig, axs = plt.subplots(n_cols, n_rows, tight_layout=True, figsize=(n_rows*4, n_cols*4) )
    for j,pt in enumerate(particles):
        for i,(m1,m2) in enumerate(m_names):
            me1 = metrics[m1][pt]
            me2 = metrics[m2][pt]
            axs[i,j].grid()
            axs[i,j].plot( me1, me2)
            axs[i,j].set(xlabel=m1, ylabel=m2)
            axs[i,j].set_title(f"{m1} vs {m2} on {pt}")

    if plot_for_all:
        tot_evs = np.sum([ particle_nums[p] for p in particles ])
        factor = {}
        for p in particles:
            factor[p] = particle_nums[p]/tot_evs
        for i,(m1,m2) in enumerate(m_names):
            metr1 = np.zeros( thresholds.shape[0] )
            metr2 = np.zeros( thresholds.shape[0] )
            for p in particles:
                metr1 += metrics[m1][p]*factor[p]
                metr2 += metrics[m2][p]*factor[p]
            axs[i,j+1].grid()
            axs[i,j+1].plot( metr1, metr2)
            axs[i,j+1].set(xlabel=m1, ylabel=m2)
            axs[i,j+1].set_title(f"{m1} vs {m2} on all")
    plt.savefig(path_save+save_prefix+'_versus.png')
    plt.close(fig)

# plot time residuals histogram
def plot_tres_hist(tres, preds, cut):

    pass_hits = preds>cut
    fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(4, 4) )
    plt.grid(True)
    axs.hist(tres[pass_hits], bins=50, range=(-20,20))

    plt.savefig(path_save+save_prefix+'_treshist.png')
    plt.close(fig)


particle_evt_idxs, particle_hit_idxs, particle_nums = get_particle_slices(h5f, particles)
for nn in nns:
    save_prefix += f"{nn.split('/')[-1]}_{mean_mode}_h{num_hit_cut}s{num_string_cut}"
    # make dataset and get predictions
    val_dataset = make_val_dataset(h5f, val_batch_size, np.array([1.,1.]), set_up_Q_lim, up_Q_lim, set_low_Q_lim, low_Q_lim, False, 20.)
    flat_preds, flat_y_true, flat_tres, event_starts = make_preds(nn, h5f, val_dataset, val_batch_size, None)
    num_evs = event_starts.shape[0]-1
    num_hits = flat_preds.shape[0]
    # get channel ids, if cuts imposed
    if num_hit_cut>0:
        with h5.File(h5f,'r') as hf:
            chs_ids = hf['val/channels/data'][:num_hits]
    else:
        chs_ids = None
    # get metrics
    metrics = metrics_over_thresholds(flat_preds, flat_y_true, flat_tres, event_starts, thresholds, particles, particle_evt_idxs, particle_hit_idxs, mean_mode,
                                chs_ids, num_hit_cut, num_string_cut)
    # plot
    plot_graphics_thresholds(metrics, thresholds, particle_nums=particle_nums, plot_for_all=True)
    plot_graphics_versus(metrics, particles, particle_nums, plot_for_all=True)
    plot_tres_hist(flat_tres, flat_preds, 0.8)