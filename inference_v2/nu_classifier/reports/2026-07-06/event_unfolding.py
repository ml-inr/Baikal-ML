#!/usr/bin/env python
"""Delay-embedding (Takens) of event hit-sequences — does the phase-space geometry
of exp false-ν differ from real ν / muons? (model-independent temporal-structure probe)

For each event: order signal hits (prob>0.8) by time; take a scalar per-hit series
s (charge Q, and depth z), z-score it, and delay-embed with window m=3:
    point_i = (s_i, s_{i+1}, s_{i+2}) in R^3.
The point cloud reconstructs the sequence dynamics. Shape descriptors from PCA of the
cloud (eigenvalues λ1≥λ2≥λ3, normalized): participation ratio (effective dim; ~1 =
structured curve, ~3 = space-filling/noisy), planarity, sphericity. Compared across
exp_hi (score>0.8), MC ν (score>0.8), MC muon, under h8s3.
Outputs tables/event_unfolding.csv, figures/event_unfolding_{clouds,shape}.png.
"""
from __future__ import annotations
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
CAT = ROOT/'data_manager/catalog_v2.duckdb'
PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'; EXH5 = ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS = ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS = ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR = 0.8
CUT = 'n_sn_hits>=8 and n_sn_strings>=3'; M = 3   # delay window

def read_hits(h5path, probs_path, df):
    """Time-ordered filtered hits (n,5)=[Q,t,x,y,z] per event."""
    out = [None]*len(df); f = h5py.File(h5path,'r',**RDCC); fp = h5py.File(probs_path,'r',**RDCC)
    for (base,pk), idx in df.groupby(['base','part_key']).groups.items():
        try:
            es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        for row,l in zip(np.array(idx), df.loc[idx,'local_idx'].to_numpy()):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<M+2: continue
            hh = ds[s:e].astype(np.float32)[m]
            out[row] = hh[np.argsort(hh[:,1])]        # order by time
    f.close(); fp.close(); return out

def delay_cloud(series, m=M):
    s = np.asarray(series, np.float64)
    sd = s.std()
    if sd < 1e-9 or len(s) < m+1: return None
    s = (s - s.mean())/sd
    return np.stack([s[i:len(s)-m+1+i] for i in range(m)], axis=1)   # (n-m+1, m)

def shape(cloud):
    if cloud is None or len(cloud) < 3: return None
    ev = np.linalg.eigvalsh(np.cov(cloud.T)); ev = np.sort(ev)[::-1]; ev = np.clip(ev,0,None)
    tot = ev.sum()+1e-12; f = ev/tot
    pr = (ev.sum()**2)/((ev**2).sum()+1e-12)        # participation ratio (eff dim of cloud)
    return dict(pr=pr, lam1=f[0], lam3=f[2], planarity=(f[1]-f[2]), sphericity=f[2]/ (f[0]+1e-12))

c = duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
def qmc(w,n): return c.execute(f"SELECT ev.data_class base,l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND {w} ORDER BY random() LIMIT {n}").df()
def qex(w,n): return c.execute(f"SELECT 'exp_full' base,l.part_key,l.local_idx FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN ('20','249')) AND {w} ORDER BY random() LIMIT {n}").df()
pops = {'exp_hi': (qex("pr.score>0.8",2500), EXH5,EXPROBS),
        'mc_nu_hi': (qmc("ev.data_class IN ('nuatm_2020','nue2_2020') AND pr.score>0.8",2500), MCH5,MCPROBS),
        'mc_muon': (qmc("ev.data_class='muatm_2020'",2500), MCH5,MCPROBS)}
c.close()

rows=[]; hits_by_pop={}
for name,(df,h5,pr) in pops.items():
    hs = read_hits(h5,pr,df); hs=[h for h in hs if h is not None]; hits_by_pop[name]=hs
    for h in hs:
        for chan,col in [('Q',0),('z',4)]:
            sh = shape(delay_cloud(h[:,col]))
            if sh: rows.append(dict(pop=name, chan=chan, nhit=len(h), **sh))
    print(f'{name:9s}: {len(hs)} events')
tab = pd.DataFrame(rows); tab.to_csv(HERE/'tables/event_unfolding.csv', index=False)
print('\n== delay-cloud shape (median) by population/channel ==')
print(tab.groupby(['chan','pop'])[['pr','lam1','planarity','sphericity']].median().round(3).to_string())

# --- shape distributions (participation ratio = effective dim of the phase cloud) ---
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,chan in zip(axes,['Q','z']):
    for p,cl in [('mc_muon','tab:red'),('mc_nu_hi','tab:green'),('exp_hi','black')]:
        v=tab[(tab.chan==chan)&(tab.pop==p)].pr
        ax.hist(v,bins=40,density=True,histtype='step',lw=2,color=cl,label=f'{p} (med {v.median():.2f})')
    ax.set_xlabel(f'participation ratio of delay cloud ({chan}, m={M})'); ax.set_ylabel('density')
    ax.set_title(f'Phase-space effective dim — {chan} series'); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE/'figures/event_unfolding_shape.png',dpi=130); print('saved figures/event_unfolding_shape.png')

# --- visualize a few delay clouds (Q series) per population ---
fig2=plt.figure(figsize=(16,9))
for r,(name,cl) in enumerate([('exp_hi','black'),('mc_nu_hi','tab:green'),('mc_muon','tab:red')]):
    big=[h for h in hits_by_pop[name] if len(h)>=18][:5]
    for k,h in enumerate(big):
        ax=fig2.add_subplot(3,5,r*5+k+1,projection='3d')
        cloud=delay_cloud(h[:,0])
        if cloud is not None:
            ax.plot(cloud[:,0],cloud[:,1],cloud[:,2],'-',lw=0.6,color=cl,alpha=0.5)
            ax.scatter(cloud[:,0],cloud[:,1],cloud[:,2],s=12,c=np.arange(len(cloud)),cmap='viridis')
        ax.set_title(f'{name} nh={len(h)}',fontsize=8,color=cl); ax.tick_params(labelsize=5)
fig2.suptitle('Delay embedding (Q series, window 3) — colour = hit order',y=1.0)
fig2.tight_layout(); fig2.savefig(HERE/'figures/event_unfolding_clouds.png',dpi=120); print('saved figures/event_unfolding_clouds.png')
