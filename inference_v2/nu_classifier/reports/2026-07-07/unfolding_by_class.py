#!/usr/bin/env python
"""CORRECT control: delay-cloud shape of score>0.8 events split by MC CLASS.
Key contrast = exp false-ν vs muatm false-ν (both background that leaked past 0.8) —
NOT exp vs real MC neutrinos (that is trivial). Also nuatm/nue2 (real ν) for reference.
z-PR, Q-PR, rvert per event; h8s3; MC from the boosted E1@ep10 preds (6.6M muatm).
Outputs tables/unfolding_by_class.csv, figures/unfolding_by_class.png.
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
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8; CUT='n_sn_hits>=8 and n_sn_strings>=3'; M=3

def read_hits(h5path, probs_path, df):
    out=[None]*len(df); f=h5py.File(h5path,'r',**RDCC); fp=h5py.File(probs_path,'r',**RDCC)
    for (base,pk),idx in df.groupby(['base','part_key']).groups.items():
        try: es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        for row,l in zip(np.array(idx),df.loc[idx,'local_idx'].to_numpy()):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<M+2: continue
            hh=ds[s:e].astype(np.float32)[m]; out[row]=hh[np.argsort(hh[:,1])]
    f.close(); fp.close(); return out

def pr_of(series):
    s=np.asarray(series,np.float64); sd=s.std()
    if sd<1e-9 or len(s)<M+1: return np.nan
    s=(s-s.mean())/sd; cloud=np.stack([s[i:len(s)-M+1+i] for i in range(M)],1)
    if len(cloud)<3: return np.nan
    ev=np.sort(np.clip(np.linalg.eigvalsh(np.cov(cloud.T)),0,None))[::-1]
    return (ev.sum()**2)/((ev**2).sum()+1e-12)
def rvert_of(h):
    sx,sy,sz=h[:,2].std(),h[:,3].std(),h[:,4].std(); return sz/(np.sqrt(sx*sx+sy*sy)+1e-3)

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
def qmc(cls,n=4000):
    return c.execute(f"SELECT '{cls}' base,l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk AND ev.data_class='{cls}' JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND pr.score>0.8 ORDER BY random() LIMIT {n}").df()
exp=c.execute(f"SELECT 'exp_full' base,l.part_key,l.local_idx FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249')) AND pr.score>0.8").df()
pops={'exp_hi (false nu)':(exp,EXH5,EXPROBS),'muatm_hi (false nu)':(qmc('muatm_2020'),MCH5,MCPROBS),
      'nuatm_hi (real)':(qmc('nuatm_2020'),MCH5,MCPROBS),'nue2_hi (real)':(qmc('nue2_2020'),MCH5,MCPROBS)}
c.close()
rows=[]
for name,(df,h5,prb) in pops.items():
    hs=read_hits(h5,prb,df)
    for h in hs:
        if h is None: continue
        rows.append(dict(pop=name, prz=pr_of(h[:,4]), prq=pr_of(h[:,0]), rvert=rvert_of(h)))
    print(f'{name:22s}: {sum(x is not None for x in hs)} events')
t=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
t.to_csv(HERE/'tables/unfolding_by_class.csv',index=False)
print('\n== median by class (score>0.8, h8s3) ==')
print(t.groupby('pop')[['prz','prq','rvert']].agg(['median','count']).round(3).to_string())

fig,ax=plt.subplots(1,3,figsize=(16,5))
order=['muatm_hi (false nu)','exp_hi (false nu)','nuatm_hi (real)','nue2_hi (real)']
cols={'exp_hi (false nu)':'black','muatm_hi (false nu)':'tab:red','nuatm_hi (real)':'tab:green','nue2_hi (real)':'tab:olive'}
for j,met in enumerate(['prz','prq','rvert']):
    for p in order:
        v=t[t['pop']==p][met].dropna()
        if len(v): ax[j].hist(v,bins=30,density=True,histtype='step',lw=2,color=cols[p],label=f'{p} ({v.median():.2f})')
    ax[j].set_xlabel(met); ax[j].legend(fontsize=7); ax[j].grid(alpha=0.3)
fig.suptitle('Delay-cloud shape at score>0.8 by MC class — exp vs muatm false-ν is the key contrast')
fig.tight_layout(); fig.savefig(HERE/'figures/unfolding_by_class.png',dpi=130); print('saved figures/unfolding_by_class.png')
