#!/usr/bin/env python
"""Does 'exp has noisier z (higher z-delay PR)' hold across ALL score bins, or only
at p>0.8? Delay-cloud shape (Takens m=3) per event, stratified by score bin, exp vs MC.
Also rvert (verticality) per bin to check the horizon confound.
Outputs tables/unfolding_by_score.csv, figures/unfolding_by_score.png.
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
BINS=[(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]; PER=1500

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
    s=(s-s.mean())/sd
    cloud=np.stack([s[i:len(s)-M+1+i] for i in range(M)],1)
    if len(cloud)<3: return np.nan
    ev=np.sort(np.clip(np.linalg.eigvalsh(np.cov(cloud.T)),0,None))[::-1]
    return (ev.sum()**2)/((ev**2).sum()+1e-12)

def rvert_of(h):
    sx,sy,sz=h[:,2].std(),h[:,3].std(),h[:,4].std(); return sz/(np.sqrt(sx*sx+sy*sy)+1e-3)

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
rows=[]
for lo,hi in BINS:
    ex=c.execute(f"SELECT pr.score,'exp_full' base,l.part_key,l.local_idx FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249')) AND pr.score>={lo} AND pr.score<{hi} ORDER BY random() LIMIT {PER}").df()
    mc=c.execute(f"SELECT pr.score,ev.data_class base,l.part_key,l.local_idx FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk WHERE {CUT} AND pr.score>={lo} AND pr.score<{hi} ORDER BY random() LIMIT {PER}").df()
    for dom,df,h5,prb in [('exp',ex,EXH5,EXPROBS),('mc',mc,MCH5,MCPROBS)]:
        hs=read_hits(h5,prb,df)
        for h in hs:
            if h is None: continue
            rows.append(dict(dom=dom, sbin=f'{lo:.1f}-{hi:.1f}', prz=pr_of(h[:,4]), prq=pr_of(h[:,0]), rvert=rvert_of(h)))
    print(f'bin {lo}-{hi}: exp {sum(x is not None for x in read_hits(EXH5,EXPROBS,ex)) if False else len(ex)} / mc {len(mc)} queried')
c.close()
t=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
t.to_csv(HERE/'tables/unfolding_by_score.csv',index=False)
g=t.groupby(['sbin','dom'])[['prz','prq','rvert']].median().round(3)
print('\n== median by score bin & domain ==\n'+g.to_string())

# figure: PR_z, PR_q, rvert vs score bin, exp vs mc
fig,ax=plt.subplots(1,3,figsize=(16,5))
xb=[f'{lo:.1f}-{hi:.1f}' for lo,hi in BINS]; x=np.arange(len(xb))
for j,(met,ttl) in enumerate([('prz','z-delay PR'),('prq','Q-delay PR'),('rvert','rvert (verticality)')]):
    for dom,cl in [('exp','black'),('mc','tab:red')]:
        med=[t[(t.sbin==b)&(t.dom==dom)][met].median() for b in xb]
        ax[j].plot(x,med,'o-',lw=2,color=cl,label=dom)
    ax[j].set_xticks(x); ax[j].set_xticklabels(xb,rotation=30); ax[j].set_title(ttl); ax[j].grid(alpha=0.3); ax[j].legend()
    ax[j].set_xlabel('score bin')
fig.suptitle('Delay-cloud shape & verticality vs score bin — exp vs MC (h8s3)')
fig.tight_layout(); fig.savefig(HERE/'figures/unfolding_by_score.png',dpi=130); print('saved figures/unfolding_by_score.png')
