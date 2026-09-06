#!/usr/bin/env python
"""Is the 128-dim embedding over-provisioned, and does exp OOD drift live in the
UNUSED (low-variance) MC directions? Tests the 'reduce d_model' hypothesis.

PCA on MC embeddings -> effective dim (participation ratio, #PCs for 90/99% var).
Then decompose exp false-ν distance-from-MC-mean into: (a) top-K PCs where MC has
variance vs (b) the tail PCs MC barely uses. If exp far-ness is in the tail (unused)
directions, a tighter bottleneck / dim reduction is motivated; if in the top PCs,
it's a genuine in-manifold data difference and reducing dim won't help.
Outputs tables/embedding_dimensionality.csv, figures/embedding_dimensionality.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores_and_embeddings

CAT = ROOT/'data_manager/catalog_v2.duckdb'
PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
CKPT = ROOT/'experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01/da_checkpoint_epoch_010.pth'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
EXH5 = ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS = ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS = ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8; CUT='n_sn_hits>=5'; DEV='cuda:3'

def read_hits(h5path, probs_path, df):
    feats=[None]*len(df); f=h5py.File(h5path,'r',**RDCC); fp=h5py.File(probs_path,'r',**RDCC)
    for (base,pk),idx in df.groupby(['base','part_key']).groups.items():
        try:
            es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        for row,l in zip(np.array(idx),df.loc[idx,'local_idx'].to_numpy()):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<2: continue
            feats[row]=ds[s:e].astype(np.float32)[m]
    f.close(); fp.close(); return feats

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
mc=c.execute(f"""SELECT ev.data_class AS base,l.part_key,l.local_idx FROM m.predictions pr
  JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
  WHERE {CUT} AND ev.data_class IN ('muatm_2020','nuatm_2020','nue2_2020') ORDER BY random() LIMIT 8000""").df()
ex=c.execute(f"""SELECT 'exp_full' AS base,l.part_key,l.local_idx FROM e.predictions pr
  JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
  WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249')) AND pr.score>0.8""").df()
c.close()

model,norm,_=load_model(str(CKPT),device=DEV)
def embed(df,h5,pr):
    fs=read_hits(h5,pr,df); k=[i for i,x in enumerate(fs) if x is not None]
    _,E=predict_scores_and_embeddings(model,[fs[i] for i in k],norm,batch_size=512,device=DEV); return E.astype(np.float64)
Emc=embed(mc,MCH5,MCPROBS); Eex=embed(ex,EXH5,EXPROBS)
print(f'MC emb {Emc.shape}, exp_hi emb {Eex.shape}')

# PCA on MC
mu=Emc.mean(0); Xc=Emc-mu
U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
var=S**2/(len(Emc)-1); cum=np.cumsum(var)/var.sum()
PR=(var.sum()**2)/(np.sum(var**2))                     # participation ratio (effective dim)
d90=int(np.searchsorted(cum,0.90)+1); d99=int(np.searchsorted(cum,0.99)+1)
print(f'\nMC embedding effective dim: participation ratio={PR:.1f}; #PCs for 90% var={d90}, 99%={d99} (of 128)')

# project exp_hi onto MC PCs; distance-from-MC-mean decomposed by PC groups
Pex=(Eex-mu)@Vt.T                                       # exp coords in MC-PC basis (N,128)
mc_std=np.sqrt(var)                                     # MC std per PC
z=Pex/ (mc_std+1e-9)                                    # whiten by MC per-PC std -> per-PC Mahalanobis contribution
d2=z**2                                                 # squared contribution per PC
# group: top d90 PCs (used) vs the rest (tail/unused)
top_share=d2[:, :d90].sum(1)/d2.sum(1)
tail_share=d2[:, d90:].sum(1)/d2.sum(1)
print(f'\nexp_hi Mahalanobis-to-MC decomposition (median over events):')
print(f'  fraction of distance^2 in TOP {d90} PCs (MC-used)   = {np.median(top_share):.3f}')
print(f'  fraction of distance^2 in TAIL {128-d90} PCs (unused) = {np.median(tail_share):.3f}')
# also raw per-PC: where is exp most anomalous? mean |z| per PC, top offenders
mean_absz=np.abs(z).mean(0)
order=np.argsort(mean_absz)[::-1][:8]
print(f'  top-8 most-anomalous PCs (by mean|z|): idx={list(order)}')
print(f'     their MC var-rank (0=highest var): {list(order)}  mean|z|={np.round(mean_absz[order],1)}')

pd.DataFrame(dict(pc=np.arange(128), mc_var_frac=var/var.sum(), cum_var=cum,
                  exp_mean_absz=mean_absz)).to_csv(HERE/'tables/embedding_dimensionality.csv',index=False)
with open(HERE/'tables/embedding_dimensionality.csv','a') as f:
    f.write(f'# participation_ratio={PR:.2f} d90={d90} d99={d99} '
            f'exp_top{d90}_share={np.median(top_share):.3f} exp_tail_share={np.median(tail_share):.3f}\n')

fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].plot(np.arange(1,129),cum,'-o',ms=3); ax[0].axhline(0.9,color='grey',ls='--'); ax[0].axhline(0.99,color='grey',ls=':')
ax[0].axvline(d90,color='tab:orange',ls='--',label=f'90% @ {d90} PCs'); ax[0].set_xlabel('PC'); ax[0].set_ylabel('cum MC variance')
ax[0].set_title(f'MC embedding spectrum (PR={PR:.1f} eff dim)'); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(np.arange(128), mean_absz,'-'); ax[1].axvline(d90,color='tab:orange',ls='--',label=f'top {d90} used')
ax[1].set_xlabel('MC PC (var-ordered)'); ax[1].set_ylabel('exp_hi mean |z| (per-PC anomaly)')
ax[1].set_title('Where exp false-ν are anomalous (per MC-PC)'); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE/'figures/embedding_dimensionality.png',dpi=130); print('saved figures/embedding_dimensionality.png')
