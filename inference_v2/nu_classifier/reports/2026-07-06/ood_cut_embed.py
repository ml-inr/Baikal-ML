#!/usr/bin/env python
"""OOD cut on score>0.8 in the MODEL's embedding space (companion to ood_cut.py).

ood_cut.py (hand-picked physical features) FAILED: exp false-ν look in-distribution
there (maha 7.5 ≈ MC), so a physical-OOD cut removes ~nothing and the excess stays —
the OOD is representational, not in simple observables. Here we repeat the survival-
cut in the 128-dim encoder embedding (E1@ep10), OOD score = kNN distance to the MC
manifold (robust, no covariance assumption). Same 3 questions:
  (1) does an embedding-OOD cut remove the excess?
  (2) how many in-distribution high-score exp events remain (candidate real ν)?
  (3) control: do real MC ν (score>0.8) survive the same cut? (retention)

Outputs tables/ood_cut_embed.csv, figures/ood_cut_embed.png.
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
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR = 0.8
CUT = 'n_sn_hits>=5'; DEV = 'cuda:3'


def read_hits(h5path, probs_path, df):
    feats = [None]*len(df)
    f = h5py.File(h5path,'r',**RDCC); fp = h5py.File(probs_path,'r',**RDCC)
    for (base,pk), idx in df.groupby(['base','part_key']).groups.items():
        try:
            es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        for row,l in zip(np.array(idx), df.loc[idx,'local_idx'].to_numpy()):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<2: continue
            feats[row]=ds[s:e].astype(np.float32)[m]
    f.close(); fp.close(); return feats


c = duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
def q_mc(where,n):
    return c.execute(f"""SELECT ev.data_class AS base, l.part_key, l.local_idx
      FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND {where} ORDER BY random() LIMIT {n}""").df()
def q_exp(where):
    return c.execute(f"""SELECT 'exp_full' AS base, l.part_key, l.local_idx
      FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249')) AND {where}""").df()
Nexp_tot = c.execute(f"SELECT count(*) FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk "
                     f"WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249'))").fetchone()[0]
pops = {'mc_ref': (q_mc("ev.data_class IN ('muatm_2020','nuatm_2020','nue2_2020')",12000), MCH5, MCPROBS),
        'mc_nu_hi': (q_mc("ev.data_class IN ('nuatm_2020','nue2_2020') AND pr.score>0.8",4000), MCH5, MCPROBS),
        'exp_hi': (q_exp("pr.score>0.8"), EXH5, EXPROBS)}
c.close()

model, norm, _ = load_model(str(CKPT), device=DEV)
emb = {}
for name,(df,h5,pr) in pops.items():
    feats = read_hits(h5,pr,df); keep=[i for i,x in enumerate(feats) if x is not None]
    sc,E = predict_scores_and_embeddings(model,[feats[i] for i in keep],norm,batch_size=512,device=DEV)
    emb[name]=E.astype(np.float64); print(f'{name:9s}: {len(keep)}/{len(df)}')

from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=20).fit(emb['mc_ref'])
knn = lambda E: nn.kneighbors(E)[0].mean(1)
d_ref = knn(emb['mc_ref']); d_nu = knn(emb['mc_nu_hi']); d_exp = knn(emb['exp_hi'])
print(f"\nkNN-to-MC (embedding OOD): MC ref med={np.median(d_ref):.2f} p99={np.percentile(d_ref,99):.2f} "
      f"| MC ν-hi med={np.median(d_nu):.2f} | exp-hi med={np.median(d_exp):.2f}")

rows=[]
for pct in [90,95,99,99.9]:
    thr=np.percentile(d_ref,pct); se=int((d_exp<=thr).sum())
    rows.append(dict(mc_keep_pct=pct, ood_thr=round(thr,2), exp_hi_survive=se, exp_hi_total=len(d_exp),
                     implied_frac=se/Nexp_tot, real_nu_retention=round(float((d_nu<=thr).mean()),3)))
tab=pd.DataFrame(rows); tab.to_csv(HERE/'tables/ood_cut_embed.csv',index=False)
print(f'\nexp held-out total ({CUT}) = {Nexp_tot:,}; physical ν ~1e-6..1e-5 => ~{Nexp_tot*1e-6:.1f}..{Nexp_tot*1e-5:.0f}')
print(tab.to_string(index=False))

fig,ax=plt.subplots(figsize=(8,5))
for d,lab,cl in [(d_ref,'MC ref','tab:red'),(d_nu,'MC ν (score>0.8)','tab:green'),(d_exp,'exp (score>0.8)','black')]:
    ax.hist(np.log10(d+0.01),bins=50,density=True,histtype='step',lw=2,color=cl,label=lab)
ax.axvline(np.log10(np.percentile(d_ref,99)+0.01),color='grey',ls='--',label='MC p99 cut')
ax.set_xlabel('log10 kNN dist to MC (embedding)'); ax.set_ylabel('density')
ax.set_title('Embedding-space OOD heuristic on score>0.8'); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE/'figures/ood_cut_embed.png',dpi=130); print('saved figures/ood_cut_embed.png')
