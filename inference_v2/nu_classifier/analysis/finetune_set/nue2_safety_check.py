#!/usr/bin/env python
"""Safety check: does our fine-tune background selection (encoder-OOD>2.5 AND rvert<0.7)
sweep up real neutrinos? Test on MC nue2 (astrophysical nu_mu, E^-2 spectrum = the BRIGHT
nu population we have), stratified by brightness.

Honest design: nue2 are part of the MC reference manifold, so typical nue2 are in-dist
by construction. To not understate the risk (self-match -> knn dist 0), the nue2 TEST
sample is drawn DISJOINT from the reference (by event_fk). Reference composition mirrors
the builder: muatm + nuatm + (a slice of) nue2.

Reports, overall and per Q_mean bin (brightness): fraction of nue2 that would be
selected as background (OOD>2.5 AND rvert<0.7), plus their score/rvert. If the bright
tail increasingly passes the cut, that extrapolates to a VHE-nu danger.

Outputs tables/nue2_safety.csv (per-event), prints binned summary.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores_and_embeddings

CAT = ROOT/'data_manager/catalog_v2.duckdb'
PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
CKPT = ROOT/'experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01/da_checkpoint_epoch_010.pth'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
MCPROBS = ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8
CUT = 'pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'
DEV = 'cuda:0'; KNN=20; OOD_CUT=2.5; RVCUT=0.7


def read_feats(df):
    feats=[None]*len(df); rvert=np.full(len(df),np.nan); qmean=np.full(len(df),np.nan); nfilt=np.zeros(len(df),np.int32)
    f=h5py.File(MCH5,'r',**RDCC); fp=h5py.File(MCPROBS,'r',**RDCC)
    for (base,pk),idx in df.groupby(['base','part_key']).groups.items():
        try: es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        idx=np.array(idx); locs=df.loc[idx,'local_idx'].to_numpy(); o=np.argsort(locs)
        for row,l in zip(idx[o],locs[o]):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<2: continue
            hh=ds[s:e].astype(np.float32)[m]; hh=hh[np.argsort(hh[:,1])]
            feats[row]=hh; nfilt[row]=len(hh)
            sx,sy,sz=hh[:,2].std(),hh[:,3].std(),hh[:,4].std()
            rvert[row]=sz/(np.sqrt(sx*sx+sy*sy)+1e-3); qmean[row]=np.clip(hh[:,0],0,100).mean()
    f.close(); fp.close(); return feats,rvert,qmean,nfilt


def embed(df):
    feats,rvert,qmean,nfilt=read_feats(df)
    keep=[i for i,x in enumerate(feats) if x is not None]; fl=[feats[i] for i in keep]
    sc,emb=predict_scores_and_embeddings(MODEL,fl,NORM,batch_size=512,device=DEV)
    sub=df.iloc[keep].copy().reset_index(drop=True)
    sub['rvert']=rvert[keep]; sub['qmean']=qmean[keep]; sub['nfilt']=nfilt[keep]; sub['score']=sc
    return sub, emb.astype(np.float32)


c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
def q(dc,n):
    return c.execute(f"""SELECT pr.event_fk, ev.data_class AS base, l.part_key, l.local_idx
      FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
      JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND ev.data_class IN ({dc}) ORDER BY random() LIMIT {n}""").df()
ref_bg = q("'muatm_2020','nuatm_2020'", 30000)   # reference: muons + atm nu (tracks)
nue2   = q("'nue2_2020'", 55000)                  # astrophysical nu_mu (E^-2)
c.close()

MODEL, NORM, _ = load_model(str(CKPT), device=DEV)

# split nue2 disjoint: 12k into reference, rest = test
rng=np.random.default_rng(1); perm=rng.permutation(len(nue2))
nue2_ref = nue2.iloc[perm[:12000]].reset_index(drop=True)
nue2_test= nue2.iloc[perm[12000:]].reset_index(drop=True)

ref = pd.concat([ref_bg, nue2_ref], ignore_index=True)
ref_sub, ref_emb = embed(ref)
test_sub, test_emb = embed(nue2_test)
print(f'reference embedded {len(ref_sub):,} (muatm+nuatm+nue2), nue2 test {len(test_sub):,}', flush=True)

from sklearn.neighbors import NearestNeighbors
nn=NearestNeighbors(n_neighbors=KNN).fit(ref_emb)
test_sub['ood']=nn.kneighbors(test_emb)[0].mean(1)
test_sub.to_csv(HERE/'tables/nue2_safety.csv',index=False)

d=test_sub
sel=(d.ood>OOD_CUT)&(d.rvert<RVCUT)
print(f"\n=== MC nue2 vs background selection (OOD>{OOD_CUT} AND rvert<{RVCUT}) ===")
print(f"nue2 test N={len(d)}  score med={d.score.median():.3f} (frac>0.8 correctly nu: {(d.score>0.8).mean():.3f})")
print(f"OVERALL selected as background: {sel.sum()} ({sel.mean()*100:.2f}%)")
print(f"  frac OOD>{OOD_CUT}: {(d.ood>OOD_CUT).mean():.3f}   frac rvert<{RVCUT}: {(d.rvert<RVCUT).mean():.3f}")
print(f"\n{'Q_mean bin':>12} {'N':>7} {'ood med':>8} {'rvert med':>9} {'score med':>9} {'%OOD>2.5':>9} {'%rvert<0.7':>10} {'% -> BG':>8}")
qbins=[0,3,5,8,15,30,1000]
rows=[]
for lo,hi in zip(qbins[:-1],qbins[1:]):
    s=d[(d.qmean>=lo)&(d.qmean<hi)]
    if len(s):
        bg=((s.ood>OOD_CUT)&(s.rvert<RVCUT)).mean()
        rows.append(dict(qlo=lo,qhi=hi,N=len(s),ood_med=s.ood.median(),rvert_med=s.rvert.median(),score_med=s.score.median(),
                         f_ood=(s.ood>OOD_CUT).mean(),f_rv=(s.rvert<RVCUT).mean(),f_bg=bg))
        print(f'{lo:>4}-{hi:<5} {len(s):>7} {s.ood.median():>8.2f} {s.rvert.median():>9.3f} {s.score.median():>9.3f} {(s.ood>OOD_CUT).mean():>9.3f} {(s.rvert<RVCUT).mean():>10.3f} {bg*100:>7.2f}%')
pd.DataFrame(rows).to_csv(HERE/'tables/nue2_safety_binned.csv',index=False)
# brightest tail explicit
for qtail in [15,30,50]:
    s=d[d.qmean>=qtail]
    if len(s): print(f"  brightest tail Q>={qtail}: N={len(s)}  -> BG {((s.ood>OOD_CUT)&(s.rvert<RVCUT)).mean()*100:.2f}%  (score med {s.score.median():.3f}, rvert med {s.rvert.median():.3f})")
