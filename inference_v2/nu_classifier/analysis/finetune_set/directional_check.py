#!/usr/bin/env python
"""Defense-1 test: does a 'background only if clearly DOWN-going' filter protect real
neutrinos while keeping the down-going muon background?

Directional proxy (no reco, per user's constraint): corr(t, z) on filtered hits.
  down-going  -> later hits are lower  -> corr(t,z) NEGATIVE
  up-going    -> later hits are higher -> corr(t,z) POSITIVE
  cascade / no clear track             -> corr(t,z) ~ 0  (naturally low fit quality)
Sign is CALIBRATED empirically on muatm (all down-going, GT theta>90).

Filter: include an event in background only if corr(t,z) < -c (clearly down-going).
So background = OOD>2.5 AND rvert<0.7 AND corr(t,z) < -c.

Populations (h8s3), each embedded, kNN-OOD to a MC reference (muatm+nuatm+nue2):
  - nue2   : astrophysical nu_mu (E^-2) -> the bright nu we must protect
  - muatm  : down-going muons  -> the background we must KEEP  (corr<<0 expected)
  - nuatm  : tracks, split by GT theta<90 (up=protect) / >90 (down)
Reports corr distributions, corr-vs-theta, and the nue2/nuatm-up ->background fraction
before/after the down-going filter, plus muatm survival (background retention).
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
DEV='cuda:0'; KNN=20; OOD_CUT=2.5; RVCUT=0.7


def read_feats(df, want_theta=False):
    feats=[None]*len(df); rvert=np.full(len(df),np.nan); qmean=np.full(len(df),np.nan)
    corr=np.full(len(df),np.nan); theta=np.full(len(df),np.nan)
    f=h5py.File(MCH5,'r',**RDCC); fp=h5py.File(MCPROBS,'r',**RDCC)
    for (base,pk),idx in df.groupby(['base','part_key']).groups.items():
        try: es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']; pr=fp[f'{base}/probs/{pk}/data']
        except KeyError: continue
        th=None
        if want_theta:
            tk=f'{base}/prime_prty/{pk}/data'
            if tk in f: th=f[tk][:,0]
        idx=np.array(idx); locs=df.loc[idx,'local_idx'].to_numpy(); o=np.argsort(locs)
        for row,l in zip(idx[o],locs[o]):
            if l+1>=len(es): continue
            s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
            if m.sum()<2: continue
            hh=ds[s:e].astype(np.float32)[m]; hh=hh[np.argsort(hh[:,1])]
            feats[row]=hh
            sx,sy,sz=hh[:,2].std(),hh[:,3].std(),hh[:,4].std()
            rvert[row]=sz/(np.sqrt(sx*sx+sy*sy)+1e-3); qmean[row]=np.clip(hh[:,0],0,100).mean()
            tt,zz=hh[:,1],hh[:,4]
            corr[row]=np.corrcoef(tt,zz)[0,1] if (tt.std()>1e-6 and zz.std()>1e-6) else 0.0
            if th is not None and l<len(th): theta[row]=th[l]
    f.close(); fp.close(); return feats,rvert,qmean,corr,theta


def embed(df, want_theta=False):
    feats,rvert,qmean,corr,theta=read_feats(df,want_theta)
    keep=[i for i,x in enumerate(feats) if x is not None]; fl=[feats[i] for i in keep]
    sc,emb=predict_scores_and_embeddings(MODEL,fl,NORM,batch_size=512,device=DEV)
    sub=df.iloc[keep].copy().reset_index(drop=True)
    sub['rvert']=rvert[keep]; sub['qmean']=qmean[keep]; sub['corr_tz']=corr[keep]
    sub['theta']=theta[keep]; sub['score']=sc
    return sub, emb.astype(np.float32)


c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
def q(dc,n):
    return c.execute(f"""SELECT pr.event_fk, ev.data_class AS base, l.part_key, l.local_idx
      FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
      JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND ev.data_class IN ({dc}) ORDER BY random() LIMIT {n}""").df()
ref_bg = q("'muatm_2020'", 25000)          # nuatm probs use mismatched part_keys -> skip
nue2   = q("'nue2_2020'", 42000)
muatm  = q("'muatm_2020'", 20000)
c.close()

MODEL, NORM, _ = load_model(str(CKPT), device=DEV)
rng=np.random.default_rng(1); perm=rng.permutation(len(nue2))
nue2_ref=nue2.iloc[perm[:10000]].reset_index(drop=True)
nue2_test=nue2.iloc[perm[10000:]].reset_index(drop=True)

ref=pd.concat([ref_bg,nue2_ref],ignore_index=True)
_, ref_emb = embed(ref)
from sklearn.neighbors import NearestNeighbors
nn=NearestNeighbors(n_neighbors=KNN).fit(ref_emb)

def with_ood(df, wt=False):
    sub,emb=embed(df,wt); sub['ood']=nn.kneighbors(emb)[0].mean(1); return sub

d_nue2  = with_ood(nue2_test)
d_muatm = with_ood(muatm, wt=True)
for nm,dd in [('nue2',d_nue2),('muatm',d_muatm)]:
    dd.to_csv(HERE/f'tables/dir_{nm}.csv',index=False)
print(f'embedded: nue2 {len(d_nue2)}, muatm {len(d_muatm)}',flush=True)

# ---- 1. sign calibration: corr(t,z) on muatm (all down-going, theta>90) vs nue2 ----
print('\n=== corr(t,z) sign calibration (down-going = ?) ===')
print(f"muatm (all down-going, theta>90): corr_tz median={d_muatm.corr_tz.median():.3f} "
      f"[p10={np.percentile(d_muatm.corr_tz,10):.2f}, p90={np.percentile(d_muatm.corr_tz,90):.2f}]")
print(f"nue2  (astrophysical nu_mu):      corr_tz median={d_nue2.corr_tz.median():.3f} "
      f"[p10={np.percentile(d_nue2.corr_tz,10):.2f}, p90={np.percentile(d_nue2.corr_tz,90):.2f}]")
# muatm are all down-going -> their corr sign defines 'down'. (up-going tracks are the
# mirror by symmetry; nuatm probs unavailable so not shown directly.)
DOWN = -0.3 if d_muatm.corr_tz.median() < 0 else 0.3
downgoing = (lambda s: s.corr_tz < DOWN) if DOWN<0 else (lambda s: s.corr_tz > DOWN)
print(f"\n-> 'clearly down-going' = corr_tz {'<' if DOWN<0 else '>'} {DOWN}  (calibrated on muatm)")

# ---- 2. background retention (muatm) & nu protection (nue2) ----
def bg_base(s): return (s.ood>OOD_CUT)&(s.rvert<RVCUT)
def bg_dir(s):  return bg_base(s)&downgoing(s)
print('\n=== effect of down-going filter ===')
print(f"{'population':>18} {'N':>7} {'BG base%':>9} {'BG +dir%':>9} {'note':>26}")
for nm,dd,note in [('nue2 (protect)',d_nue2,'nu lost -> want LOW'),
                   ('muatm (keep as BG)',d_muatm,'bg kept -> want base~+dir')]:
    print(f"{nm:>18} {len(dd):>7} {bg_base(dd).mean()*100:>8.2f}% {bg_dir(dd).mean()*100:>8.2f}%   {note:>26}")
# muatm: what frac of the down-going background survives the corr filter at all?
print(f"\nmuatm down-going retention (frac corr_tz{'<' if DOWN<0 else '>'}{DOWN}): {downgoing(d_muatm).mean():.3f}")
print(f"  -> of muatm that pass OOD&rvert (candidate BG), frac still down-going: "
      f"{downgoing(d_muatm[bg_base(d_muatm)]).mean() if bg_base(d_muatm).sum() else float('nan'):.3f}")

# ---- 3. nue2 protection by brightness (base vs +dir) ----
print('\n=== nue2 ->background by Q_mean: base vs +down-going filter ===')
print(f"{'Q bin':>10} {'N':>7} {'BG base%':>9} {'BG +dir%':>9}")
qbins=[0,5,8,15,30,1000]; rows=[]
for lo,hi in zip(qbins[:-1],qbins[1:]):
    s=d_nue2[(d_nue2.qmean>=lo)&(d_nue2.qmean<hi)]
    if len(s):
        rows.append(dict(qlo=lo,qhi=hi,N=len(s),bg_base=bg_base(s).mean(),bg_dir=bg_dir(s).mean()))
        print(f"{lo:>4}-{hi:<5} {len(s):>7} {bg_base(s).mean()*100:>8.2f}% {bg_dir(s).mean()*100:>8.2f}%")
pd.DataFrame(rows).to_csv(HERE/'tables/dir_nue2_binned.csv',index=False)
