#!/usr/bin/env python
"""Is the 'bright (Q>8) AND horizontal (rvert<0.7)' corner occupied by MC-nu (so the net
learned bright+horizontal -> nu), and do the exp false-nu sit at the MC-nu EDGE (mildly OOD,
SNGP only partial) rather than in an empty void? Tests the user's correction to the
'empty corner' framing.

Populations (h8s3, E1@ep10 embeddings): MC muatm / nuatm / nue2 (nuatm restricted to the
100 parts that have precomputed SN probs), exp score>0.8 (false-nu), 0.5-0.8, <0.2.
nuatm & nue2 = same physics (single mu-neutrino), nue2 higher energy.

A. corner composition & scores: per class, among rvert<0.7 & Q>8 -> N, score med, frac>0.8.
B. distance to the MC-nu manifold: kNN(embed) to nuatm+nue2 for nu(calib) / nu bright-horiz /
   exp_hi / exp_lo -> are exp false-nu at the nu edge or beyond?
C. nearest MC-nu identity: fraction of exp_hi whose nearest MC-nu neighbour is itself
   bright+horizontal (literally 'look like energetic horizontal nu').
Outputs tables/mc_nu_corner*.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores_and_embeddings

CAT = ROOT/'data_manager/catalog_v2.duckdb'
E1_PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
CKPT = ROOT/'experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01/da_checkpoint_epoch_010.pth'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
EXH5 = ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS = ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS = ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8
CUT='pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'; DEV='cuda:2'; KNN=20
RV_HZ=0.7; Q_BRIGHT=8.0


_SN=None
def _sn():
    """Lazily load the canonical sig-noise model for on-the-fly filtering."""
    global _SN
    if _SN is None:
        from inference_v2.shared.model_utils import load_sn_model
        from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import predict_flat
        m,_,dev=load_sn_model(device=DEV); _SN=(m,dev,predict_flat)
    return _SN

def _feat_row(hh):
    hh=hh[np.argsort(hh[:,1])]
    sx,sy,sz=hh[:,2].std(),hh[:,3].std(),hh[:,4].std()
    return hh, sz/(np.sqrt(sx*sx+sy*sy)+1e-3), np.clip(hh[:,0],0,100).mean()

def read(h5path, probs_path, df, sn_on_the_fly=False):
    """Filtered hits (prob>THR) per event. probs from precomputed file, or SN on-the-fly
    (needed where precomputed probs cover only a subset of parts, e.g. nuatm: 5%)."""
    feats=[None]*len(df); rvert=np.full(len(df),np.nan); qm=np.full(len(df),np.nan)
    f=h5py.File(h5path,'r',**RDCC); fp=None if sn_on_the_fly else h5py.File(probs_path,'r',**RDCC)
    for (base,pk),idx in df.groupby(['base','part_key']).groups.items():
        try:
            es=f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds=f[f'{base}/raw/data/{pk}/data']
        except KeyError: continue
        idx=np.array(idx); locs=df.loc[idx,'local_idx'].to_numpy(); o=np.argsort(locs)
        if sn_on_the_fly:
            # run SN once on this part's selected events
            m_,dev,predict_flat=_sn()
            sl=[(int(es[l]),int(es[l+1])) for l in locs[o] if l+1<len(es)]
            valid=[l for l in locs[o] if l+1<len(es)]
            if not sl: continue
            data=np.concatenate([ds[s:e].astype(np.float32) for s,e in sl],axis=0)
            nh=np.array([e-s for s,e in sl],dtype=np.int64); starts=np.zeros(len(sl)+1,dtype=np.int64); np.cumsum(nh,out=starts[1:])
            prob=predict_flat(model=m_,data=data,ev_starts=starts,batch_size=512,device=dev,normalize=True)
            for j,(row,l) in enumerate([(r,l) for r,l in zip(idx[o],locs[o]) if l+1<len(es)]):
                a,b=int(starts[j]),int(starts[j+1]); p=prob[a:b].astype(np.float32); mm=p>THR
                if mm.sum()<2: continue
                hh,rv,q=_feat_row(data[a:b][mm]); feats[row]=hh; rvert[row]=rv; qm[row]=q
        else:
            try: pr=fp[f'{base}/probs/{pk}/data']
            except KeyError: continue
            for row,l in zip(idx[o],locs[o]):
                if l+1>=len(es): continue
                s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
                if m.sum()<2: continue
                hh,rv,q=_feat_row(ds[s:e].astype(np.float32)[m]); feats[row]=hh; rvert[row]=rv; qm[row]=q
    f.close();
    if fp is not None: fp.close()
    return feats,rvert,qm

MODEL,NORM,_=load_model(str(CKPT),device=DEV)
def embed(h5,pr,df,sn_on_the_fly=False):
    feats,rvert,qm=read(h5,pr,df,sn_on_the_fly=sn_on_the_fly); keep=[i for i,x in enumerate(feats) if x is not None]
    sc,emb=predict_scores_and_embeddings(MODEL,[feats[i] for i in keep],NORM,batch_size=512,device=DEV)
    s=df.iloc[keep].copy().reset_index(drop=True); s['rvert']=rvert[keep]; s['qmean']=qm[keep]; s['score']=sc
    return s, emb.astype(np.float32)

c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{E1_PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{E1_PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
# restrict muatm/nue2 to parts that actually have precomputed probs -> no wasted reads
_fp=h5py.File(MCPROBS,'r')
def _plist(g): return "("+",".join(f"'{p}'" for p in _fp[f'{g}/probs'].keys())+")"
PARTS={'muatm_2020':_plist('muatm_2020'),'nue2_2020':_plist('nue2_2020')}
def probs_parts(g):
    with h5py.File(MCPROBS,'r') as fp: return list(fp[f'{g}/probs'].keys())
# Restrict muatm/nue2 queries to parts that HAVE precomputed probs (no wasted reads).
# muatm 48% coverage, nue2 100%; nuatm 0% -> SN on-the-fly instead.
PARTS={g:"("+",".join(f"'{p}'" for p in probs_parts(g))+")" for g in ('muatm_2020','nue2_2020')}
def qmc(dc,n,extra=""):
    return c.execute(f"""SELECT pr.score sc0, ev.data_class base, l.part_key, l.local_idx
      FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
      JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND ev.data_class IN ({dc}) {extra} ORDER BY random() LIMIT {n}""").df()
def qex(w,n):
    return c.execute(f"""SELECT pr.score sc0,'exp_full' base,l.part_key,l.local_idx
      FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
      JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND NOT(ev.cluster=2 AND ev.run IN('20','249')) AND {w} ORDER BY random() LIMIT {n}""").df()
dfs={
 'muatm':qmc("'muatm_2020'",45000,extra=f"AND l.part_key IN {PARTS['muatm_2020']}"),
 'nuatm':qmc("'nuatm_2020'",20000),   # 0 h8s3 in probs-parts -> SN on-the-fly
 'nue2': qmc("'nue2_2020'",45000,extra=f"AND l.part_key IN {PARTS['nue2_2020']}"),
 'exp_hi':qex("pr.score>0.8",4000),
 'exp_mid':qex("pr.score>0.5 AND pr.score<=0.8",4000),
 'exp_lo':qex("pr.score<0.2",5000),
}
c.close()

E={}; M={}
for name,df in dfs.items():
    h5,pr=(EXH5,EXPROBS) if name.startswith('exp') else (MCH5,MCPROBS)
    sn_otf = (name=='nuatm')   # nuatm precomputed probs cover only 5% of parts -> SN on-the-fly
    s,emb=embed(h5,pr,df,sn_on_the_fly=sn_otf); M[name]=s; E[name]=emb
    print(f'{name:8s}: {len(s):>6} events'+(' [SN on-the-fly]' if sn_otf else ''),flush=True)

# ---- A. corner composition & scores ----
print("\n=== A. bright (Q>%g) AND horizontal (rvert<%g): per class ==="%(Q_BRIGHT,RV_HZ))
print(f"{'class':>8} {'N_all':>7} {'N_horiz':>8} {'N_br_hz':>8} {'score med(br_hz)':>16} {'frac>0.8(br_hz)':>15}")
rowsA=[]
for name in ['muatm','nuatm','nue2']:
    s=M[name]; hz=s.rvert<RV_HZ; brhz=hz&(s.qmean>Q_BRIGHT); sub=s[brhz]
    rowsA.append(dict(cls=name,N_all=len(s),N_horiz=int(hz.sum()),N_br_hz=int(brhz.sum()),
        score_med_brhz=float(sub.score.median()) if len(sub) else np.nan,
        frac_gt08_brhz=float((sub.score>0.8).mean()) if len(sub) else np.nan))
    print(f"{name:>8} {len(s):>7} {int(hz.sum()):>8} {int(brhz.sum()):>8} "
          f"{(sub.score.median() if len(sub) else np.nan):>16.3f} {((sub.score>0.8).mean() if len(sub) else np.nan):>15.3f}")
pd.DataFrame(rowsA).to_csv(HERE/'tables/mc_nu_corner_A.csv',index=False)
# exp reference: exp_hi in that corner
eh=M['exp_hi']; ehc=eh[(eh.rvert<RV_HZ)&(eh.qmean>Q_BRIGHT)]
print(f"  (ref) exp_hi bright-horiz: N={len(ehc)}/{len(eh)}  Q med={eh.qmean.median():.2f} rvert med={eh.rvert.median():.3f}")

# ---- B. distance to MC-nu manifold ----
from sklearn.neighbors import NearestNeighbors
nu_emb=np.vstack([E['nuatm'],E['nue2']]); nu_meta=pd.concat([M['nuatm'],M['nue2']],ignore_index=True)
rng=np.random.default_rng(0); perm=rng.permutation(len(nu_emb)); nref=int(0.8*len(nu_emb))
ref=nu_emb[perm[:nref]]; cal_i=perm[nref:]
nn=NearestNeighbors(n_neighbors=KNN).fit(ref)
def kdist(emb): return nn.kneighbors(emb)[0].mean(1)
print("\n=== B. kNN distance to MC-nu (nuatm+nue2) manifold ===")
nu_brhz=(nu_meta.rvert<RV_HZ)&(nu_meta.qmean>Q_BRIGHT)
groups={
 'MC-nu (in-dist calib)':nu_emb[cal_i],
 'MC-nu bright-horiz':nu_emb[nu_brhz.to_numpy()],
 'exp_hi (false-nu)':E['exp_hi'],
 'exp_mid':E['exp_mid'],
 'exp_lo':E['exp_lo'],
 'muatm':E['muatm'],
}
rowsB=[]
for g,emb in groups.items():
    d=kdist(emb); rowsB.append(dict(group=g,N=len(emb),knn_nu_med=float(np.median(d)),knn_nu_p25=float(np.percentile(d,25)),knn_nu_p75=float(np.percentile(d,75))))
    print(f"  {g:>24}: N={len(emb):>6}  kNN-to-nu median={np.median(d):.3f}  [p25={np.percentile(d,25):.2f}, p75={np.percentile(d,75):.2f}]")
pd.DataFrame(rowsB).to_csv(HERE/'tables/mc_nu_corner_B.csv',index=False)

# ---- C. nearest MC-nu identity for exp_hi ----
nn1=NearestNeighbors(n_neighbors=5).fit(nu_emb)
di,ii=nn1.kneighbors(E['exp_hi'])
nearest=ii[:,0]
near_rvert=nu_meta.rvert.to_numpy()[nearest]; near_q=nu_meta.qmean.to_numpy()[nearest]
frac_near_brhz=float(np.mean((near_rvert<RV_HZ)&(near_q>Q_BRIGHT)))
frac_near_hz=float(np.mean(near_rvert<RV_HZ))
print("\n=== C. nearest MC-nu neighbour of exp_hi (false-nu) ===")
print(f"  frac whose nearest MC-nu is horizontal (rvert<{RV_HZ}): {frac_near_hz:.3f}")
print(f"  frac whose nearest MC-nu is bright+horizontal:          {frac_near_brhz:.3f}")
print(f"  nearest-nu median rvert={np.median(near_rvert):.3f}, median Q={np.median(near_q):.2f}")
pd.DataFrame([dict(frac_near_horiz=frac_near_hz,frac_near_bright_horiz=frac_near_brhz,
    near_rvert_med=float(np.median(near_rvert)),near_q_med=float(np.median(near_q)))]).to_csv(HERE/'tables/mc_nu_corner_C.csv',index=False)
print("\ndone.")
