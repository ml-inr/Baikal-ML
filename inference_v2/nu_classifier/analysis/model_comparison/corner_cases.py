#!/usr/bin/env python
"""Do the two candidate corner-cases exist in MC muatm, or are they exp-specific?

Case 1 — an isolated passed-noise hit far from the track (top/bottom of detector) inflates
verticality (rvert) and can flip direction (corr(t,z)). Measured geometrically by the
leave-one-out rvert drop (remove the single largest z-outlier hit) and, for muatm, by TRUTH:
per-hit labels (raw/labels: muon-hit id >0 vs noise ≤0) → count passed-noise hits among the
filtered hits.

Case 2 — two weak near-horizontal muons, one early+low and one late+high, together look
up-going. Measured geometrically by a two-cluster gap (early vs late time-half centroids
separated in space) with up-going corr; for muatm by TRUTH: muon multiplicity
prime_prty[:,4] (n_muons) — is a high-score up-going muatm event a real multi-muon, and does
MC (single-shower, correlated bundles) ever fake up-going?

Compares MC muatm (with truth) to the exp super-confident candidates (geometry only).
Outputs tables/corner_cases_muatm.csv + prints the comparison.
"""
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parents[0]; ROOT = HERE.parents[3]
PREDS = ROOT/'inference_v2/nu_classifier/preds'; CAT = ROOT/'data_manager/catalog_v2.duckdb'
FT='260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model'
MCH5=ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
MCPROBS=ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXH5=ROOT/'data_manager/data/h5datasets/exp_full.h5'
EXPROBS=ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC=dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR=0.8
CUT='pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'


def geom(hh):
    """rvert, corr(t,z), leave-one-out rvert drop (remove biggest z-outlier), two-cluster gap."""
    t,x,y,z=hh[:,1],hh[:,2],hh[:,3],hh[:,4]
    sx,sy,sz=x.std(),y.std(),z.std()
    rvert=sz/(np.sqrt(sx*sx+sy*sy)+1e-3)
    corr=np.corrcoef(t,z)[0,1] if (t.std()>1e-6 and z.std()>1e-6) else 0.0
    # case 1: remove the single hit farthest from median z, recompute rvert
    j=np.argmax(np.abs(z-np.median(z))); keep=np.ones(len(hh),bool); keep[j]=False
    hk=hh[keep]; sxk,syk,szk=hk[:,2].std(),hk[:,3].std(),hk[:,4].std()
    rvert_lo=szk/(np.sqrt(sxk*sxk+syk*syk)+1e-3)
    lo_drop=1.0-rvert_lo/(rvert+1e-9)                       # ~1 => rvert driven by one hit
    corr_lo=np.corrcoef(hk[:,1],hk[:,4])[0,1] if (hk[:,1].std()>1e-6 and hk[:,4].std()>1e-6) else 0.0
    corr_flip=int(np.sign(corr)!=np.sign(corr_lo) and abs(corr)>0.3)
    # case 2: split by median time into early/late, spatial gap between the two centroids
    order=np.argsort(t); half=len(hh)//2
    e=hh[order[:half],2:5]; l=hh[order[half:],2:5]
    gap=np.linalg.norm(l.mean(0)-e.mean(0))
    spread=0.5*(np.linalg.norm(e.std(0))+np.linalg.norm(l.std(0)))+1e-3
    two_clu=gap/spread                                      # high => two separated blobs
    return rvert,corr,lo_drop,corr_flip,two_clu


# ---- MC muatm: FT-scored sample with truth ----
c=duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PREDS/FT/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
mu=c.execute(f"""SELECT pr.score, l.part_key, l.local_idx FROM m.predictions pr
  JOIN cat.events ev ON ev.id=pr.event_fk JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
  WHERE {CUT} AND ev.data_class='muatm_2020' AND pr.score>0.5 ORDER BY random() LIMIT 8000""").df()
c.close()
print(f'muatm FT>0.5 sample: {len(mu)}')

rows=[]
f=h5py.File(MCH5,'r',**RDCC); fp=h5py.File(MCPROBS,'r',**RDCC)
for pk,idx in mu.groupby('part_key').groups.items():
    try:
        es=f[f'muatm_2020/raw/ev_starts/{pk}/data'][:]; ds=f[f'muatm_2020/raw/data/{pk}/data']
        lbl=f[f'muatm_2020/raw/labels/{pk}/data']; pp=f[f'muatm_2020/prime_prty/{pk}/data']
        pr=fp[f'muatm_2020/probs/{pk}/data']
    except KeyError: continue
    for row,l in zip(np.array(idx), mu.loc[idx,'local_idx'].to_numpy()):
        if l+1>=len(es): continue
        s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
        if m.sum()<2: continue
        hh=ds[s:e].astype(np.float32)[m]; hh=hh[np.argsort(hh[:,1])]
        lab=lbl[s:e][m]                              # per-filtered-hit truth label
        n_noise=int((lab<=0).sum())                  # passed-noise hits among filtered
        n_mu=int(pp[l,4])                            # muon multiplicity (truth)
        rv,co,lod,cflip,tclu=geom(hh)
        rows.append(dict(score=mu.loc[row,'score'],rvert=rv,corr=co,lo_drop=lod,corr_flip=cflip,
                         two_clu=tclu,n_noise=n_noise,n_filt=len(hh),n_mu=n_mu))
f.close(); fp.close()
d=pd.DataFrame(rows); d.to_csv(HERE/'tables/corner_cases_muatm.csv',index=False)
print(f'muatm processed: {len(d)}')

def frac(df,m): return float(m.mean()) if len(df) else np.nan
hi=d[d.score>0.9]; up=d[(d.score>0.9)&(d['corr']>0.3)]
print(f"\n=== MC muatm high-FT-score (>0.9): N={len(hi)} ===")
print(f"  up-going-like (corr>0.3):        {frac(hi,hi['corr']>0.3):.3f}")
print(f"  vertical (rvert>0.9):            {frac(hi,hi.rvert>0.9):.3f}")
print(f"\n  CASE 1 (isolated noise hit):")
print(f"    has >=1 passed-noise hit:      {frac(hi,hi.n_noise>0):.3f}")
print(f"    rvert driven by 1 hit(lo_drop>0.5): {frac(hi,hi.lo_drop>0.5):.3f}")
print(f"    direction flips on 1-hit removal:   {frac(hi,hi.corr_flip>0):.3f}")
print(f"\n  CASE 2 (multi-muon / two-cluster):")
print(f"    n_muons>=2 (truth):           {frac(hi,hi.n_mu>=2):.3f}")
print(f"    two-cluster gap>2:            {frac(hi,hi.two_clu>2):.3f}")
print(f"    up-going AND n_muons>=2:      {frac(hi,(hi['corr']>0.3)&(hi.n_mu>=2)):.3f}")
print(f"  --- among up-going high-score muatm (N={len(up)}): n_mu>=2 frac={frac(up,up.n_mu>=2):.3f}, has-noise frac={frac(up,up.n_noise>0):.3f} ---")

# ---- exp super-confident candidates (geometry only) ----
cand=pd.read_csv(HERE/'tables/nu_candidates_full.csv')
ec=[]
f=h5py.File(EXH5,'r',**RDCC); fp=h5py.File(EXPROBS,'r',**RDCC)
for pk,idx in cand.groupby('part_key').groups.items():
    try: es=f[f'exp_full/raw/ev_starts/{pk}/data'][:]; ds=f[f'exp_full/raw/data/{pk}/data']; pr=fp[f'exp_full/probs/{pk}/data']
    except KeyError: continue
    for row,l in zip(np.array(idx), cand.loc[idx,'local_idx'].to_numpy()):
        if l+1>=len(es): continue
        s,e=int(es[l]),int(es[l+1]); p=pr[s:e].astype(np.float32); m=p>THR
        if m.sum()<2: continue
        hh=ds[s:e].astype(np.float32)[m]; hh=hh[np.argsort(hh[:,1])]
        rv,co,lod,cflip,tclu=geom(hh); ec.append(dict(score=cand.loc[row,'score'],rvert=rv,corr=co,lo_drop=lod,corr_flip=cflip,two_clu=tclu))
f.close(); fp.close()
ce=pd.DataFrame(ec)
print(f"\n=== exp super-confident candidates (score>0.9): N={len(ce)} (geometry only) ===")
print(f"  CASE 1  rvert driven by 1 hit(lo_drop>0.5): {frac(ce,ce.lo_drop>0.5):.3f}   direction flips: {frac(ce,ce.corr_flip>0):.3f}")
print(f"  CASE 2  two-cluster gap>2:                  {frac(ce,ce.two_clu>2):.3f}")
print(f"\ncompare (case1 lo_drop>0.5 | case2 two_clu>2):  muatm[{frac(hi,hi.lo_drop>0.5):.2f} | {frac(hi,hi.two_clu>2):.2f}]  exp-cand[{frac(ce,ce.lo_drop>0.5):.2f} | {frac(ce,ce.two_clu>2):.2f}]")
