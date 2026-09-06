#!/usr/bin/env python
"""Simple model-independent OOD heuristic on the score>0.8 region, used as a CUT.

Question: the exp neutrino-like excess is a narrow OOD tail (see §4c). If we build a
simple OOD score in PHYSICAL-observable space (not embeddings, to avoid circularity)
measuring how far an exp event is from the MC training manifold, and cut on it —
    (1) does the excess disappear?
    (2) how many in-distribution high-score exp events remain (candidate real ν)?
    (3) control: do real MC neutrinos (high-score) survive the same cut?

Physical feature vector per event (filtered hits, prob>0.8):
  log10(nhits), nstrings, log10(<Q>), log10(maxQ), std(Q)/<Q>, std(t),
  std(z), std(xy), rvert=std_z/std_xy.
OOD score = Mahalanobis to the MC (muon+ν) distribution in this standardized space.
Threshold calibrated on held-out MC percentiles (keep X% of MC).

Restricted to score>0.8 for exp (the relevant, small set). Outputs
tables/ood_cut.csv, figures/ood_cut.png.
"""
from __future__ import annotations
from pathlib import Path
import duckdb, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
CAT = ROOT/'data_manager/catalog_v2.duckdb'
PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
MCH5 = ROOT/'data_manager/data/h5datasets/baikal_mc_merged.h5'
EXH5 = ROOT/'data_manager/data/h5datasets/exp_full.h5'
MCPROBS = ROOT/'data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
EXPROBS = ROOT/'data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5'
RDCC = dict(rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003); THR = 0.8
CUT = 'n_sn_hits>=5'    # h5s0 (inclusive) — keep the full relevant set

FEATNAMES = ['log_nhits','nstrings','log_meanQ','log_maxQ','Qrelstd','t_std','z_std','xy_std','rvert']

def feats_from_hits(h5path, probs_path, df):
    """Physical feature matrix (N, 9) from filtered hits; rows with <2 hits -> NaN."""
    X = np.full((len(df), len(FEATNAMES)), np.nan, np.float32)
    f = h5py.File(h5path,'r',**RDCC); fp = h5py.File(probs_path,'r',**RDCC)
    for (base,pk), idx in df.groupby(['base','part_key']).groups.items():
        try:
            es = f[f'{base}/raw/ev_starts/{pk}/data'][:]; ds = f[f'{base}/raw/data/{pk}/data']
            pr = fp[f'{base}/probs/{pk}/data']
        except KeyError:
            continue
        for row, l in zip(np.array(idx), df.loc[idx,'local_idx'].to_numpy()):
            if l+1 >= len(es): continue
            s,e = int(es[l]), int(es[l+1]); p = pr[s:e].astype(np.float32); m = p > THR
            if m.sum() < 2: continue
            hh = ds[s:e].astype(np.float32)[m]
            Q, t = hh[:,0], hh[:,1]; x,y,z = hh[:,2], hh[:,3], hh[:,4]
            zs, xys = z.std(), np.sqrt(x.std()**2 + y.std()**2)
            X[row] = [np.log10(len(hh)), 0.0,  # nstrings filled from DB below
                      np.log10(max(Q.mean(),1e-3)), np.log10(max(Q.max(),1e-3)),
                      Q.std()/max(Q.mean(),1e-3), t.std(), zs, xys, zs/(xys+1e-3)]
    f.close(); fp.close(); return X

c = duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")

def q_mc(where, n):
    return c.execute(f"""SELECT pr.score, pr.n_sn_strings, ev.data_class AS base, l.part_key, l.local_idx
      FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
      JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND {where} ORDER BY random() LIMIT {n}""").df()
def q_exp(where, n=None):
    lim = f"ORDER BY random() LIMIT {n}" if n else ""
    return c.execute(f"""SELECT pr.score, pr.n_sn_strings, 'exp_full' AS base, l.part_key, l.local_idx
      FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
      JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249')) AND {where} {lim}""").df()

Nexp_tot = c.execute(f"SELECT count(*) FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk "
                     f"WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249'))").fetchone()[0]

pops = {
    'mc_ref':    q_mc("ev.data_class IN ('muatm_2020','nuatm_2020','nue2_2020')", 12000),  # manifold
    'mc_nu_hi':  q_mc("ev.data_class IN ('nuatm_2020','nue2_2020') AND pr.score>0.8", 4000),# real ν control
    'exp_hi':    q_exp("pr.score>0.8"),          # ALL held-out exp false-ν candidates
}
c.close()

data = {}
for name, df in pops.items():
    h5, pr = (EXH5, EXPROBS) if name.startswith('exp') else (MCH5, MCPROBS)
    X = feats_from_hits(h5, pr, df)
    X[:,1] = df['n_sn_strings'].to_numpy()           # nstrings from DB
    ok = np.isfinite(X).all(1)
    data[name] = (df[ok].reset_index(drop=True), X[ok])
    print(f'{name:9s}: {ok.sum():>5}/{len(df)} events')

# standardize on mc_ref, Mahalanobis to mc_ref
Xref = data['mc_ref'][1]
mu, sd = Xref.mean(0), Xref.std(0)+1e-6
Z = lambda X: (X-mu)/sd
cov = np.cov(Z(Xref).T) + 1e-3*np.eye(len(FEATNAMES)); P = np.linalg.pinv(cov)
maha = lambda X: np.einsum('ij,jk,ik->i', Z(X), P, Z(X))
d_ref = maha(Xref); d_nu = maha(data['mc_nu_hi'][1]); d_exp = maha(data['exp_hi'][1])

print(f'\nMahalanobis (physical-feature OOD):')
print(f'  MC ref manifold : median={np.median(d_ref):.1f}  p95={np.percentile(d_ref,95):.1f}  p99={np.percentile(d_ref,99):.1f}  p99.9={np.percentile(d_ref,99.9):.1f}')
print(f'  MC ν high-score : median={np.median(d_nu):.1f}  frac>p99(ref)={np.mean(d_nu>np.percentile(d_ref,99)):.3f}')
print(f'  exp high-score  : median={np.median(d_exp):.1f}  frac>p99(ref)={np.mean(d_exp>np.percentile(d_ref,99)):.3f}  (N={len(d_exp)})')

# OOD cut sweep: keep exp_hi with maha <= MC percentile threshold
rows=[]
for pct in [90,95,99,99.9]:
    thr = np.percentile(d_ref, pct)
    surv_exp = int((d_exp <= thr).sum())
    surv_nu  = float((d_nu <= thr).mean())      # real-ν retention (control)
    rows.append(dict(mc_keep_pct=pct, ood_thr=round(thr,1),
                     exp_hi_survive=surv_exp, exp_hi_total=len(d_exp),
                     implied_frac=surv_exp/Nexp_tot, real_nu_retention=round(surv_nu,3)))
tab=pd.DataFrame(rows); tab.to_csv(HERE/'tables/ood_cut.csv', index=False)
print(f'\nexp held-out total (score-any, {CUT}) = {Nexp_tot:,}; physical ν expectation ~1e-6..1e-5 => ~{Nexp_tot*1e-6:.1f}..{Nexp_tot*1e-5:.0f} events')
print(tab.to_string(index=False))

# figure
fig,ax=plt.subplots(figsize=(8,5))
for d,lab,cl in [(d_ref,'MC ref manifold','tab:red'),(d_nu,'MC ν (score>0.8)','tab:green'),(d_exp,'exp (score>0.8)','black')]:
    ax.hist(np.log10(d+1), bins=50, density=True, histtype='step', lw=2, color=cl, label=lab)
ax.axvline(np.log10(np.percentile(d_ref,99)+1), color='grey', ls='--', label='MC p99 cut')
ax.set_xlabel('log10 physical-feature Mahalanobis to MC'); ax.set_ylabel('density')
ax.set_title('OOD heuristic (physical features) on the score>0.8 region')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE/'figures/ood_cut.png', dpi=130)
print('saved figures/ood_cut.png')
