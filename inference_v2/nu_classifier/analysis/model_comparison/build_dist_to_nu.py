#!/usr/bin/env python
"""Paper figure built around the *distance to the MC-nu manifold* (Report 2026-07-08 §8):
the high-score experimental false-neutrinos are bright near-horizontal events attracted to
the neutrino cluster in feature space --- close to MC-nu (kNN~3) and far from muons (kNN~11),
i.e. just beyond the nu edge, not in a void.

Two separate reference manifolds (base E1@ep10 encoder embeddings):
  d_nu = mean kNN(k=20) distance to a MC-nu (nuatm+nue2) reference,
  d_mu = mean kNN(k=20) distance to a MC-EAS (muatm) reference.
Held-out eval draws (disjoint from references). Populations: MC nu, MC EAS, exp bulk, and
exp false-nu (score>0.8).

Panel (a): distribution of d_nu for MC-nu / MC-EAS / exp false-nu / exp bulk --- the
0.8 / 3 / 11 ordering. Panel (b): d_mu vs d_nu, exp coloured by score, with MC-nu and
MC-EAS medians --- false-nu sit on the nu side (low d_nu, high d_mu) but offset from the core.
Saves tables/dist_to_nu.csv, figures/nu_classifier_dist_to_nu.png. cuda:0.
"""
from pathlib import Path
import sys, duckdb, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores_and_embeddings
from inference_v2.nu_classifier.exp_finetuning.build_exp_bg_ood import (
    read_filtered, MCH5, EXH5, MCPROBS, EXPROBS, CAT)

CKPT = ROOT/'experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01/da_checkpoint_epoch_010.pth'
PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
CUT = 'pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'
DEV = 'cuda:0'; BS = 256; SN = 0.8; KNN = 20
NU_REF, MU_REF, EVAL_NU, EVAL_MU, EXP_RAND, EXP_HI = 15000, 15000, 10000, 10000, 30000, 3000

c = duckdb.connect(); c.execute('PRAGMA disable_progress_bar')
c.execute(f"ATTACH '{PRED/'exp_full_thr0p8.duckdb'}' AS e (READ_ONLY)")
c.execute(f"ATTACH '{PRED/'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
def q_mc(classes, n):
    cl = ",".join(f"'{x}'" for x in classes)
    return c.execute(f"""SELECT pr.score, ev.data_class AS base, l.part_key, l.local_idx
        FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
        JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
        WHERE {CUT} AND ev.data_class IN ({cl}) ORDER BY random() LIMIT {n}""").df()
def q_exp(extra, n):
    return c.execute(f"""SELECT pr.score, 'exp_full' AS base, l.part_key, l.local_idx
        FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
        JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
        WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249')) {extra}
        ORDER BY random() LIMIT {n}""").df()
nu_pool = q_mc(['nuatm_2020','nue2_2020'], NU_REF + EVAL_NU)
mu_pool = q_mc(['muatm_2020'], MU_REF + EVAL_MU)
exp_rand = q_exp('', EXP_RAND)
exp_hi   = q_exp('AND pr.score > 0.8', EXP_HI)
c.close()
print(f'nu_pool {len(nu_pool)}, mu_pool {len(mu_pool)}, exp_rand {len(exp_rand)}, exp_hi {len(exp_hi)}', flush=True)

model, norm, _ = load_model(str(CKPT), device=DEV)
def embed(df, h5, probs):
    feats, *_ , rvert = read_filtered(h5, True, probs, df, SN)
    keep = np.array([i for i, x in enumerate(feats) if x is not None])
    sc, emb = predict_scores_and_embeddings(model, [feats[i] for i in keep], norm, batch_size=BS, device=DEV)
    return emb.astype(np.float32), df.iloc[keep]['score'].to_numpy(), rvert[keep]

nu_emb, _, _        = embed(nu_pool, MCH5, MCPROBS)
mu_emb, _, _        = embed(mu_pool, MCH5, MCPROBS)
er_emb, er_sc, er_rv = embed(exp_rand, EXH5, EXPROBS)
eh_emb, eh_sc, eh_rv = embed(exp_hi, EXH5, EXPROBS)
print('embedded all', flush=True)

def split(E, frac=0.7):  # robust to read_filtered survival: split by fraction
    k = int(frac * len(E)); return E[:k], E[k:]
nu_ref, nu_eval = split(nu_emb)
mu_ref, mu_eval = split(mu_emb)
print(f'nu_ref {len(nu_ref)} nu_eval {len(nu_eval)} | mu_ref {len(mu_ref)} mu_eval {len(mu_eval)}', flush=True)
nn_nu = NearestNeighbors(n_neighbors=KNN).fit(nu_ref)
nn_mu = NearestNeighbors(n_neighbors=KNN).fit(mu_ref)
dnu = lambda E: nn_nu.kneighbors(E)[0].mean(1)
dmu = lambda E: nn_mu.kneighbors(E)[0].mean(1)

pops = {
    'MC $\\nu$':        (dnu(nu_eval), dmu(nu_eval), None),
    'MC EAS':           (dnu(mu_eval), dmu(mu_eval), None),
    'exp bulk':         (dnu(er_emb),  dmu(er_emb),  er_sc),
    'exp false-$\\nu$': (dnu(eh_emb),  dmu(eh_emb),  eh_sc),  # score>0.8
}
for k, (dv, dm, _) in pops.items():
    print(f'{k:16s} d_nu median {np.median(dv):.2f}  d_mu median {np.median(dm):.2f}', flush=True)
pd.concat([pd.DataFrame(dict(pop=k, d_nu=dv, d_mu=dm,
                            score=(sc if sc is not None else np.full(len(dv), np.nan))))
           for k,(dv,dm,sc) in pops.items()]).to_csv(HERE/'tables/dist_to_nu.csv', index=False)

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
b = np.linspace(0, 13, 70)
for k, cl in [('MC $\\nu$','tab:green'), ('MC EAS','tab:blue'), ('exp false-$\\nu$','black'), ('exp bulk','0.6')]:
    ax[0].hist(pops[k][0], bins=b, density=True, histtype='step', lw=2, color=cl, label=k)
ax[0].set_xlabel('distance to MC-$\\nu$ manifold  (mean kNN, $k{=}20$)'); ax[0].set_ylabel('normalised density')
ax[0].set_title('(a) Attraction to the $\\nu$ cluster'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].scatter(pops['MC EAS'][1], pops['MC EAS'][0], s=4, alpha=0.06, color='tab:blue', edgecolors='none')
ax[1].scatter(pops['MC $\\nu$'][1], pops['MC $\\nu$'][0], s=4, alpha=0.12, color='tab:green', edgecolors='none')
dv, dm, sc = pops['exp bulk']
o = np.argsort(er_sc); scat = ax[1].scatter(dm[o], dv[o], c=er_sc[o], s=np.where(er_sc[o]>0.8,20,5),
                                             alpha=0.5, cmap='viridis', vmin=0, vmax=1, edgecolors='none')
ax[1].plot(np.median(pops['MC $\\nu$'][1]), np.median(pops['MC $\\nu$'][0]), '*', ms=18, color='tab:green', mec='k', label='MC $\\nu$ (median)')
ax[1].plot(np.median(pops['MC EAS'][1]),   np.median(pops['MC EAS'][0]),   '*', ms=18, color='tab:blue',  mec='k', label='MC EAS (median)')
ax[1].plot(np.median(pops['exp false-$\\nu$'][1]), np.median(pops['exp false-$\\nu$'][0]), 'X', ms=13, color='red', mec='k', label='exp false-$\\nu$ (median)')
ax[1].set_xlabel('distance to MC-EAS manifold  $d_\\mu$'); ax[1].set_ylabel('distance to MC-$\\nu$ manifold  $d_\\nu$')
ax[1].set_title('(b) Which cluster? ($\\nu$ vs EAS)'); ax[1].legend(fontsize=8, loc='upper right'); ax[1].grid(alpha=0.3)
cb = fig.colorbar(scat, ax=ax[1]); cb.set_label('classifier score $\\xi$')
fig.tight_layout(); fig.savefig(HERE/'figures/nu_classifier_dist_to_nu.png', dpi=140)
print('saved figures/nu_classifier_dist_to_nu.png', flush=True)
