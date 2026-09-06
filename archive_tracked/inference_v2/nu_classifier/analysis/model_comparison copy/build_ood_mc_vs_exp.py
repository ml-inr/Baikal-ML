#!/usr/bin/env python
"""Modified OOD-selection figure: encoder-OOD compared between MC and exp.

encoder-OOD = mean kNN(k=20) distance to a COMBINED Monte Carlo reference manifold
(random muatm+nuatm+nue2, exactly as used to build the fine-tune background — the base
network's encoder defines the space). The reference is one and the same for all
populations; MC EAS and MC ν are evaluated held-out (disjoint random draws) so both
in-distribution simulations are shown separately alongside exp.

Panel (a): OOD distribution for MC EAS (muatm), MC ν (nuatm+nue2) and exp, one shared
combined reference; cut at OOD>2.5. Panel (b): rvert vs OOD for exp, coloured by score.

Saves tables/ood_mc_vs_exp.csv and figures/nu_classifier_ood_selection.png (base ckpt,
cuda:0). Reuses read_filtered + model load from the build script.
"""
from pathlib import Path
import sys, duckdb, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from inference_v2.shared.model_utils import load_model, predict_scores_and_embeddings
from archive_tracked.inference_v2.nu_classifier.exp_finetuning.build_exp_bg_ood import (
    read_filtered, MCH5, EXH5, MCPROBS, EXPROBS, CAT)

CKPT = ROOT/'experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01/da_checkpoint_epoch_010.pth'
PRED = ROOT/'inference_v2/nu_classifier/preds/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010'
CUT = 'pr.n_sn_hits>=8 AND pr.n_sn_strings>=3'
DEV = 'cuda:0'; BS = 256; SN = 0.8; KNN = 20; OOD_CUT, RV_CUT = 2.5, 0.7
# Class-BALANCED reference (1:1 EAS:nu, matching the classifier's 2:1:1 training mixture).
# muatm and nu each drawn as one pool then split ref/eval disjoint (no self-match).
N_REF_MU, N_REF_NU, N_MU, N_NU, N_EXP = 20000, 20000, 15000, 15000, 40000

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

# draw pools, split ref/eval disjoint
mu_pool = q_mc(['muatm_2020'], N_REF_MU + N_MU)
nu_pool = q_mc(['nuatm_2020','nue2_2020'], N_REF_NU + N_NU)
ref = pd.concat([mu_pool.iloc[:N_REF_MU], nu_pool.iloc[:N_REF_NU]], ignore_index=True)  # 1:1 EAS:nu
mu  = mu_pool.iloc[N_REF_MU:].reset_index(drop=True)   # held-out EAS
nu  = nu_pool.iloc[N_REF_NU:].reset_index(drop=True)   # held-out nu
exp  = c.execute(f"""SELECT pr.score, 'exp_full' AS base, l.part_key, l.local_idx
    FROM e.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
    JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
    WHERE {CUT} AND NOT (ev.cluster=2 AND ev.run IN ('20','249'))
    ORDER BY random() LIMIT {N_EXP}""").df()
c.close()
print(f'ref balanced {len(ref)} (muatm frac {(ref.base=="muatm_2020").mean():.2f}); '
      f'eval mu {len(mu)}, nu {len(nu)}, exp {len(exp)}', flush=True)

model, norm, _ = load_model(str(CKPT), device=DEV)

def embed(df, h5, probs):
    feats, nsh, nss, rvert = read_filtered(h5, True, probs, df, SN)
    keep = np.array([i for i, x in enumerate(feats) if x is not None])
    fl = [feats[i] for i in keep]
    sc, emb = predict_scores_and_embeddings(model, fl, norm, batch_size=BS, device=DEV)
    return keep, emb.astype(np.float32), rvert[keep], sc

kr, ref_emb, _, _  = embed(ref, MCH5, MCPROBS);  print(f'ref embedded {len(ref_emb)}', flush=True)
km, mu_emb, mu_rv, _   = embed(mu, MCH5, MCPROBS)
kn, nu_emb, nu_rv, _   = embed(nu, MCH5, MCPROBS)
ke, ex_emb, ex_rv, ex_sc = embed(exp, EXH5, EXPROBS)
print('all embedded', flush=True)

nn = NearestNeighbors(n_neighbors=KNN).fit(ref_emb)
mu_ood = nn.kneighbors(mu_emb)[0].mean(1)
nu_ood = nn.kneighbors(nu_emb)[0].mean(1)
ex_ood = nn.kneighbors(ex_emb)[0].mean(1)

# save per-event OOD + rvert for cheap replotting
pd.concat([
    pd.DataFrame(dict(pop='MC EAS', ood=mu_ood, rvert=mu_rv)),
    pd.DataFrame(dict(pop='MC nu',  ood=nu_ood, rvert=nu_rv)),
    pd.DataFrame(dict(pop='exp',    ood=ex_ood, rvert=ex_rv, score=ex_sc)),
]).to_csv(HERE/'tables/ood_mc_vs_exp.csv', index=False)
for nm, o, r in [('MC EAS', mu_ood, mu_rv), ('MC nu', nu_ood, nu_rv), ('exp', ex_ood, ex_rv)]:
    h = r < RV_CUT
    print(f'{nm:8s} OOD med {np.median(o):.2f} frac>2.5 {(o>OOD_CUT).mean():.3f} | '
          f'horizontal(rvert<0.7): med {np.median(o[h]):.2f} frac>2.5 {(o[h]>OOD_CUT).mean():.3f} (N={h.sum()})', flush=True)

# ---- figure: panel (a) conditioned on NEAR-HORIZONTAL events (rvert<0.7), matching the selection ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
b = np.linspace(0, 8, 70)
for o, r, cl, lab in [(mu_ood, mu_rv, 'tab:blue', 'MC EAS (muatm)'),
                      (nu_ood, nu_rv, 'tab:green', 'MC $\\nu$ (nuatm+nue2)'),
                      (ex_ood, ex_rv, 'black', 'experiment')]:
    ax[0].hist(o[r < RV_CUT], bins=b, density=True, histtype='step', lw=2, color=cl, label=lab)
ax[0].axvline(OOD_CUT, ls='--', color='tab:red'); ax[0].text(OOD_CUT+0.08, ax[0].get_ylim()[1]*0.9, 'cut = 2.5', rotation=90, color='tab:red', fontsize=9, va='top')
ax[0].set_yscale('log'); ax[0].set_xlabel('encoder-OOD  (mean kNN distance to MC manifold, $k{=}20$)')
ax[0].set_ylabel('normalised density'); ax[0].set_title('(a) OOD of near-horizontal events ($r_\\mathrm{vert}<0.7$)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which='both')

d = pd.DataFrame(dict(rvert=ex_rv, ood=ex_ood, score=ex_sc)).sort_values('score')
sizes = np.where(d.score > 0.8, 22, 6)
sc = ax[1].scatter(d.rvert, d.ood, c=d.score, s=sizes, alpha=0.55, cmap='viridis', vmin=0, vmax=1, edgecolors='none')
ax[1].add_patch(Rectangle((0, OOD_CUT), RV_CUT, 8-OOD_CUT, fill=False, ec='tab:red', lw=2, ls='--'))
ax[1].axvline(RV_CUT, ls=':', color='0.4'); ax[1].axhline(OOD_CUT, ls=':', color='0.4')
ax[1].text(RV_CUT*0.5, 6.8, 'selected\nbackground', color='tab:red', ha='center', fontsize=9, fontweight='bold')
ax[1].set_xlim(0, 3); ax[1].set_ylim(0, 8)
ax[1].set_xlabel('verticality  $r_\\mathrm{vert}=\\sigma_z/\\sigma_{xy}$'); ax[1].set_ylabel('encoder-OOD')
ax[1].set_title('(b) Selection corner (experiment)'); ax[1].grid(alpha=0.3)
cb = fig.colorbar(sc, ax=ax[1]); cb.set_label('classifier score $\\xi$')
fig.tight_layout(); fig.savefig(HERE/'figures/nu_classifier_ood_selection.png', dpi=140)
print('saved figures/nu_classifier_ood_selection.png', flush=True)
