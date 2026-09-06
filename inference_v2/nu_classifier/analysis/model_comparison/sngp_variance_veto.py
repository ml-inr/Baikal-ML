#!/usr/bin/env python
"""Is the SNGP predictive variance useful as an explicit OOD veto?

Stage 2 (sngp_working_point.py) showed the mean-field score alone does not reduce the
experimental excess: shrinking logits by 1/sqrt(1+lambda*var) is too weak when the false
neutrinos sit at the *edge* of the simulated-neutrino distribution rather than far outside
it. But the variance is still measured per event, and the pilot found it ~3x higher for the
experimental false neutrinos than for genuine simulated neutrinos. So the question this
script answers is different: used as a separate cut, does gp_var remove experimental excess
faster than it removes genuine neutrinos?

Method: fix the working point on SNGP's own out-of-training muatm scores (mu = muon
survival). Among events above that score cut, additionally require gp_var < v, scanning v
over quantiles of the MC-nu variance distribution. For each v report the excess (exp/mu,
recomputed with the muon survival *after* the same variance cut, so the working point stays
matched) and the equal-weighted nu efficiency. A useful veto lowers the excess while keeping
nu efficiency; a useless one moves both together.

Outputs tables/sngp_variance_veto.csv + figures/sngp_variance_veto.png.
"""
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 18, "axes.labelsize": 18, "axes.titlesize": 18,
                     "legend.fontsize": 14, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "figure.dpi": 120})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREDS = ROOT / "inference_v2/nu_classifier/preds"
CAT = ROOT / "data_manager/catalog_v2.duckdb"
FT_BG = ROOT / ("inference_v2/nu_classifier/exp_finetuning/exp_bg_datasets/"
                "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010"
                "_ood2p5_rv0p7/exp_bg_event_fks.npy")
SNGP = "sngp_nu_classifier_baseline@best_sngp_model"
CUT = "p.n_sn_hits>=8 AND p.n_sn_strings>=3"
MU_POINTS = [1e-2, 3e-3, 1e-3]          # where muon statistics are solid
VAR_Q = [1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3]   # quantiles of the MC-nu variance

ft_bg = set(np.load(FT_BG).astype(np.int64).tolist())


def fetch(source, classes=None):
    c = duckdb.connect()
    c.execute("PRAGMA disable_progress_bar")
    c.execute(f"ATTACH '{PREDS / SNGP / f'{source}_thr0p8.duckdb'}' AS d (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    where = CUT
    if classes:
        where += " AND ev.data_class IN (" + ",".join(f"'{x}'" for x in classes) + ")"
    else:
        where += " AND NOT (ev.cluster=2 AND ev.run IN ('20','249'))"
    df = c.execute(f"""SELECT p.event_fk, p.score, p.gp_var, ev.data_class
                       FROM d.predictions p JOIN cat.events ev ON ev.id=p.event_fk
                       WHERE {where}""").df()
    c.close()
    return df


mc = fetch("mc_merged", ["muatm_2020", "nuatm_2020", "nue2_2020"])
ex = fetch("exp_full")
ex = ex[~ex.event_fk.isin(ft_bg)]
mu = mc[mc.data_class == "muatm_2020"]
na = mc[mc.data_class == "nuatm_2020"]
ne = mc[mc.data_class == "nue2_2020"]
print(f"muatm={len(mu):,}  nuatm={len(na):,}  nue2={len(ne):,}  exp={len(ex):,}", flush=True)

nu_var = np.concatenate([na.gp_var.to_numpy(), ne.gp_var.to_numpy()])
print("gp_var medians — MC nu %.4f | MC mu %.4f | exp %.4f"
      % (np.median(nu_var), mu.gp_var.median(), ex.gp_var.median()), flush=True)

rows = []
for mu_target in MU_POINTS:
    score_cut = float(np.quantile(mu.score.to_numpy(), 1 - mu_target))
    for q in VAR_Q:
        v_cut = float(np.quantile(nu_var, q)) if q < 1.0 else np.inf
        keep = lambda d: (d.score.to_numpy() > score_cut) & (d.gp_var.to_numpy() < v_cut)
        n_mu = int(keep(mu).sum())
        mu_eff = n_mu / len(mu)                       # muon survival after both cuts
        exp_frac = float(keep(ex).mean())
        eff = 0.5 * (float(keep(na).mean()) + float(keep(ne).mean()))
        rows.append(dict(mu_target=mu_target, var_q=q, var_cut=v_cut, score_cut=score_cut,
                         n_mu=n_mu, mu_surv=mu_eff, exp_frac=exp_frac,
                         excess=exp_frac / mu_eff if mu_eff > 0 else np.nan,
                         nu_eff=eff, n_exp=int(keep(ex).sum())))

tab = pd.DataFrame(rows)
tab.to_csv(HERE / "tables/sngp_variance_veto.csv", index=False)

for mu_target in MU_POINTS:
    t = tab[tab.mu_target == mu_target]
    print(f"\n=== working point mu={mu_target:.0e} (score cut {t.score_cut.iloc[0]:.4f}) ===")
    print(t[["var_q", "var_cut", "n_mu", "excess", "nu_eff", "n_exp"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
for mu_target, cl in zip(MU_POINTS, ["steelblue", "darkorange", "crimson"]):
    t = tab[tab.mu_target == mu_target].sort_values("nu_eff")
    ax[0].plot(t.nu_eff, t.excess, "o-", color=cl, lw=1.8, label=rf"$\mu={mu_target:.0e}$")
    t2 = tab[tab.mu_target == mu_target].sort_values("var_q")
    ax[1].plot(t2.var_q, t2.excess, "o-", color=cl, lw=1.8, label=rf"$\mu={mu_target:.0e}$")
ax[0].axhline(1.0, ls=":", color="grey")
ax[0].set_xlabel(r"$\nu$ efficiency"); ax[0].set_ylabel(r"excess  exp$/\mu$")
ax[0].set_title("Veto trade-off")
ax[1].axhline(1.0, ls=":", color="grey")
ax[1].set_xlabel(r"variance cut (quantile of MC-$\nu$ variance)")
ax[1].set_ylabel(r"excess  exp$/\mu$")
ax[1].set_title("Excess vs variance cut")
for a in ax:
    a.legend(); a.minorticks_on()
    a.grid(True, which="major", alpha=0.35); a.grid(True, which="minor", alpha=0.18, ls=":")
fig.tight_layout()
fig.savefig(HERE / "figures/sngp_variance_veto.png")
print("\nsaved figures/sngp_variance_veto.png")
