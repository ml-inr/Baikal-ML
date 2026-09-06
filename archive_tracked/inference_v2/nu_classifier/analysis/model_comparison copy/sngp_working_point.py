#!/usr/bin/env python
"""SNGP vs base vs fine-tuned at MATCHED working points.

The decisive test the pilot (sngp_exp_eval.py) could not answer. SNGP's mean-field
shrinks *all* scores, so any fixed-threshold statement about it is confounded by score
compression -- the trap documented for spectral norm (report 2026-07-06 §1b). Here every
model is compared at a fixed MC-muon survival mu, i.e. the threshold is re-derived per
model as the (1-mu) quantile of its own out-of-training muatm scores.

Two quantities per working point:
  excess  = (fraction of exp events above the cut) / mu     -- 1.0 means exp behaves
            exactly like the simulated muon background; >1 is the domain-gap excess.
  nu_eff  = equal-weighted (nuatm, nue2) efficiency above the same cut.

All models are restricted to the SAME events (intersection of scored event_fks, h8s3),
so differences are model effects, not sampling. MC is out-of-training for every model;
exp excludes the fine-tuning background (label-leaked) but keeps the DA-target events,
which were unlabelled and therefore valid for evaluation.

Outputs tables/sngp_working_point.csv + figures/sngp_working_point.png.
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
CUT = "p.n_sn_hits>=8 AND p.n_sn_strings>=3"
E1 = "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010"
MODELS = {
    "base": E1,
    "fine-tuned": E1 + "_ood2p5_rv0p7_finetuned@best_finetuned_model",
    "SNGP": "sngp_nu_classifier_baseline@best_sngp_model",
}
MU_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
STY = {"base": "steelblue", "fine-tuned": "crimson", "SNGP": "darkorange"}

ft_bg = set(np.load(FT_BG).astype(np.int64).tolist())
print(f"fine-tuning background event_fks excluded from exp: {len(ft_bg):,}", flush=True)


def fetch(ck, source, classes=None):
    """{event_fk: score} for one model/source at h8s3."""
    db = PREDS / ck / f"{source}_thr0p8.duckdb"
    if not db.exists():
        return None
    c = duckdb.connect()
    c.execute("PRAGMA disable_progress_bar")
    c.execute(f"ATTACH '{db}' AS d (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    where = CUT
    if classes:
        where += " AND ev.data_class IN (" + ",".join(f"'{x}'" for x in classes) + ")"
    else:  # exp: drop the two artefact runs, as everywhere else
        where += " AND NOT (ev.cluster=2 AND ev.run IN ('20','249'))"
    df = c.execute(f"""SELECT p.event_fk, p.score, ev.data_class
                       FROM d.predictions p JOIN cat.events ev ON ev.id=p.event_fk
                       WHERE {where}""").df()
    c.close()
    return df


# ---- collect, then intersect on event_fk so all models see identical events ----
data = {}
for name, ck in MODELS.items():
    mc = fetch(ck, "mc_merged", ["muatm_2020", "nuatm_2020", "nue2_2020"])
    ex = fetch(ck, "exp_full")
    if mc is None or ex is None:
        raise SystemExit(f"missing predictions for {name}")
    ex = ex[~ex.event_fk.isin(ft_bg)]
    data[name] = (mc, ex)
    print(f"{name:11s} raw: mc={len(mc):,} exp={len(ex):,}", flush=True)

common_mc = set.intersection(*[set(d[0].event_fk) for d in data.values()])
common_ex = set.intersection(*[set(d[1].event_fk) for d in data.values()])
print(f"\ncommon events: mc={len(common_mc):,}  exp={len(common_ex):,}", flush=True)

rows = []
for name in MODELS:
    mc, ex = data[name]
    mc = mc[mc.event_fk.isin(common_mc)]
    ex = ex[ex.event_fk.isin(common_ex)]
    mu_s = mc[mc.data_class == "muatm_2020"].score.to_numpy()
    na_s = mc[mc.data_class == "nuatm_2020"].score.to_numpy()
    ne_s = mc[mc.data_class == "nue2_2020"].score.to_numpy()
    ex_s = ex.score.to_numpy()
    print(f"{name:11s} common: muatm={len(mu_s):,} nuatm={len(na_s):,} "
          f"nue2={len(ne_s):,} exp={len(ex_s):,}", flush=True)
    for mu in MU_GRID:
        cut = float(np.quantile(mu_s, 1 - mu))
        n_mu = int((mu_s > cut).sum())
        ex_frac = float((ex_s > cut).mean())
        rows.append(dict(
            model=name, mu=mu, cut=cut, n_mu_above=n_mu,
            exp_frac=ex_frac, excess=ex_frac / mu,
            n_exp_above=int((ex_s > cut).sum()),
            nuatm_eff=float((na_s > cut).mean()), nue2_eff=float((ne_s > cut).mean()),
            nu_eff=0.5 * (float((na_s > cut).mean()) + float((ne_s > cut).mean()))))

tab = pd.DataFrame(rows)
tab.to_csv(HERE / "tables/sngp_working_point.csv", index=False)

print("\n=== excess (exp/mu) and nu efficiency at matched working points ===")
piv = tab.pivot(index="mu", columns="model", values="excess").sort_index(ascending=False)
eff = tab.pivot(index="mu", columns="model", values="nu_eff").sort_index(ascending=False)
print("\nexcess:\n" + piv.round(2).to_string())
print("\nnu_eff:\n" + eff.round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
for name in MODELS:
    t = tab[tab.model == name].sort_values("mu")
    ax[0].plot(t.mu, t.excess, "o-", color=STY[name], lw=1.8, label=name)
    ax[1].plot(t.mu, t.nu_eff, "o-", color=STY[name], lw=1.8, label=name)
ax[0].axhline(1.0, ls=":", color="grey")
ax[0].set_xscale("log"); ax[0].invert_xaxis(); ax[0].set_yscale("log")
ax[0].set_xlabel(r"MC muon survival $\mu$"); ax[0].set_ylabel(r"excess  exp$/\mu$")
ax[0].set_title("Experimental excess")
ax[1].set_xscale("log"); ax[1].invert_xaxis()
ax[1].set_xlabel(r"MC muon survival $\mu$"); ax[1].set_ylabel(r"$\nu$ efficiency")
ax[1].set_title("Signal efficiency")
for a in ax:
    a.legend(); a.minorticks_on()
    a.grid(True, which="major", alpha=0.35); a.grid(True, which="minor", alpha=0.18, ls=":")
fig.tight_layout()
fig.savefig(HERE / "figures/sngp_working_point.png")
print("\nsaved figures/sngp_working_point.png")
