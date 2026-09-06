#!/usr/bin/env python
"""Fig 15 (single figure, 3 equal panels in a row) for the neutrino candidate extractor:
 (a) EAS suppression vs nu efficiency;
 (b) nu efficiency vs zenith angle Theta;
 (c) nu efficiency vs primary energy.
Panels (b) and (c) share the nu-efficiency y-axis. Base vs fine-tuned.

ALL panels are computed strictly out-of-training: Monte Carlo events seen during training
are removed through the NPY back-links (part_key|local_idx), identically for the muon cut
(muatm) and the signal classes (nuatm, nue2). Excluding training changes every efficiency by
< 0.02 pp (verified) -- the curves are unchanged, but the figure now matches the "excluded
from training" statement in the text. GT theta = prime_prty[:,0], GT energy = prime_prty[:,2].
No GPU. Output figures/nu_classifier_efficiency_3panel.png + tables/paper_suppression_vs_eff.csv.
"""
from pathlib import Path

import duckdb
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 20, "axes.labelsize": 20, "axes.titlesize": 20,
                     "legend.fontsize": 15, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "figure.dpi": 120})
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PREDS = ROOT / "inference_v2/nu_classifier/preds"
CAT = ROOT / "data_manager/catalog_v2.duckdb"
MCH5 = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"
NPY = ROOT / "data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
RDCC = dict(rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=1_000_003)
CUT = "pr.n_sn_hits>=8 AND pr.n_sn_strings>=3"
MODELS = {"base": "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01@da_checkpoint_epoch_010",
          "Fine-tuned": "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01"
                        "@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model"}
SUPP = 1e6
SURV = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
STY = [("base", "steelblue", "Base network"), ("Fine-tuned", "crimson", "After fine-tuning")]

train_pk = np.load(NPY / "h5_part_keys.npy", allow_pickle=True).astype(str)
train_li = np.load(NPY / "h5_local_event_ids.npy").astype(np.int64)
TRAIN = set(f"{a}|{b}" for a, b in zip(train_pk, train_li))
print(f"training keys: {len(TRAIN):,}", flush=True)


def load(ck):
    """Out-of-training scores for muatm (cut) and nuatm/nue2 (signal), with back-links."""
    c = duckdb.connect()
    c.execute("PRAGMA disable_progress_bar")
    c.execute(f"ATTACH '{PREDS / ck / 'mc_merged_thr0p8.duckdb'}' AS m (READ_ONLY)")
    c.execute(f"ATTACH '{CAT}' AS cat (READ_ONLY)")
    d = c.execute(f"""SELECT pr.score, ev.data_class base, l.part_key, l.local_idx
      FROM m.predictions pr JOIN cat.events ev ON ev.id=pr.event_fk
      JOIN cat.h5_locations l ON l.event_fk=pr.event_fk
      WHERE {CUT} AND ev.data_class IN ('muatm_2020','nuatm_2020','nue2_2020')""").df()
    c.close()
    key = d.part_key.astype(str) + "|" + d.local_idx.astype(str)
    d = d[~key.isin(TRAIN).to_numpy()].reset_index(drop=True)
    mu = d[d.base == "muatm_2020"].score.to_numpy()
    nu = d[d.base != "muatm_2020"].reset_index(drop=True)
    return mu, nu


def read_truth(nu):
    theta = np.full(len(nu), np.nan)
    energy = np.full(len(nu), np.nan)
    with h5py.File(MCH5, "r", **RDCC) as f:
        for (base, pk), idx in nu.groupby(["base", "part_key"]).groups.items():
            k = f"{base}/prime_prty/{pk}/data"
            if k not in f:
                continue
            arr = f[k][:]
            loc = nu.loc[idx, "local_idx"].to_numpy()
            ok = loc < arr.shape[0]
            theta[np.array(idx)[ok]] = arr[loc[ok], 0]
            energy[np.array(idx)[ok]] = arr[loc[ok], 2]
    return theta, energy


def eff_binned(nu, col, edges):
    cen = 0.5 * (edges[:-1] + edges[1:])
    e, err = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        vals = [nu[(nu.base == cls) & (nu[col] >= a) & (nu[col] < b)].passc.mean()
                for cls in ["nuatm_2020", "nue2_2020"]
                if len(nu[(nu.base == cls) & (nu[col] >= a) & (nu[col] < b)]) >= 5]
        if vals:
            e.append(np.mean(vals))
            err.append(np.std(vals) / max(len(vals), 1) ** 0.5 if len(vals) > 1 else 0.0)
        else:
            e.append(np.nan)
            err.append(0.0)
    return cen, np.array(e), np.array(err)


# ---- load once per model (out-of-training) ----
DATA, rows = {}, []
for name, ck in MODELS.items():
    mu, nu = load(ck)
    th, en = read_truth(nu)
    nu = nu.assign(theta=th, energy=en).dropna(subset=["theta", "energy"])
    nu = nu.assign(logE=np.log10(np.clip(nu.energy, 1, None)))
    DATA[name] = (mu, nu)
    na = int((nu.base == "nuatm_2020").sum())
    ne = int((nu.base == "nue2_2020").sum())
    print(f"{name}: out-of-training muatm={len(mu):,} nuatm={na:,} nue2={ne:,}", flush=True)
    # ---- panel (a): suppression vs equal-weighted efficiency ----
    for s in SURV:
        cut = np.quantile(mu, 1 - s)
        ea = float((nu[nu.base == "nuatm_2020"].score.to_numpy() > cut).mean())
        ec = float((nu[nu.base == "nue2_2020"].score.to_numpy() > cut).mean())
        rows.append(dict(model=name, supp=1.0 / s, eff=0.5 * (ea + ec),
                         nuatm=ea, nue2=ec, cut=cut, n_mu_at_cut=int((mu > cut).sum())))

tab = pd.DataFrame(rows)
tab.to_csv(HERE / "tables/paper_suppression_vs_eff.csv", index=False)
hl = tab[(tab.model == "base") & (np.isclose(tab.supp, 1e6))].iloc[0]
print(f"headline: base equal-weighted nu-eff at 1e6 = {100*hl.eff:.2f}% "
      f"(nuatm {100*hl.nuatm:.2f}%, nue2 {100*hl.nue2:.2f}%, n_mu@cut {int(hl.n_mu_at_cut)})",
      flush=True)

# ---- plot: 3 equal panels; (b),(c) share y ----
fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
for name, cl, lab in STY:                                    # (a)
    t = tab[tab.model == name].sort_values("eff")
    ax[0].plot(t.eff, t.supp, "o-", color=cl, lw=1.8, label=lab)
ax[0].axhline(1e6, ls=":", color="grey")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"$\nu$ efficiency")
ax[0].set_ylabel("EAS suppression factor")
ax[0].legend()
th_all = np.concatenate([DATA[n][1].theta.to_numpy() for n in MODELS])
th_lo, th_hi = np.nanpercentile(th_all, [1, 99])
th_edges = np.linspace(th_lo, th_hi, 13)
en_edges = np.linspace(1.0, 7.0, 13)
for name, cl, lab in STY:                                    # (b),(c)
    mu, nu = DATA[name]
    cut = np.quantile(mu, 1 - 1 / SUPP)
    nu = nu.assign(passc=(nu.score > cut).astype(float))
    cth, eth, erth = eff_binned(nu, "theta", th_edges)
    cen, een, eren = eff_binned(nu, "logE", en_edges)
    ax[1].errorbar(cth, eth, yerr=erth, fmt="o-", color=cl, lw=1.8, capsize=2, label=lab)
    ax[2].errorbar(cen, een, yerr=eren, fmt="o-", color=cl, lw=1.8, capsize=2, label=lab)
ax[1].set_ylim(0, 1.02)
ax[2].set_ylim(0, 1.02)
ax[1].set_xlabel(r"zenith angle $\Theta$ [deg]")
ax[1].set_ylabel(r"$\nu$ efficiency")
ax[1].legend()
ax[2].set_xlabel(r"$\log_{10}(E/\mathrm{GeV})$")
ax[2].set_ylabel(r"$\nu$ efficiency")
ax[2].legend()
for a, lb in zip(ax, ["(a)", "(b)", "(c)"]):
    a.set_title(lb, loc="left", fontweight="bold")
    a.minorticks_on()
    a.grid(True, which="major", alpha=0.35)
    a.grid(True, which="minor", alpha=0.18, ls=":")
    a.tick_params(which="major", length=6)
    a.tick_params(which="minor", length=3)
fig.tight_layout()
fig.savefig(HERE / "figures/nu_classifier_efficiency_3panel.png")
print("saved figures/nu_classifier_efficiency_3panel.png")
