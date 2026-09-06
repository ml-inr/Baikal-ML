#!/usr/bin/env python
"""Compare the three exp_full DA nu-classifier models (E1/E2/E3b).

For each checkpoint, on the *out-of-training* test predictions:

  MC (mc_merged, per-class, training events excluded via NPY back-links):
    - AUC (signal = nuatm+nue2  vs  background = muatm)
    - signal (neutrino) efficiency at score > 0.8
    - muon survival at score > 0.8  and  muon suppression = 1 / survival
    - signal efficiency at a fixed muon working point (muon survival = 1e-3)

  EXP (exp_full, out-of-training, bad runs c02_r20/r249 dropped):
    - neutrino-like fraction at score > 0.8  (physical expectation ~1e-6)

Outputs a printed table, a CSV, and two overlay plots (MC survival curves,
exp score tail) into this directory.

Usage:
    python inference_v2/nu_classifier/analysis/model_comparison/compare_da_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

PREDS = ROOT / "inference_v2/nu_classifier/preds"
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"
MC_TRAIN_NPY = ROOT / "data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
EXP_TRAIN_NPY = ROOT / "data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8"

_E1 = "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01"
_E2 = "260705_0708_da_nu_classifier_exp_full_E2_lambda0.01"
_E3b = "260705_0721_da_nu_classifier_exp_full_E3b_lambda0.01"

# Preset 'best': each run's best_da_model — but these sit at very different
# epochs (E1=23, E2=43, E3b=37), so NOT a matched working point.
# Preset 'matched': checkpoints matched to E1 @ epoch 10 on val_loss (primary)
# and F1 (control); E2 bracketed at 22 (loss/AUC) and 28 (F1).
PRESETS = {
    "best": {
        "E1 baseline":      f"{_E1}@best_da_model",
        "E2 spectral-norm": f"{_E2}@best_da_model",
        "E3b afterpulse":   f"{_E3b}@best_da_model",
    },
    "matched": {
        "E1 @ep10":            f"{_E1}@da_checkpoint_epoch_010",
        "E2 @ep22 (loss/AUC)": f"{_E2}@da_checkpoint_epoch_022",
        "E2 @ep28 (F1)":       f"{_E2}@da_checkpoint_epoch_028",
        "E3b @ep10":           f"{_E3b}@da_checkpoint_epoch_010",
    },
    # E5 (horizon θ-loss) @ep5 vs E1 refs: same-epoch (5) + metric-matched (1).
    # E5's aggregate MC metrics are deflated by the deliberate horizon abstention,
    # so bracket E1 between same-budget (ep5) and matched-F1/AUC (ep1).
    "e5vse1": {
        "E5 horizon @ep5": "260707_0226_da_nu_classifier_exp_full_E5_horizon_lambda0.01@da_checkpoint_epoch_005",
        "E1 @ep5 (same)":  f"{_E1}@da_checkpoint_epoch_005",
        "E1 @ep1 (matched)": f"{_E1}@da_checkpoint_epoch_001",
    },
    # DA on/off: FixedDA λ=0 (pure MC classifier, DA fully off) @ep15 vs E1 (λ=0.01)
    # @ep10, matched on val_loss. Tests whether DA drives the exp excess.
    "lambda0": {
        "E1 λ0.01 @ep10":   f"{_E1}@da_checkpoint_epoch_010",
        "FixedDA λ0 @ep15": "260531_1722_da_nu_classifier_h5s0_lambda0.0_thr0.8_FixedDA@da_checkpoint_epoch_015",
    },
}

SCORE_THR = 0.8          # decision threshold for "neutrino-like"
MU_WP = 1e-3             # fixed muon-survival working point for efficiency compare
MIN_HITS = 8             # default analysis topology cut (h8s3): n_sn_hits >= 8
MIN_STRINGS = 3          #                                       n_sn_strings >= 3
# NB: prediction DBs are scored at h5s0 (min_hits=5, min_strings=0); h8s3 is an
# ANALYSIS-time tightening via the n_sn_hits/n_sn_strings columns (no re-infer).


def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("PRAGMA disable_progress_bar")
    return c


def _train_keys(npy_dir: Path, part_f: str, loc_f: str) -> set[str]:
    """All training 'part_key|local_idx' keys (every class)."""
    pk = np.asarray(np.load(npy_dir / part_f, allow_pickle=True), dtype=str)
    li = np.asarray(np.load(npy_dir / loc_f), dtype=np.int64)
    return set(np.char.add(np.char.add(pk, "|"), li.astype(str)).tolist())


def load_mc(ckpt: str) -> pd.DataFrame:
    """Out-of-training MC scores with data_class; training events removed."""
    db = PREDS / ckpt / "mc_merged_thr0p8.duckdb"
    c = _con()
    c.execute(f"ATTACH '{db}' AS p (READ_ONLY)")
    c.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    df = c.execute(f"""
        SELECT pr.event_fk, pr.score, e.data_class, l.part_key, l.local_idx
        FROM p.predictions pr
        JOIN cat.events e       ON e.id = pr.event_fk
        JOIN cat.h5_locations l ON l.event_fk = pr.event_fk
        WHERE pr.n_sn_hits >= {MIN_HITS} AND pr.n_sn_strings >= {MIN_STRINGS}
    """).df()
    c.close()
    train = _train_keys(MC_TRAIN_NPY, "h5_part_keys.npy", "h5_local_event_ids.npy")
    key = df["part_key"].astype(str) + "|" + df["local_idx"].astype(str)
    df = df[~key.isin(train)].copy()
    df["is_signal"] = df["data_class"].isin(["nuatm_2020", "nue2_2020"]).astype(int)
    return df.drop(columns=["part_key", "local_idx"])


def load_exp(ckpt: str) -> pd.DataFrame:
    """Out-of-training exp_full scores; training events + bad runs removed."""
    db = PREDS / ckpt / "exp_full_thr0p8.duckdb"
    c = _con()
    c.execute(f"ATTACH '{db}' AS p (READ_ONLY)")
    c.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    # drop bad runs (cluster 2, runs 20 & 249) in SQL; hold out training in pandas
    df = c.execute(f"""
        SELECT pr.event_fk, pr.score, l.part_key, l.local_idx
        FROM p.predictions pr
        JOIN cat.events e       ON e.id = pr.event_fk
        JOIN cat.h5_locations l ON l.event_fk = pr.event_fk
        WHERE NOT (e.cluster = 2 AND e.run IN ('20', '249'))
          AND pr.n_sn_hits >= {MIN_HITS} AND pr.n_sn_strings >= {MIN_STRINGS}
    """).df()
    c.close()
    train = _train_keys(EXP_TRAIN_NPY, "exp_h5_part_keys.npy",
                        "exp_h5_local_event_ids.npy")
    key = df["part_key"].astype(str) + "|" + df["local_idx"].astype(str)
    df = df[~key.isin(train)].copy()
    return df.drop(columns=["part_key", "local_idx"])


def eff_at_mu_wp(sig: np.ndarray, bg: np.ndarray, mu_surv: float) -> tuple[float, float]:
    """Signal efficiency at the score cut that gives muon survival = mu_surv."""
    if len(bg) == 0:
        return np.nan, np.nan
    cut = np.quantile(bg, 1.0 - mu_surv)          # cut leaving mu_surv of muons
    return float((sig > cut).mean()), float(cut)


def main(preset: str = "best") -> None:
    models = PRESETS[preset]
    thr = np.unique(np.concatenate([np.linspace(0, 0.9, 46),
                                    np.linspace(0.9, 0.9999, 80)]))
    rows, mc_curves, exp_scores = [], {}, {}
    mu_scores, sig_scores, wp_cuts = {}, {}, {}

    for name, ckpt in models.items():
        mc = load_mc(ckpt)
        sig = mc.loc[mc.is_signal == 1, "score"].to_numpy()
        bg = mc.loc[mc.is_signal == 0, "score"].to_numpy()
        y = mc.is_signal.to_numpy()
        auc = roc_auc_score(y, mc.score.to_numpy()) if y.min() != y.max() else np.nan

        sig_eff = float((sig > SCORE_THR).mean())
        mu_surv = float((bg > SCORE_THR).mean())
        eff_wp, cut_wp = eff_at_mu_wp(sig, bg, MU_WP)

        exp = load_exp(ckpt)
        exp_hi = int((exp.score > SCORE_THR).sum())
        exp_frac = exp_hi / len(exp) if len(exp) else np.nan
        # exp fraction ABOVE the cut giving MC muon-survival = MU_WP.
        # Normalises each model's global score conservatism (spectral norm
        # lowers scores everywhere), isolating the OOD-specific effect.
        exp_frac_wp = (float((exp.score > cut_wp).mean())
                       if np.isfinite(cut_wp) else np.nan)

        rows.append({
            "model": name,
            "mc_sig_N": len(sig), "mc_bg_N": len(bg), "auc": auc,
            f"sig_eff@{SCORE_THR}": sig_eff,
            f"mu_surv@{SCORE_THR}": mu_surv,
            f"mu_suppr@{SCORE_THR}": (1.0 / mu_surv) if mu_surv > 0 else np.inf,
            f"sig_eff@mu={MU_WP:g}": eff_wp,
            "exp_N": len(exp), "exp_hi_N": exp_hi, "exp_frac>0.8": exp_frac,
            f"exp_frac@mu={MU_WP:g}": exp_frac_wp,
        })
        mc_curves[name] = {
            "sig": np.array([(sig > t).mean() for t in thr]),
            "bg":  np.array([(bg > t).mean() for t in thr]),
        }
        exp_scores[name] = exp.score.to_numpy()
        mu_scores[name]  = bg
        sig_scores[name] = sig
        wp_cuts[name]    = cut_wp
        print(f"[done] {name}: MC sig={len(sig):,} bg={len(bg):,} AUC={auc:.4f} "
              f"| exp {exp_hi:,}/{len(exp):,} = {exp_frac:.3%}")

    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n" + "=" * 100)
    print(tab.to_string(index=False))
    tab.to_csv(HERE / f"comparison_table_{preset}.csv", index=False)
    print(f"\nsaved {HERE / f'comparison_table_{preset}.csv'}")

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(models)}

    # ── MC survival curves (signal solid, muon dashed) ──────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, cur in mc_curves.items():
        ax.plot(thr, cur["sig"], color=colors[name], lw=2, label=f"{name} — ν")
        ax.plot(thr, np.clip(cur["bg"], 1e-7, 1), color=colors[name], lw=1.6,
                ls="--", label=f"{name} — μ")
    ax.axvline(SCORE_THR, color="grey", ls=":", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("score cut"); ax.set_ylabel("survival fraction")
    ax.set_title("MC survival (out-of-training): ν signal (solid) vs μ background (dashed)")
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / f"mc_survival_curves_{preset}.png", dpi=130)
    print(f"saved {HERE / f'mc_survival_curves_{preset}.png'}")

    # ── exp score tail (neutrino-like region) ───────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    bins = np.linspace(0, 1, 51)
    for name, s in exp_scores.items():
        ax2.hist(s, bins=bins, histtype="step", lw=2, color=colors[name],
                 label=f"{name}  (>0.8: {(s > SCORE_THR).mean():.3%})")
    ax2.axvline(SCORE_THR, color="grey", ls=":", lw=1)
    ax2.set_yscale("log"); ax2.set_xlabel("score"); ax2.set_ylabel("exp events")
    ax2.set_title("Exp out-of-training score distribution (bad runs c02_r20/r249 dropped)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(HERE / f"exp_score_tail_{preset}.png", dpi=130)
    print(f"saved {HERE / f'exp_score_tail_{preset}.png'}")

    # ── Overlay: MC-muatm vs exp score density per model (log y) ─────────────
    # Density-normalised so the *shape/location* shift is visible independent of
    # sample size. E2 (spectral norm) shifts BOTH muatm and exp left by the same
    # amount → global score compression, not OOD-specific suppression.
    bins = np.linspace(0, 1, 61)
    fig3, (axm, axe) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for name in models:
        c = colors[name]
        axm.hist(mu_scores[name], bins=bins, density=True, histtype="step",
                 lw=2, color=c, label=name)
        axe.hist(exp_scores[name], bins=bins, density=True, histtype="step",
                 lw=2, color=c, label=name)
    for ax, ttl in ((axm, "MC muatm (muon) score"), (axe, "exp score")):
        ax.axvline(SCORE_THR, color="grey", ls=":", lw=1.2)
        ax.set_yscale("log"); ax.set_xlabel("score"); ax.set_title(ttl)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axm.set_ylabel("density (log)")
    fig3.suptitle("Score distributions: spectral norm (E2) shifts muatm AND exp "
                  "left by the same amount (global compression)")
    fig3.tight_layout(rect=(0, 0, 1, 0.96))
    fig3.savefig(HERE / f"score_hist_muatm_vs_exp_{preset}.png", dpi=130)
    print(f"saved {HERE / f'score_hist_muatm_vs_exp_{preset}.png'}")

    # ── Per-model panels: muatm vs exp with 0.8 threshold + μ=1e-3 WP cut ────
    # Shows the exp↔muatm relationship is the same across models once each
    # model's own working-point cut (dashed) is used instead of a fixed 0.8.
    n = len(models)
    fig4, axes = plt.subplots(1, n, figsize=(4.6 * n, 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        row = tab.loc[tab.model == name].iloc[0]
        ax.hist(mu_scores[name], bins=bins, density=True, histtype="stepfilled",
                lw=1.5, color="tab:red", alpha=0.35, label="MC muatm")
        ax.hist(exp_scores[name], bins=bins, density=True, histtype="step",
                lw=2, color="black", label="exp")
        ax.axvline(SCORE_THR, color="grey", ls=":", lw=1.2, label="thr 0.8")
        ax.axvline(wp_cuts[name], color="tab:blue", ls="--", lw=1.5,
                   label=f"μ-WP cut={wp_cuts[name]:.3f}")
        ax.set_yscale("log"); ax.set_xlabel("score")
        ax.set_title(f"{name}\nexp>0.8={row['exp_frac>0.8']:.3%} | "
                     f"exp@μWP={row[f'exp_frac@mu={MU_WP:g}']:.3%}", fontsize=9)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("density (log)")
    fig4.suptitle("Per-model exp vs MC-muatm. At a FIXED 0.8 threshold E2 looks "
                  "better; at each model's μ=1e-3 working-point cut (dashed) the "
                  "exp fraction is equal (~0.25%)", fontsize=11)
    fig4.tight_layout(rect=(0, 0, 1, 0.93))
    fig4.savefig(HERE / f"score_hist_per_model_{preset}.png", dpi=130)
    print(f"saved {HERE / f'score_hist_per_model_{preset}.png'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="best")
    main(ap.parse_args().preset)
