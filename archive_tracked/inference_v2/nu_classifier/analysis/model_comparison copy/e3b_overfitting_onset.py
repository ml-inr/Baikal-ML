#!/usr/bin/env python
"""When does the exp neutrino-like *excess* emerge during training?

Probe E3b (the slowest learner) at early epochs {1,3,5} plus the matched-round
epoch 10. For each epoch, on the out-of-training test predictions:

    excess(thr) = exp_frac(>thr) / mc_muatm_frac(>thr)

The ratio is taken with the SAME model at the SAME threshold, so the model's
global score scale cancels -> a clean, scale-robust domain-gap measure. excess≈1
means exp looks like the MC muon background; excess>1 is the OOD false-neutrino
excess. We watch from which epoch the high-threshold excess starts to grow.

Outputs: excess-vs-threshold curves (one per epoch), an excess-vs-epoch summary
at fixed thresholds, and a CSV.

Usage:
    python inference_v2/nu_classifier/analysis/model_comparison/e3b_overfitting_onset.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from compare_da_models import load_mc, load_exp, PREDS  # same dir

HERE = Path(__file__).resolve().parent
RUNS = {
    "e3b": "260705_0721_da_nu_classifier_exp_full_E3b_lambda0.01",
    "e2":  "260705_0708_da_nu_classifier_exp_full_E2_lambda0.01",
    "e1":  "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01",
}
EPOCHS = [1, 3, 5, 10]
FIXED_THR = [0.5, 0.8, 0.9, 0.95, 0.99]


def _surv(scores: np.ndarray, thr: np.ndarray) -> np.ndarray:
    n = max(len(scores), 1)
    return np.array([(scores > t).sum() / n for t in thr])


def main(tag: str = "e3b") -> None:
    run = RUNS[tag]
    thr = np.unique(np.concatenate([np.linspace(0.0, 0.9, 46),
                                    np.linspace(0.9, 0.999, 60)]))
    curves, rows = {}, []
    for ep in EPOCHS:
        ckpt = f"{run}@da_checkpoint_epoch_{ep:03d}"
        if not (PREDS / ckpt / "exp_full_thr0p8.duckdb").exists():
            print(f"[skip] ep{ep}: preds not found yet ({ckpt})")
            continue
        try:
            mc = load_mc(ckpt)
            mu = mc.loc[mc.is_signal == 0, "score"].to_numpy()   # MC muatm background
            exp = load_exp(ckpt).score.to_numpy()
        except Exception as e:                                   # DB still being written
            print(f"[skip] ep{ep}: DB not readable yet ({type(e).__name__})")
            continue

        exp_s = _surv(exp, thr)
        mu_s = _surv(mu, thr)
        with np.errstate(divide="ignore", invalid="ignore"):
            excess = np.where(mu_s > 0, exp_s / mu_s, np.nan)
        curves[ep] = {"exp_s": exp_s, "mu_s": mu_s, "excess": excess,
                      "n_exp": len(exp), "n_mu": len(mu)}

        row = {"epoch": ep, "n_exp": len(exp), "n_mu": len(mu)}
        for t in FIXED_THR:
            fe = (exp > t).mean()
            fm = (mu > t).mean()
            row[f"exp>{t}"] = fe
            row[f"mu>{t}"] = fm
            row[f"excess@{t}"] = fe / fm if fm > 0 else np.nan
        rows.append(row)
        print(f"[done] ep{ep}: exp N={len(exp):,} mu N={len(mu):,} "
              f"| excess@0.8={row['excess@0.8']:.2f} @0.9={row['excess@0.9']:.2f} "
              f"@0.95={row['excess@0.95']:.2f}")

    if not rows:
        print("no epochs ready — rerun once inference finishes")
        return

    TAG = tag.upper()
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n" + "=" * 90 + "\n" + tab.to_string(index=False))
    tab.to_csv(HERE / f"{tag}_excess_table.csv", index=False)
    print(f"\nsaved {HERE / f'{tag}_excess_table.csv'}")

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(curves)))

    # ── excess vs threshold ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for c, (ep, d) in zip(cmap, curves.items()):
        m = np.isfinite(d["excess"]) & (d["mu_s"] > 0)
        ax.plot(thr[m], d["excess"][m], color=c, lw=2,
                label=f"epoch {ep}  (exp N={d['n_exp']:,})")
    ax.axhline(1.0, color="grey", ls="--", lw=1, label="excess = 1 (exp ≈ MC μ)")
    ax.axvline(0.8, color="grey", ls=":", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("score threshold")
    ax.set_ylabel("excess = exp_frac / mc_muatm_frac")
    ax.set_title(f"{TAG}: exp neutrino-like excess vs threshold, by training epoch")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / f"{tag}_excess_vs_threshold.png", dpi=130)
    print(f"saved {HERE / f'{tag}_excess_vs_threshold.png'}")

    # ── excess vs epoch at fixed thresholds ────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    eps = tab.epoch.to_numpy()
    for t in FIXED_THR:
        ax2.plot(eps, tab[f"excess@{t}"].to_numpy(), "o-", lw=2, label=f"thr {t}")
    ax2.axhline(1.0, color="grey", ls="--", lw=1)
    ax2.set_xlabel("training epoch"); ax2.set_ylabel("excess (exp / MC μ)")
    ax2.set_yscale("log"); ax2.set_xticks(eps)
    ax2.set_title(f"{TAG}: onset of exp neutrino-like excess with training")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(HERE / f"{tag}_excess_vs_epoch.png", dpi=130)
    print(f"saved {HERE / f'{tag}_excess_vs_epoch.png'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=list(RUNS), default="e3b")
    main(ap.parse_args().tag)
