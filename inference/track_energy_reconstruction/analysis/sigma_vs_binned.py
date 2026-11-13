#!/usr/bin/env python3
"""Compare per-event sigma with a sigma that depends only on predicted energy.

The baseline is the simplest alternative to a learned per-event uncertainty: fit
one residual scale per predicted-energy bin on a calibration half of the test
parts, then evaluate it on the other half.  The per-event model is calibrated
with one scalar on the same fit half.  Both are scored with a proper Gaussian
NLL on held-out parts, so a gain cannot come from coverage alone.

Only Gaussian checkpoints are used; Student-t scale parameters are not directly
comparable with a Gaussian bin RMS.  Any stronger baseline should add only
observable hit-derived covariates; the MC ``theta`` column is truth and must
never be a training feature or an operational selection variable.  Run from the
repository root.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from inference.track_energy_reconstruction.analysis.reporting import (
    PRED_ROOT,
    ROOT,
    RUN_SPECS,
)

BINS = np.arange(-0.5, 8.01, 0.5)
FALLBACK_IDS = ("D-G", "AC-G", "B-G", "H-G")


def _gaussian_nll(err: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    return (
        0.5 * np.log(2.0 * np.pi)
        + np.log(sigma)
        + 0.5 * (np.asarray(err, dtype=np.float64) / sigma) ** 2
    )


def _load_test(run_name: str) -> pd.DataFrame:
    db_path = PRED_ROOT / f"{run_name}@best_model/mc_merged_h8s3_thr0p8.duckdb"
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(
            """
            SELECT pred_log10e, true_log10e, pred_sigma_dex, h5_part_key
            FROM predictions WHERE split = 'test'
            """
        ).df()
    df["err"] = df["pred_log10e"] - df["true_log10e"]
    df["abs_err"] = df["err"].abs()
    return df


def _bin_index(values: pd.Series) -> pd.Series:
    return pd.cut(values, bins=BINS, labels=False)


def _fit_bin_baseline(
    fit: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, float, float, float]:
    """Return per-bin RMS/q68/q95 and global fallbacks from the fit half."""
    grouped = fit.groupby("bin", observed=True)
    rms = grouped["err"].apply(lambda x: float(np.sqrt(np.mean(x * x))))
    q68 = grouped["abs_err"].apply(lambda x: float(np.quantile(x, 0.68)))
    q95 = grouped["abs_err"].apply(lambda x: float(np.quantile(x, 0.95)))
    return (
        rms,
        q68,
        q95,
        float(np.sqrt(np.mean(fit["err"] ** 2))),
        float(np.quantile(fit["abs_err"], 0.68)),
        float(np.quantile(fit["abs_err"], 0.95)),
    )


def _bootstrap_dnll(
    test: pd.DataFrame,
    delta: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Part-level bootstrap CI for mean(model NLL - bin NLL)."""
    by_part = (
        pd.DataFrame({"part": test["h5_part_key"].to_numpy(), "d": delta})
        .groupby("part", observed=True)
        .agg(sum=("d", "sum"), n=("d", "size"))
    )
    rng = np.random.RandomState(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, len(by_part), len(by_part))
        vals[i] = by_part["sum"].iloc[idx].sum() / by_part["n"].iloc[idx].sum()
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def _compare(run_id: str, run_name: str, seed: int, n_boot: int) -> dict:
    cfg_path = (
        ROOT / "trained_models/track_energy_reconstruction" / run_name
        / "train_config.yaml"
    )
    cfg = yaml.safe_load(cfg_path.read_text())
    hetero = (cfg.get("model") or {}).get("heteroscedastic") or {}
    if str(hetero.get("distribution", "gaussian")).lower() != "gaussian":
        raise ValueError(f"{run_id}: only Gaussian checkpoints are supported")

    df = _load_test(run_name)
    df["bin"] = _bin_index(df["pred_log10e"])
    parts = np.array(sorted(df["h5_part_key"].unique()))
    if len(parts) < 2:
        raise RuntimeError(f"{run_id}: need at least two test parts")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(parts))
    fit_parts = set(parts[perm[: len(parts) // 2]].tolist())
    fit = df[df["h5_part_key"].isin(fit_parts)].copy()
    test = df[~df["h5_part_key"].isin(fit_parts)].copy()

    bin_rms, bin_q68, bin_q95, global_rms, global_q68, global_q95 = (
        _fit_bin_baseline(fit)
    )
    test["sigma_bin"] = test["bin"].map(bin_rms).fillna(global_rms)
    test["q68_bin"] = test["bin"].map(bin_q68).fillna(global_q68)
    test["q95_bin"] = test["bin"].map(bin_q95).fillna(global_q95)

    # One global scale for the learned sigma, fitted on the same half.
    sigma = test["pred_sigma_dex"].to_numpy(np.float64)
    fit_sigma = fit["pred_sigma_dex"].to_numpy(np.float64)
    k_nll = float(
        np.sqrt(np.mean((fit["err"].to_numpy() / fit_sigma) ** 2))
    )
    k68 = float(np.quantile(fit["abs_err"].to_numpy() / fit_sigma, 0.68))
    k95 = float(np.quantile(fit["abs_err"].to_numpy() / fit_sigma, 0.95))
    test["sigma_model"] = k_nll * sigma

    rho_within = []
    n_within = 0
    for _, group in test.groupby("bin", observed=True):
        if len(group) >= 1000 and group["pred_sigma_dex"].nunique() > 10:
            rho_within.append(
                float(spearmanr(group["pred_sigma_dex"], group["abs_err"]).correlation)
            )
            n_within += len(group)

    nll_model = _gaussian_nll(test["err"], test["sigma_model"])
    nll_bin = _gaussian_nll(test["err"], test["sigma_bin"])
    delta = nll_model - nll_bin
    ci_lo, ci_hi = _bootstrap_dnll(test, delta, n_boot, seed)
    return {
        "id": run_id,
        "run": run_name,
        "n_fit": int(len(fit)),
        "n_eval": int(len(test)),
        "n_parts_eval": int(test["h5_part_key"].nunique()),
        "nll_model": float(nll_model.mean()),
        "nll_bin": float(nll_bin.mean()),
        "nll_model_minus_bin": float(delta.mean()),
        "dnll_ci_lo": ci_lo,
        "dnll_ci_hi": ci_hi,
        "n_boot": int(n_boot),
        "cov68_model": float((test["abs_err"] <= sigma * k68).mean()),
        "cov68_bin": float((test["abs_err"] <= test["q68_bin"]).mean()),
        "cov95_model": float((test["abs_err"] <= sigma * k95).mean()),
        "cov95_bin": float((test["abs_err"] <= test["q95_bin"]).mean()),
        "k_nll": k_nll,
        "k68": k68,
        "k95": k95,
        "rho_sigma_abs": float(
            spearmanr(test["pred_sigma_dex"], test["abs_err"]).correlation
        ),
        "rho_within_bins": (
            float(np.mean(rho_within)) if rho_within else float("nan")
        ),
        "n_events_within_bins": int(n_within),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", default=",".join(FALLBACK_IDS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument(
        "--output",
        default=str(
            ROOT / "inference/track_energy_reconstruction/analysis/reports/"
            "sigma_vs_binned.csv"
        ),
    )
    args = ap.parse_args()
    wanted = [s.strip() for s in args.ids.split(",") if s.strip()]
    run_by_id = {rid: spec[0] for rid, spec in RUN_SPECS.items()}
    rows = []
    for run_id in wanted:
        if run_id not in run_by_id:
            raise KeyError(f"unknown run id {run_id}")
        row = _compare(run_id, run_by_id[run_id], args.seed, args.n_boot)
        rows.append(row)
        print(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
