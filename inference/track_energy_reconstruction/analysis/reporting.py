"""Shared reporting functions for one energy checkpoint.

The summary notebook and the per-checkpoint notebooks intentionally share this
module.  A per-run notebook only selects ``RUN_ID`` and calls
``make_run_report``; it does not duplicate the metric or plotting logic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.stats import norm, t as student_t, pearsonr, spearmanr
from IPython.display import display

ROOT = Path(__file__).resolve().parents[3]
PRED_ROOT = ROOT / "inference/track_energy_reconstruction/preds"
TRAIN_ROOT = ROOT / "trained_models/track_energy_reconstruction"
REPORT_ROOT = ROOT / "inference/track_energy_reconstruction/analysis/reports/by_checkpoint"
CLASS_NAMES = {1: "nuatm_2020", 2: "nue2_2020"}
BINS = np.arange(-0.5, 8.01, 0.5)

RUN_SPECS = {
    "BASE": ("261027_0935_mu_energy_BASE_cont_partsplit_s42", "mean-only"),
    "A": ("261026_2057_mu_energy_A_lam0p01_detachedcritic_partsplit_scratch_s42", "DA mean-only"),
    "D-G": ("261027_0935_mu_energy_D-G_partsplit_scratch_s42", "joint detached"),
    "D-T5": ("261027_0935_mu_energy_D-T5_partsplit_scratch_s42", "joint detached"),
    "C-G": ("261027_0935_mu_energy_C-G_partsplit_scratch_s42", "joint full two-opt"),
    "C-T5": ("261027_0935_mu_energy_C-T5_partsplit_scratch_s42", "joint full two-opt"),
    "H-G": ("261027_0935_mu_energy_H-G_d0base_s42", "frozen head"),
    "H-T5": ("261027_0935_mu_energy_H-T5_d0base_s42", "frozen head"),
    "AH-G": ("261027_0935_mu_energy_AH-G_abase_s42", "frozen head on DA base"),
    "AH-T5": ("261027_0935_mu_energy_AH-T5_abase_s42", "frozen head on DA base"),
    "B-G": ("261027_0935_mu_energy_B-G_partsplit_scratch_s42", "joint full one-opt"),
    "B-T5": ("261027_0935_mu_energy_B-T5_partsplit_scratch_s42", "joint full one-opt"),
    "AB-G": ("261027_0935_mu_energy_AB-G_partsplit_scratch_s42", "DA joint full one-opt"),
    "AB-T5": ("261027_0935_mu_energy_AB-T5_partsplit_scratch_s42", "DA joint full one-opt"),
    "AD-G": ("261027_0935_mu_energy_AD-G_partsplit_scratch_s42", "DA joint detached"),
    "AD-T5": ("261027_0935_mu_energy_AD-T5_partsplit_scratch_s42", "DA joint detached"),
    "AC-G": ("261027_0935_mu_energy_AC-G_partsplit_scratch_s42", "DA joint full two-opt"),
    "AC-T5": ("261027_0935_mu_energy_AC-T5_partsplit_scratch_s42", "DA joint full two-opt"),
}

DA_PAIRS = {
    "A": "BASE",
    "AD-G": "D-G",
    "AD-T5": "D-T5",
    "AC-G": "C-G",
    "AC-T5": "C-T5",
    "AB-G": "B-G",
    "AB-T5": "B-T5",
    "AH-G": "H-G",
    "AH-T5": "H-T5",
}
DA_PROBE_PATH = ROOT / "inference/track_energy_reconstruction/analysis/reports/da_probe.csv"


def _nll(err, sigma, distribution, nu=5.0):
    err = np.asarray(err, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    if distribution == "student_t":
        z = err / sigma
        const = (-math.lgamma(0.5 * (nu + 1.0)) + math.lgamma(0.5 * nu)
                 + 0.5 * math.log(nu * math.pi))
        return const + np.log(sigma) + 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
    return 0.5 * math.log(2.0 * math.pi) + np.log(sigma) + 0.5 * (err / sigma) ** 2


def _pit(err, sigma, distribution, nu=5.0):
    # Callers pass err = pred - true.  PIT is F(true), i.e. evaluate the
    # predictive CDF at (true - pred), not at (pred - true).
    z = -np.asarray(err, dtype=np.float64) / np.maximum(
        np.asarray(sigma, dtype=np.float64), 1e-12
    )
    return student_t.cdf(z, df=nu) if distribution == "student_t" else norm.cdf(z)


def _pit_ks(pit):
    pit = np.asarray(pit, dtype=np.float64)
    hist, _ = np.histogram(pit, bins=np.linspace(0, 1, 21))
    cum = np.cumsum(hist) / max(len(pit), 1)
    return float(np.max(np.abs(cum - np.arange(1, 21) / 20.0)))


def _bootstrap_mae(df, n_boot=300, seed=42):
    g = (df.groupby("h5_part_key", observed=True)
           .agg(abs_sum=("abs_err", "sum"), n=("abs_err", "size")))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(g), size=len(g))
        vals[i] = g["abs_sum"].iloc[idx].sum() / g["n"].iloc[idx].sum()
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def load_run(run_id: str, run_name: Optional[str] = None) -> dict:
    run_name = run_name or RUN_SPECS[run_id][0]
    kind = RUN_SPECS.get(run_id, ("", ""))[1]
    run_dir = TRAIN_ROOT / run_name
    summary = yaml.safe_load((run_dir / "training_summary.yaml").read_text())
    cfg = yaml.safe_load((run_dir / "train_config.yaml").read_text())
    db_path = PRED_ROOT / f"{run_name}@best_model/mc_merged_h8s3_thr0p8.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("""
            SELECT pred_log10e, true_log10e, pred_sigma_dex,
                   particle_type, n_sig_hits, theta, h5_part_key, split
            FROM predictions WHERE split = 'test'
        """).df()
    if df.empty or not (df["split"] == "test").all():
        raise RuntimeError(f"{run_id}: invalid/empty test population")
    h = cfg.get("model", {}).get("heteroscedastic", {}) or {}
    dist = str(h.get("distribution", "gaussian")).lower()
    has_sigma = bool(bool(h.get("enabled", False)) and bool(df["pred_sigma_dex"].notna().any()))
    factors = (summary.get("calibration", {}) or {}).get("scale_factors", {}) or {}
    q68 = float(factors.get("scale_factor_68", 1.0))
    q95 = float(factors.get("scale_factor_95", 1.0))
    err = (df["pred_log10e"] - df["true_log10e"]).to_numpy(np.float64)
    abs_err = np.abs(err)
    sigma = df["pred_sigma_dex"].to_numpy(np.float64)
    df = df.copy()
    df["err"] = err
    df["abs_err"] = abs_err
    df["class_name"] = df["particle_type"].map(CLASS_NAMES).fillna("unknown")
    rec = {
        "id": run_id, "run": run_name, "kind": kind, "db": db_path,
        "summary": summary, "cfg": cfg, "df": df,
        "n": len(df), "n_parts": int(df["h5_part_key"].nunique()),
        "has_sigma": has_sigma, "distribution": dist,
        "nu": float(h.get("student_t_nu", 5.0)),
        "strategy": summary.get("sigma_strategy", ""),
        "monitor": summary.get("monitor"),
        "best_epoch": summary.get("best_epoch"),
        "best_metric": summary.get("best_metric"),
        "q68": q68, "q95": q95, "factors": factors,
        "sigma": sigma, "err": err, "abs_err": abs_err,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(err.mean()),
    }
    if has_sigma:
        rec.update({
            "nll_raw": float(np.mean(_nll(err, sigma, dist, rec["nu"]))),
            "nll_cal": float(np.mean(_nll(err, sigma * q68, dist, rec["nu"]))),
            "raw68": float(np.mean(abs_err <= sigma)),
            "raw95": float(np.mean(abs_err <= 1.96 * sigma)),
            "cal68": float(np.mean(abs_err <= sigma * q68)),
            "cal95": float(np.mean(abs_err <= sigma * q95)),
            "mean_sigma": float(np.mean(sigma)),
            "pit_raw": _pit(err, sigma, dist, rec["nu"]),
            "pit_cal": _pit(err, sigma * q68, dist, rec["nu"]),
        })
        rec["pit_ks_raw"] = _pit_ks(rec["pit_raw"])
        rec["pit_ks_cal"] = _pit_ks(rec["pit_cal"])
    return rec


def bin_table(r: dict, column: str = "true_log10e", min_n: int = 1) -> pd.DataFrame:
    """Per-bin statistics, binned either by true or by predicted energy."""
    df = r["df"]
    d = df[[column, "abs_err", "err", "pred_sigma_dex"]].copy()
    d["bin_mid"] = pd.cut(
        d[column], bins=BINS, labels=(BINS[:-1] + BINS[1:]) / 2
    ).astype(float)
    rows = []
    for mid, g in d.dropna(subset=["bin_mid"]).groupby("bin_mid", observed=True):
        row = {
            "bin_mid": float(mid), "n": len(g),
            "mae": float(g["abs_err"].mean()),
            "bias": float(g["err"].mean()),
            "rmse": float(np.sqrt(np.mean(g["err"] ** 2))),
            "q16": float(np.quantile(g["err"], 0.16)),
            "median": float(np.quantile(g["err"], 0.5)),
            "q84": float(np.quantile(g["err"], 0.84)),
        }
        if r["has_sigma"]:
            sig = g["pred_sigma_dex"].to_numpy(np.float64)
            e = g["abs_err"].to_numpy(np.float64)
            row.update({
                "mean_sigma": float(np.mean(sig)),
                "raw68": float(np.mean(e <= sig)),
                "cal68": float(np.mean(e <= sig * r["q68"])),
                "raw95": float(np.mean(e <= 1.96 * sig)),
                "cal95": float(np.mean(e <= sig * r["q95"])),
            })
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("bin_mid")
    return out[out["n"] >= min_n]


def make_run_report(run_id: str, run_name: Optional[str] = None) -> dict:
    r = load_run(run_id, run_name)
    out_dir = REPORT_ROOT / f"{r['run']}@best_model"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = r["df"]
    bins = bin_table(r, "true_log10e")
    bins_pred = bin_table(r, "pred_log10e")
    bins.to_csv(out_dir / "by_energy_bin.csv", index=False)
    bins_pred.to_csv(out_dir / "by_predicted_energy_bin.csv", index=False)

    meta = pd.DataFrame([{
        "id": r["id"], "run": r["run"], "kind": r["kind"],
        "checkpoint_epoch": r["best_epoch"], "monitor": r["monitor"],
        "best_metric": r["best_metric"], "n_test": r["n"],
        "n_parts": r["n_parts"], "has_sigma": r["has_sigma"],
        "distribution": r["distribution"] if r["has_sigma"] else "",
        "sigma_strategy": r["strategy"], "q68": r["q68"] if r["has_sigma"] else np.nan,
        "q95": r["q95"] if r["has_sigma"] else np.nan,
        "db": str(r["db"]),
    }])
    display(meta)

    # Converged independent domain probe: same small held-out muatm/exp events,
    # fitted on the current encoder features.  This is the metric to compare DA
    # with no-DA; the old per-epoch history MLP is under-trained.
    if DA_PROBE_PATH.exists():
        probe_all = pd.read_csv(DA_PROBE_PATH)
        cur = probe_all[probe_all["id"] == r["id"]]
        if len(cur):
            pair_id = DA_PAIRS.get(r["id"])
            if pair_id is None:
                pair_id = next((k for k, v in DA_PAIRS.items() if v == r["id"]), None)
            pair = probe_all[probe_all["id"] == pair_id] if pair_id else pd.DataFrame()
            cols = ["id", "da_enabled", "linear_auc", "linear_auc_part",
                    "original_mlp_auc", "mlp_auc_mean", "mlp_auc_mean_part",
                    "mlp_auc_std", "mlp_auc_std_part", "mlp_n_iter_mean",
                    "mlp_n_iter_mean_part", "n_source", "n_target",
                    "probe_split"]
            show = pd.concat([cur[cols], pair[cols]], ignore_index=True)
            display(show.round(4))
            if len(pair):
                cur_mlp = float(cur["mlp_auc_mean"].iloc[0])
                pair_mlp = float(pair["mlp_auc_mean"].iloc[0])
                cur_part = float(cur["mlp_auc_mean_part"].iloc[0])
                pair_part = float(pair["mlp_auc_mean_part"].iloc[0])
                print(f"probe delta vs {pair_id} (event split): "
                      f"{cur_mlp - pair_mlp:+.4f} "
                      f"({pair_id} {pair_mlp:.4f} -> {r['id']} {cur_mlp:.4f})")
                print(f"probe delta vs {pair_id} (part split):  "
                      f"{cur_part - pair_part:+.4f} "
                      f"({pair_id} {pair_part:.4f} -> {r['id']} {cur_part:.4f})")
            fig, ax = plt.subplots(figsize=(7.0, 4.3))
            x = np.arange(len(show))
            ax.errorbar(x - 0.06, show["mlp_auc_mean"], yerr=show["mlp_auc_std"],
                        fmt="o", ms=6, capsize=4, lw=1.2,
                        label="MLP, event split")
            ax.errorbar(x + 0.06, show["mlp_auc_mean_part"],
                        yerr=show["mlp_auc_std_part"], fmt="s", ms=6,
                        capsize=4, lw=1.2, label="MLP, part-disjoint")
            ax.scatter(x - 0.06, show["linear_auc"], marker="x", s=55,
                       label="linear, event split")
            ax.scatter(x + 0.06, show["linear_auc_part"], marker="+", s=70,
                       label="linear, part-disjoint")
            ax.set_xticks(x)
            ax.set_xticklabels(show["id"])
            ax.axhline(0.5, color="black", ls=":", lw=1, label="chance")
            ax.set_ylim(0.50, 0.78)
            ax.set_ylabel("held-out domain AUC")
            ax.set_title(f"{r['id']}: converged domain probe")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="lower left")
            fig.tight_layout()
            fig.savefig(fig_dir / "da_probe_auc.png", dpi=150, bbox_inches="tight")
            plt.show()
        else:
            print("converged DA probe: no row yet for", r["id"])
    else:
        print("converged DA probe CSV is not available")

    global_row = {
        "mae": r["mae"], "rmse": r["rmse"], "bias": r["bias"],
        "within_x2": float(np.mean(r["abs_err"] <= math.log10(2.0))),
        "within_x5": float(np.mean(r["abs_err"] <= math.log10(5.0))),
        "within_x10": float(np.mean(r["abs_err"] <= 1.0)),
    }
    if r["has_sigma"]:
        global_row.update({
            "mean_sigma": r["mean_sigma"], "raw68": r["raw68"],
            "raw95": r["raw95"], "cal68": r["cal68"], "cal95": r["cal95"],
            "nll_raw": r["nll_raw"], "nll_cal": r["nll_cal"],
            "pit_ks_raw": r["pit_ks_raw"], "pit_ks_cal": r["pit_ks_cal"],
        })
    global_df = pd.DataFrame([global_row])
    global_df.to_csv(out_dir / "global_metrics.csv", index=False)
    display(global_df.round(4))

    class_df = (df.groupby("class_name")
                  .agg(n=("abs_err", "size"), mae=("abs_err", "mean"),
                       bias=("err", "mean"))
                  .reset_index())
    class_df.to_csv(out_dir / "class_metrics.csv", index=False)
    display(class_df.round(4))

    # 2D: true energy vs residual
    fig, ax = plt.subplots(figsize=(9, 5.5))
    h = ax.hist2d(df["true_log10e"], df["err"], bins=[BINS, np.linspace(-2, 2, 81)],
                  cmap="viridis", norm=LogNorm())
    fig.colorbar(h[3], ax=ax, label="events / bin")
    g = bins[bins["n"] >= 50]
    ax.plot(g["bin_mid"], g["median"], color="white", lw=1.7, label="median")
    ax.plot(g["bin_mid"], g["q16"], color="white", lw=1, ls="--", label="16/84%")
    ax.plot(g["bin_mid"], g["q84"], color="white", lw=1, ls="--")
    ax.axhline(0, color="white", lw=0.8, ls=":")
    ax.set_xlabel("true log10(E_mu / GeV)")
    ax.set_ylabel("pred - true (dex)")
    ax.set_title(f"{r['id']}: residual vs true energy")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_vs_true_energy_2d.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 2D: predicted energy vs residual (operational binning)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    h = ax.hist2d(df["pred_log10e"], df["err"], bins=[BINS, np.linspace(-2, 2, 81)],
                  cmap="viridis", norm=LogNorm())
    fig.colorbar(h[3], ax=ax, label="events / bin")
    g = bins_pred[bins_pred["n"] >= 50]
    ax.plot(g["bin_mid"], g["median"], color="white", lw=1.7, label="median")
    ax.plot(g["bin_mid"], g["q16"], color="white", lw=1, ls="--", label="16/84%")
    ax.plot(g["bin_mid"], g["q84"], color="white", lw=1, ls="--")
    ax.axhline(0, color="white", lw=0.8, ls=":")
    ax.set_xlabel("predicted log10(E_mu / GeV)")
    ax.set_ylabel("pred - true (dex)")
    ax.set_title(f"{r['id']}: residual vs predicted energy")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_vs_predicted_energy_2d.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 2D: true vs predicted
    fig, ax = plt.subplots(figsize=(6.5, 6))
    h = ax.hist2d(df["true_log10e"], df["pred_log10e"], bins=[BINS, BINS],
                  cmap="viridis", norm=LogNorm())
    fig.colorbar(h[3], ax=ax, label="events / bin")
    ax.plot([BINS[0], BINS[-1]], [BINS[0], BINS[-1]], color="white", lw=1)
    ax.set_xlabel("true log10(E_mu / GeV)")
    ax.set_ylabel("predicted log10(E_mu / GeV)")
    ax.set_title(f"{r['id']}: prediction vs truth")
    fig.tight_layout()
    fig.savefig(fig_dir / "pred_vs_true_2d.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Per-bin MAE and bias
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(bins["bin_mid"], bins["mae"], marker="o", ms=3, lw=1.5)
    axes[0].set_ylabel("MAE (dex)")
    axes[1].plot(bins["bin_mid"], bins["bias"], marker="o", ms=3, lw=1.5)
    axes[1].axhline(0, color="black", lw=0.8, ls=":")
    axes[1].set_ylabel("bias (dex)")
    for ax in axes:
        ax.set_xlabel("true log10(E_mu / GeV)")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{r['id']}: per-bin error")
    fig.tight_layout()
    fig.savefig(fig_dir / "error_vs_energy.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Predicted-energy profile: this is the operational binning (no truth needed)
    display(bins_pred[["bin_mid", "n", "mae", "bias", "rmse"]].round(4))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(bins_pred["bin_mid"], bins_pred["mae"], marker="o", ms=3, lw=1.5)
    axes[0].set_ylabel("MAE (dex)")
    axes[1].plot(bins_pred["bin_mid"], bins_pred["bias"], marker="o", ms=3, lw=1.5)
    axes[1].axhline(0, color="black", lw=0.8, ls=":")
    axes[1].set_ylabel("bias (dex)")
    for ax in axes:
        ax.set_xlabel("predicted log10(E_mu / GeV)")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{r['id']}: error binned by predicted energy")
    fig.tight_layout()
    fig.savefig(fig_dir / "error_vs_predicted_energy.png", dpi=150, bbox_inches="tight")
    plt.show()

    # DA history, if applicable
    da_cfg = r["cfg"].get("domain_adaptation", {}) or {}
    hist_path = TRAIN_ROOT / r["run"] / "training_history.csv"
    if da_cfg.get("enabled", False) and hist_path.exists():
        hist = pd.read_csv(hist_path)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for col, label in [
            ("val_domain_probe_auc", "probe linear"),
            ("val_domain_probe_mlp_auc", "probe MLP (old)"),
            ("val_domain_critic_auc", "critic"),
        ]:
            if col in hist:
                ax.plot(hist["epoch"], hist[col], lw=1.5, label=label)
        if "da_coef" in hist:
            eps = hist.loc[hist["da_coef"].fillna(0) > 0, "epoch"]
            if len(eps):
                ax.axvline(float(eps.min()), color="black", ls="--", lw=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.48, 0.75)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(f"{r['id']}: DA history")
        fig.tight_layout()
        fig.savefig(fig_dir / "da_history.png", dpi=150, bbox_inches="tight")
        plt.show()

    (out_dir / "report.json").write_text(json.dumps({
        "id": r["id"], "run": r["run"], "checkpoint_epoch": r["best_epoch"],
        "monitor": r["monitor"], "best_metric": r["best_metric"],
        "n_test": r["n"], "n_parts": r["n_parts"],
        "has_sigma": bool(r["has_sigma"]), "distribution": r["distribution"],
        "sigma_strategy": r["strategy"], "q68": r["q68"], "q95": r["q95"],
        "db": str(r["db"]),
    }, indent=2))
    print("report directory:", out_dir)
    return r


def _coverage_figure(
    bins: pd.DataFrame,
    xlabel: str,
    title: str,
    path: Path,
    min_n: int = 50,
) -> None:
    """Coverage at 68% and 95% versus one energy axis."""
    d = bins[bins["n"] >= min_n].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    specs = [
        ("68%", "raw68", "cal68", 0.68, (0.40, 0.95)),
        ("95%", "raw95", "cal95", 0.95, (0.75, 1.0)),
    ]
    for ax, (level, raw_col, cal_col, nominal, ylim) in zip(axes, specs):
        n = d["n"].to_numpy(np.float64)
        for col, label, color in [
            (raw_col, "raw", "#1f77b4"),
            (cal_col, "calibrated", "#d62728"),
        ]:
            p = d[col].to_numpy(np.float64)
            err = np.sqrt(p * (1.0 - p) / n)
            ax.errorbar(d["bin_mid"], p, yerr=err, marker="o", ms=3,
                        lw=1.5, capsize=2, color=color, label=label)
        ax.axhline(nominal, color="black", ls="--", lw=1,
                   label=f"nominal {level}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{level} coverage")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(f"{level} coverage")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def _pit_bin_table(
    energy: np.ndarray,
    pit: np.ndarray,
    min_n: int = 50,
) -> pd.DataFrame:
    """Per-energy-bin PIT moments and a binned-uniformity statistic."""
    d = pd.DataFrame({
        "energy": np.asarray(energy, dtype=np.float64),
        "pit": np.asarray(pit, dtype=np.float64),
    })
    d["bin_mid"] = pd.cut(
        d["energy"], bins=BINS, labels=(BINS[:-1] + BINS[1:]) / 2
    ).astype(float)
    rows = []
    for mid, g in d.dropna(subset=["bin_mid"]).groupby("bin_mid", observed=True):
        p = g["pit"].to_numpy(np.float64)
        p = p[np.isfinite(p)]
        if len(p) == 0:
            continue
        rows.append({
            "bin_mid": float(mid),
            "n": int(len(p)),
            "pit_mean": float(np.mean(p)),
            "pit_std": float(np.std(p)),
            "pit_ks": float(_pit_ks(p)),
        })
    return pd.DataFrame(rows).sort_values("bin_mid")


def _pit_heatmap(
    energy: np.ndarray,
    pit: np.ndarray,
    xlabel: str,
    title: str,
    path: Path,
    min_n: int = 50,
) -> None:
    """PIT density per energy bin; each row is normalized to unit area."""
    p_edges = np.linspace(0.0, 1.0, 21)
    density = np.full((len(BINS) - 1, len(p_edges) - 1), np.nan)
    energy = np.asarray(energy, dtype=np.float64)
    pit = np.asarray(pit, dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        mask = (energy >= lo) & (energy < hi) & np.isfinite(pit)
        n = int(mask.sum())
        if n < min_n:
            continue
        h, _ = np.histogram(pit[mask], bins=p_edges)
        density[i] = h / n * len(p_edges)  # equal-width bins -> density
    fig, ax = plt.subplots(figsize=(9, 5.5))
    mesh = ax.pcolormesh(
        p_edges, BINS, np.ma.masked_invalid(density),
        shading="flat", cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=2.0),
    )
    fig.colorbar(mesh, ax=ax, label="PIT density (1 = uniform)")
    ax.set_xlabel("PIT")
    ax.set_ylabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def _sigma_ranking(r: dict) -> pd.DataFrame:
    """Global and within-predicted-bin correlations for the sigma ranking."""
    sigma = np.asarray(r["sigma"], dtype=np.float64)
    err = np.asarray(r["err"], dtype=np.float64)
    abs_err = np.asarray(r["abs_err"], dtype=np.float64)
    pred = r["df"]["pred_log10e"].to_numpy(np.float64)
    mask = np.isfinite(sigma) & np.isfinite(err) & np.isfinite(pred)
    sigma, err, abs_err, pred = sigma[mask], err[mask], abs_err[mask], pred[mask]

    def corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        fn = spearmanr if method == "spearman" else pearsonr
        return float(fn(x, y).correlation)

    bin_idx = pd.cut(pred, bins=BINS, labels=False)
    within = []
    for _, g in pd.DataFrame({"bin": bin_idx, "s": sigma, "a": abs_err}).dropna().groupby("bin"):
        if len(g) >= 500 and g["s"].nunique() > 10:
            rho = corr(g["s"].to_numpy(), g["a"].to_numpy(), "spearman")
            if np.isfinite(rho):
                within.append(rho)
    return pd.DataFrame([{
        "n": int(len(sigma)),
        "pearson_sigma_signed_error": corr(sigma, err, "pearson"),
        "spearman_sigma_signed_error": corr(sigma, err, "spearman"),
        "pearson_sigma_abs_error": corr(sigma, abs_err, "pearson"),
        "spearman_sigma_abs_error": corr(sigma, abs_err, "spearman"),
        "spearman_abs_within_pred_bins_mean": (
            float(np.mean(within)) if within else float("nan")
        ),
        "n_pred_bins_used": int(len(within)),
    }])


def make_sigma_report(
    run_id: str,
    run_name: Optional[str] = None,
    r: Optional[dict] = None,
) -> dict:
    """Separate per-checkpoint block: coverage, PIT and sigma ranking.

    The block is deliberately independent of ``make_run_report`` so the main
    report stays about the energy prediction.  It uses the same loaded record
    when the wrapper passes ``r``; otherwise it loads the checkpoint itself.
    """
    r = load_run(run_id, run_name) if r is None else r
    out_dir = REPORT_ROOT / f"{r['run']}@best_model"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not r["has_sigma"]:
        print(f"{r['id']}: no sigma head; sigma analysis skipped")
        return r

    df = r["df"]
    bins_true = bin_table(r, "true_log10e")
    bins_pred = bin_table(r, "pred_log10e")

    print("Coverage is shown versus true and predicted energy.  Raw is the",
          "model scale; calibrated uses the trainer's global q68/q95 factors.")
    if r["distribution"] == "student_t":
        print("Student-t: raw coverage uses normal +/-1/+/-1.96 cutoffs; "
              "use calibrated factors or t quantiles for interval semantics.")
    _coverage_figure(
        bins_true,
        "true log10(E_mu / GeV)",
        f"{r['id']}: sigma coverage vs true energy",
        fig_dir / "sigma_coverage_vs_energy.png",
    )
    _coverage_figure(
        bins_pred,
        "predicted log10(E_mu / GeV)",
        f"{r['id']}: sigma coverage vs predicted energy",
        fig_dir / "sigma_coverage_vs_predicted_energy.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, label in [
        (axes[0], "pit_raw", "raw"),
        (axes[1], "pit_cal", "calibrated"),
    ]:
        ax.hist(r[key], bins=np.linspace(0, 1, 21), density=True,
                histtype="step", lw=2)
        ax.axhline(1.0, color="black", ls="--", lw=1)
        ax.set_title(f"{label} PIT; KS={_pit_ks(r[key]):.3f}")
        ax.set_xlabel("PIT")
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
    fig.suptitle(f"{r['id']}: global PIT")
    fig.tight_layout()
    fig.savefig(fig_dir / "pit.png", dpi=150, bbox_inches="tight")
    plt.show()

    pit_true = _pit_bin_table(
        df["true_log10e"].to_numpy(np.float64), r["pit_cal"]
    )
    pit_pred = _pit_bin_table(
        df["pred_log10e"].to_numpy(np.float64), r["pit_cal"]
    )
    pit_true.to_csv(out_dir / "pit_by_true_energy.csv", index=False)
    pit_pred.to_csv(out_dir / "pit_by_predicted_energy.csv", index=False)
    _pit_heatmap(
        df["true_log10e"].to_numpy(np.float64), r["pit_cal"],
        "true log10(E_mu / GeV)",
        f"{r['id']}: calibrated PIT per true-energy bin",
        fig_dir / "pit_by_true_energy_2d.png",
    )
    _pit_heatmap(
        df["pred_log10e"].to_numpy(np.float64), r["pit_cal"],
        "predicted log10(E_mu / GeV)",
        f"{r['id']}: calibrated PIT per predicted-energy bin",
        fig_dir / "pit_by_predicted_energy_2d.png",
    )
    display(pit_true[pit_true["n"] >= 50].round(4))
    display(pit_pred[pit_pred["n"] >= 50].round(4))

    fig, ax = plt.subplots(figsize=(7, 5))
    h = ax.hist2d(df["pred_log10e"], r["sigma"], bins=[BINS, 50],
                  cmap="viridis", norm=LogNorm())
    fig.colorbar(h[3], ax=ax, label="events / bin")
    ax.set_xlabel("predicted log10(E_mu / GeV)")
    ax.set_ylabel("predicted sigma (dex)")
    ax.set_title(f"{r['id']}: sigma vs predicted energy")
    fig.tight_layout()
    fig.savefig(fig_dir / "sigma_vs_predicted_energy_2d.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    ranking = _sigma_ranking(r)
    ranking.to_csv(out_dir / "sigma_ranking.csv", index=False)
    display(ranking.round(4))

    sigma = np.asarray(r["sigma"], dtype=np.float64)
    err = np.asarray(r["err"], dtype=np.float64)
    abs_err = np.asarray(r["abs_err"], dtype=np.float64)
    mask = np.isfinite(sigma) & np.isfinite(err)
    sigma, err, abs_err = sigma[mask], err[mask], abs_err[mask]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    h = axes[0].hist2d(sigma, err, bins=[60, 80], cmap="viridis",
                       norm=LogNorm())
    fig.colorbar(h[3], ax=axes[0], label="events / bin")
    axes[0].axhline(0, color="white", lw=0.8, ls=":")
    axes[0].set_xlabel("predicted sigma (dex)")
    axes[0].set_ylabel("pred - true (dex)")
    axes[0].set_title(
        f"signed error; Spearman={ranking['spearman_sigma_signed_error'].iloc[0]:+.3f}"
    )
    h = axes[1].hist2d(sigma, abs_err, bins=[60, 80], cmap="viridis",
                       norm=LogNorm())
    fig.colorbar(h[3], ax=axes[1], label="events / bin")
    axes[1].set_xlabel("predicted sigma (dex)")
    axes[1].set_ylabel("|pred - true| (dex)")
    axes[1].set_title(
        f"absolute error; Spearman={ranking['spearman_sigma_abs_error'].iloc[0]:+.3f}"
    )
    fig.suptitle(f"{r['id']}: sigma ranking of errors")
    fig.tight_layout()
    fig.savefig(fig_dir / "sigma_vs_error_2d.png", dpi=150, bbox_inches="tight")
    plt.show()
    return r


def make_wrapper(run_id: str, run_name: Optional[str] = None) -> Path:
    """Write a thin per-checkpoint notebook under analysis/by_checkpoint."""
    from nbformat import v4 as nbf
    run_name = run_name or RUN_SPECS[run_id][0]
    out_dir = ROOT / "inference/track_energy_reconstruction/analysis/by_checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_name}@best_model.ipynb"
    nb = nbf.new_notebook()
    nb["cells"] = [
        nbf.new_markdown_cell(
            f"# {run_id}: per-checkpoint test analysis\n\n"
            f"Run: `{run_name}`.  The report reads the test-part DuckDB and calls "
            f"`analysis/reporting.py::make_run_report`."
        ),
        nbf.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "ROOT = Path.cwd().resolve()\n"
            "while not (ROOT / 'inference/track_energy_reconstruction').exists():\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT / 'inference/track_energy_reconstruction/analysis'))\n"
            "import reporting\n"
            f"report = reporting.make_run_report({run_id!r}, {run_name!r})"
        ),
        nbf.new_markdown_cell(
            "## Sigma analysis\n\n"
            "Coverage versus true and predicted energy, global and energy-binned "
            "PIT, and sigma ranking of signed and absolute errors.  This block is "
            "separate from the mean-energy report."
        ),
        nbf.new_code_cell(
            f"reporting.make_sigma_report({run_id!r}, {run_name!r}, r=report)"
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "baikal25", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    from nbformat import write
    write(nb, str(out))
    return out
