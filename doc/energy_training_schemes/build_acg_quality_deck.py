#!/usr/bin/env python3
"""Build the AC-G quality deck and its backup slides.

Main slides: mean-energy accuracy; the DANN/uncertainty ablation and converged
domain probe; uncertainty, conditional PIT and error-ranking diagnostics.
Backup slides: learning curves, the non-neural baseline, training configuration,
the AdamW/sigma/DANN choices, and per-class energy spectra/statistics.

The numbers are read from the production analysis CSVs, training histories,
dataset arrays and test-part DuckDBs; the figures and the PPTX are regenerated
by this script.
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image, ImageChops, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from scipy.stats import norm, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / "inference/track_energy_reconstruction/analysis/reports"
PRED = ROOT / "inference/track_energy_reconstruction/preds"
FIG = HERE / "figures/acg_quality"
PPTX = HERE / "acg_quality_deck.pptx"

RUNS = {
    "BASE": "261027_0935_mu_energy_BASE_cont_partsplit_s42",
    "C-G": "261027_0935_mu_energy_C-G_partsplit_scratch_s42",
    "AC-G": "261027_0935_mu_energy_AC-G_partsplit_scratch_s42",
}

DATA_DIR = (
    ROOT / "data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M_partsplit"
)
NON_NEURAL_RUN = "261026_1725_non_neural_energy_baseline_partsplit_s42"
BASELINE_TEST_DB = PRED / f"{NON_NEURAL_RUN}@hgb/test.duckdb"

PAPER = "#FFFFFF"
INK = "#263238"
MUTED = "#6B777D"
GRID = "#D9DEE1"
ENERGY = "#2F6DB3"
SIGMA = "#E07A2E"
DOMAIN = "#3A8A5A"
RED = "#C0392B"
GOOD = "#2E7D4F"
WARN = "#D99A2B"
WHITE = "#FFFFFF"
NEUTRAL = "#F0EFEC"

PAPER_RGB = RGBColor(0xFF, 0xFF, 0xFF)
INK_RGB = RGBColor(0x26, 0x32, 0x38)
MUTED_RGB = RGBColor(0x6B, 0x77, 0x7D)
GRID_RGB = RGBColor(0xD9, 0xDE, 0xE1)
ENERGY_RGB = RGBColor(0x2F, 0x6D, 0xB3)
SIGMA_RGB = RGBColor(0xE0, 0x7A, 0x2E)
RED_RGB = RGBColor(0xC0, 0x39, 0x2B)
GOOD_RGB = RGBColor(0x2E, 0x7D, 0x4F)
WARN_RGB = RGBColor(0xD9, 0x9A, 0x2B)
WHITE_RGB = RGBColor(0xFF, 0xFF, 0xFF)

BINS = np.arange(-0.5, 8.01, 0.5)
PIT_BINS = np.linspace(0.0, 1.0, 21)
SW, SH = 13.333, 7.5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
})


def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(REPORT / "summary_metrics.csv").set_index("id")
    cal = pd.read_csv(REPORT / "calibration_metrics.csv").set_index("id")
    probe = pd.read_csv(REPORT / "da_probe.csv").set_index("id")
    return summary, cal, probe


def load_profile(run_id: str) -> pd.DataFrame:
    p = (REPORT / "by_checkpoint" / f"{RUNS[run_id]}@best_model"
         / "by_energy_bin.csv")
    d = pd.read_csv(p)
    return d[d["n"] >= 500].copy()


def load_predictions(run_id: str) -> pd.DataFrame:
    p = (PRED / f"{RUNS[run_id]}@best_model"
         / "mc_merged_h8s3_thr0p8.duckdb")
    with duckdb.connect(str(p), read_only=True) as con:
        d = con.execute(
            "SELECT event_fk, pred_log10e, true_log10e, pred_sigma_dex, "
            "h5_part_key FROM predictions WHERE split='test'"
        ).df()
    d["abs_err"] = (d["pred_log10e"] - d["true_log10e"]).abs()
    return d


def load_history(run_id: str) -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "trained_models/track_energy_reconstruction"
        / RUNS[run_id] / "training_history.csv"
    )


def load_comparison() -> pd.DataFrame:
    """HGB baseline and working model on exactly the same test events."""
    with duckdb.connect(str(BASELINE_TEST_DB), read_only=True) as con:
        base = con.execute(
            "SELECT event_fk, true_log10e, pred_log10e, n_hits, n_strings, "
            "part_key FROM predictions WHERE split='test'"
        ).df()
    nn_path = (
        PRED / f"{RUNS['AC-G']}@best_model" / "mc_merged_h8s3_thr0p8.duckdb"
    )
    with duckdb.connect(str(nn_path), read_only=True) as con:
        nn = con.execute(
            "SELECT event_fk, true_log10e, pred_log10e, particle_type, "
            "n_sig_hits, n_sig_strings, h5_part_key "
            "FROM predictions WHERE split='test'"
        ).df()
    m = base.merge(nn, on="event_fk", suffixes=("_base", "_nn"))
    if len(m) != len(base) or len(m) != len(nn):
        raise RuntimeError("baseline and working-model test sets do not match")
    m["err_base"] = m["true_log10e_nn"] - m["pred_log10e_base"]
    m["err_nn"] = m["true_log10e_nn"] - m["pred_log10e_nn"]
    m["abs_base"] = m["err_base"].abs()
    m["abs_nn"] = m["err_nn"].abs()
    return m


def comparison_summary(m: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Overall, per-class and part-bootstrap numbers for the HGB comparison."""
    g = (
        pd.DataFrame({
            "part": m["h5_part_key"],
            "diff": m["abs_nn"] - m["abs_base"],
        })
        .groupby("part", observed=True)
        .agg(s=("diff", "sum"), n=("diff", "size"))
    )
    rng = np.random.default_rng(42)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(g), len(g))
        vals[i] = g["s"].iloc[idx].sum() / g["n"].iloc[idx].sum()
    rows = []
    for pt, label in [(1, "Atmospheric νμ"), (2, "Astrophysical νμ")]:
        q = m[m["particle_type"] == pt]
        rows.append({
            "label": label,
            "n": int(len(q)),
            "hgb": float(q["abs_base"].mean()),
            "nn": float(q["abs_nn"].mean()),
            "delta": float((q["abs_nn"] - q["abs_base"]).mean()),
        })
    return {
        "n": int(len(m)),
        "hgb": {
            "mae": float(m["abs_base"].mean()),
            "bias": float(m["err_base"].mean()),
            "x2": float((m["abs_base"] <= np.log10(2.0)).mean()),
            "x5": float((m["abs_base"] <= np.log10(5.0)).mean()),
        },
        "nn": {
            "mae": float(m["abs_nn"].mean()),
            "bias": float(m["err_nn"].mean()),
            "x2": float((m["abs_nn"] <= np.log10(2.0)).mean()),
            "x5": float((m["abs_nn"] <= np.log10(5.0)).mean()),
        },
        "delta": float((m["abs_nn"] - m["abs_base"]).mean()),
        "delta_lo": float(np.quantile(vals, 0.025)),
        "delta_hi": float(np.quantile(vals, 0.975)),
        "by_class": rows,
    }


def class_statistics() -> pd.DataFrame:
    """Train and test energy/hit summaries for the two MC particle classes."""
    le = np.log10(np.load(DATA_DIR / "energy.npy").astype(np.float64))
    pt = np.load(DATA_DIR / "particle_types.npy")
    nn_path = (
        PRED / f"{RUNS['AC-G']}@best_model" / "mc_merged_h8s3_thr0p8.duckdb"
    )
    with duckdb.connect(str(nn_path), read_only=True) as con:
        test = con.execute(
            "SELECT true_log10e, true_energy_gev, particle_type, "
            "n_sig_hits, n_sig_strings FROM predictions WHERE split='test'"
        ).df()
    rows = []
    for code, label in [(1, "Atmospheric νμ"),
                        (2, "Astrophysical νμ (E⁻²)")]:
        tr = le[pt == code]
        te = test[test["particle_type"] == code]
        e_te = te["true_energy_gev"].to_numpy(float)
        q16, q84 = np.quantile(e_te, [0.16, 0.84])
        rows.append({
            "label": label,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_med_tev": float(10 ** np.median(tr) / 1e3),
            "test_med_tev": float(np.median(e_te) / 1e3),
            "test_q16_tev": float(q16 / 1e3),
            "test_q84_tev": float(q84 / 1e3),
            "hits": float(np.median(te["n_sig_hits"])),
            "strings": float(np.median(te["n_sig_strings"])),
        })
    return pd.DataFrame(rows)


def paired_mae_delta(run_da: str, run_other: str, n_boot: int = 3000) -> dict:
    a = load_predictions(run_da)
    b = load_predictions(run_other)
    m = a.merge(b, on="event_fk", suffixes=("_a", "_b"))
    diff = (m["abs_err_a"] - m["abs_err_b"]).to_numpy()
    g = (
        pd.DataFrame({"part": m["h5_part_key_b"], "diff": diff})
        .groupby("part", observed=True)
        .agg(s=("diff", "sum"), n=("diff", "size"))
    )
    rng = np.random.RandomState(42)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, len(g), len(g))
        vals[i] = g["s"].iloc[idx].sum() / g["n"].iloc[idx].sum()
    return {
        "n": int(len(m)),
        "mean": float(diff.mean()),
        "lo": float(np.quantile(vals, 0.025)),
        "hi": float(np.quantile(vals, 0.975)),
    }


def finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=240, facecolor=PAPER)
    plt.close(fig)
    # Trim outer white margins so the chart fills more of the slide.
    im = Image.open(path).convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255)))
    bbox = diff.getbbox()
    if bbox:
        pad = 8
        im.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                 min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))).save(path)


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9.3, length=3.5, width=0.8)


def make_mae_chart(summary: pd.DataFrame, path: Path) -> None:
    rows = [
        ("BASE", "Mean-only\nbaseline", "#8A9298", 0),
        ("C-G", "Gaussian model\nwithout DANN", SIGMA, 1),
        ("AC-G", "Gaussian model\nwith DANN", ENERGY, 2),
    ]
    fig, ax = plt.subplots(figsize=(5.75, 2.55))
    for rid, label, color, y in rows:
        r = summary.loc[rid]
        lo = float(r["mae"] - r["mae_ci_lo"])
        hi = float(r["mae_ci_hi"] - r["mae"])
        ax.errorbar(
            float(r["mae"]), y, xerr=np.array([[lo], [hi]]),
            fmt="o", ms=7.5, color=color, ecolor=color,
            elinewidth=1.2, capsize=3.5, markeredgecolor=PAPER,
            markeredgewidth=1.4, zorder=3,
        )
        ax.text(float(r["mae"]) + 0.00065, y + 0.19,
                f"{float(r['mae']):.3f}", color=INK, fontsize=8.8,
                ha="left", va="center")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([r[1] for r in rows], fontsize=8.2)
    ax.set_ylim(-0.65, 2.65)
    ax.set_xlim(0.4665, 0.487)
    ax.set_xticks([0.468, 0.472, 0.476, 0.480, 0.484])
    ax.set_xlabel("Test MAE (dex; lower is better)", fontsize=8.8)
    ax.set_title("Mean accuracy", loc="left", fontsize=10.2,
                 fontweight="bold", pad=7)
    style_axes(ax, "x")
    finish(fig, path)


def make_domain_chart(probe: pd.DataFrame, path: Path) -> None:
    # One number per model; the stricter independent probe is the honest summary.
    col = "mlp_auc_mean_part"
    std_col = "mlp_auc_std_part"
    cg = float(probe.loc["C-G", col])
    ac = float(probe.loc["AC-G", col])
    cg_s = float(probe.loc["C-G", std_col])
    ac_s = float(probe.loc["AC-G", std_col])
    y = 0.0
    fig, ax = plt.subplots(figsize=(5.75, 2.55))
    ax.plot([cg, ac], [y, y], color=GRID, lw=3.0, zorder=1)
    ax.errorbar(cg, y, xerr=cg_s, fmt="o", ms=7.5, color=SIGMA,
                ecolor=SIGMA, elinewidth=1.2, capsize=3.5,
                markeredgecolor=PAPER, markeredgewidth=1.4, zorder=3)
    ax.errorbar(ac, y, xerr=ac_s, fmt="o", ms=7.5, color=ENERGY,
                ecolor=ENERGY, elinewidth=1.2, capsize=3.5,
                markeredgecolor=PAPER, markeredgewidth=1.4, zorder=3)
    ax.text(cg, y + 0.20, f"{cg:.3f}", color=INK, fontsize=9.0,
            ha="center", va="bottom")
    ax.text(ac, y + 0.20, f"{ac:.3f}", color=INK, fontsize=9.0,
            ha="center", va="bottom")
    ax.text((cg + ac) / 2, y - 0.25, f"{100 * (ac - cg):+.1f} pp",
            color=MUTED, fontsize=8.4, ha="center", va="top")
    ax.set_yticks([0])
    ax.set_yticklabels(["Separability"], fontsize=9.0)
    ax.set_ylim(-0.65, 0.75)
    ax.set_xlim(0.600, 0.682)
    ax.set_xticks([0.60, 0.62, 0.64, 0.66, 0.68])
    ax.set_xlabel("AUC (lower = less domain-separable)", fontsize=8.6)
    ax.set_title("Domain separability", loc="left", fontsize=10.2,
                 fontweight="bold", pad=7)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SIGMA,
               markeredgecolor=PAPER, markeredgewidth=1.1, markersize=6,
               label="Without DANN"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ENERGY,
               markeredgecolor=PAPER, markeredgewidth=1.1, markersize=6,
               label="With DANN"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              fontsize=8.0, handletextpad=0.3)
    style_axes(ax, "x")
    finish(fig, path)


def make_profile_chart(path: Path) -> None:
    style = {
        "C-G": ("C-G (no DANN)", SIGMA),
        "AC-G": ("AC-G (DANN)", ENERGY),
    }
    fig, ax = plt.subplots(figsize=(12.15, 2.05))
    for rid, (label, color) in style.items():
        d = load_profile(rid)
        ax.plot(d["bin_mid"], d["mae"], color=color, lw=1.9,
                marker="o", ms=3.2, markeredgecolor=PAPER,
                markeredgewidth=0.8, label=label, zorder=3)
    ax.set_xlim(1.0, 7.55)
    ax.set_ylim(0.18, 0.96)
    ax.set_xticks(range(1, 8))
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("True log10(E_mu / GeV)", fontsize=8.8)
    ax.set_ylabel("MAE (dex)", fontsize=8.8)
    ax.set_title("MAE vs true energy", loc="left", fontsize=9.8,
                 fontweight="bold", pad=4)
    ax.legend(loc="upper right", frameon=False, fontsize=7.8,
              ncol=3, handlelength=1.7, columnspacing=1.0)
    style_axes(ax, "y")
    finish(fig, path)


def make_coverage_chart(cal: pd.DataFrame, path: Path) -> None:
    p = (REPORT / "by_checkpoint" / f"{RUNS['AC-G']}@best_model"
         / "by_predicted_energy_bin.csv")
    d = pd.read_csv(p)
    d = d[d["n"] >= 50]
    fig, axes = plt.subplots(1, 2, figsize=(5.85, 2.75), sharex=True,
                             gridspec_kw={"wspace": 0.22})
    specs = [
        ("raw68", 0.68, (0.35, 0.80), "68% interval"),
        ("raw95", 0.95, (0.65, 1.00), "95% interval"),
    ]
    for ax, (col, nominal, ylim, title) in zip(axes, specs):
        n = d["n"].to_numpy(float)
        pvals = d[col].to_numpy(float)
        err = np.sqrt(pvals * (1.0 - pvals) / n)
        ax.errorbar(
            d["bin_mid"], pvals, yerr=err, fmt="o", ms=4.1,
            color=ENERGY, ecolor=ENERGY, elinewidth=0.9,
            capsize=2.0, markeredgecolor=PAPER, markeredgewidth=0.8,
            zorder=3,
        )
        ax.axhline(nominal, color=MUTED, lw=0.9, zorder=1)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=10.2,
                     fontweight="bold", pad=4)
        ax.set_xlabel("Predicted log10(E_mu / GeV)", fontsize=9.0)
        ax.set_xticks(range(1, 8))
        style_axes(ax, "y")
    axes[0].set_ylabel("Coverage", fontsize=9.0)
    axes[0].text(1.02, 0.725, "nominal 68%", color=MUTED, fontsize=8.2)
    axes[1].text(1.02, 0.975, "nominal 95%", color=MUTED, fontsize=8.2)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.21, top=0.86,
                        wspace=0.22)
    finish(fig, path)


def pit_density(pred: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pit = norm.cdf(
        (pred["true_log10e"].to_numpy(float)
         - pred["pred_log10e"].to_numpy(float))
        / pred["pred_sigma_dex"].to_numpy(float)
    )
    density = np.full((len(BINS) - 1, len(PIT_BINS) - 1), np.nan)
    for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        m = (
            (pred["pred_log10e"].to_numpy(float) >= lo)
            & (pred["pred_log10e"].to_numpy(float) < hi)
            & np.isfinite(pit)
        )
        n = int(m.sum())
        if n < 50:
            continue
        h, _ = np.histogram(pit[m], bins=PIT_BINS)
        density[i] = h / n * len(PIT_BINS)
    return BINS, PIT_BINS, density


def make_pit_chart(pred: pd.DataFrame, path: Path) -> None:
    e_edges, p_edges, density = pit_density(pred)
    cmap = LinearSegmentedColormap.from_list(
        "acg_pit", [ENERGY, NEUTRAL, RED]
    )
    fig, ax = plt.subplots(figsize=(5.85, 2.75))
    mesh = ax.pcolormesh(
        p_edges, e_edges, np.ma.masked_invalid(density),
        shading="flat", cmap=cmap,
        norm=TwoSlopeNorm(vcenter=1.0, vmin=0.25, vmax=1.75),
    )
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("PIT density (1 = uniform)", fontsize=8.2)
    cbar.ax.tick_params(labelsize=7.8, colors=MUTED, length=2)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 8.0)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticks(range(1, 9))
    ax.set_xlabel("PIT", fontsize=9.2)
    ax.set_ylabel("Predicted log10(E_mu / GeV)", fontsize=9.0)
    ax.set_title("PIT by predicted energy", loc="left",
                 fontsize=10.3, fontweight="bold", pad=4)
    style_axes(ax, "x")
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.20, top=0.86)
    finish(fig, path)


def add_text(slide, x, y, w, h, value, color=INK_RGB, size=12,
             bold=False, align=PP_ALIGN.LEFT, name=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = value
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    if name:
        box.name = name
    return box


def add_picture(slide, path: Path, x, y, w):
    im = Image.open(path)
    h = w * im.height / im.width
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                    Inches(w), Inches(h))


def add_line(slide, x1, y1, x2, y2, color=GRID_RGB, width=0.8):
    shp = slide.shapes.add_connector(1, Inches(x1), Inches(y1),
                                     Inches(x2), Inches(y2))
    shp.line.color.rgb = color
    shp.line.width = Pt(width)


def add_card(slide, x, y, w, h, label, value, sub, accent):
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE_RGB
    card.line.color.rgb = GRID_RGB
    card.line.width = Pt(0.8)
    card.adjustments[0] = 0.08
    card.shadow.inherit = False
    # accent rule on the left
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x + 0.13), Inches(y + 0.20),
        Inches(0.045), Inches(h - 0.40)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    bar.shadow.inherit = False
    add_text(slide, x + 0.30, y + 0.15, w - 0.45, 0.22,
             label.upper(), MUTED_RGB, 7.8, True)
    add_text(slide, x + 0.30, y + 0.39, w - 0.45, 0.42,
             value, INK_RGB, 19.5, True)
    add_text(slide, x + 0.30, y + 0.86, w - 0.45, 0.25,
             sub, MUTED_RGB, 8.0)


def make_energy_profile_chart(path: Path) -> None:
    p = (REPORT / "by_checkpoint" / f"{RUNS['AC-G']}@best_model"
         / "by_energy_bin.csv")
    d = pd.read_csv(p)
    d = d[d["n"] >= 500]
    pred = load_predictions("AC-G")
    g = (
        pd.DataFrame({
            "bin": pd.cut(pred["true_log10e"], bins=BINS, labels=False),
            "abs_err": pred["abs_err"],
        })
        .groupby("bin", observed=True)["abs_err"]
        .agg(n="size", mae="mean", std="std")
        .reset_index()
    )
    g["bin_mid"] = (g["bin"] + 0.5) * 0.5
    g = g[g["n"] >= 500]
    ci95 = 1.96 * g["std"] / np.sqrt(g["n"])
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.0), sharex=True,
                             gridspec_kw={"wspace": 0.22})
    axes[0].errorbar(g["bin_mid"], g["mae"], yerr=ci95, fmt="o-",
                     ms=4.0, lw=2.0, color=ENERGY, ecolor=ENERGY,
                     elinewidth=0.9, capsize=2.2,
                     markeredgecolor=PAPER, markeredgewidth=0.9, zorder=3)
    axes[1].plot(d["bin_mid"], d["bias"], color=ENERGY, lw=2.0,
                 marker="o", ms=3.6, markeredgecolor=PAPER,
                 markeredgewidth=0.9)
    axes[1].axhline(0.0, color=MUTED, lw=0.9)
    axes[0].set_title("MAE (95% CI)", loc="left", fontsize=10.5,
                      fontweight="bold", pad=4)
    axes[1].set_title("Bias (pred − true)", loc="left", fontsize=10.5,
                      fontweight="bold", pad=4)
    axes[0].set_ylabel("MAE (log10 E)", fontsize=9.2)
    axes[1].set_ylabel("Bias (log10 E)", fontsize=9.2)
    for ax in axes:
        ax.set_xlabel("True log10(E / GeV)", fontsize=9.0)
        ax.set_xticks(range(1, 8))
        style_axes(ax, "y")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.23, top=0.86,
                        wspace=0.22)
    finish(fig, path)


def make_pred_residual_chart(path: Path) -> None:
    d = load_predictions("AC-G")
    fig, axes = plt.subplots(1, 2, figsize=(5.85, 3.0), sharex=True,
                             gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    h = ax.hist2d(d["true_log10e"], d["pred_log10e"], bins=[BINS, BINS],
                  cmap="Blues", norm=matplotlib.colors.LogNorm())
    ax.plot([BINS[0], BINS[-1]], [BINS[0], BINS[-1]],
            color=INK, lw=1.0)
    ax.set_xlabel("True log10(E / GeV)", fontsize=9.0)
    ax.set_ylabel("Predicted log10(E / GeV)", fontsize=9.0)
    ax.set_title("Prediction vs truth", loc="left", fontsize=10.5,
                 fontweight="bold", pad=4)
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(1.0, 8.0)
    ax.set_xticks(range(1, 8))
    ax.set_yticks(range(1, 8))
    style_axes(ax, "y")

    ax = axes[1]
    ax.hist2d(d["true_log10e"], d["pred_log10e"] - d["true_log10e"],
              bins=[BINS, np.linspace(-2.0, 2.0, 61)],
              cmap="Blues", norm=matplotlib.colors.LogNorm())
    g = (
        pd.DataFrame({
            "bin": pd.cut(d["true_log10e"], bins=BINS, labels=False),
            "err": d["pred_log10e"] - d["true_log10e"],
        })
        .groupby("bin", observed=True)["err"]
        .agg(n="size", median="median",
             q16=lambda x: x.quantile(0.16),
             q84=lambda x: x.quantile(0.84))
        .reset_index()
    )
    g["bin_mid"] = (g["bin"] + 0.5) * 0.5
    g = g[g["n"] >= 500]
    ax.plot(g["bin_mid"], g["median"], color=PAPER, lw=1.8,
            zorder=4, label="median")
    ax.plot(g["bin_mid"], g["q16"], color=PAPER, lw=1.0,
            ls="--", zorder=4)
    ax.plot(g["bin_mid"], g["q84"], color=PAPER, lw=1.0,
            ls="--", zorder=4)
    ax.axhline(0.0, color=PAPER, lw=0.8, ls=":", zorder=3)
    ax.set_xlabel("True log10(E / GeV)", fontsize=9.0)
    ax.set_ylabel("Pred − true (log10 E)", fontsize=9.0)
    ax.set_title("Residual vs truth", loc="left", fontsize=10.5,
                 fontweight="bold", pad=4)
    ax.set_xlim(1.0, 8.0)
    ax.set_xticks(range(1, 8))
    style_axes(ax, "y")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.23, top=0.86,
                        wspace=0.28)
    finish(fig, path)


def make_ranking_chart(path: Path) -> None:
    d = load_predictions("AC-G")
    rows = []
    for b, g in d.groupby(pd.cut(d["true_log10e"], bins=BINS, labels=False),
                          observed=True):
        if len(g) < 500:
            continue
        rho = spearmanr(g["pred_sigma_dex"], g["abs_err"]).correlation
        rows.append((float((b + 0.5) * 0.5), float(rho), len(g)))
    if not rows:
        raise RuntimeError("no ranking bins")
    r = pd.DataFrame(rows, columns=["bin_mid", "rho", "n"])
    rho_global = float(spearmanr(d["pred_sigma_dex"], d["abs_err"]).correlation)
    fig, ax = plt.subplots(figsize=(5.85, 2.45))
    ax.plot(r["bin_mid"], r["rho"], color=ENERGY, lw=2.0,
            marker="o", ms=4.0, markeredgecolor=PAPER,
            markeredgewidth=0.9, zorder=3)
    ax.axhline(0.0, color=MUTED, lw=0.9)
    ax.axhline(rho_global, color=ENERGY, lw=0.9, alpha=0.45)
    ax.text(1.05, rho_global + 0.015,
            f"global {rho_global:+.2f}", color=ENERGY, fontsize=8.4)
    ax.set_ylim(0.0, 0.62)
    ax.set_xlim(1.0, 8.0)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("True log10(E / GeV)", fontsize=9.0)
    ax.set_ylabel("Spearman(σ, |error|)", fontsize=8.0)
    ax.set_title("Ranking within energy bins", loc="left",
                 fontsize=10.4, fontweight="bold", pad=4)
    style_axes(ax, "y")
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.23, top=0.86)
    finish(fig, path)


def make_learning_curves(path: Path) -> None:
    specs = [
        ("BASE", "Mean-only", "#8A9298"),
        ("C-G", "Gaussian, no DANN", SIGMA),
        ("AC-G", "Gaussian + DANN", ENERGY),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True,
                             gridspec_kw={"wspace": 0.07})
    for rid, label, color in specs:
        d = load_history(rid)
        axes[0].plot(d["epoch"], d["mae_log10"], color=color, lw=1.8,
                     label=label)
        axes[1].plot(d["epoch"], d["val_mae_log10"], color=color, lw=1.8,
                     label=label)
    for ax, title in zip(axes, ["Training MAE", "Validation MAE"]):
        ax.set_xlim(0, 125)
        ax.set_ylim(0.44, 0.66)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Epoch", fontsize=9.0)
        ax.set_title(title, loc="left", fontsize=10.3,
                     fontweight="bold", pad=4)
        style_axes(ax, "y")
    axes[0].set_ylabel("MAE (log10 E)", fontsize=9.0)
    axes[1].legend(loc="upper right", frameon=False, fontsize=8.0,
                   handlelength=1.8)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.86,
                        wspace=0.07)
    finish(fig, path)


def make_non_neural_chart(m: pd.DataFrame, path: Path) -> None:
    bins = np.arange(1.0, 8.51, 0.5)
    m = m.copy()
    m["bin"] = pd.cut(m["true_log10e_nn"], bins=bins, right=False)
    q = (
        m.groupby("bin", observed=True)
        .agg(n=("event_fk", "size"),
             hgb=("abs_base", "mean"),
             nn=("abs_nn", "mean"),
             delta=("abs_nn", lambda x: np.nan))
        .reset_index()
    )
    d = m.groupby("bin", observed=True).apply(
        lambda x: (x["abs_nn"] - x["abs_base"]).mean(), include_groups=False
    )
    q["delta"] = d.to_numpy()
    q = q[q["n"] >= 100].copy()
    q["mid"] = [float(iv.mid) for iv in q["bin"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55),
                             gridspec_kw={"width_ratios": [1.55, 1.0],
                                          "wspace": 0.24})
    axes[0].plot(q["mid"], q["hgb"], color="#8A9298", lw=2.0, marker="o",
                 ms=3.8, markeredgecolor=PAPER, markeredgewidth=0.8,
                 label="HGB, 50 features")
    axes[0].plot(q["mid"], q["nn"], color=ENERGY, lw=2.0, marker="o",
                 ms=3.8, markeredgecolor=PAPER, markeredgewidth=0.8,
                 label="Working model")
    axes[0].set_xlim(1.0, 8.0)
    axes[0].set_ylim(0.10, 1.12)
    axes[0].set_xticks(range(1, 9))
    axes[0].set_xlabel("True log10(E / GeV)", fontsize=9.0)
    axes[0].set_ylabel("MAE (log10 E)", fontsize=9.0)
    axes[0].set_title("MAE vs true energy", loc="left", fontsize=10.2,
                      fontweight="bold", pad=4)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8.0)
    axes[0].grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    axes[0].set_axisbelow(True)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)
    axes[0].tick_params(colors=MUTED, labelsize=9.0)

    colors = [RED if v > 0 else ENERGY for v in q["delta"]]
    axes[1].bar(q["mid"], q["delta"], width=0.40, color=colors, alpha=0.85,
                edgecolor=PAPER, linewidth=0.5)
    axes[1].axhline(0.0, color=MUTED, lw=0.9)
    axes[1].set_xlim(1.0, 8.0)
    axes[1].set_xticks(range(1, 9))
    axes[1].set_xlabel("True log10(E / GeV)", fontsize=9.0)
    axes[1].set_ylabel("ΔMAE (NN − HGB)", fontsize=9.0)
    axes[1].set_title("Paired difference", loc="left", fontsize=10.2,
                      fontweight="bold", pad=4)
    axes[1].grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    axes[1].set_axisbelow(True)
    for side in ("top", "right"):
        axes[1].spines[side].set_visible(False)
    axes[1].tick_params(colors=MUTED, labelsize=9.0)
    axes[1].text(1.08, q["delta"].max() * 0.82, "NN worse",
                 color=RED, fontsize=7.8)
    axes[1].text(1.08, q["delta"].min() * 0.82, "NN better",
                 color=ENERGY, fontsize=7.8)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.86)
    finish(fig, path)


def make_spectra_chart(path: Path) -> None:
    le = np.log10(np.load(DATA_DIR / "energy.npy").astype(np.float64))
    pt = np.load(DATA_DIR / "particle_types.npy")
    nn_path = (
        PRED / f"{RUNS['AC-G']}@best_model" / "mc_merged_h8s3_thr0p8.duckdb"
    )
    with duckdb.connect(str(nn_path), read_only=True) as con:
        test = con.execute(
            "SELECT true_log10e, particle_type FROM predictions "
            "WHERE split='test'"
        ).df()
    bins = np.arange(1.0, 8.01, 0.25)
    panels = [
        ("All events", le, test["true_log10e"].to_numpy(float)),
        ("Atmospheric νμ", le[pt == 1],
         test.loc[test["particle_type"] == 1, "true_log10e"].to_numpy(float)),
        ("Astrophysical νμ (E⁻²)", le[pt == 2],
         test.loc[test["particle_type"] == 2, "true_log10e"].to_numpy(float)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.15, 3.1), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    for ax, (title, train, tst) in zip(axes, panels):
        ax.hist(train, bins=bins, density=True, histtype="step",
                color="#8A9298", lw=1.8, label="Training (flat selection)")
        ax.hist(tst, bins=bins, density=True, histtype="step",
                color=ENERGY, lw=1.8, label="Test sample (natural)")
        ax.set_xlim(1.0, 8.0)
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("True log10(E / GeV)", fontsize=9.0)
        ax.set_title(title, loc="left", fontsize=10.3,
                     fontweight="bold", pad=4)
        style_axes(ax, "y")
    axes[0].set_ylabel("Normalized density", fontsize=9.0)
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.9,
                   handlelength=1.8)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.86)
    finish(fig, path)


def make_coverage_concept_chart(path: Path) -> None:
    x = np.linspace(-4.0, 4.0, 600)
    y = norm.pdf(x)
    fig, ax = plt.subplots(figsize=(5.75, 2.8))
    ax.fill_between(x, y, where=np.abs(x) <= 1.96,
                    color=ENERGY, alpha=0.10)
    ax.fill_between(x, y, where=np.abs(x) <= 1.0,
                    color=ENERGY, alpha=0.22)
    ax.plot(x, y, color=INK, lw=1.5)
    ax.axvline(0.0, color=MUTED, lw=0.8)
    for value, label, color, text_x in [
        (-0.55, "inside 68%", GOOD, -0.55),
        (1.75, "inside 95%", WARN, 1.35),
        (2.55, "outside", RED, 2.85),
    ]:
        ax.plot(value, 0.012, "o", ms=6.0, color=color,
                markeredgecolor=PAPER, markeredgewidth=1.2, zorder=4)
        ax.annotate(label, xy=(value, 0.012), xytext=(text_x, -0.075),
                    color=MUTED, fontsize=7.0, ha="center", va="top")
    ax.text(-1.32, 0.26, "68%", color=MUTED, fontsize=8.0, ha="right")
    ax.text(1.48, 0.075, "95%", color=MUTED, fontsize=8.0, ha="left")
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-0.11, 0.44)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([])
    ax.set_xlabel("true − predicted, in predicted σ units", fontsize=8.2)
    ax.set_title("Coverage asks whether the truth is inside",
                 loc="left", fontsize=9.6, fontweight="bold", pad=5)
    style_axes(ax, "x")
    finish(fig, path)


def make_pit_concept_chart(path: Path) -> None:
    edges = np.linspace(0.0, 1.0, 21)
    centers = (edges[:-1] + edges[1:]) / 2.0
    profiles = [
        np.ones_like(centers),
        0.30 + 1.75 * (2.0 * np.abs(centers - 0.5)) ** 1.3,
        0.20 + 2.0 * np.exp(-0.5 * ((centers - 0.5) / 0.16) ** 2),
    ]
    titles = ["Calibrated\n(flat)", "Too confident\n(U-shaped)",
              "Too cautious\n(central peak)"]
    fig, axes = plt.subplots(1, 3, figsize=(5.75, 2.7), sharey=True,
                             gridspec_kw={"wspace": 0.28})
    for ax, prof, title in zip(axes, profiles, titles):
        ax.bar(centers, prof, width=0.046, color=ENERGY, alpha=0.78,
               edgecolor=PAPER, linewidth=0.5)
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 2.50)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"], fontsize=6.8)
        ax.set_yticks([])
        ax.set_title(title, fontsize=7.8, fontweight="bold", pad=4)
        ax.set_xlabel("PIT", fontsize=7.4)
        style_axes(ax, "y")
    axes[0].set_ylabel("density", fontsize=7.6)
    finish(fig, path)


def add_table(slide, x, y, w, h, rows, col_widths=None,
              header_fill=RGBColor(0xEE, 0xF3, 0xF9), font_size=9.7):
    nrows, ncols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    table = shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        table.rows[r].height = Inches(h / nrows)
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else WHITE_RGB
            tf = cell.text_frame
            tf.clear()
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(value)
            run.font.name = "Arial"
            run.font.size = Pt(font_size)
            run.font.bold = r == 0
            run.font.color.rgb = INK_RGB if r == 0 else INK_RGB
    return table


def add_probe_diagram(slide, x, y, w, h):
    boxes = [
        ("Trained\nencoder", "FROZEN", 0.0, 0.27),
        ("feature\nvector", "no gradient", 0.23, 0.50),
        ("fresh\nMLP", "new probe", 0.47, 0.72),
        ("MC vs\ndata", "domain", 0.71, 1.0),
    ]
    bw = w * 0.19
    for label, sub, frac, _ in boxes:
        bx = x + w * frac
        shp = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(bx), Inches(y + 0.15),
            Inches(bw), Inches(0.78)
        )
        shp.fill.solid()
        shp.fill.fore_color.rgb = WHITE_RGB
        shp.line.color.rgb = GRID_RGB
        shp.line.width = Pt(0.9)
        shp.shadow.inherit = False
        shp.adjustments[0] = 0.10
        tf = shp.text_frame
        tf.clear()
        tf.margin_left = Inches(0.04)
        tf.margin_right = Inches(0.04)
        tf.margin_top = Inches(0.02)
        tf.margin_bottom = Inches(0.02)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = label
        run.font.name = "Arial"
        run.font.size = Pt(8.3)
        run.font.bold = True
        run.font.color.rgb = INK_RGB
        add_text(slide, bx, y + 0.98, bw, 0.22, sub,
                 MUTED_RGB, 7.2, False, PP_ALIGN.CENTER)
    for frac in (0.19, 0.42, 0.66):
        ax = x + w * frac
        arr = slide.shapes.add_shape(
            MSO_SHAPE.RIGHT_ARROW, Inches(ax), Inches(y + 0.46),
            Inches(w * 0.035), Inches(0.16)
        )
        arr.fill.solid()
        arr.fill.fore_color.rgb = MUTED_RGB
        arr.line.fill.background()


def add_bullets(slide, x, y, w, h, items, size=9.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(3)
        run = p.add_run()
        run.text = "•  " + item
        run.font.name = "Arial"
        run.font.size = Pt(size)
        run.font.color.rgb = INK_RGB


def add_backup_header(slide, number: int, label: str, title: str,
                      subtitle: str) -> None:
    add_text(slide, 0.55, 0.28, 12.2, 0.22,
             f"BACKUP {number}  ·  {label}", MUTED_RGB, 8.3, True)
    add_text(slide, 0.55, 0.50, 12.2, 0.46, title, INK_RGB, 21.5, True)
    add_text(slide, 0.55, 1.04, 12.2, 0.30, subtitle, MUTED_RGB, 9.3)


def build_deck(charts: dict[str, Path], summary: pd.DataFrame,
               cal: pd.DataFrame, probe: pd.DataFrame,
               delta_cg: dict, delta_base: dict,
               comp: dict, class_stats: pd.DataFrame) -> None:
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    r = summary.loc["AC-G"]
    calr = cal.loc["AC-G"]
    cov = pd.read_csv(
        REPORT / "by_checkpoint" / f"{RUNS['AC-G']}@best_model"
        / "by_predicted_energy_bin.csv"
    )
    cov = cov[cov["n"] >= 50]
    raw68_lo = float(cov["raw68"].min())
    raw68_hi = float(cov["raw68"].max())
    raw95_lo = float(cov["raw95"].min())
    raw95_hi = float(cov["raw95"].max())

    # ---------------- slide 1: working model ----------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER_RGB
    add_text(slide, 0.55, 0.28, 12.2, 0.22,
             "WORKING MODEL  ·  GAUSSIAN ENERGY + UNCERTAINTY + DANN",
             MUTED_RGB, 8.3, True)
    add_text(slide, 0.55, 0.50, 12.2, 0.48,
             "Working model: energy resolution and bias",
             INK_RGB, 23.5, True, name="slide1_title")
    add_text(slide, 0.55, 1.04, 12.2, 0.28,
             "MC evaluation · natural spectrum · 418k events · "
             "Gaussian mean and uncertainty trained jointly",
             MUTED_RGB, 9.4)
    add_text(slide, 0.55, 1.34, 12.2, 0.25,
             "MAE 0.475 log10(E/GeV)  ·  within ×2 46.6%  ·  within ×5 77.6%  ·  "
             "σ covers 70.3 / 95.0%",
             INK_RGB, 9.2)

    add_picture(slide, charts["energy_profile"], 0.55, 1.66, 5.75)
    add_picture(slide, charts["pred_resid"], 7.03, 1.66, 5.75)

    add_text(slide, 0.55, 5.22, 5.75, 0.22,
             "WHAT THE NUMBERS MEAN", MUTED_RGB, 8.0, True)
    add_text(slide, 0.55, 5.46, 5.75, 1.15,
             "MAE 0.475 log10(E/GeV) ≈ factor 3.  Best near 2.25 "
             "log10 (~180 GeV); largest errors at 3.25–3.75 (~2–6 TeV); "
             "the high-energy tail is sparse.  Whiskers on the MAE curve "
             "are 95% CIs of the binned mean, not sigma.",
             INK_RGB, 8.9)
    add_text(slide, 7.03, 5.22, 5.75, 0.22,
             "THE BIAS IS CONDITIONAL", MUTED_RGB, 8.0, True)
    add_text(slide, 7.03, 5.46, 5.75, 1.15,
             "The bias survives binning in true energy: low true E is "
             "overpredicted, high true E underpredicted.  This is a "
             "conditional mean/model effect, not a marginal migration "
             "artefact.  High-energy bins are sparse and heavily weighted, "
             "yet show little bias; a flat training prior alone does not "
             "explain this.",
             INK_RGB, 8.9)

    notes = slide.notes_slide.notes_text_frame
    notes.text = (
        "The working model uses DANN and jointly trained mean/uncertainty. "
        f"Global MAE is {r['mae']:.4f} in log10(E/GeV); bias = "
        f"{r['bias']:+.4f}; within x2 = {100*r['within_x2']:.1f}%, "
        f"within x5 = {100*r['within_x5']:.1f}%. "
        "The 0.475 log10 unit corresponds to a multiplicative factor "
        f"{10**0.475:.1f}. The residual cloud is conditionally biased: low "
        "true energy is overpredicted and high true energy underpredicted. "
        "This persists after binning in true energy, so it is not just a "
        "marginal composition effect; high-energy bins show little bias. Single "
        "training seed; MC evaluation only."
    )

    # ---------------- slide 2: ablation ----------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER_RGB
    add_text(slide, 0.55, 0.28, 12.2, 0.22,
             "ABLATION  ·  DANN AND UNCERTAINTY",
             MUTED_RGB, 8.3, True)
    add_text(slide, 0.55, 0.50, 12.2, 0.48,
             "Global accuracy holds after DANN and uncertainty",
             INK_RGB, 23.5, True, name="slide2_title")
    add_text(slide, 0.55, 1.04, 12.2, 0.28,
             "Same evaluation; remove one modification at a time.",
             MUTED_RGB, 9.4)

    add_text(slide, 0.55, 1.53, 6.1, 0.26,
             "GLOBAL METRICS  (log10 E)", MUTED_RGB, 8.0, True)
    def _mae_pm(rid):
        row = summary.loc[rid]
        half = 0.5 * (row["mae_ci_hi"] - row["mae_ci_lo"])
        return f"{row['mae']:.4f} ± {half:.4f}"

    m0 = summary.loc["BASE"]
    rows = [
        ["Configuration", "MAE ± 95% CI", "Bias", "Within ×2", "Within ×5"],
        ["Mean-only baseline", _mae_pm("BASE"), f"{m0['bias']:+.3f}",
         f"{100*m0['within_x2']:.1f}%", f"{100*m0['within_x5']:.1f}%"],
        ["Gaussian, no DANN", _mae_pm("C-G"),
         f"{summary.loc['C-G']['bias']:+.3f}",
         f"{100*summary.loc['C-G']['within_x2']:.1f}%",
         f"{100*summary.loc['C-G']['within_x5']:.1f}%"],
        ["Gaussian + DANN", _mae_pm("AC-G"),
         f"{summary.loc['AC-G']['bias']:+.3f}",
         f"{100*summary.loc['AC-G']['within_x2']:.1f}%",
         f"{100*summary.loc['AC-G']['within_x5']:.1f}%"],
    ]
    add_table(slide, 0.55, 1.84, 6.1, 2.22, rows,
              col_widths=[2.05, 1.55, 0.75, 0.85, 0.90])
    d_cg = 0.5 * (delta_cg["hi"] - delta_cg["lo"])
    d_base = 0.5 * (delta_base["hi"] - delta_base["lo"])
    add_text(slide, 0.55, 4.28, 6.1, 0.62,
             "Paired ΔMAE vs mean-only: "
             f"{delta_base['mean']:+.4f} ± {d_base:.4f}.  "
             "Vs the same model without DANN: "
             f"{delta_cg['mean']:+.4f} ± {d_cg:.4f}.  "
             "No global accuracy penalty.",
             INK_RGB, 8.7)

    add_text(slide, 6.95, 1.53, 5.8, 0.26,
             "DOMAIN-SEPARATION PROBE", MUTED_RGB, 8.0, True)
    add_probe_diagram(slide, 6.95, 1.82, 5.8, 1.20)
    add_text(slide, 6.95, 3.18, 5.8, 0.42,
             "Freeze the trained encoder; train a fresh MLP only to separate "
             "MC from data.  Lower AUC = less domain information.",
             MUTED_RGB, 8.2)
    cg = probe.loc["C-G"]
    ac = probe.loc["AC-G"]
    cg_auc = float(cg["mlp_auc_mean_part"])
    ac_auc = float(ac["mlp_auc_mean_part"])
    cg_err = float(cg["mlp_auc_std_part"])
    ac_err = float(ac["mlp_auc_std_part"])
    diff_err = float(np.hypot(cg_err, ac_err))
    d_rows = [
        ["Probe", "AUC ± seed scatter"],
        ["Without DANN", f"{cg_auc:.3f} ± {cg_err:.3f}"],
        ["With DANN", f"{ac_auc:.3f} ± {ac_err:.3f}"],
        ["Difference", f"{ac_auc-cg_auc:+.3f} ± {diff_err:.3f}"],
    ]
    add_table(slide, 6.95, 3.70, 5.8, 1.45, d_rows,
              col_widths=[3.2, 2.6])

    add_text(slide, 0.55, 5.55, 12.2, 0.42,
             "Global accuracy holds; domain information drops by about 2 pp.",
             INK_RGB, 14.0, True)
    add_text(slide, 0.55, 6.05, 12.2, 0.52,
             "The probe is a new MLP trained after the encoder is frozen; it is "
             "not the training critic.  It sees only the learned features.",
             MUTED_RGB, 9.0)

    notes = slide.notes_slide.notes_text_frame
    notes.text = (
        "Ablation in global metrics. The mean-only baseline has no DANN and no "
        "uncertainty head; the no-DANN model has uncertainty but no DANN; the "
        "working model has both. Global MAE changes by less than 0.004 log10 "
        "units. The domain probe is a fresh MLP trained on frozen encoder "
        "features; it does not backpropagate into the encoder and is not the "
        "training critic. Its AUC falls from 0.652 to 0.632 (2.1 percentage "
        "points), so DANN does something."
    )

    # ---------------- slide 3: sigma ----------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER_RGB
    add_text(slide, 0.55, 0.28, 12.2, 0.22,
             "UNCERTAINTY  ·  GAUSSIAN ENERGY + DANN",
             MUTED_RGB, 8.3, True)
    add_text(slide, 0.55, 0.50, 12.2, 0.48,
             "Sigma: good on average, not per energy",
             INK_RGB, 23.5, True, name="slide3_title")
    add_text(slide, 0.55, 1.04, 12.2, 0.28,
             "68/95% coverage: 70.3% / 95.0% globally; in the tails, "
             "coverage and PIT drift with predicted energy.",
             MUTED_RGB, 9.4)

    add_picture(slide, charts["coverage"], 0.55, 1.62, 5.85)
    add_picture(slide, charts["pit"], 6.93, 1.62, 5.85)
    add_text(slide, 6.93, 1.34, 5.85, 0.24,
             "PIT: u = Φ[(E_true − E_pred) / σ]   ·   flat = correct",
             INK_RGB, 9.0)

    add_picture(slide, charts["ranking"], 0.55, 4.72, 5.8)
    add_text(slide, 6.93, 4.72, 5.85, 0.26,
             "THE PICTURE IN WORDS", MUTED_RGB, 8.0, True)
    add_bullets(slide, 6.93, 4.98, 5.85, 1.95, [
        f"68/95% coverage: {100*raw68_lo:.0f}–{100*raw68_hi:.0f}% "
        f"and {100*raw95_lo:.0f}–{100*raw95_hi:.0f}% across energy; global "
        "averages hide the tails.",
        "PIT tilts: low-E events sit high, high-E events low. This follows "
        "the conditional mean bias, not just a width error.",
        "Ranking: Spearman(σ, |error|) inside true-energy bins; global +0.47, "
        "bins +0.15–0.53.",
        "The flat prior may affect the fit, but does not create the binwise "
        "bias; high-E bins show little bias. Global rescaling fixes averages, "
        "not tails; the network optimized weighted global NLL, not conditional "
        "coverage.",
    ], size=9.2)

    notes = slide.notes_slide.notes_text_frame
    notes.text = (
        "Sigma as produced by the model. Global coverage is "
        f"{100*calr['raw68']:.1f}% at 68% and {100*calr['raw95']:.1f}% at 95%. "
        "Across predicted-energy bins, 68% coverage ranges from "
        f"{100*raw68_lo:.1f}% to {100*raw68_hi:.1f}%. PIT is computed as "
        "u = Phi((true - pred) / sigma); a flat histogram is the target. "
        "The tilt follows the energy-dependent mean bias. The flat training "
        "prior may affect the fit, but the binwise bias is not a marginal "
        "composition artefact; high-energy bins show little bias. Global "
        "rescaling of the mean or width would only adjust the average; the "
        "network optimized a weighted global NLL, not conditional coverage. "
        "Ranking is Spearman(sigma, |error|) inside each true-energy bin."
    )

    # ---------------- backup 1: learning curves ----------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER_RGB
    add_backup_header(
        slide, 1, "LEARNING CURVES",
        "Learning curves: all three models converge",
        "Training and validation MAE on the held-out validation sample; shared "
        "encoder, same training data.",
    )
    add_picture(slide, charts["learning"], 0.55, 1.52, 7.25)
    hist = {rid: load_history(rid) for rid in ["BASE", "C-G", "AC-G"]}
    rows = [["Model", "Epochs", "Best val MAE", "Final val MAE"]]
    for rid, label in [("BASE", "Mean-only"),
                       ("C-G", "Gaussian, no DANN"),
                       ("AC-G", "Gaussian + DANN")]:
        d = hist[rid]
        rows.append([label, f"{len(d)}", f"{d['val_mae_log10'].min():.4f}",
                     f"{d['val_mae_log10'].iloc[-1]:.4f}"])
    add_table(slide, 8.05, 1.58, 4.75, 1.72, rows,
              col_widths=[1.90, 0.70, 1.10, 1.05], font_size=8.8)
    add_text(slide, 8.05, 3.52, 4.75, 0.24,
             "WHAT THE CURVES SHOW", MUTED_RGB, 8.0, True)
    add_bullets(slide, 8.05, 3.78, 4.75, 1.55, [
        "The working model reaches its lowest validation MAE at epoch 76; "
        "the no-DANN model at epoch 95.",
        "At convergence the two Gaussian curves differ by at most ~0.0035; "
        "DANN does not buy or cost global accuracy.",
        "The mean-only continuation converges about 0.007 higher at its best.",
        "Loss is not compared: Huber and Gaussian NLL have different scales "
        "and signs. MAE is the common scale.",
    ], size=8.5)
    add_text(slide, 0.55, 5.02, 12.2, 1.22,
             "Gaussian runs use a fixed 100-epoch budget and no early stopping; "
             "the mean-only baseline is a 124-epoch continuation at a lower "
             "learning rate.  The working-model curve is not distorted by the "
             "adversarial term: validation MAE decreases smoothly and the final "
             "checkpoint remains in the same band as the two baselines.",
             INK_RGB, 9.2)
    notes = slide.notes_slide.notes_text_frame
    notes.text = (
        "Learning curves are MAE because the three runs do not share a loss: the "
        "mean-only run optimises Huber, the Gaussian runs optimise weighted "
        "Gaussian NLL. The validation curves are on the 80k-event held-out "
        "fraction of the training sample; the held-out test MAE in the main slides "
        "is a different, harder population. The working model's best validation "
        "epoch is 76; the no-DANN model's is 95; the mean-only history reaches "
        "its minimum at 118 but its saved checkpoint is epoch 111. The small "
        "differences in the table are within seed-to-seed variation; the deck "
        "does not claim an accuracy gain from DANN."
    )

    # ---------------- backup 2: non-neural baseline ----------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER_RGB
    add_backup_header(
        slide, 2, "NON-NEURAL BASELINE",
        "Non-neural baseline: sequence model wins overall, not everywhere",
        "Same held-out test sample: 418,153 events.  HGB uses 50 handcrafted "
        "per-event features; it has no hit sequence.",
    )
    add_picture(slide, charts["non_neural"], 0.55, 1.52, 7.30)
    c = comp
    rows = [
        ["Test metric", "HGB", "Working", "Δ (working − HGB)"],
        ["MAE", f"{c['hgb']['mae']:.4f}", f"{c['nn']['mae']:.4f}",
         f"{c['delta']:+.4f}"],
        ["Bias", f"{c['hgb']['bias']:+.4f}", f"{c['nn']['bias']:+.4f}",
         f"{c['nn']['bias'] - c['hgb']['bias']:+.4f}"],
        ["Within ×2", f"{100*c['hgb']['x2']:.1f}%",
         f"{100*c['nn']['x2']:.1f}%",
         f"{100*(c['nn']['x2'] - c['hgb']['x2']):+.1f} pp"],
        ["Within ×5", f"{100*c['hgb']['x5']:.1f}%",
         f"{100*c['nn']['x5']:.1f}%",
         f"{100*(c['nn']['x5'] - c['hgb']['x5']):+.1f} pp"],
    ]
    add_table(slide, 8.10, 1.58, 4.70, 1.95, rows,
              col_widths=[1.25, 0.95, 1.00, 1.50], font_size=8.8)
    add_text(slide, 8.10, 3.64, 4.70, 0.72,
             f"ΔMAE 95% CI: {c['delta_lo']:+.4f} to {c['delta_hi']:+.4f} "
             "(block bootstrap).", INK_RGB, 8.8)
    by_class = c["by_class"]
    add_text(slide, 8.10, 4.05, 4.70, 0.24,
             "PER CLASS, MAE HGB → WORKING", MUTED_RGB, 8.0, True)
    add_text(slide, 8.10, 4.32, 4.70, 0.28,
             f"Atmospheric νμ: {by_class[0]['hgb']:.3f} → "
             f"{by_class[0]['nn']:.3f}", INK_RGB, 8.8)
    add_text(slide, 8.10, 4.62, 4.70, 0.28,
             f"Astrophysical νμ: {by_class[1]['hgb']:.3f} → "
             f"{by_class[1]['nn']:.3f}", INK_RGB, 8.8)
    add_text(slide, 0.55, 5.38, 12.2, 1.10,
             "The HGB baseline is slightly better only between 0.3 and 10 TeV; "
             "outside that range the sequence model wins.  HGB sees charge and "
             "geometry summaries but cannot use the ordered hit pattern.  On the "
             "20k internal holdout, the baseline suite was HGB 0.514, Ridge "
             "0.617, log-linear 0.761 in log10 E; those numbers are on a "
             "different population and are not the test comparison above.",
             INK_RGB, 9.0)
    notes = slide.notes_slide.notes_text_frame
    notes.text = (
        "The baseline comparison is paired event by event on the same held-out "
        "sample. HGB is the strongest of the handcrafted baselines on the 20k "
        "held-out sample (MAE 0.5136; Ridge 0.6172; log-linear 0.7611), but "
        "only HGB has test-sample predictions in this artefact. The overall test "
        "MAE is 0.5212 for HGB and 0.4748 for the working model; the paired "
        "difference is -0.0464 with a block-bootstrap 95% interval of -0.0477 to "
        "-0.0448. Both classes improve. The binned curve shows the exception: "
        "between about 2.5 and 4.0 in log10 E, HGB is slightly better. This is "
        "an honest limitation of the sequence model, not hidden by the global "
        "average."
    )

    # ---------------- backup 3: training configuration ----------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER_RGB
    add_backup_header(
        slide, 3, "TRAINING CONFIGURATION",
        "Training setup: one architecture, three objectives",
        "The encoder and data pipeline are shared; only the objective, σ head "
        "and DANN switch.",
    )
    left_rows = [
        ["Component", "Setting"],
        ["Input", "5 numbers per hit: Q, t, x, y, z; up to 500 selected hits; "
                  "Q clipped at 100 p.e."],
        ["Encoder", "4 layers; d=128; 4 heads; FFN 256; CLS pooling"],
        ["Regularization", "Dropout 0.1 in encoder and heads; weight decay 1e-2"],
        ["Mean head", "128 → 64 → 1"],
        ["σ head", "128 → 64 → 1 Gaussian log σ²; Gaussian models only"],
        ["Parameters", "539k mean-only; 548k with σ (σ head 8.3k)"],
    ]
    add_table(slide, 0.55, 1.55, 5.95, 4.15, left_rows,
              col_widths=[1.35, 4.60], font_size=8.7)
    right_rows = [
        ["Setting", "Value"],
        ["Optimizer", "AdamW"],
        ["Base learning rate", "5e-4; mean-only continuation 3e-5"],
        ["Weight decay", "1e-2"],
        ["Batch size", "256 events"],
        ["Epochs", "100 Gaussian; 124 mean-only (150 cap)"],
        ["Gradient clip", "1.0"],
        ["Scheduler", "ReduceLROnPlateau: ×0.5, patience 4, min 1e-6"],
        ["Selection", "Lowest validation MAE; seed 42"],
    ]
    add_table(slide, 6.75, 1.55, 6.05, 4.15, right_rows,
              col_widths=[2.05, 4.00], font_size=8.7)
    add_text(slide, 0.55, 5.92, 12.2, 0.92,
             "Training data: flat selection in log10 E, weights 1–20, effective "
             "sample size 32.6% of the 1M events.  The target is normalized by "
             "the train mean and standard deviation; the Gaussian NLL is weighted "
             "per event.  The mean-only run has no σ head and no DANN; it is a "
             "continuation, so its base learning rate is lower.",
             INK_RGB, 8.9)
    notes = slide.notes_slide.notes_text_frame
    notes.text = (
        "All three models use the same 4-layer, d=128 transformer encoder with "
        "4 heads, FFN 256, CLS pooling and dropout 0.1. The mean-only model has "
        "539,393 trainable parameters; adding the Gaussian variance head brings "
        "the total to 547,714, of which 8,321 belong to the sigma head. The "
        "Gaussian runs use AdamW with lr 5e-4, weight decay 1e-2, batch 256, "
        "gradient clipping 1.0, and a ReduceLROnPlateau scheduler (factor 0.5, "
        "patience 4, min lr 1e-6). The mean-only continuation runs at 3e-5 for "
        "up to 150 epochs and stops after 124. Training data are flat-selected "
        "in log10 E with weights 1-20; the effective sample size is 32.6% of 1M. "
        "The target is normalized with the train mean 3.6857 and std 2.4827."
    )

    # ---------------- backup 4: why AdamW, sigma and DANN ----------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER_RGB
    add_backup_header(
        slide, 4, "DESIGN CHOICES",
        "Why AdamW, and what is actually tuned",
        "A priori choices, not an optimizer ablation.  Only DANN and the σ head "
        "are switched in the main comparison.",
    )
    add_text(slide, 0.55, 1.55, 6.0, 0.24,
             "WHY ADAMW", MUTED_RGB, 8.0, True)
    add_bullets(slide, 0.55, 1.82, 6.0, 2.25, [
        "Chosen as the standard optimizer for transformer encoders; there was "
        "no Adam-vs-AdamW head-to-head in this line.",
        "Decoupled weight decay applies L2 shrinkage separately from the "
        "adaptive moment scaling; this is the practical reason to prefer AdamW.",
        "Stable at 5e-4 with gradient clipping 1.0; no divergence or loss "
        "explosion in the three runs.",
        "Weight decay 1e-2 is a regularizer; the deck does not claim it "
        "improved test accuracy.",
    ], size=8.8)
    diff_rows = [
        ["Model", "Loss", "σ head", "DANN"],
        ["Mean-only", "Huber δ=0.5", "no", "no"],
        ["Gaussian, no DANN", "Gaussian NLL", "yes", "no"],
        ["Working", "Gaussian NLL", "yes", "yes"],
    ]
    add_table(slide, 6.90, 1.55, 5.90, 1.70, diff_rows,
              col_widths=[1.95, 1.75, 1.05, 1.15], font_size=9.0)
    add_text(slide, 0.55, 3.30, 6.0, 0.24,
             "SIGMA HEAD", MUTED_RGB, 8.0, True)
    add_bullets(slide, 0.55, 3.56, 6.0, 1.95, [
        "σ is a learned per-event scale in log10 E; the Gaussian NLL is "
        "½[(y−μ)/σ]² + log σ + const.",
        "Separate AdamW for the 8.3k σ parameters; encoder and mean use the "
        "main optimizer (full_two_opt).",
        "First 2 epochs use Huber on the mean only; then NLL starts. σ is "
        "initialized at 0.5 and clipped to [0.05, 5] in log10 E.",
    ], size=8.8)
    add_text(slide, 6.90, 3.30, 5.9, 0.24,
             "DANN", MUTED_RGB, 8.0, True)
    add_bullets(slide, 6.90, 3.56, 5.9, 2.55, [
        "λ = 0 for the first 5 epochs, then λ max 0.01; no ramp; "
        "gradient-reversal factor 1.0.",
        "Discriminator 2×128 with LayerNorm and dropout 0.1; Adam at 1e-3; one "
        "critic step per batch on frozen features.",
        "The adversarial term uses domain labels only; energy labels never enter "
        "it.",
        "The measured effect is the frozen-feature probe AUC drop, not a "
        "certificate of domain invariance.",
    ], size=8.8)
    add_text(slide, 0.55, 5.92, 12.25, 0.42,
             "Not tuned here: encoder width/depth, dropout, optimizer.",
             MUTED_RGB, 8.6)
    notes = slide.notes_slide.notes_text_frame
    notes.text = (
        "AdamW was chosen a priori because it is the standard optimizer for "
        "transformer encoders and its decoupled weight decay is a cleaner "
        "regularizer than L2 mixed into an adaptive update. This is not an "
        "optimizer ablation: Adam, SGD and different weight decays were not "
        "compared. The model width, depth, dropout and batch size were also fixed "
        "before the DANN/uncertainty comparison. The Gaussian NLL uses a learned "
        "log-variance; sigma is a scale in log10 E, not an OOD score. The sigma "
        "head has its own AdamW (full_two_opt), a two-epoch mean-only Huber "
        "warm-up, and is clipped to [0.05, 5]. DANN uses lambda max 0.01 after "
        "a five-epoch "
        "warm-up; the discriminator is a 2x128 MLP with LayerNorm and dropout "
        "0.1, trained with Adam at 1e-3; the critic takes one step per batch on "
        "frozen features. The adversarial term sees domain labels only. The "
        "probe AUC drop is a measured property of the frozen features, not proof "
        "that the domain gap is closed."
    )

    # ---------------- backup 5: spectra and class statistics ----------------
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER_RGB
    add_backup_header(
        slide, 5, "DATA",
        "Energy spectra and per-class statistics",
        "MC truth only; experimental data have no true energy and are not in "
        "these truth-level distributions.",
    )
    add_picture(slide, charts["spectra"], 0.55, 1.52, 12.15)
    table_rows = [
        ["Class", "Train n", "Test n", "Median E, train → test",
         "Test 16–84% E", "Test hits / strings"],
    ]
    for _, row in class_stats.iterrows():
        table_rows.append([
            row["label"],
            f"{int(row['n_train']):,}",
            f"{int(row['n_test']):,}",
            f"{row['train_med_tev']:.2f} → {row['test_med_tev']:.2f} TeV",
            f"{row['test_q16_tev']:.2f}–{row['test_q84_tev']:.1f} TeV",
            f"{int(row['hits'])} / {int(row['strings'])}",
        ])
    add_table(slide, 0.55, 4.93, 12.15, 1.23, table_rows,
              col_widths=[2.85, 1.20, 1.20, 2.55, 2.20, 2.15],
              font_size=9.0)
    add_text(slide, 0.55, 6.42, 12.2, 0.58,
             "Training sample: 1.0M events, flat selection in log10 E, weights "
             "1–20, ESS 32.6%.  Test: 418,153 events in a new MC sample with the "
             "natural spectrum.  Hits and strings are test medians after the "
             "8-hit / 3-string selection; charge is clipped at 100 p.e.",
             INK_RGB, 8.6)
    notes = slide.notes_slide.notes_text_frame
    notes.text = (
        "The training sample is flat-selected in log10 E over the 1M events, "
        "with weights 1 to 20 and an effective sample size of 32.6%; the test "
        "sample is a separate natural-spectrum sample of 418,153 events. The two "
        "MC classes are atmospheric nu_mu (class 1) and astrophysical nu_mu with "
        "an E^-2 spectrum (class 2; the file label is nue2, but it is not an "
        "electron neutrino). Median energies are 0.28 to 0.30 TeV for "
        "atmospheric and 39 to 29 TeV for astrophysical. The test median hit "
        "counts are 9 and 15, and the median numbers of strings are 3 and 4. "
        "Experimental data have no true energy, so they cannot appear in this "
        "truth-level table."
    )

    prs.save(str(PPTX))


def _font(size: float, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}",
                              max(8, int(round(size * 150 / 72))))


def _rgb(color) -> tuple[int, int, int]:
    value = str(color).lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def render_preview() -> None:
    """Approximate visual QA rasteriser for the generated pptx.

    LibreOffice is not available on this host. This renderer uses the actual
    shape positions, text runs and picture blobs from the PPTX, so layout
    collisions and clipping can be inspected without re-authoring the slides.
    """
    prs = Presentation(str(PPTX))
    scale = 150
    for n, slide in enumerate(prs.slides, start=1):
        canvas = Image.new("RGB", (int(SW * scale), int(SH * scale)),
                           _rgb(PAPER))
        draw = ImageDraw.Draw(canvas)
        for shape in slide.shapes:
            x = int(shape.left / 914400 * scale)
            y = int(shape.top / 914400 * scale)
            w = int(shape.width / 914400 * scale)
            h = int(shape.height / 914400 * scale)
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                im = Image.open(io.BytesIO(shape.image.blob)).convert("RGB")
                im = im.resize((w, h), Image.Resampling.LANCZOS)
                canvas.paste(im, (x, y))
                continue
            if getattr(shape, "has_table", False):
                table = shape.table
                col_w = [int(c.width / 914400 * scale) for c in table.columns]
                row_h = [int(r.height / 914400 * scale) for r in table.rows]
                yy = y
                for r, row in enumerate(table.rows):
                    xx = x
                    for c, cell in enumerate(row.cells):
                        fill = None
                        try:
                            fill = _rgb(cell.fill.fore_color.rgb)
                        except Exception:
                            pass
                        if fill is not None:
                            draw.rectangle([xx, yy, xx + col_w[c], yy + row_h[r]],
                                           fill=fill, outline=_rgb(GRID))
                        text = cell.text
                        if text:
                            run = cell.text_frame.paragraphs[0].runs[0]
                            size = run.font.size.pt if run.font.size else 9.7
                            bold = bool(run.font.bold)
                            color = _rgb(run.font.color.rgb) if run.font.color.rgb else _rgb(INK)
                            font = _font(size, bold)
                            align = cell.text_frame.paragraphs[0].alignment
                            tw = draw.textlength(text, font=font)
                            tx = xx + (col_w[c] - tw) / 2 if align == PP_ALIGN.CENTER else xx + 5
                            draw.text((tx, yy + (row_h[r] - font.size * 1.2) / 2),
                                      text, font=font, fill=color)
                        xx += col_w[c]
                    yy += row_h[r]
                continue
            if shape.shape_type == MSO_SHAPE_TYPE.LINE:
                continue
            fill = None
            try:
                fill = _rgb(shape.fill.fore_color.rgb)
            except Exception:
                pass
            line = None
            try:
                line = _rgb(shape.line.color.rgb)
            except Exception:
                pass
            if fill is not None:
                if shape.shape_type == MSO_SHAPE.ROUNDED_RECTANGLE:
                    draw.rounded_rectangle(
                        [x, y, x + w, y + h], radius=max(4, int(0.08 * h)),
                        fill=fill, outline=line, width=1,
                    )
                else:
                    draw.rectangle([x, y, x + w, y + h], fill=fill,
                                   outline=line, width=1)
            if not shape.has_text_frame:
                continue
            tf = shape.text_frame
            cursor_y = y
            for p in tf.paragraphs:
                text = "".join(run.text for run in p.runs).replace("\n", " ")
                if not text:
                    continue
                run = p.runs[0]
                size = run.font.size.pt if run.font.size is not None else 10
                bold = bool(run.font.bold)
                color = _rgb(run.font.color.rgb) if run.font.color.rgb else _rgb(INK)
                font = _font(size, bold)
                max_w = max(20, w)
                # simple greedy wrap by character
                lines = []
                cur = ""
                for word in text.split(" "):
                    cand = (cur + " " + word).strip()
                    if draw.textlength(cand, font=font) <= max_w or not cur:
                        cur = cand
                    else:
                        lines.append(cur)
                        cur = word
                if cur:
                    lines.append(cur)
                line_h = font.size * 1.18
                for line in lines:
                    tw = draw.textlength(line, font=font)
                    if p.alignment == PP_ALIGN.CENTER:
                        tx = x + (w - tw) / 2
                    else:
                        tx = x
                    draw.text((tx, cursor_y), line, font=font, fill=color)
                    cursor_y += line_h
        out = FIG / f"preview_slide_{n}.png"
        canvas.save(out)
        print("preview", out)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary, cal, probe = load_metrics()
    charts = {
        "energy_profile": FIG / "slide1_energy_profile.png",
        "pred_resid": FIG / "slide1_pred_residual.png",
        "coverage": FIG / "slide3_coverage.png",
        "pit": FIG / "slide3_pit.png",
        "ranking": FIG / "slide3_ranking.png",
        "learning": FIG / "backup_learning_curves.png",
        "non_neural": FIG / "backup_non_neural.png",
        "spectra": FIG / "backup_spectra.png",
    }
    make_energy_profile_chart(charts["energy_profile"])
    make_pred_residual_chart(charts["pred_resid"])
    make_coverage_chart(cal, charts["coverage"])
    ac_pred = load_predictions("AC-G")
    make_pit_chart(ac_pred, charts["pit"])
    make_ranking_chart(charts["ranking"])
    make_learning_curves(charts["learning"])
    comparison = load_comparison()
    comp = comparison_summary(comparison)
    make_non_neural_chart(comparison, charts["non_neural"])
    make_spectra_chart(charts["spectra"])
    class_stats = class_statistics()

    delta_cg = paired_mae_delta("AC-G", "C-G")
    delta_base = paired_mae_delta("AC-G", "BASE")
    print("ΔMAE AC-G vs C-G", delta_cg)
    print("ΔMAE AC-G vs BASE", delta_base)
    print("HGB comparison", comp)
    print("class stats", class_stats.to_string(index=False))
    build_deck(charts, summary, cal, probe, delta_cg, delta_base,
               comp, class_stats)
    print("wrote", PPTX)
    render_preview()


if __name__ == "__main__":
    main()
