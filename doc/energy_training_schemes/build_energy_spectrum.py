#!/usr/bin/env python3
"""Training vs natural MC energy spectrum for the muon-energy dataset.

The "natural" curve is the distribution of all h8s3 + single-track candidates in
the train parts, before flat selection.  The training curve uses the flat-selection
weights actually passed to the loss, so it represents the effective training
spectrum rather than the raw row counts.  Both curves are normalized to unit area
per dex, because the candidate and training populations have different sizes.

Source of the per-bin counts: ``selection_info.json`` written by
``data_manager/mu_energy_ds_builder``.  This is MC, not experimental data; the
experiment has no energy truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET = ROOT / "data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M_partsplit"
OUT = HERE / "figures/energy_spectrum_train_vs_real"

PAPER = "#FAFAF8"
INK = "#263238"
MUTED = "#6B777D"
GRID = "#D9DEE1"
ENERGY = "#2F6DB3"
SIGMA = "#E07A2E"


def load_spectrum() -> dict:
    info = json.loads((DATASET / "selection_info.json").read_text())
    report = info["per_bin_report"]
    edges = np.array([r["lo"] for r in report] + [report[-1]["hi"]], dtype=float)
    width = np.diff(edges)
    if not np.allclose(width, width[0], rtol=0, atol=1e-12):
        raise ValueError("flat-selection bins are not equal width in log10 E")

    n_candidates = np.array([r["n"] for r in report], dtype=float)
    n_selected = np.array([r["selected"] for r in report], dtype=float)
    weight = np.array([r["weight"] for r in report], dtype=float)

    if int(n_candidates.sum()) != int(info["n_candidates"]):
        raise ValueError("per-bin candidate counts do not match n_candidates")
    if int(n_selected.sum()) != int(info["n_selected"]):
        raise ValueError("per-bin selected counts do not match n_selected")
    if not np.all(weight > 0):
        raise ValueError("empty bin in flat-selection report")

    # Weighted counts are the effective training mass; unweighted row counts are
    # the sampled dataset, not the distribution the weighted loss sees.
    training_counts = n_selected * weight

    # Cross-check the report against the stored arrays.  A handful of events can
    # fall on a bin edge because energy.npy is float32; the report is canonical.
    energy = np.load(DATASET / "energy.npy").astype(np.float64)
    flat_weights = np.load(DATASET / "flat_weights.npy").astype(np.float64)
    log_e = np.log10(energy)
    hist_selected, _ = np.histogram(log_e, bins=edges)
    hist_weighted, _ = np.histogram(log_e, bins=edges, weights=flat_weights)
    max_selected_delta = int(np.max(np.abs(hist_selected - n_selected)))
    max_weighted_delta = float(
        np.max(np.abs(hist_weighted - training_counts)) / max(training_counts.max(), 1)
    )

    real_density = n_candidates / n_candidates.sum() / width
    training_density = training_counts / training_counts.sum() / width

    return {
        "edges": edges,
        "width": width,
        "real_density": real_density,
        "training_density": training_density,
        "n_candidates": int(n_candidates.sum()),
        "n_selected": int(n_selected.sum()),
        "sum_weights": float(training_counts.sum()),
        "ess_fraction": float(info["ess_fraction"]),
        "weight_range": (float(weight.min()), float(weight.max())),
        "max_selected_delta": max_selected_delta,
        "max_weighted_delta": max_weighted_delta,
        "n_bins": len(report),
    }


def draw() -> None:
    d = load_spectrum()
    edges = d["edges"]
    width = d["width"][0]
    fig = plt.figure(figsize=(10, 11))#, facecolor=PAPER)
    ax = fig.add_axes([0.115, 0.13, 0.855, 0.68])
    #ax.set_facecolor(PAPER)

    ax.stairs(
        d["real_density"], edges, baseline=None, color=ENERGY, lw=2.2, zorder=3,
        label=f"MC before flatten — {d['n_candidates']/1e6:.2f}M candidates",
    )
    ax.stairs(
        d["training_density"], edges, baseline=None, color=SIGMA, lw=2.6, zorder=4,
        label=(
            "training, flat selection — "
            f"{d['n_selected']/1e6:.2f}M events"#, Σw = {d['sum_weights']/1e6:.2f}M"
        ),
    )

    ax.set_xlim(edges[0] - 0.12, edges[-1] + 0.12)
    ax.set_ylim(0.0, 0.42)

    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xticklabels(["1 GeV", "100 GeV", "10 TeV", "1 PeV", "100 PeV"])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel(r"$log_{10}(E_{entry} / GeV)$", fontsize=20)
    ax.set_ylabel("Density", fontsize=20)

    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.9)
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=MUTED, labelsize=13.5, length=5, width=1.0)

    ax.legend(
        loc="upper left", frameon=True, 
        #facecolor=PAPER, 
        edgecolor=GRID,
        fontsize=16, handlelength=2.2,
    )

    # ax.annotate(
    #     "Natural MC: peak at 10–100 TeV",
    #     xy=(4.35, d["real_density"][14]),
    #     xytext=(4.70, 0.385),
    #     color=ENERGY,
    #     fontsize=12.0,
    #     ha="left",
    #     va="center",
    #     arrowprops=dict(arrowstyle="-", color=ENERGY, lw=1.0),
    # )

    fig.text(0.075, 0.94, "Training vs MC energy spectrum",
             fontsize=25.0, fontweight="bold", color=INK, ha="left")
    # fig.text(
    #     0.077, 0.885,
    #     "Train parts, h8s3 + single-track, MC only. "
    #     f"25 bins x {width:.3f} dex; both curves normalized to unit area.",
    #     fontsize=12.0, color=MUTED, ha="left",
    # )
    # fig.text(
    #     0.077, 0.035,
    #     f"ESS training = {100*d['ess_fraction']:.1f}%; flat weights "
    #     f"{d['weight_range'][0]:.0f}–{d['weight_range'][1]:.0f}.  "
    #     "Source: train1M_partsplit / selection_info.json.",
    #     fontsize=10.0, color=MUTED, ha="left",
    # )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".svg"), dpi=220)#, facecolor=PAPER)
    fig.savefig(OUT.with_suffix(".png"), dpi=220)#, facecolor=PAPER)
    plt.close(fig)

    print(
        f"bins={d['n_bins']}  width={width:.4f} dex  "
        f"candidates={d['n_candidates']:,}  selected={d['n_selected']:,}  "
        f"sum_w={d['sum_weights']:.0f}  ESS={100*d['ess_fraction']:.1f}%  "
        f"report/array max deltas: selected={d['max_selected_delta']} "
        f"weighted={100*d['max_weighted_delta']:.4f}%"
    )
    print("wrote", OUT.with_suffix(".png"), "and", OUT.with_suffix(".svg"))


if __name__ == "__main__":
    draw()
