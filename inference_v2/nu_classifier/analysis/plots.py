"""Standard plots for nu-classifier analysis.

All functions return (fig, ax) and optionally save to file if save_path is given.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from inference_v2.shared.metrics import efficiency_rejection_curve


def plot_score_dist(
    dfs: List[pd.DataFrame],
    labels: List[str],
    signal_col: Optional[str] = None,
    n_bins: int = 80,
    log_y: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Score distributions for one or more prediction sets.

    Args:
        dfs: List of DataFrames from load_preds().
        labels: Legend labels per DataFrame.
        signal_col: If set, split each df into signal/background by this column.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, n_bins + 1)

    for df, lbl in zip(dfs, labels):
        if signal_col is not None and signal_col in df.columns:
            sig = df[df[signal_col] == 1]["score"].values
            bg  = df[df[signal_col] == 0]["score"].values
            ax.hist(sig, bins=bins, histtype="step", label=f"{lbl} signal",  density=True)
            ax.hist(bg,  bins=bins, histtype="step", label=f"{lbl} bg",      density=True, linestyle="--")
        else:
            ax.hist(df["score"].values, bins=bins, histtype="step", label=lbl, density=True)

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Nu-classifier score distribution")
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_efficiency_rejection(
    dfs: List[pd.DataFrame],
    labels: List[str],
    label_col: str = "label",
    n_points: int = 500,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Signal efficiency vs background rejection curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for df, lbl in zip(dfs, labels):
        if label_col not in df.columns:
            continue
        _, sig_eff, bg_rej = efficiency_rejection_curve(
            df["score"].values, df[label_col].values, n_points=n_points
        )
        finite = np.isfinite(bg_rej)
        ax.semilogy(sig_eff[finite], bg_rej[finite], label=lbl)

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_title("Nu-classifier: efficiency vs rejection")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_roc(
    dfs: List[pd.DataFrame],
    labels: List[str],
    label_col: str = "label",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """ROC curve (FPR vs TPR)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for df, lbl in zip(dfs, labels):
        if label_col not in df.columns:
            continue
        y_true = (df[label_col].values > 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_true, df["score"].values)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{lbl} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


_CLASS_COLORS: dict = {
    "muatm_2020":        "#1f77b4",
    "nuatm_2020":        "#ff7f0e",
    "nue2_2020":         "#d62728",
    "nuatm_conv_2020":   "#2ca02c",
    "nuatm_prompt_2020": "#9467bd",
    "exp":               "#8c564b",
    "exp_reco":          "#e377c2",
}

_CUT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def plot_score_dist_by_class(
    dfs_per_cut: Dict[str, pd.DataFrame],
    class_colors: Optional[Dict[str, str]] = None,
    n_bins: int = 80,
    log_y: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """2×2 subplots — one panel per cut, histograms per data_class.

    Args:
        dfs_per_cut: Ordered dict {cut_label: DataFrame from load_moe_preds()}.
    """
    colors = {**_CLASS_COLORS, **(class_colors or {})}
    cut_labels = list(dfs_per_cut.keys())
    bins = np.linspace(0, 1, n_bins + 1)

    n_cuts = len(cut_labels)
    ncols = 2
    nrows = (n_cuts + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharex=True)
    axes_flat = np.array(axes).flatten()

    all_classes: list = []
    for df in dfs_per_cut.values():
        for cls in df["data_class"].unique():
            if cls not in all_classes:
                all_classes.append(cls)

    for i, (label, df) in enumerate(dfs_per_cut.items()):
        ax = axes_flat[i]
        for cls in all_classes:
            mask = df["data_class"] == cls
            if not mask.any():
                continue
            ax.hist(
                df.loc[mask, "score"].values,
                bins=bins, histtype="step", density=True,
                color=colors.get(cls, "#aaaaaa"),
                label=f"{cls} ({mask.sum():,})",
                linewidth=1.4,
            )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title or "Nu-classifier score distributions by class", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes_flat


def plot_suppression_vs_efficiency(
    dfs_per_cut: Dict[str, pd.DataFrame],
    signal_classes: List[str],
    bg_classes: List[str],
    n_points: int = 500,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """FPR⁻¹ (suppression factor) vs TPR (signal efficiency) — one curve per cut.

    Signal = rows with data_class in signal_classes.
    Background = rows with data_class in bg_classes.
    Quality cut is baked into each df; this function sweeps the score threshold.
    """
    from inference_v2.shared.metrics import efficiency_rejection_curve

    fig, ax = plt.subplots(figsize=(8, 6))

    for (label, df), color in zip(dfs_per_cut.items(), _CUT_COLORS):
        sig_mask = df["data_class"].isin(signal_classes)
        bg_mask  = df["data_class"].isin(bg_classes)
        if not sig_mask.any() or not bg_mask.any():
            continue

        scores = np.concatenate([df.loc[sig_mask, "score"].values,
                                  df.loc[bg_mask,  "score"].values])
        labels = np.concatenate([np.ones(sig_mask.sum()), np.zeros(bg_mask.sum())])

        _, sig_eff, bg_rej = efficiency_rejection_curve(scores, labels, n_points=n_points)
        finite = np.isfinite(bg_rej)
        ax.semilogy(sig_eff[finite], bg_rej[finite],
                    label=f"{label}  (sig={sig_mask.sum():,}, bg={bg_mask.sum():,})",
                    color=color, linewidth=1.8)

    ax.set_xlabel("Neutrino selection efficiency (TPR)")
    ax.set_ylabel("Muon suppression factor (1/FPR)")
    ax.set_title(title or "Suppression vs efficiency")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
