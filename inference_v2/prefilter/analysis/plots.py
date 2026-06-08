"""Standard plots for prefilter analysis."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sklearn_auc

from inference_v2.shared.metrics import efficiency_rejection_curve


def plot_score_dist(
    dfs: List[pd.DataFrame],
    labels: List[str],
    signal_col: Optional[str] = None,
    n_bins: int = 80,
    log_y: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, n_bins + 1)

    for df, lbl in zip(dfs, labels):
        if signal_col is not None and signal_col in df.columns:
            sig = df[df[signal_col] == 1]["score"].values
            bg  = df[df[signal_col] == 0]["score"].values
            ax.hist(sig, bins=bins, histtype="step", label=f"{lbl} signal", density=True)
            ax.hist(bg,  bins=bins, histtype="step", label=f"{lbl} bg",     density=True, linestyle="--")
        else:
            ax.hist(df["score"].values, bins=bins, histtype="step", label=lbl, density=True)

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Prefilter score distribution")
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
    ax.set_title("Prefilter: efficiency vs rejection")
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
    fig, ax = plt.subplots(figsize=(7, 6))

    for df, lbl in zip(dfs, labels):
        if label_col not in df.columns:
            continue
        y_true = (df[label_col].values > 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_true, df["score"].values)
        roc_auc = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{lbl} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Prefilter ROC curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
