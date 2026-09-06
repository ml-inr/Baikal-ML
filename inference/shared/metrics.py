"""Thin analysis metric wrappers for use in analysis notebooks and scripts."""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC for binary classification."""
    return float(roc_auc_score(labels, scores))


def at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Precision, recall, accuracy, background rejection at a fixed score threshold."""
    preds = (scores >= threshold).astype(int)
    labels_int = (labels > 0.5).astype(int)
    tp = int(((preds == 1) & (labels_int == 1)).sum())
    tn = int(((preds == 0) & (labels_int == 0)).sum())
    fp = int(((preds == 1) & (labels_int == 0)).sum())
    fn = int(((preds == 0) & (labels_int == 1)).sum())
    n_bg = tn + fp
    return {
        "threshold":          threshold,
        "precision":          tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall":             tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "accuracy":           (tp + tn) / len(labels) if len(labels) > 0 else 0.0,
        "bg_rejection":       n_bg / fp if fp > 0 else float("inf"),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def efficiency_rejection_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signal efficiency vs background rejection sweep.

    Returns:
        (thresholds, sig_efficiency, bg_rejection)
        bg_rejection = n_bg_total / n_bg_passing — how many times background is reduced.
    """
    labels_bin = (labels > 0.5).astype(bool)
    n_sig = labels_bin.sum()
    n_bg  = (~labels_bin).sum()

    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    sig_eff = np.array([(scores[labels_bin] >= t).mean() for t in thresholds], dtype=np.float64)
    bg_pass = np.array([(scores[~labels_bin] >= t).sum() for t in thresholds], dtype=np.float64)
    bg_rej  = np.where(bg_pass > 0, n_bg / bg_pass, np.inf)

    return thresholds, sig_eff, bg_rej
