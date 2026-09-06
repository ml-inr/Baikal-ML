"""Analysis metrics for nu-classifier predictions.

Thin wrappers that accept DataFrames from load_preds().
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from inference_v2.shared.metrics import (
    auc_score as _auc,
    at_threshold as _at_thr,
    efficiency_rejection_curve as _eff_rej,
)


def auc(df: pd.DataFrame, label_col: str = "label") -> float:
    """AUC from a predictions DataFrame that includes a label column."""
    return _auc(df["score"].values, df[label_col].values)


def at_threshold(df: pd.DataFrame, threshold: float, label_col: str = "label") -> Dict[str, Any]:
    """Precision, recall, bg_rejection at a fixed threshold."""
    return _at_thr(df["score"].values, df[label_col].values, threshold)


def efficiency_rejection_curve(
    df: pd.DataFrame,
    label_col: str = "label",
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signal efficiency vs background rejection curve."""
    return _eff_rej(df["score"].values, df[label_col].values, n_points=n_points)
