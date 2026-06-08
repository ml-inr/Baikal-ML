"""Analysis metrics for prefilter predictions."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from inference_v2.shared.metrics import (
    auc_score as _auc,
    at_threshold as _at_thr,
    efficiency_rejection_curve as _eff_rej,
)


def auc(df: pd.DataFrame, label_col: str = "label") -> float:
    return _auc(df["score"].values, df[label_col].values)


def at_threshold(df: pd.DataFrame, threshold: float, label_col: str = "label") -> Dict[str, Any]:
    return _at_thr(df["score"].values, df[label_col].values, threshold)


def efficiency_rejection_curve(
    df: pd.DataFrame,
    label_col: str = "label",
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _eff_rej(df["score"].values, df[label_col].values, n_points=n_points)
