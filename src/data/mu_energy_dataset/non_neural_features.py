"""Event-level handcrafted features for the non-neural energy baseline.

This module is deliberately separate from the hit-sequence collate path used by the
neural energy model.  It reduces every event to one row of physically motivated
aggregates (counts, charge, time moments and geometry) so that Ridge and
HistGradientBoosting can be compared with the neural baseline on exactly the same
selected hits.

Charge is clipped at ``amp_clip`` (default 100 p.e.) before aggregation, matching
the neural model's input preprocessing.  The raw charge is used only to count how
many hits saturate the clip; the baseline therefore does not see information that
the neural model is denied.

The expected input is the hit-level layout of ``MuEnergyNpyDataset``:
``features`` is ``(n_hits, 5)`` with ``[Q, t, x, y, z]`` and ``offsets`` maps
events to rows.  The returned matrix has one row per event, in the same order as
the offsets.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

AMP_CLIP_DEFAULT = 100.0

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "counts": (
        "n_hits",
        "n_strings",
        "log1p_n_hits",
        "log1p_n_strings",
        "hits_per_string",
    ),
    "charge": (
        "sumQ",
        "meanQ",
        "stdQ",
        "log10_sumQ",
        "log10_meanQ",
        "log10_stdQ",
        "maxQ",
        "frac_maxQ",
        "frac_Q_gt_1",
        "frac_Q_gt_10",
        "frac_Q_gt_50",
        "n_clip",
        "frac_clip",
        "log1p_n_clip",
    ),
    "time": (
        "mean_t",
        "std_t",
        "range_t",
        "qweighted_mean_t",
        "qweighted_std_t",
        "corr_Q_t",
    ),
    "geometry": (
        "mean_x",
        "std_x",
        "range_x",
        "mean_y",
        "std_y",
        "range_y",
        "mean_z",
        "std_z",
        "range_z",
        "lambda1",
        "lambda2",
        "lambda3",
        "linearity",
        "sqrt_lambda1",
        "transverse",
        "range_xy",
        "range_3d",
        "mean_r",
        "std_r",
        "frac_r_lt20",
        "frac_r_20_50",
        "frac_r_gt50",
        "frac_z_pos",
        "hit_density",
        "charge_density",
    ),
}

FEATURE_NAMES: tuple[str, ...] = tuple(
    name for names in FEATURE_GROUPS.values() for name in names
)


def feature_names() -> list[str]:
    """Return feature names in the column order of the feature matrix."""
    return list(FEATURE_NAMES)


def feature_groups() -> dict[str, list[str]]:
    """Return a copy of the group -> feature names mapping."""
    return {group: list(names) for group, names in FEATURE_GROUPS.items()}


def _reduce_sum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(np.asarray(values, dtype=np.float64), starts)


def _reduce_sum_sq(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.add.reduceat(arr * arr, starts)


def _reduce_min(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.minimum.reduceat(np.asarray(values, dtype=np.float64), starts)


def _reduce_max(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.maximum.reduceat(np.asarray(values, dtype=np.float64), starts)


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    out = np.full_like(np.asarray(numerator, dtype=np.float64), fill)
    np.divide(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
        out=out,
        where=np.asarray(denominator) != 0,
    )
    return out


def compute_non_neural_features(
    features: np.ndarray,
    offsets: np.ndarray,
    n_sig_strings: np.ndarray,
    *,
    amp_clip: float = AMP_CLIP_DEFAULT,
    n_sig_hits: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Aggregate hit sequences into one row per event.

    Args:
        features: ``(n_hits, 5)`` array with ``[Q, t, x, y, z]``; raw units.
        offsets: ``(n_events + 1,)`` event boundaries into ``features``.
        n_sig_strings: ``(n_events,)`` number of selected strings.
        amp_clip: raw charge clip in p.e.; must be positive.
        n_sig_hits: optional expected hit counts for a consistency check.

    Returns:
        ``(n_events, n_features)`` float32 matrix in ``FEATURE_NAMES`` order.
    """
    if features.ndim != 2 or features.shape[1] != 5:
        raise ValueError(f"features must be (n_hits, 5), got {features.shape}")
    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("offsets must be a 1-D array with at least two entries")
    if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets must start at 0 and be strictly increasing")
    if offsets[-1] != len(features):
        raise ValueError(
            f"offsets[-1]={offsets[-1]} does not match {len(features)} hits"
        )
    if amp_clip <= 0.0:
        raise ValueError(f"amp_clip must be positive, got {amp_clip}")

    n_events = len(offsets) - 1
    n_sig_strings = np.asarray(n_sig_strings, dtype=np.float64)
    if len(n_sig_strings) != n_events:
        raise ValueError(
            f"n_sig_strings has {len(n_sig_strings)} rows, expected {n_events}"
        )
    lengths = np.diff(offsets).astype(np.float64)
    if n_sig_hits is not None:
        given = np.asarray(n_sig_hits, dtype=np.int64)
        if not np.array_equal(given, lengths.astype(np.int64)):
            raise ValueError("n_sig_hits does not match offsets lengths")

    starts = offsets[:-1]
    nh = lengths
    ns = n_sig_strings

    q_raw = np.asarray(features[:, 0], dtype=np.float64)
    q = np.minimum(q_raw, float(amp_clip))
    sum_q = _reduce_sum(q, starts)
    sum_q2 = _reduce_sum_sq(q, starts)
    max_q = _reduce_max(q, starts)
    mean_q = sum_q / nh
    var_q = np.maximum(sum_q2 / nh - mean_q * mean_q, 0.0)
    std_q = np.sqrt(var_q)

    n_clip = _reduce_sum(q_raw >= float(amp_clip), starts)
    frac_max_q = _safe_divide(max_q, sum_q)
    frac_q_gt_1 = _safe_divide(_reduce_sum(q > 1.0, starts), nh)
    frac_q_gt_10 = _safe_divide(_reduce_sum(q > 10.0, starts), nh)
    frac_q_gt_50 = _safe_divide(_reduce_sum(q > 50.0, starts), nh)
    frac_clip = _safe_divide(n_clip, nh)

    t = np.asarray(features[:, 1], dtype=np.float64)
    sum_t = _reduce_sum(t, starts)
    sum_t2 = _reduce_sum_sq(t, starts)
    mean_t = sum_t / nh
    var_t = np.maximum(sum_t2 / nh - mean_t * mean_t, 0.0)
    std_t = np.sqrt(var_t)
    min_t = _reduce_min(t, starts)
    max_t = _reduce_max(t, starts)
    sum_qt = _reduce_sum(q * t, starts)
    sum_qt2 = _reduce_sum(q * t * t, starts)
    qmean_t = _safe_divide(sum_qt, sum_q)
    qvar_t = np.maximum(_safe_divide(sum_qt2, sum_q) - qmean_t * qmean_t, 0.0)
    qstd_t = np.sqrt(qvar_t)
    cov_qt = _safe_divide(sum_qt, nh) - mean_q * mean_t
    corr_qt = _safe_divide(
        cov_qt, np.sqrt(np.maximum(var_q * var_t, 0.0))
    )

    x = np.asarray(features[:, 2], dtype=np.float64)
    y = np.asarray(features[:, 3], dtype=np.float64)
    z = np.asarray(features[:, 4], dtype=np.float64)

    def moments(values: np.ndarray) -> tuple[np.ndarray, ...]:
        s1 = _reduce_sum(values, starts)
        s2 = _reduce_sum_sq(values, starts)
        mean = s1 / nh
        var = np.maximum(s2 / nh - mean * mean, 0.0)
        return mean, np.sqrt(var), _reduce_min(values, starts), _reduce_max(values, starts)

    mean_x, std_x, min_x, max_x = moments(x)
    mean_y, std_y, min_y, max_y = moments(y)
    mean_z, std_z, min_z, max_z = moments(z)
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z

    cov = np.empty((n_events, 3, 3), dtype=np.float64)
    cov[:, 0, 0] = _reduce_sum_sq(x, starts) / nh - mean_x * mean_x
    cov[:, 1, 1] = _reduce_sum_sq(y, starts) / nh - mean_y * mean_y
    cov[:, 2, 2] = _reduce_sum_sq(z, starts) / nh - mean_z * mean_z
    cov_xy = _reduce_sum(x * y, starts) / nh - mean_x * mean_y
    cov_xz = _reduce_sum(x * z, starts) / nh - mean_x * mean_z
    cov_yz = _reduce_sum(y * z, starts) / nh - mean_y * mean_z
    cov[:, 0, 1] = cov[:, 1, 0] = cov_xy
    cov[:, 0, 2] = cov[:, 2, 0] = cov_xz
    cov[:, 1, 2] = cov[:, 2, 1] = cov_yz
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    lambda3 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda1 = eigenvalues[:, 2]
    trace = lambda1 + lambda2 + lambda3
    linearity = _safe_divide(lambda1, trace)
    sqrt_lambda1 = np.sqrt(lambda1)
    transverse = np.sqrt(lambda2 + lambda3)

    radius = np.sqrt(x * x + y * y)
    mean_r = _reduce_sum(radius, starts) / nh
    std_r = np.sqrt(
        np.maximum(_reduce_sum_sq(radius, starts) / nh - mean_r * mean_r, 0.0)
    )
    frac_r_lt20 = _safe_divide(_reduce_sum(radius < 20.0, starts), nh)
    frac_r_20_50 = _safe_divide(
        _reduce_sum((radius >= 20.0) & (radius < 50.0), starts), nh
    )
    frac_r_gt50 = _safe_divide(_reduce_sum(radius >= 50.0, starts), nh)
    frac_z_pos = _safe_divide(_reduce_sum(z > 0.0, starts), nh)
    length_proxy = np.maximum(sqrt_lambda1, 1e-3)
    hit_density = nh / length_proxy
    charge_density = sum_q / length_proxy

    values: dict[str, np.ndarray] = {
        "n_hits": nh,
        "n_strings": ns,
        "log1p_n_hits": np.log1p(nh),
        "log1p_n_strings": np.log1p(ns),
        "hits_per_string": _safe_divide(nh, np.maximum(ns, 1.0)),
        "sumQ": sum_q,
        "meanQ": mean_q,
        "stdQ": std_q,
        "log10_sumQ": np.log10(np.maximum(sum_q, 1e-9)),
        "log10_meanQ": np.log10(np.maximum(mean_q, 1e-9)),
        "log10_stdQ": np.log10(np.maximum(std_q, 1e-9)),
        "maxQ": max_q,
        "frac_maxQ": frac_max_q,
        "frac_Q_gt_1": frac_q_gt_1,
        "frac_Q_gt_10": frac_q_gt_10,
        "frac_Q_gt_50": frac_q_gt_50,
        "n_clip": n_clip,
        "frac_clip": frac_clip,
        "log1p_n_clip": np.log1p(n_clip),
        "mean_t": mean_t,
        "std_t": std_t,
        "range_t": max_t - min_t,
        "qweighted_mean_t": qmean_t,
        "qweighted_std_t": qstd_t,
        "corr_Q_t": corr_qt,
        "mean_x": mean_x,
        "std_x": std_x,
        "range_x": range_x,
        "mean_y": mean_y,
        "std_y": std_y,
        "range_y": range_y,
        "mean_z": mean_z,
        "std_z": std_z,
        "range_z": range_z,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda3": lambda3,
        "linearity": linearity,
        "sqrt_lambda1": sqrt_lambda1,
        "transverse": transverse,
        "range_xy": np.sqrt(range_x * range_x + range_y * range_y),
        "range_3d": np.sqrt(
            range_x * range_x + range_y * range_y + range_z * range_z
        ),
        "mean_r": mean_r,
        "std_r": std_r,
        "frac_r_lt20": frac_r_lt20,
        "frac_r_20_50": frac_r_20_50,
        "frac_r_gt50": frac_r_gt50,
        "frac_z_pos": frac_z_pos,
        "hit_density": hit_density,
        "charge_density": charge_density,
    }
    missing = [name for name in FEATURE_NAMES if name not in values]
    if missing:
        raise RuntimeError(f"missing feature columns: {missing}")
    matrix = np.column_stack([values[name] for name in FEATURE_NAMES]).astype(
        np.float32
    )
    if not np.isfinite(matrix).all():
        bad = np.where(~np.isfinite(matrix))
        raise ValueError(
            "non-finite feature values at rows/columns "
            f"{bad[0][:5].tolist()} / {bad[1][:5].tolist()}"
        )
    return matrix
