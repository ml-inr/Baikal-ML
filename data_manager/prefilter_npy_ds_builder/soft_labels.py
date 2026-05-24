"""Soft label computation for the prefilter classification task."""

from typing import Optional

import numpy as np


def compute_soft_label(
    n_signal_hits: np.ndarray,
    is_neutrino: bool,
    mode: str = "linear",
    h_saturate: int = 15,
    steepness: Optional[float] = None,
) -> np.ndarray:
    """Vectorised soft-label computation for an array of events.

    Args:
        n_signal_hits: (n_events,) int array of signal hit counts.
        is_neutrino: Whether these events are neutrino (True) or muon (False).
        mode: "linear" or "sigmoid".
        h_saturate: Saturation threshold.
        steepness: Sigmoid steepness (default: 30 / h_saturate).

    Returns:
        (n_events,) float32 array of labels in [0, 1].
    """
    out = np.zeros(len(n_signal_hits), dtype=np.float32)
    if not is_neutrino:
        return out

    if mode == "linear":
        if h_saturate <= 0:
            out[n_signal_hits > 0] = 1.0
        else:
            out[:] = np.minimum(1.0, n_signal_hits / h_saturate)
    elif mode == "sigmoid":
        k = steepness if steepness is not None else 30.0 / max(h_saturate, 1)
        x = k * (n_signal_hits - h_saturate / 2.0)
        out[:] = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    else:
        raise ValueError(f"Unknown soft label mode: {mode}")
    return out
