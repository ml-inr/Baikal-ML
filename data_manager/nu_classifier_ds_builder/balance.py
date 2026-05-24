"""Class balancing for the nu-classifier dataset.

Target distribution (all counts equal after balancing):
    signal:     nuatm_2020  (n each)
                nue2_2020   (n each)   → 2n total signal
    background: muatm_2020             → 2n total background

Constraint: n_nuatm == n_nue2 == n_muatm_half,  total_signal == total_background.
n is limited by the smallest of the three available pools.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

PARTICLE_ENCODE = {
    "muatm_2020": 0,
    "nuatm_2020": 1,
    "nue2_2020":  2,
}


def balance_classes(
    particle_types: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return indices of selected events after class balancing.

    Selects n events from each of muatm, nuatm, nue2 where
    n = min(available muatm, available nuatm, available nue2).
    Final dataset: 2n signal (n nuatm + n nue2) vs 2n background (2n muatm).

    Args:
        particle_types: (N,) int8 encoded particle type.
        rng: Random state for reproducible selection.

    Returns:
        1-D int64 array of selected event indices (unshuffled — caller shuffles).
    """
    idx_muatm = np.where(particle_types == PARTICLE_ENCODE["muatm_2020"])[0]
    idx_nuatm = np.where(particle_types == PARTICLE_ENCODE["nuatm_2020"])[0]
    idx_nue2  = np.where(particle_types == PARTICLE_ENCODE["nue2_2020"])[0]

    logger.info(
        f"Before balancing: "
        f"muatm={len(idx_muatm):,}, nuatm={len(idx_nuatm):,}, nue2={len(idx_nue2):,}"
    )

    n = min(len(idx_muatm) // 2, len(idx_nuatm), len(idx_nue2))
    if n == 0:
        logger.warning("One class is empty after cuts — cannot balance.")
        return np.concatenate([idx_muatm, idx_nuatm, idx_nue2])

    for arr in [idx_muatm, idx_nuatm, idx_nue2]:
        rng.shuffle(arr)

    selected = np.concatenate([
        idx_muatm[:2 * n],   # 2n muatm = background
        idx_nuatm[:n],       # n nuatm  = signal
        idx_nue2[:n],        # n nue2   = signal
    ])

    logger.info(
        f"After balancing: "
        f"muatm={2*n:,}, nuatm={n:,}, nue2={n:,} — total={4*n:,}"
    )
    return selected
