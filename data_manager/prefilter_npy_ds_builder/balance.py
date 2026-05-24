"""Class balancing for the prefilter dataset.

Operates on per-event metadata arrays — no hit data is touched.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Must match encoding in io.py / build.py
PARTICLE_ENCODE = {
    "muatm_2020": 0,
    "nuatm_2020": 1,
    "nue2_2020": 2,
}


def balance_classes(
    labels: np.ndarray,
    particle_types: np.ndarray,
    n_signal_hits: np.ndarray,
    rng: np.random.RandomState,
    mu_split_hits_threshold: Optional[int],
) -> np.ndarray:
    """Return indices of selected events after class balancing.

    Signal:     neutrino events with label >= 0.5
    Background: muon events + neutrino events with label < 0.5

    When ``mu_split_hits_threshold`` is not None, muons are split into
    "high" (>= threshold signal hits) and "low" (< threshold) background groups.
    Pass None to skip the split — appropriate for hard labeling where the
    threshold has no meaning.

    Constraints:
        1. signal_nuatm == signal_nue2
        2. total_background ≈ total_signal

    Args:
        labels: (n_events,) float32 soft labels.
        particle_types: (n_events,) int8 encoded particle type.
        n_signal_hits: (n_events,) int32 signal hit counts.
        rng: Random state for reproducible selection.
        mu_split_hits_threshold: Muons with >= this many signal hits go into
            the "high" background group. None = no split, all muons are "high".

    Returns:
        1-D int64 array of selected event indices.
    """
    is_signal = labels >= 0.5
    is_muatm = particle_types == PARTICLE_ENCODE["muatm_2020"]
    is_nuatm = particle_types == PARTICLE_ENCODE["nuatm_2020"]
    is_nue2 = particle_types == PARTICLE_ENCODE["nue2_2020"]

    sig_nuatm = np.where(is_signal & is_nuatm)[0]
    sig_nue2 = np.where(is_signal & is_nue2)[0]
    bg_nuatm = np.where(~is_signal & is_nuatm)[0]
    bg_nue2 = np.where(~is_signal & is_nue2)[0]

    if mu_split_hits_threshold is None:
        bg_muon_high = np.where(~is_signal & is_muatm)[0]
        bg_muon_low = np.array([], dtype=np.int64)
        split_desc = "no split"
    else:
        bg_muon_high = np.where(
            ~is_signal & is_muatm & (n_signal_hits >= mu_split_hits_threshold)
        )[0]
        bg_muon_low = np.where(
            ~is_signal & is_muatm & (n_signal_hits < mu_split_hits_threshold)
        )[0]
        split_desc = f">={mu_split_hits_threshold} sig hits"

    logger.info(
        f"Before balancing: "
        f"sig_nuatm={len(sig_nuatm):,}, sig_nue2={len(sig_nue2):,}, "
        f"bg_muon_high={len(bg_muon_high):,} ({split_desc}), "
        f"bg_muon_low={len(bg_muon_low):,}, "
        f"bg_nuatm={len(bg_nuatm):,}, bg_nue2={len(bg_nue2):,}"
    )

    n_sig_per = min(len(sig_nuatm), len(sig_nue2))
    if n_sig_per == 0:
        logger.warning("No signal events of one type — cannot balance")
        return np.arange(len(labels))

    # Background quotas
    n_bg_nu = n_sig_per // 2
    n_bg_nuatm = min(n_bg_nu // 2, len(bg_nuatm))
    n_bg_nue2 = min(n_bg_nu - n_bg_nuatm, len(bg_nue2))

    if mu_split_hits_threshold is None:
        # Hard mode: signal = nuatm + nue2 (2*n_sig_per), background = muons only.
        # Need 2*n_sig_per muons for 1:1 balance.
        n_muon_target = 2 * n_sig_per
        n_muon_high = min(n_muon_target, len(bg_muon_high))
        n_muon_low = 0

        limiting = n_muon_high / n_muon_target if n_muon_target > 0 else 1.0
        if limiting < 1.0:
            logger.warning(f"Insufficient muons, scaling by {limiting:.3f}")
            n_sig_per = int(n_sig_per * limiting)
            n_muon_high = 2 * n_sig_per

        n_bg_nuatm = 0
        n_bg_nue2 = 0
    else:
        n_muon_high = min(n_sig_per, len(bg_muon_high))
        n_muon_low = min(n_sig_per // 2, len(bg_muon_low))

        targets = [
            (n_muon_high, n_sig_per),
            (n_muon_low, n_sig_per // 2),
            (n_bg_nuatm, n_bg_nu // 2),
            (n_bg_nue2, n_bg_nu - n_bg_nu // 2),
        ]
        limiting = min(
            (actual / target if target > 0 else 1.0)
            for actual, target in targets
        )
        if limiting < 1.0:
            logger.warning(f"Insufficient bg, scaling by {limiting:.3f}")
            n_sig_per = int(n_sig_per * limiting)
            n_muon_high = n_sig_per
            n_muon_low = n_sig_per // 2
            n_bg_nu = n_sig_per // 2
            n_bg_nuatm = n_bg_nu // 2
            n_bg_nue2 = n_bg_nu - n_bg_nuatm

    for arr in [sig_nuatm, sig_nue2, bg_muon_high, bg_muon_low,
                bg_nuatm, bg_nue2]:
        rng.shuffle(arr)

    selected = np.concatenate([
        sig_nuatm[:n_sig_per],
        sig_nue2[:n_sig_per],
        bg_muon_high[:n_muon_high],
        bg_muon_low[:n_muon_low],
        bg_nuatm[:n_bg_nuatm],
        bg_nue2[:n_bg_nue2],
    ])

    total_sig = 2 * n_sig_per
    total_bg = n_muon_high + n_muon_low + n_bg_nuatm + n_bg_nue2
    logger.info(
        f"After balancing: signal={total_sig:,} "
        f"(nuatm={n_sig_per:,}, nue2={n_sig_per:,}), "
        f"background={total_bg:,} "
        f"(muon_high={n_muon_high:,}, muon_low={n_muon_low:,}, "
        f"nuatm={n_bg_nuatm:,}, nue2={n_bg_nue2:,}), "
        f"total={len(selected):,}"
    )
    return selected
