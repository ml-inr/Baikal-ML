"""HDF5 reading and mmap writing for the muon-energy dataset builder.

This is a deliberate, self-contained copy of the sig-noise / hit-writing part of
``data_manager/nu_classifier_ds_builder/io.py`` (copied 2026-10-21).  The two datasets are
kept independent on purpose: changing the classifier builder must not silently change the
energy dataset.  If a bug is fixed here, check whether the classifier copy needs the same
fix.

Two-pass design
---------------
Pass 1 (per part, sequential):
    Load ``raw/data`` + ``raw/ev_starts`` + ``raw/channels`` for one part.
    Run the sig-noise model -> per-hit probabilities.
    Compute per-event ``n_sig_hits`` and ``n_sig_strings`` (prob > threshold).
    Keep only per-event scalars in RAM; hit features are discarded after each part.

Pass 2 (per selected part only):
    Re-run the sig-noise model on parts that contain at least one selected event.
    Write prob-filtered hit features (prob > threshold) into pre-allocated memory maps.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import h5py
import numpy as np
import torch

from trained_models.sig_noise.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
    predict_flat,
)

logger = logging.getLogger(__name__)

STRING_DIVISOR = 36


class EventLocation(NamedTuple):
    """Where to find an event's hits in the source HDF5 file."""

    particle_type: str       # e.g. "muatm_2020"
    part_key: str            # e.g. "part_0"
    hit_start: int           # within the part
    hit_end: int
    event_idx_in_part: int   # 0-based local index within the h5 part


def _run_model(
    data_raw: np.ndarray,
    ev_starts: np.ndarray,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run the sig-noise model with built-in normalization; return per-hit probs."""
    return predict_flat(
        model=model,
        data=data_raw,
        ev_starts=ev_starts,
        batch_size=batch_size,
        device=device,
        normalize=True,
        max_gpu_hits=2_000_000,
    )


def _count_sig_hits_strings(
    mask: np.ndarray,
    channels: np.ndarray,
    hit_starts: np.ndarray,
    n_hits: np.ndarray,
    n_events: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised per-event signal-hit and unique-string counts from a boolean mask."""
    intp_starts = hit_starts.astype(np.intp)
    n_sig_hits = np.add.reduceat(mask.astype(np.int32), intp_starts)

    if mask.any():
        event_idx = np.repeat(np.arange(n_events), n_hits)
        sig_ev = event_idx[mask]
        sig_str = (channels[mask] // STRING_DIVISOR).astype(np.int32)
        combined = sig_ev.astype(np.int64) * 1000 + sig_str.astype(np.int64)
        sort_idx = combined.argsort()
        combined_s = combined[sort_idx]
        uniq = np.empty(len(combined_s), dtype=bool)
        uniq[0] = True
        uniq[1:] = combined_s[1:] != combined_s[:-1]
        n_sig_strings = np.bincount(
            sig_ev[sort_idx[uniq]], minlength=n_events
        ).astype(np.int32)
    else:
        n_sig_strings = np.zeros(n_events, dtype=np.int32)

    return n_sig_hits.astype(np.int32), n_sig_strings


def read_part_metadata(
    particle_group: h5py.Group,
    part_key: str,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> Optional[Dict]:
    """Pass 1 for one part: run the model, compute per-event sig-hit metadata.

    Returns dict with keys:
        n_hits        (n_events,) int32
        n_sig_hits    (n_events,) int32
        n_sig_strings (n_events,) int32
        hit_starts    (n_events,) int64
        hit_ends      (n_events,) int64
    or ``None`` if the part is absent or empty.
    """
    if f"raw/ev_starts/{part_key}" not in particle_group:
        logger.warning("  Part %s not found in h5 group — skipping", part_key)
        return None
    ev_starts = particle_group[f"raw/ev_starts/{part_key}/data"][:].astype(np.int64)
    n_events = len(ev_starts) - 1
    if n_events == 0:
        return None

    hit_starts = ev_starts[:-1]
    hit_ends = ev_starts[1:]
    n_hits = (hit_ends - hit_starts).astype(np.int32)

    data_raw = particle_group[f"raw/data/{part_key}/data"][:].astype(np.float32)
    channels = particle_group[f"raw/channels/{part_key}/data"][:].astype(np.int32)

    prob = _run_model(data_raw, ev_starts, model, batch_size, device)
    sig_mask = prob > threshold

    n_sig_hits, n_sig_strings = _count_sig_hits_strings(
        sig_mask, channels, hit_starts, n_hits, n_events,
    )

    return {
        "n_hits": n_hits,
        "n_sig_hits": n_sig_hits,
        "n_sig_strings": n_sig_strings,
        "hit_starts": hit_starts,
        "hit_ends": hit_ends,
    }


def write_selected_features(
    h5_path: str,
    locations: List[EventLocation],
    selected_idx: np.ndarray,
    offsets: np.ndarray,
    features_path: str,
    probs_path: str,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> None:
    """Pass 2: write prob-filtered hits to ``features.npy`` and ``probs.npy``.

    Selected events are grouped by (particle_type, part_key), so each part is processed at
    most once.  Only parts with at least one selected event are re-visited.
    """
    total_sig_hits = int(offsets[-1])
    features = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=np.float32, shape=(total_sig_hits, 5),
    )
    probs_out = np.lib.format.open_memmap(
        probs_path, mode="w+", dtype=np.float32, shape=(total_sig_hits,),
    )

    part_groups: Dict[Tuple[str, str], List[Tuple[int, EventLocation]]] = {}
    for out_idx, global_idx in enumerate(selected_idx):
        loc = locations[global_idx]
        key = (loc.particle_type, loc.part_key)
        part_groups.setdefault(key, []).append((out_idx, loc))

    n_parts = len(part_groups)
    logger.info(
        "  Writing features: %s events from %s parts",
        f"{len(selected_idx):,}", f"{n_parts:,}",
    )

    done_parts = 0
    t0 = time.time()

    with h5py.File(h5_path, "r") as h5:
        for (ptype, pk), events in part_groups.items():
            grp = h5[ptype]
            ev_starts = grp[f"raw/ev_starts/{pk}/data"][:].astype(np.int64)
            data_raw = grp[f"raw/data/{pk}/data"][:].astype(np.float32)

            prob = _run_model(data_raw, ev_starts, model, batch_size, device)
            sig_mask = prob > threshold

            for out_idx, loc in events:
                s, e = loc.hit_start, loc.hit_end
                ev_mask = sig_mask[s:e]
                dst = offsets[out_idx]
                n = offsets[out_idx + 1] - dst
                features[dst:dst + n] = data_raw[s:e][ev_mask]
                probs_out[dst:dst + n] = prob[s:e][ev_mask]

            done_parts += 1
            if done_parts % 20 == 0 or done_parts == n_parts:
                elapsed = time.time() - t0
                rate = done_parts / max(elapsed, 1e-3)
                eta = (n_parts - done_parts) / rate
                logger.info(
                    "  [%d/%d] (%.0fs elapsed, ETA %.0fs)",
                    done_parts, n_parts, elapsed, eta,
                )

    features.flush()
    probs_out.flush()
    logger.info(
        "  Written: %s sig hits, %.2f GB features + %.1f MB probs",
        f"{total_sig_hits:,}", features.nbytes / 1e9, probs_out.nbytes / 1e6,
    )


# ---------------------------------------------------------------------------
# Precomputed-probability source (the MC probs HDF5)
# ---------------------------------------------------------------------------

def read_part_metadata_from_probs(
    particle_group: h5py.Group,
    probs_group: h5py.Group,
    part_key: str,
    threshold: float,
) -> Optional[Dict]:
    """Pass 1 for one part, using the precomputed SN probabilities HDF5.

    The probs file stores ``n_sig_hits_0.8`` and ``n_sig_strings_0.8`` per event; the
    hit boundaries come from the main HDF5.  These are the same h8s3 numbers the model
    produces at batch size 256 (checked against a direct threshold on the stored probs).
    If the requested threshold has no stored counts, fail loudly rather than recompute
    silently with a different definition.
    """
    if f"raw/ev_starts/{part_key}" not in particle_group:
        logger.warning("  Part %s not found in h5 group — skipping", part_key)
        return None
    ev_starts = particle_group[f"raw/ev_starts/{part_key}/data"][:].astype(np.int64)
    n_events = len(ev_starts) - 1
    if n_events == 0:
        return None

    hit_starts = ev_starts[:-1]
    hit_ends = ev_starts[1:]
    n_hits = (hit_ends - hit_starts).astype(np.int32)

    key_h = f"n_sig_hits_{threshold:g}"
    key_s = f"n_sig_strings_{threshold:g}"
    if key_h not in probs_group or key_s not in probs_group:
        raise KeyError(
            f"{key_h}/{key_s} not stored in the probs file; available: "
            f"{[k for k in probs_group.keys() if k.startswith('n_sig_hits_')]}"
        )
    n_sig_hits = probs_group[f"{key_h}/{part_key}/data"][:].astype(np.int32)
    n_sig_strings = probs_group[f"{key_s}/{part_key}/data"][:].astype(np.int32)
    if len(n_sig_hits) != n_events or len(n_sig_strings) != n_events:
        raise ValueError(
            f"{part_key}: stored sig counts have {len(n_sig_hits)}/"
            f"{len(n_sig_strings)} entries, expected {n_events}"
        )

    return {
        "n_hits": n_hits,
        "n_sig_hits": n_sig_hits,
        "n_sig_strings": n_sig_strings,
        "hit_starts": hit_starts,
        "hit_ends": hit_ends,
    }


def write_selected_features_from_probs(
    h5_path: str,
    probs_h5_path: str,
    locations: List[EventLocation],
    selected_idx: np.ndarray,
    offsets: np.ndarray,
    features_path: str,
    probs_out_path: str,
    threshold: float,
) -> None:
    """Pass 2 using the precomputed per-hit probabilities.

    The hit arrays come from the main HDF5; the probabilities are read from
    ``<class>/probs/<part>``.  The selected hits are the same ``prob > threshold``
    selection as in the model path, without loading or running the model.
    """
    total_sig_hits = int(offsets[-1])
    features = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=np.float32, shape=(total_sig_hits, 5),
    )
    probs_out = np.lib.format.open_memmap(
        probs_out_path, mode="w+", dtype=np.float32, shape=(total_sig_hits,),
    )

    part_groups: Dict[Tuple[str, str], List[Tuple[int, EventLocation]]] = {}
    for out_idx, global_idx in enumerate(selected_idx):
        loc = locations[global_idx]
        key = (loc.particle_type, loc.part_key)
        part_groups.setdefault(key, []).append((out_idx, loc))

    n_parts = len(part_groups)
    logger.info(
        "  Writing features from probs: %s events from %s parts",
        f"{len(selected_idx):,}", f"{n_parts:,}",
    )

    done_parts = 0
    t0 = time.time()
    with h5py.File(h5_path, "r") as h5, h5py.File(probs_h5_path, "r") as pf:
        for (ptype, pk), events in part_groups.items():
            data_raw = h5[ptype][f"raw/data/{pk}/data"][:].astype(np.float32)
            prob = pf[ptype][f"probs/{pk}/data"][:].astype(np.float32)
            if len(prob) != len(data_raw):
                raise ValueError(
                    f"{ptype}/{pk}: probs have {len(prob)} entries, hits {len(data_raw)}"
                )
            sig_mask = prob > threshold

            for out_idx, loc in events:
                s, e = loc.hit_start, loc.hit_end
                ev_mask = sig_mask[s:e]
                dst = offsets[out_idx]
                n = offsets[out_idx + 1] - dst
                if int(ev_mask.sum()) != n:
                    raise ValueError(
                        f"{ptype}/{pk}: event {loc.event_idx_in_part} has "
                        f"{int(ev_mask.sum())} selected hits, offsets say {n}"
                    )
                features[dst:dst + n] = data_raw[s:e][ev_mask]
                probs_out[dst:dst + n] = prob[s:e][ev_mask]

            done_parts += 1
            if done_parts % 20 == 0 or done_parts == n_parts:
                elapsed = time.time() - t0
                rate = done_parts / max(elapsed, 1e-3)
                eta = (n_parts - done_parts) / rate
                logger.info(
                    "  [%d/%d] (%.0fs elapsed, ETA %.0fs)",
                    done_parts, n_parts, elapsed, eta,
                )

    features.flush()
    probs_out.flush()
    logger.info(
        "  Written: %s sig hits, %.2f GB features + %.1f MB probs",
        f"{total_sig_hits:,}", features.nbytes / 1e9, probs_out.nbytes / 1e6,
    )
