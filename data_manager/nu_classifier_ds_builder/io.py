"""HDF5 reading and mmap writing for the nu-classifier dataset builder.

Two-pass design
---------------
Pass 1 (per part, sequential):
    Load raw/data + raw/ev_starts + raw/channels for one part at a time.
    Run the sig-noise model → per-hit probabilities.
    Compute per-event: n_sig_hits, n_sig_strings (hits with prob > threshold).
    Keep only per-event scalars in RAM — hit features discarded after each part.

Pass 2 (per selected part only):
    Re-run sig-noise model on parts that contain at least one selected event.
    Write prob-filtered hit features (prob > threshold) and prob values
    into pre-allocated memory-mapped output arrays.
"""

import logging
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import h5py
import numpy as np
import torch

from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
    predict_flat,
)

logger = logging.getLogger(__name__)

STRING_DIVISOR = 36


# ---------------------------------------------------------------------------
# Location record
# ---------------------------------------------------------------------------

class EventLocation(NamedTuple):
    """Where to find an event's hits in the source HDF5 file."""
    particle_type: str   # e.g. "muatm_2020"
    part_key: str        # e.g. "part_0"
    hit_start: int       # within the part
    hit_end: int


# ---------------------------------------------------------------------------
# Shared model runner
# ---------------------------------------------------------------------------

def _run_model(
    data_raw: np.ndarray,      # (n_hits, 5) float32, un-normalized
    ev_starts: np.ndarray,     # (n_events+1,) int64
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run sig-noise model with built-in normalization. Returns (n_hits,) float32."""
    return predict_flat(
        model=model,
        data=data_raw,
        ev_starts=ev_starts,
        batch_size=batch_size,
        device=device,
        normalize=True,
        max_gpu_hits=2_000_000,
    )


# ---------------------------------------------------------------------------
# Pass 1: per-part inference + metadata
# ---------------------------------------------------------------------------

def _count_sig_hits_strings(
    mask: np.ndarray,       # (total_hits,) bool
    channels: np.ndarray,   # (total_hits,) int32
    hit_starts: np.ndarray, # (n_events,) int64
    n_hits: np.ndarray,     # (n_events,) int32
    n_events: int,
) -> tuple:
    """Vectorised per-event signal hit + unique string counts from a boolean mask."""
    intp_starts  = hit_starts.astype(np.intp)
    n_sig_hits   = np.add.reduceat(mask.astype(np.int32), intp_starts)

    if mask.any():
        event_idx  = np.repeat(np.arange(n_events), n_hits)
        sig_ev     = event_idx[mask]
        sig_str    = (channels[mask] // STRING_DIVISOR).astype(np.int32)
        combined   = sig_ev.astype(np.int64) * 1000 + sig_str.astype(np.int64)
        sort_idx   = combined.argsort()
        combined_s = combined[sort_idx]
        uniq       = np.empty(len(combined_s), dtype=bool)
        uniq[0]    = True
        uniq[1:]   = combined_s[1:] != combined_s[:-1]
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
    """Pass 1 for one part: run model, compute per-event sig-hit metadata.

    Returns dict with keys:
        n_hits              (n_events,) int32  — total hits before filtering
        n_sig_hits          (n_events,) int32  — hits with prob > threshold
        n_sig_strings       (n_events,) int32  — unique strings among sig hits
        n_gt_sig_hits       (n_events,) int32  — GT signal hits (labels != 0)
        n_gt_sig_strings    (n_events,) int32  — unique GT signal strings
        hit_starts          (n_events,) int64  — within-part hit boundaries
        hit_ends            (n_events,) int64
    or None if the part is empty.
    """
    if f"raw/ev_starts/{part_key}" not in particle_group:
        logger.warning(f"  Part {part_key} not found in h5 group — skipping")
        return None
    ev_starts = particle_group[f"raw/ev_starts/{part_key}/data"][:].astype(np.int64)
    n_events = len(ev_starts) - 1
    if n_events == 0:
        return None

    hit_starts = ev_starts[:-1]
    hit_ends   = ev_starts[1:]
    n_hits     = (hit_ends - hit_starts).astype(np.int32)

    data_raw = particle_group[f"raw/data/{part_key}/data"][:].astype(np.float32)
    channels = particle_group[f"raw/channels/{part_key}/data"][:].astype(np.int32)
    gt_labels = particle_group[f"raw/labels/{part_key}/data"][:]

    prob     = _run_model(data_raw, ev_starts, model, batch_size, device)
    sig_mask = prob > threshold
    gt_mask  = gt_labels != 0

    n_sig_hits, n_sig_strings = _count_sig_hits_strings(
        sig_mask, channels, hit_starts, n_hits, n_events,
    )
    n_gt_sig_hits, n_gt_sig_strings = _count_sig_hits_strings(
        gt_mask, channels, hit_starts, n_hits, n_events,
    )

    return {
        "n_hits":           n_hits,
        "n_sig_hits":       n_sig_hits,
        "n_sig_strings":    n_sig_strings,
        "n_gt_sig_hits":    n_gt_sig_hits,
        "n_gt_sig_strings": n_gt_sig_strings,
        "hit_starts":       hit_starts,
        "hit_ends":         hit_ends,
    }


def read_all_metadata(
    h5_file: h5py.File,
    parts_dict: Dict[str, List[int]],
    particle_map: Dict[str, str],
    particle_encode: Dict[str, int],
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> Dict:
    """Pass 1: iterate all parts, collect per-event metadata.

    Returns dict with per-event arrays (concatenated across all parts):
        n_hits           (N,) int32
        n_sig_hits       (N,) int32
        n_sig_strings    (N,) int32
        particle_types   (N,) int8
        locations        list[EventLocation] of length N
    """
    all_n_hits:       List[np.ndarray] = []
    all_n_sig:        List[np.ndarray] = []
    all_n_str:        List[np.ndarray] = []
    all_n_gt_sig:     List[np.ndarray] = []
    all_n_gt_str:     List[np.ndarray] = []
    all_ptypes:       List[np.ndarray] = []
    all_locations:    List[EventLocation] = []

    for ptype_short, part_nums in parts_dict.items():
        ptype_full = particle_map[ptype_short]
        ptype_code = particle_encode[ptype_full]
        grp        = h5_file[ptype_full]
        n_parts    = len(part_nums)
        t0         = time.time()

        for i, pn in enumerate(sorted(part_nums)):
            pk   = f"part_{pn}"
            meta = read_part_metadata(grp, pk, model, batch_size, device, threshold)
            if meta is None:
                continue

            n_ev = len(meta["n_hits"])
            all_n_hits.append(meta["n_hits"])
            all_n_sig.append(meta["n_sig_hits"])
            all_n_str.append(meta["n_sig_strings"])
            all_n_gt_sig.append(meta["n_gt_sig_hits"])
            all_n_gt_str.append(meta["n_gt_sig_strings"])
            all_ptypes.append(np.full(n_ev, ptype_code, dtype=np.int8))

            for j in range(n_ev):
                all_locations.append(EventLocation(
                    particle_type=ptype_full,
                    part_key=pk,
                    hit_start=int(meta["hit_starts"][j]),
                    hit_end=int(meta["hit_ends"][j]),
                ))

            if (i + 1) % 20 == 0 or i == n_parts - 1:
                elapsed = time.time() - t0
                rate    = (i + 1) / max(elapsed, 1e-3)
                eta     = (n_parts - i - 1) / rate
                logger.info(
                    f"  [{i+1}/{n_parts}] {ptype_full}/{pk} "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
                )

        n_ev_total = sum(len(a) for a in all_n_hits)
        logger.info(f"  {ptype_full}: done ({n_ev_total:,} events so far)")

    return {
        "n_hits":           np.concatenate(all_n_hits),
        "n_sig_hits":       np.concatenate(all_n_sig),
        "n_sig_strings":    np.concatenate(all_n_str),
        "n_gt_sig_hits":    np.concatenate(all_n_gt_sig),
        "n_gt_sig_strings": np.concatenate(all_n_gt_str),
        "particle_types":   np.concatenate(all_ptypes),
        "locations":        all_locations,
    }


# ---------------------------------------------------------------------------
# Pass 2: write filtered features + probs to mmap
# ---------------------------------------------------------------------------

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
    """Pass 2: write prob-filtered hits to features.npy and probs.npy.

    Groups selected events by (particle_type, part_key) so each part is
    processed at most once. Only parts with at least one selected event
    are re-visited.
    """
    total_sig_hits = int(offsets[-1])
    features  = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=np.float32, shape=(total_sig_hits, 5),
    )
    probs_out = np.lib.format.open_memmap(
        probs_path, mode="w+", dtype=np.float32, shape=(total_sig_hits,),
    )

    # Group selected events by part
    part_groups: Dict[Tuple[str, str], List[Tuple[int, EventLocation]]] = {}
    for out_idx, global_idx in enumerate(selected_idx):
        loc = locations[global_idx]
        key = (loc.particle_type, loc.part_key)
        part_groups.setdefault(key, []).append((out_idx, loc))

    n_parts = len(part_groups)
    logger.info(
        f"  Writing features: {len(selected_idx):,} events from {n_parts:,} parts"
    )

    done_parts = 0
    t0 = time.time()

    with h5py.File(h5_path, "r") as h5:
        for (ptype, pk), events in part_groups.items():
            grp        = h5[ptype]
            ev_starts  = grp[f"raw/ev_starts/{pk}/data"][:].astype(np.int64)
            data_raw   = grp[f"raw/data/{pk}/data"][:].astype(np.float32)

            prob     = _run_model(data_raw, ev_starts, model, batch_size, device)
            sig_mask = prob > threshold

            for out_idx, loc in events:
                s, e    = loc.hit_start, loc.hit_end
                ev_mask = sig_mask[s:e]
                dst     = offsets[out_idx]
                n       = offsets[out_idx + 1] - dst
                features[dst:dst + n]  = data_raw[s:e][ev_mask]
                probs_out[dst:dst + n] = prob[s:e][ev_mask]

            done_parts += 1
            if done_parts % 20 == 0 or done_parts == n_parts:
                elapsed = time.time() - t0
                rate    = done_parts / max(elapsed, 1e-3)
                eta     = (n_parts - done_parts) / rate
                logger.info(
                    f"  [{done_parts}/{n_parts}] "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
                )

    features.flush()
    probs_out.flush()
    logger.info(
        f"  Written: {total_sig_hits:,} sig hits, "
        f"{features.nbytes / 1e9:.2f} GB features + "
        f"{probs_out.nbytes / 1e6:.1f} MB probs"
    )
