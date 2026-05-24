"""Sequential HDF5 reading for the prefilter dataset builder.

Two-pass design:
    Pass 1 (metadata): read ev_starts, labels, channels per part.
        Compute n_signal_hits, n_signal_strings, n_hits per event.
        No hit features loaded — peak RAM ≈ per-event metadata only.

    Pass 2 (features): given final event selection, read hit features
        part-by-part and write directly into a pre-allocated mmap.
"""

import logging
import time
from typing import Dict, List, NamedTuple, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

STRING_DIVISOR = 36


class EventLocation(NamedTuple):
    """Where to find an event's hits in the HDF5 file."""
    particle_type: str
    part_num: int
    hit_start: int
    hit_end: int


# ---------------------------------------------------------------------------
# Pass 1: metadata only (no hit features)
# ---------------------------------------------------------------------------

def read_part_metadata(
    particle_group: h5py.Group,
    part_name: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Read per-event metadata from one part. No hit features loaded.

    Returns dict with keys:
        n_hits           (n_events,) int32
        n_signal_hits    (n_events,) int32
        n_signal_strings (n_events,) int32
        hit_starts       (n_events,) int64  — start index within part
        hit_ends         (n_events,) int64  — end index within part
    or ``None`` if the part has no events.
    """
    ev_starts = particle_group[f"raw/ev_starts/{part_name}/data"][:]
    n_events = len(ev_starts) - 1
    if n_events == 0:
        return None

    hit_start = ev_starts[:-1].astype(np.int64)
    hit_end = ev_starts[1:].astype(np.int64)
    n_hits = (hit_end - hit_start).astype(np.int32)

    labels_raw = particle_group[f"raw/labels/{part_name}/data"][:]
    channels = particle_group[f"raw/channels/{part_name}/data"][:]

    # Signal hit count per event
    signal_mask = labels_raw != 0
    n_signal = np.add.reduceat(signal_mask.astype(np.int32), hit_start.astype(np.intp))

    # Unique signal strings per event
    string_ids = np.where(signal_mask, channels // STRING_DIVISOR, -1)
    event_idx = np.repeat(np.arange(n_events), n_hits)

    if signal_mask.any():
        sig_events = event_idx[signal_mask]
        sig_strings = string_ids[signal_mask]
        combined = (
            sig_events.astype(np.int64) * 1000
            + sig_strings.astype(np.int64)
        )
        sort_idx = combined.argsort()
        combined_sorted = combined[sort_idx]
        unique_mask = np.empty(len(combined_sorted), dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = combined_sorted[1:] != combined_sorted[:-1]
        unique_events = sig_events[sort_idx[unique_mask]]
        n_unique_signal_strings = np.bincount(
            unique_events, minlength=n_events
        ).astype(np.int32)
    else:
        n_unique_signal_strings = np.zeros(n_events, dtype=np.int32)

    return {
        "n_hits": n_hits,
        "n_signal_hits": n_signal.astype(np.int32),
        "n_signal_strings": n_unique_signal_strings,
        "hit_starts": hit_start,
        "hit_ends": hit_end,
    }


def read_all_metadata(
    h5_file: h5py.File,
    parts_dict: Dict[str, List[int]],
    particle_map: Dict[str, str],
    particle_encode: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Pass 1: read metadata for all events across all particle types.

    Args:
        h5_file: Open HDF5 file handle.
        parts_dict: ``{ptype_short: [part_num, ...]}`` from JSON.
        particle_map: ``{"muatm": "muatm_2020", ...}``.
        particle_encode: ``{"muatm_2020": 0, ...}``.

    Returns dict with per-event arrays (all concatenated):
        n_hits           (N,) int32
        n_signal_hits    (N,) int32
        n_signal_strings (N,) int32
        particle_types   (N,) int8
        locations        list[EventLocation] of length N
    """
    all_n_hits: List[np.ndarray] = []
    all_n_signal: List[np.ndarray] = []
    all_n_strings: List[np.ndarray] = []
    all_ptypes: List[np.ndarray] = []
    all_locations: List[EventLocation] = []

    for ptype_short, part_nums in parts_dict.items():
        ptype_full = particle_map[ptype_short]
        grp = h5_file[ptype_full]
        n_parts = len(part_nums)
        ptype_code = particle_encode[ptype_full]
        t0 = time.time()

        for i, pn in enumerate(sorted(part_nums)):
            pk = f"part_{pn}"
            meta = read_part_metadata(grp, pk)
            if meta is None:
                continue

            n_ev = len(meta["n_hits"])
            all_n_hits.append(meta["n_hits"])
            all_n_signal.append(meta["n_signal_hits"])
            all_n_strings.append(meta["n_signal_strings"])
            all_ptypes.append(np.full(n_ev, ptype_code, dtype=np.int8))

            for j in range(n_ev):
                all_locations.append(EventLocation(
                    particle_type=ptype_full,
                    part_num=pn,
                    hit_start=int(meta["hit_starts"][j]),
                    hit_end=int(meta["hit_ends"][j]),
                ))

            if (i + 1) % 50 == 0 or i == n_parts - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_parts - i - 1) / rate
                logger.info(
                    f"  [{i+1}/{n_parts}] {ptype_full}/{pk} "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
                )

        n_ev_total = sum(len(a) for a in all_n_hits)
        logger.info(f"  {ptype_full}: done ({n_ev_total:,} events so far)")

    return {
        "n_hits": np.concatenate(all_n_hits),
        "n_signal_hits": np.concatenate(all_n_signal),
        "n_signal_strings": np.concatenate(all_n_strings),
        "particle_types": np.concatenate(all_ptypes),
        "locations": all_locations,
    }


# ---------------------------------------------------------------------------
# Pass 2: write selected events' features into pre-allocated mmap
# ---------------------------------------------------------------------------

def write_selected_features(
    h5_path: str,
    locations: List[EventLocation],
    selected_idx: np.ndarray,
    offsets: np.ndarray,
    output_path: str,
    max_hits: Optional[int] = None,
) -> None:
    """Pass 2: read hit features for selected events, write to mmap .npy.

    Groups reads by (particle_type, part_num) so each part is read at most
    once. Within each part, copies event slices into the output array.

    Args:
        h5_path: Path to source HDF5 file.
        locations: Full list of EventLocation (from pass 1).
        selected_idx: Indices into ``locations`` in desired output order.
        offsets: (n_selected + 1,) int64 — output event boundaries.
        output_path: Path for the output features.npy file.
        max_hits: Truncate events to this many hits (or None).
    """
    total_hits = int(offsets[-1])
    features = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(total_hits, 5),
    )

    # Build mapping: (particle_type, part_num) → [(selected_order_idx, EventLocation)]
    part_groups: Dict[tuple, List[tuple]] = {}
    for out_idx, global_idx in enumerate(selected_idx):
        loc = locations[global_idx]
        key = (loc.particle_type, loc.part_num)
        part_groups.setdefault(key, []).append((out_idx, loc))

    n_parts = len(part_groups)
    logger.info(f"  Writing features: {len(selected_idx):,} events "
                f"from {n_parts:,} parts")

    done_parts = 0
    t0 = time.time()

    with h5py.File(h5_path, "r") as h5:
        for (ptype, pnum), events in part_groups.items():
            pk = f"part_{pnum}"
            part_data = h5[ptype][f"raw/data/{pk}/data"][:]

            for out_idx, loc in events:
                dst_start = offsets[out_idx]
                n = offsets[out_idx + 1] - dst_start
                features[dst_start:dst_start + n] = (
                    part_data[loc.hit_start:loc.hit_start + n]
                )

            done_parts += 1
            if done_parts % 50 == 0 or done_parts == n_parts:
                elapsed = time.time() - t0
                rate = done_parts / elapsed
                eta = (n_parts - done_parts) / rate
                logger.info(
                    f"  [{done_parts}/{n_parts}] "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
                )

    features.flush()
    logger.info(f"  Features written: {total_hits:,} hits, "
                f"{features.nbytes / 1e9:.2f} GB")
