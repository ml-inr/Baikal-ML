"""Reading hits out of the source HDF5 files, and the conventions around them.

The layout is the same in all four sources: hits lie flat, and `raw/ev_starts` is
a prefix sum, so event `i` owns hits `[starts[i] : starts[i+1]]`.  The
`*_probs_*.h5` files are aligned to it one-for-one -- same `ev_starts`, same
`channels` -- so hit `k` of a part has sig-noise probability `probs[k]`.

What is worth having in one place is the cost model.  `raw/data` is chunked
`(72461, 1)` with gzip, one chunk per column, so the price of a read is the
number of chunks it touches, not the number of hits it wants:

    one random event      ~37 ms      (5 chunks decompressed for ~70 hits)
    one contiguous block  ~0.07 ms per event

Anything that pulls scattered events one at a time is five hundred times slower
than it needs to be.  `read_hit_block` is the fix: it reads a run of events in
one slice, and splits the run wherever the gap between wanted events exceeds a
chunk, because reading across a gap buys nothing.
"""
from __future__ import annotations

import numpy as np

#: Columns of `raw/data`, in order: charge (p.e.), time (ns), and the three
#: cluster-centred coordinates (m).  Time is mean-centred per event by the
#: converter (`center_times: true`), so absolute hit time is not in the file.
HIT_VARS: tuple[str, ...] = ("q", "t", "x", "y", "z")

#: Columns of `prime_prty`, MC only.
PRIME_PRTY_COLUMNS: tuple[str, ...] = (
    "theta_deg", "phi_deg", "energy_gev", "nucleon_n", "response_muons_n",
    "event_weight")

STRING_DIVISOR = 36          # string id = channel // STRING_DIVISOR
SN_THRESHOLD = 0.8           # a hit is signal when prob > this
HDF5_CHUNK_HITS = 72_461     # `raw/data` chunk length, measured


def signal_mask(probs: np.ndarray, threshold: float = SN_THRESHOLD
                ) -> np.ndarray:
    """Which hits the sig-noise filter calls signal.

    Strictly greater, and compared in float32, because that is what the
    pipeline does everywhere it counts signal hits
    (`data_manager/nu_classifier_ds_builder/io.py`, `inference_v2/*/predict_*.py`).
    The distinction is not academic: with `>=`, one event in twenty thousand --
    one holding a hit whose probability is exactly `float32(0.8)` -- gets a
    different signal-hit count here than the one stored in the database.
    """
    return np.asarray(probs) > np.float32(threshold)


def event_lengths(ev_starts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Hit count of each event named by `idx`.  Costs no hit reads."""
    starts = np.asarray(ev_starts, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    return starts[idx + 1] - starts[idx]


def read_hit_block(datasets: dict, ev_starts: np.ndarray, idx: np.ndarray, *,
                   max_gap_hits: int = HDF5_CHUNK_HITS,
                   max_hits_per_read: int = 8_000_000) -> dict[str, np.ndarray]:
    """Hits of the events named by `idx`, read in as few slices as possible.

    Parameters
    ----------
    datasets:
        Name -> open h5py dataset, each indexed by hit.  Two-dimensional ones
        (`raw/data`) are expanded into one array per column of `HIT_VARS`.
    ev_starts:
        The part's `raw/ev_starts`, read in full -- it is small.
    idx:
        Event positions inside the part, **ascending**.  They need not be
        consecutive; large gaps are simply not read.

    Returns
    -------
    dict with `event` (position within `idx`, one entry per hit) and one array
    per requested dataset or hit variable.

    Raises
    ------
    MemoryError
        If one contiguous run would exceed `max_hits_per_read`.  That means the
        caller asked for events that are far apart in the file; lower the block
        size or `max_gap_hits` rather than raising the limit.
    """
    starts = np.asarray(ev_starts, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("read_hit_block: no events requested")
    if np.any(np.diff(idx) < 0):
        raise ValueError("read_hit_block: `idx` must be ascending")

    gaps = starts[idx[1:]] - starts[idx[:-1] + 1]
    runs = np.split(np.arange(idx.size), np.flatnonzero(gaps > max_gap_hits) + 1)

    out: dict[str, list[np.ndarray]] = {}
    for run in runs:
        rows = idx[run]
        lo, hi = starts[rows[0]], starts[rows[-1] + 1]
        if hi - lo > max_hits_per_read:
            raise MemoryError(
                f"one contiguous run spans {hi - lo:,} hits; lower the block "
                f"size or `max_gap_hits`, or raise `max_hits_per_read`")
        take = np.concatenate([np.arange(starts[i] - lo, starts[i + 1] - lo)
                               for i in rows])
        out.setdefault("event", []).append(
            np.repeat(run, event_lengths(starts, rows)))
        for name, dataset in datasets.items():
            block = dataset[lo:hi]
            if block.ndim == 2:
                for col, var in enumerate(HIT_VARS[:block.shape[1]]):
                    out.setdefault(var, []).append(block[take, col])
            else:
                out.setdefault(name, []).append(block[take])
    return {name: np.concatenate(parts) for name, parts in out.items()}
