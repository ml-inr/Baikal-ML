"""The part map, and reading hits out of one part.

Two facts drive everything here, both measured.

**`event_fk` runs contiguously inside a part** -- checked on all 47,521 parts of the
four sources, `max - min + 1 == count`.  So a part is a range of the primary key, the
map is small, and rows ordered by `event_fk` arrive already grouped by part.

**A read costs chunks, not hits.**  `raw/data` is gzip-chunked at 72,461 values per
column, and less than one chunk cannot be read.  So wanted rows are read as
*runs*: maximal groups separated by less than a chunk, since reading across a small
gap is cheaper than decompressing the surrounding chunk twice.

There is deliberately no "read the whole part" alternative.  An earlier draft chose
between the two by a per-source density threshold; measuring it showed the threshold
was useless.  Merged runs give **byte-identical** frames at every density and are
never slower -- on a 32,837-event part of `mc_merged`:

    fraction wanted   whole-part   merged runs   runs reads
              100%       0.495 s       0.280 s    2,049,485 hits
                1%       0.178 s       0.180 s    2,034,694
              0.1%       0.179 s       0.066 s      591,752

At full coverage the runs collapse to a single slice covering exactly the wanted
span, which is the whole part minus its unwanted edges; below that they start
skipping.  A threshold could only pick the worse option.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd

from .schema import HDF5_CHUNK_HITS, HIT_VARS, STRING_DIVISOR, signal_mask
from .spec import CATALOG, Source, h5_group

_PART_MAP_CACHE: dict[str, pd.DataFrame] = {}


def part_map(source: str, refresh: bool = False) -> pd.DataFrame:
    """`(data_class, group, part_key, lo, hi, n)` for every part of a source.

    Built from the catalog in one query and memoised for the process.  Measured:
    0.1 s for `exp_full`, 2.8 s for `mc_reco`, 5.5 s for `exp_reco`, 26.7 s for
    `mc_merged`.  Nothing is written to disk -- this is metadata about the catalog,
    cheap enough to rebuild each session.
    """
    if not refresh and source in _PART_MAP_CACHE:
        return _PART_MAP_CACHE[source]
    with duckdb.connect(str(CATALOG), read_only=True) as con:
        con.execute("PRAGMA threads=8")
        con.execute("PRAGMA disable_progress_bar")
        frame = con.execute("""
            SELECT e.data_class, l.part_key, count(*) AS n,
                   min(l.event_fk) AS lo, max(l.event_fk) AS hi
            FROM h5_locations l JOIN events e ON e.id = l.event_fk
            WHERE e.source = ?
            GROUP BY 1, 2 ORDER BY 3 DESC
        """, [source]).df()
    if frame.empty:
        raise KeyError(f"catalog holds no events with source = {source!r}")
    frame["group"] = [h5_group(source, c) for c in frame["data_class"]]
    gaps = int((frame.hi - frame.lo + 1 != frame.n).sum())
    if gaps:
        raise AssertionError(
            f"{source}: {gaps} parts do not hold a contiguous event_fk range. The "
            f"reader's part slicing assumes they do -- investigate before using it.")
    _PART_MAP_CACHE[source] = frame
    return frame


class Handles:
    """Open HDF5 files once, and hoist the groups that hold the parts.

    The hoisting is not a micro-optimisation.  Measured on `mc_merged`, per part:
    resolving `group/raw/data/{part}/data` from the top takes **6.0 ms**, while
    keeping `group/raw/data` and resolving only `{part}/data` takes **0.071 ms** --
    85x.  Caching the leaf datasets on top adds nothing, because each part is read
    once.  Over 22,404 parts and four datasets that is nine minutes against six
    seconds of pure name resolution.
    """

    def __init__(self) -> None:
        self._files: dict[Path, h5py.File] = {}
        self._groups: dict[tuple, h5py.Group] = {}

    def file(self, path: Path) -> h5py.File:
        if path not in self._files:
            self._files[path] = h5py.File(str(path), "r")
        return self._files[path]

    def group(self, path: Path, *keys: str) -> h5py.Group:
        key = (path, keys)
        node = self._groups.get(key)
        if node is None:
            node = self.file(path)
            for step in keys:
                node = node[step]
            self._groups[key] = node
        return node

    def dataset(self, path: Path, *keys: str, part: str):
        """`file[keys...][part]["data"]`, with everything above `part` hoisted."""
        return self.group(path, *keys)[part]["data"]

    def has(self, path: Path, group: str, name: str) -> bool:
        return name in self.file(path)[group]

    def close(self) -> None:
        self._groups.clear()
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def __enter__(self) -> "Handles":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def contiguous_runs(idx: np.ndarray, ev_starts: np.ndarray,
                    max_gap_hits: int = HDF5_CHUNK_HITS) -> list[np.ndarray]:
    """Split ascending event indices into groups worth reading in one slice.

    Two wanted events are kept in the same slice when the hits between them are fewer
    than a chunk: reading across a small gap is cheaper than decompressing the
    surrounding chunk twice.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []
    starts = np.asarray(ev_starts, dtype=np.int64)
    gaps = starts[idx[1:]] - starts[idx[:-1] + 1]
    return np.split(np.arange(idx.size), np.flatnonzero(gaps > max_gap_hits) + 1)


@dataclass
class PartHits:
    """Hits of some events of one part, plus what the read cost."""

    frame: pd.DataFrame
    n_runs: int              # contiguous slices actually read
    hits_read: int           # what came off disk, including the gaps read across
    hits_kept: int


def read_hits(handles: Handles, spec: Source, group: str, part: str,
              idx: np.ndarray, *, sn_threshold: float,
              max_gap_hits: int = HDF5_CHUNK_HITS) -> PartHits:
    """Hits of the events named by `idx` (ascending positions within the part).

    Read as merged runs; see the module docstring for why there is no whole-part
    alternative.  The caller has already limited `idx` to fit a memory budget.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("read_hits: no events requested")

    starts = handles.dataset(spec.h5, group, "raw", "ev_starts", part=part)[:]
    starts = starts.astype(np.int64)
    data_ds = handles.dataset(spec.h5, group, "raw", "data", part=part)
    chan_ds = handles.dataset(spec.h5, group, "raw", "channels", part=part)
    prob_ds = handles.dataset(spec.probs, group, "probs", part=part)

    slices = []
    for run in contiguous_runs(idx, starts, max_gap_hits):
        rows = idx[run]
        slices.append((int(starts[rows[0]]), int(starts[rows[-1] + 1]), run))

    pieces, hits_read = [], 0
    for lo, hi, run in slices:
        hits_read += hi - lo
        block = data_ds[lo:hi]
        channels = chan_ds[lo:hi]
        probs = prob_ds[lo:hi]
        rows = idx[run]
        take = np.concatenate([np.arange(starts[i] - lo, starts[i + 1] - lo)
                               for i in rows])
        lengths = (starts[rows + 1] - starts[rows]).astype(np.int64)
        pieces.append(pd.DataFrame({
            "event": np.repeat(run, lengths),
            **{name: block[take, column]
               for column, name in enumerate(HIT_VARS[:block.shape[1]])},
            "channel": channels[take],
            "prob": probs[take],
        }))

    frame = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]
    frame["string"] = frame["channel"] // STRING_DIVISOR
    frame["is_sig"] = signal_mask(frame["prob"].to_numpy(), sn_threshold)
    return PartHits(frame=frame, n_runs=len(slices), hits_read=hits_read,
                    hits_kept=len(frame))


def event_scalars(handles: Handles, spec: Source, group: str, part: str,
                  idx: np.ndarray) -> dict[str, np.ndarray]:
    """Per-event arrays that live in HDF5 rather than in any database.

    `reco_prty` and `prime_prty` are small two-dimensional arrays, one row per event,
    so the whole part is read and the wanted rows taken.  For the reco sources this is
    the only place these values exist: their prediction databases hold nothing but
    `event_fk`, `score` and the two hit counts.
    """
    from .schema import PRIME_PRTY_COLUMNS, reco_columns

    idx = np.asarray(idx, dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    if spec.has_prime and handles.has(spec.h5, group, "prime_prty"):
        prime = handles.dataset(spec.h5, group, "prime_prty", part=part)[:][idx]
        for column, name in enumerate(PRIME_PRTY_COLUMNS):
            out[f"prime_{name}"] = prime[:, column]
    if spec.has_reco and handles.has(spec.h5, group, "reco_prty"):
        reco = handles.dataset(spec.h5, group, "reco_prty", part=part)[:][idx]
        for column, name in enumerate(reco_columns(reco.shape[1])):
            out[f"reco_{name}"] = reco[:, column]
    return out


def fragment_mask(handles: Handles, spec: Source, group: str, part: str,
                  idx: np.ndarray) -> np.ndarray:
    """Keep-mask dropping multi-cluster fragments.  See `spec.py` for what they are."""
    if spec.fragment_cut is None:
        return np.ones(len(idx), dtype=bool)
    counts = handles.dataset(spec.probs, group, "n_gt_sig_hits", part=part)[:]
    return counts[np.asarray(idx, dtype=np.int64)] > spec.fragment_cut
