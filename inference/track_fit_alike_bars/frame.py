"""DataFrame entry points: fit events you already have in a notebook.

Two ways in, depending on what the frame holds:

    from inference.track_fit_alike_bars import fit_events, fit_hits

    ev = r.events(where=[...])              # reader.events(): no hits
    ev = ev.join(fit_events(ev, n_workers=8))

    for chunk in r.stream(...):             # hits already in memory
        fits = fit_hits(chunk.hits)         # indexed by chunk.hits["event"]
        chunk.events.join(fits)

`fit_events` reads the raw pulses itself -- all of them, not the sig-noise
selection -- from the HDF5 source named in each row.  `fit_hits` fits whatever
hits it is given, so it is on the caller to pass all pulses: `chunk.hits` from the
reader contains all of them (the sig-noise flag is a column, not a filter).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd

from inference.reader.parts import contiguous_runs
from inference.reader.spec import SOURCES, h5_group

from .config import FitConfig
from .fit import TrackFit, fit_event

@dataclass(frozen=True)
class FitProgress:
    """Where a fit has got to.  Reported, never printed.

    Same contract as `inference.reader.stream.Progress`, and for the same reason:
    this package is a library, so printing would corrupt logs and ignore that the
    caller may not be a terminal.  Unlike the reader, the fit knows its totals
    before it starts, so they travel with the report instead of costing a second
    call.

    `stage` is one of:

        "fitting"   one more part is done;
        "done"      every part is done.
    """

    stage: str
    parts: int = 0
    events: int = 0
    total_parts: int = 0
    total_events: int = 0
    part_key: str | None = None


@contextmanager
def tqdm_progress(desc: str = "fit", **kwargs):
    """A ready-made progress callable, drawing a tqdm bar.

        with tqdm_progress() as progress:
            fits = fit_events(ev, n_workers=16, progress=progress)

    tqdm is imported here rather than at module level: the package does not depend
    on it, and a caller that does not want a bar never loads it.
    """
    from tqdm.auto import tqdm

    bar = tqdm(total=0, unit=" ev", desc=desc, **kwargs)

    def report(state: FitProgress) -> None:
        bar.total = state.total_events
        bar.n = state.events
        if state.part_key:
            bar.set_description(f"{desc} {state.part_key}")
        bar.refresh()

    try:
        yield report
    finally:
        bar.close()


HIT_COLUMNS = ("q", "t", "x", "y", "z", "channel")
EVENT_COLUMNS = ("source", "part_key", "local_idx")

# A channel mask argument is one of: None (no mask), a `(n_channels,)` boolean
# array, a `(n_rows, n_channels)` array aligned with the frame's rows, or a
# callable taking the rows of one part (in `fit_hits`, of one group) and returning
# either shape or None.  See `masks` for where a mask comes from.


def _mask_for(mask_arg, rows: pd.DataFrame):
    """The mask rows of one part: `(n_rows, n_channels)`, `(n_channels,)` or None."""
    if mask_arg is None:
        return None
    if callable(mask_arg):
        mask = mask_arg(rows)
    else:
        mask = np.asarray(mask_arg)
        if mask.ndim == 2:
            mask = mask[rows["_row"].to_numpy(np.int64)]
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        return np.broadcast_to(mask, (len(rows), mask.size)).copy()
    if mask.ndim != 2 or len(mask) != len(rows):
        raise ValueError(f"channel mask has shape {mask.shape} for {len(rows)} rows")
    return mask


def _check_rows(mask_arg, n_rows: int) -> None:
    if mask_arg is None or callable(mask_arg):
        return
    mask = np.asarray(mask_arg)
    if mask.ndim == 2 and len(mask) != n_rows:
        raise ValueError(f"channel mask has {len(mask)} rows, the frame has {n_rows}")


def fit_hits(hits: pd.DataFrame, *, by: str = "event",
             config: FitConfig | None = None, n_workers: int = 1,
             keep_objects: bool = False, channel_mask=None,
             progress=None) -> pd.DataFrame:
    """One fit per group of a long hit table with columns q, t, x, y, z, channel.

    Returns one row per group, indexed by the group key.  With `keep_objects`, a
    column `fit` holds the `TrackFit` with its per-hit masks and residuals.

    `channel_mask` may also be a callable taking `(group key, sub-table)`.
    """
    missing = [c for c in (*HIT_COLUMNS, by) if c not in hits.columns]
    if missing:
        raise KeyError(f"fit_hits: missing columns {missing}")
    frame = hits.copy()
    if "_row" not in frame.columns:
        frame["_row"] = np.arange(len(frame))
    groups = [(key, frame.loc[g.index]) for key, g in frame.groupby(by, sort=False)]
    keys = [k for k, _ in groups]
    if callable(channel_mask):
        masks = [channel_mask(key, g) for key, g in groups]
    else:
        _check_rows(channel_mask, len(frame))
        masks = [_mask_for(channel_mask, g) for _, g in groups]
    arrays = [tuple(g[c].to_numpy() for c in HIT_COLUMNS) for _, g in groups]
    fits = _run(arrays, config, n_workers, masks, progress)
    return _frame(fits, pd.Index(keys, name=by), keep_objects)


def fit_events(events: pd.DataFrame, *, config: FitConfig | None = None,
               n_workers: int = 1, keep_objects: bool = False,
               channel_mask=None, progress=None) -> pd.DataFrame:
    """Fit every event of a frame that carries its HDF5 address.

    Needs `source`, `part_key`, `local_idx` (as `reader.events()` returns), and
    `data_class` when the source has more than one HDF5 group (the MC sources).
    Returns a frame with the same index as `events`.

    `channel_mask` is BARS' channel mask for the events, in one of the forms
    `MaskArg` lists.  `masks.dqm_masks` gives the right one per source; a callable
    is the convenient form when the mask differs per event::

        ids = header event ids
        cache = {}
        def mask(rows):
            part = rows["part_key"].iloc[0]
            cache.setdefault(part, masks.dqm_masks(source, part, ids))
            ...
        fit_events(ev, channel_mask=mask)
    """
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise KeyError(f"fit_events: missing columns {missing}")
    frame = events.copy()
    if "data_class" not in frame.columns:
        multi = {s for s in frame["source"].unique() if len(SOURCES[s].groups) > 1}
        if multi:
            raise KeyError(f"fit_events: 'data_class' needed for {sorted(multi)}")
        frame["data_class"] = frame["source"].map(lambda s: SOURCES[s].groups[0])

    frame["_row"] = np.arange(len(frame))
    _check_rows(channel_mask, len(frame))
    tasks = []
    for (source, data_class, part), rows in frame.groupby(
            ["source", "data_class", "part_key"], sort=False):
        rows = rows.sort_values("local_idx")
        tasks.append((source, data_class, part,
                      rows["local_idx"].to_numpy(np.int64),
                      rows["_row"].to_numpy(np.int64),
                      _mask_for(channel_mask, rows)))

    report = progress if progress is not None else (lambda state: None)
    total_events = len(frame)
    fits: list[TrackFit | None] = [None] * total_events
    done_parts = done_events = 0
    report(FitProgress("fitting", 0, 0, len(tasks), total_events))
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            # as_completed, not map: results are collected by row index, so the
            # order they arrive in does not matter, and a bar that moves on every
            # finished part beats one that waits for the slowest part in front.
            pending = {pool.submit(_fit_part, task, config): task for task in tasks}
            for future in as_completed(pending):
                task = pending[future]
                for r, f in zip(task[4], future.result()):
                    fits[r] = f
                done_parts += 1
                done_events += len(task[4])
                report(FitProgress("fitting", done_parts, done_events, len(tasks),
                                   total_events, task[2]))
    else:
        for task in tasks:
            for r, f in zip(task[4], _fit_part(task, config)):
                fits[r] = f
            done_parts += 1
            done_events += len(task[4])
            report(FitProgress("fitting", done_parts, done_events, len(tasks),
                               total_events, task[2]))
    report(FitProgress("done", done_parts, done_events, len(tasks), total_events))
    return _frame(fits, events.index, keep_objects)


def read_event_hits(source: str, data_class: str, part: str,
                    local_idx: np.ndarray) -> list[tuple[np.ndarray, ...]]:
    """Raw pulses (q, t, x, y, z, channel) of events of one part, in `local_idx` order.

    `local_idx` must be ascending.
    """
    spec = SOURCES[source]
    group = h5_group(source, data_class)
    out = []
    with h5py.File(spec.h5, "r") as f:
        g = f[group]
        starts = g["raw/ev_starts"][part]["data"][:].astype(np.int64)
        data = g["raw/data"][part]["data"]
        chan = g["raw/channels"][part]["data"]
        for run in contiguous_runs(local_idx, starts):
            rows = local_idx[run]
            lo, hi = int(starts[rows[0]]), int(starts[rows[-1] + 1])
            block, channels = data[lo:hi], chan[lo:hi]
            for i in rows:
                a, b = starts[i] - lo, starts[i + 1] - lo
                d = block[a:b]
                out.append((d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4],
                            channels[a:b]))
    return out


def _fit_part(task, config) -> list[TrackFit]:
    source, data_class, part, local_idx, _, masks = task
    hits = read_event_hits(source, data_class, part, local_idx)
    if masks is None:
        return [fit_event(*h, config=config) for h in hits]
    return [fit_event(*h, config=config, channel_mask=m)
            for h, m in zip(hits, masks)]


def _fit_one(payload) -> TrackFit:
    arrays, config, mask = payload
    return fit_event(*arrays, config=config, channel_mask=mask)


def _run(arrays: list, config, n_workers: int, masks=None,
         progress=None) -> list[TrackFit]:
    masks = [None] * len(arrays) if masks is None else masks
    payloads = list(zip(arrays, [config] * len(arrays), masks))
    report = progress if progress is not None else (lambda state: None)
    total = len(payloads)
    report(FitProgress("fitting", 0, 0, total, total))
    if n_workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            chunk = max(1, len(payloads) // (4 * n_workers))
            out = []
            for done, fit in enumerate(pool.map(_fit_one, payloads, chunksize=chunk),
                                       start=1):
                out.append(fit)
                if done % chunk == 0 or done == total:
                    report(FitProgress("fitting", done, done, total, total))
            report(FitProgress("done", total, total, total, total))
            return out
    out = [_fit_one(p) for p in payloads]
    report(FitProgress("done", total, total, total, total))
    return out


def _frame(fits: list[TrackFit], index: pd.Index, keep_objects: bool) -> pd.DataFrame:
    out = pd.DataFrame([f.scalars() for f in fits], index=index)
    if keep_objects:
        out["fit"] = fits
    return out
