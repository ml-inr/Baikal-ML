"""Drawing representative samples of h8s3 events and loading their hits.

The sampling is *block (cluster) sampling*, and that is forced by the file
format rather than chosen: `raw/data` is gzip-chunked at 72,461 hits per column,
so one random event costs ~37 ms while an event read as part of a contiguous run
costs ~0.07 ms (`inference_v2/shared/h5_hits.py` carries the measurement).
Drawing 20,000 scattered events would take twelve minutes per source.

So anchors are placed at random in a part's list of eligible events, and each
anchor contributes `block` consecutive eligible events.  Every eligible event
still has the same chance of being drawn, so the estimator is unbiased -- but
events inside a block are neighbours in the file, and in experiment that means
neighbours in time.  Variance is therefore larger than for a simple random
sample of the same size.  Spread the draw over many parts, which the default
allocation does, and keep `block` modest.  `block=1` is exact simple random
sampling at roughly 500x the cost.
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd

from inference_v2.shared.h5_hits import (
    HIT_VARS, PRIME_PRTY_COLUMNS, SN_THRESHOLD, STRING_DIVISOR,
    read_hit_block, signal_mask,
)
from inference_v2.shared.reco_schema import columns_for

from . import paths
from .sources import QUALITY, SOURCES, SourceSpec, h5_group

#: Per-event statistics computed for every hit variable.  `ptp` is derived.
STATS = ("mean", "std", "min", "max")

#: Events per statistics batch.  Grouped aggregation is almost all fixed cost at
#: this scale -- 64 events take 29 ms and 512 take 28 -- so computing statistics
#: once per 64-event block costs 458 s per million events against 24 s when
#: batched.  Bigger batches buy nothing beyond a few thousand events and only
#: hold more hits in memory.
STATS_BATCH_EVENTS = 4096


# ── Progress ──────────────────────────────────────────────────────────────────
#: Seconds between progress lines while reading parts.  A draw of a million
#: events touches ten thousand parts and takes minutes; silence for minutes is
#: indistinguishable from a hang.
REPORT_EVERY_S = 5.0


class _Progress:
    """Stage and per-part progress on one line, or nothing when `verbose` is off."""

    def __init__(self, source: str, verbose: bool) -> None:
        self.source, self.verbose = source, verbose
        self.started = self.last = self.phase = time.time()
        # Overwriting with a carriage return only works on a terminal.  Piped to
        # a file or captured by a runner it just concatenates, so there every
        # line stands on its own.
        self.overwrite = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def say(self, message: str, *, newline: bool = True) -> None:
        if not self.verbose:
            return
        elapsed = time.time() - self.started
        line = f"[{self.source}] {elapsed:6.1f}s  {message}"
        if self.overwrite:
            print(f"\r{line:<78}", end="\n" if newline else "", flush=True)
        else:
            # Not a terminal -- a notebook, a log file, a captured run.  There
            # is no overwriting here, so a progress line has to be its own line
            # or it is lost: dropping them turned a minute of reading into a
            # minute of silence with no way to tell work from a hang.
            print(line, flush=True)

    def begin_phase(self) -> None:
        """Start the clock the estimate of remaining time is measured against.

        Without this the estimate divides the setup time -- ten seconds of
        scanning and querying -- by the first part's share of the work and
        announces eight hours.
        """
        self.phase = time.time()

    @contextmanager
    def stage(self, message: str):
        self.say(f"{message} ...", newline=False)
        yield
        self.say(message + " done")

    def tick(self, message: str, done: float = 0.0, force: bool = False) -> None:
        """Progress, at most every `REPORT_EVERY_S`, with an estimate of what is
        left when `done` (a fraction in 0..1) says how far along we are."""
        now = time.time()
        if not (force or now - self.last >= REPORT_EVERY_S):
            return
        self.last = now
        if 0.0 < done < 1.0 and not force:
            left = (now - self.phase) * (1 - done) / done
            message = f"{message}  ~{left / 60:.1f} min left"
        self.say(message, newline=force)


# ── Databases ─────────────────────────────────────────────────────────────────
def connect(source: str, threads: int = 8) -> duckdb.DuckDBPyConnection:
    """Read-only connection to one prediction database, catalog as `cat`."""
    con = duckdb.connect(str(SOURCES[source].preds), read_only=True)
    con.execute(f"PRAGMA threads={threads}")
    # DuckDB draws its own progress bar on a tty, which in a notebook shows up as
    # unexplained stripes next to our own output.  One reporter is enough.
    con.execute("PRAGMA disable_progress_bar")
    con.execute(f"ATTACH '{paths.CATALOG}' AS cat (READ_ONLY)")
    return con


def _select(source: str, *, quality: bool, drop_training: bool,
            apply_masks: bool, extra: str | None) -> tuple[str, str]:
    """FROM-joins and WHERE clause of the eligible population.

    Identity comes from `splits` where the database has it and from the catalog
    otherwise; `h5_locations` is the only key that means the same thing for
    every source, so it always carries part_key and local_idx.
    """
    spec = SOURCES[source]
    joins = ["JOIN cat.events e ON e.id = p.event_fk",
             "JOIN cat.h5_locations l ON l.event_fk = p.event_fk"]
    where: list[str] = []
    if spec.has_splits:
        joins.insert(0, "JOIN splits s USING (event_fk)")
    if quality:
        where.append(QUALITY)
    if drop_training and spec.train_filter:
        where.append(spec.train_filter)
    if apply_masks:
        where.extend(spec.flag_masks)
        if spec.cluster_blacklist:
            bad = ", ".join(str(c) for c in spec.cluster_blacklist)
            where.append(f"e.cluster NOT IN ({bad})")
        if spec.run_blacklist:
            bad = ", ".join(f"'{r}'" for r in spec.run_blacklist)
            where.append(f"l.part_key NOT IN ({bad})")
    if extra:
        where.append(f"({extra})")
    return " ".join(joins), " AND ".join(where) or "TRUE"


def part_sizes(source: str, *, quality: bool = True, drop_training: bool = True,
               apply_masks: bool = True, extra: str | None = None
               ) -> pd.DataFrame:
    """Eligible event count per (data_class, part_key), in one scan.

    The `mc_reco` fragment cut is not applied here -- it lives in the
    probability file, not in SQL -- so those counts are about 12% optimistic.
    """
    joins, where = _select(source, quality=quality, drop_training=drop_training,
                           apply_masks=apply_masks, extra=extra)
    with connect(source) as con:
        frame = con.execute(f"""
            SELECT e.data_class, l.part_key, count(*) AS n
            FROM predictions p {joins} WHERE {where}
            GROUP BY 1, 2 ORDER BY 1, 2
        """).df()
    frame["group"] = [h5_group(source, c) for c in frame["data_class"]]
    return frame


def _eligible_in_parts(con, source: str, chosen: pd.DataFrame, joins: str,
                       where: str) -> dict[tuple[str, str], pd.DataFrame]:
    """Every eligible event of the chosen parts, in one pass over the database.

    One query per part is what this replaces, and the difference is not small:
    `predictions` has 94M rows in `mc_merged` with no index on `part_key`, so a
    per-part query is a full scan.  Batching turned a four-minute draw into
    twenty seconds.
    """
    classes = ", ".join(f"'{c}'" for c in sorted(set(chosen["data_class"])))
    parts = ", ".join(f"'{p}'" for p in sorted(set(chosen["part_key"])))
    role = ", s.role" if SOURCES[source].has_splits else ""
    frame = con.execute(f"""
        SELECT e.data_class, l.part_key, l.local_idx, p.event_fk, p.score,
               p.n_sn_hits AS db_n_sn_hits, p.n_sn_strings AS db_n_sn_strings,
               e.cluster, e.run{role}
        FROM predictions p {joins}
        WHERE {where}
          AND e.data_class IN ({classes}) AND l.part_key IN ({parts})
        ORDER BY e.data_class, l.part_key, l.local_idx
    """).df()
    return {key: sub.reset_index(drop=True)
            for key, sub in frame.groupby(["data_class", "part_key"])}


# ── HDF5 handles ──────────────────────────────────────────────────────────────
class _Handles:
    """Open each HDF5 file once, and remember the datasets looked up in it.

    Both halves matter.  Opening a 900 GB file repeatedly is obviously wasteful,
    but so is walking `group/raw/data/part_x/data` again for every block: a draw
    of a million events does that tens of thousands of times, in groups holding
    twelve thousand parts.
    """

    def __init__(self) -> None:
        self._open: dict[Path, h5py.File] = {}
        self._datasets: dict[tuple, h5py.Dataset] = {}

    def get(self, path: Path) -> h5py.File:
        if path not in self._open:
            self._open[path] = h5py.File(str(path), "r")
        return self._open[path]

    def dataset(self, path: Path, group: str, *keys: str):
        """`file[group][keys...][part]["data"]`, looked up once and kept."""
        cache_key = (path, group, keys)
        dataset = self._datasets.get(cache_key)
        if dataset is None:
            node = self.get(path)[group]
            for key in keys:
                node = node[key]
            dataset = self._datasets[cache_key] = node["data"]
        return dataset

    def has(self, path: Path, group: str, name: str) -> bool:
        return name in self.get(path)[group]

    def close(self) -> None:
        self._datasets.clear()
        for handle in self._open.values():
            handle.close()
        self._open.clear()


def _fragment_ok(handles: _Handles, spec: SourceSpec, group: str, part: str,
                 idx: np.ndarray) -> np.ndarray:
    """Keep-mask dropping multi-cluster fragments.  MC reco only."""
    if spec.fragment_cut is None:
        return np.ones(len(idx), dtype=bool)
    n_gt = handles.dataset(spec.probs, group, "n_gt_sig_hits", part)[:]
    return n_gt[idx] > spec.fragment_cut


def _anchors(rng: np.random.Generator, n_eligible: int, n_blocks: int,
             block: int) -> np.ndarray:
    """Distinct block starting positions inside a part's eligible list."""
    if n_blocks * block >= n_eligible:              # take the whole part
        return np.arange(0, n_eligible, block)[:n_blocks]
    span = max(1, n_eligible - block + 1)
    return np.sort(rng.choice(span, size=min(n_blocks, span), replace=False))


# ── The sample ────────────────────────────────────────────────────────────────
@dataclass
class Sample:
    """One drawn sample: per-event table, per-hit table, and how it was drawn."""

    source: str
    events: pd.DataFrame
    hits: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"<Sample {self.source}: {len(self.events):,} events, "
                f"{len(self.hits):,} hits, {self.meta.get('n_parts', 0)} parts>")

    def group(self, name: str) -> "Sample":
        """The sub-sample of one HDF5 group, hits kept in step with events."""
        keep = self.events["group"].to_numpy() == name
        events = self.events[keep].reset_index(drop=True)
        renumber = np.cumsum(keep) - 1
        hits = self.hits[keep[self.hits["event"].to_numpy()]].copy()
        hits["event"] = renumber[hits["event"].to_numpy()]
        return Sample(self.source, events, hits.reset_index(drop=True),
                      {**self.meta, "group": name, "drawn": len(events)})


def draw(source: str, n_events: int = 20_000, *, block: int = 64, seed: int = 0,
         quality: bool = True, drop_training: bool = True,
         apply_masks: bool = True, extra: str | None = None,
         sn_threshold: float = SN_THRESHOLD, with_hits: bool = True,
         keep_hits: bool = True, verbose: bool = True,
         stats_batch_events: int = STATS_BATCH_EVENTS,
         max_hits_per_read: int = 8_000_000,
         max_gap_hits: int = 100_000) -> Sample:
    """Draw eligible events and load their hits.  See the module docstring.

    Parameters
    ----------
    n_events:
        Target size.  The achieved size is in `Sample.meta` and is smaller when
        a source runs out of eligible events, or -- for `mc_reco` -- when
        multi-cluster fragments are dropped after the allocation.
    block:
        How many consecutive eligible events one anchor contributes, so the draw
        is `ceil(n_events / block)` anchors rather than `n_events` independent
        picks.  This is the speed/variance dial described in the module
        docstring: it costs nothing in bias and something in variance, because
        events inside a block are neighbours in the file.  Measure that cost on
        a real sample with `design_effect` instead of guessing at it; every
        event carries the `block_id` it came from.  `block=1` is a simple random
        sample at roughly 500x the cost.
    drop_training:
        Remove what the model was fitted on: labelled `mc_merged` events and
        `exp_full` domain-adaptation targets.  Leave on for anything the excess
        is measured from.
    apply_masks:
        Apply the source's run and cluster blacklists and its fragment cut.
    keep_hits:
        Keep the per-hit table.  Per-event statistics are computed either way,
        block by block, so turning this off costs nothing but the hits
        themselves -- and they are the bulk: about 85 hits per event, roughly
        4 GB per million events per source.  Turn it off unless the hit-level
        histograms are the point.
    verbose:
        Print progress.  A million-event draw touches ten thousand parts and
        runs for minutes.
    extra:
        Extra SQL, e.g. `"e.data_class = 'nuatm_conv_2020'"`.  Proportional
        allocation gives rare classes very few events, so draw them separately.

    Returns
    -------
    Sample, whose `events` carry identity, `score`, the database hit counts,
    multiplicities, per-event statistics named
    `{raw|sig}_{q,t,x,y,z}_{mean,std,min,max,ptp}`, `prob_*`, and whatever truth
    or reconstruction the source has; and whose `hits` carry one row per raw hit
    with `event, q, t, x, y, z, channel, string, prob, is_sig`.
    """
    spec = SOURCES[source]
    rng = np.random.default_rng(seed)
    progress = _Progress(source, verbose)

    with progress.stage("scanning parts"):
        sizes = part_sizes(source, quality=quality, drop_training=drop_training,
                           apply_masks=apply_masks, extra=extra)
    if sizes.empty:
        raise RuntimeError(f"{source}: no eligible events under this selection")

    n_blocks = max(1, int(np.ceil(n_events / block)))
    weights = sizes["n"].to_numpy(float)
    per_part = rng.multinomial(n_blocks, weights / weights.sum())
    chosen = sizes[per_part > 0].copy()
    chosen["n_blocks"] = per_part[per_part > 0]
    progress.say(f"{int(sizes['n'].sum()):,} eligible in {len(sizes):,} parts; "
                 f"{n_blocks:,} blocks of {block} over {len(chosen):,} parts")
    if with_hits and keep_hits:
        projected = n_events * 85 * 45 / 1e9        # ~85 hits, ~45 B per hit
        if projected > 2.0:
            progress.say(f"note: keeping hits will need about {projected:.1f} GB; "
                          "pass keep_hits=False for statistics only")

    joins, where = _select(source, quality=quality, drop_training=drop_training,
                           apply_masks=apply_masks, extra=extra)
    handles = _Handles()
    ev_frames: list[pd.DataFrame] = []
    stat_frames: list[pd.DataFrame] = []
    hit_batches: list[pd.DataFrame] = []
    batch: list[pd.DataFrame] = []          # hit frames awaiting statistics
    batch_events: list[int] = []
    n_parts = n_drawn = n_hits = 0

    def flush() -> None:
        """Statistics for the accumulated blocks, then let their hits go.

        Batching is what makes a large draw finish: grouped aggregation at this
        scale is nearly all fixed cost, so paying it once per block would cost
        twenty times more than paying it once per few thousand events.
        """
        if not batch:
            return
        offsets = np.cumsum([0] + batch_events[:-1])
        frame = pd.concat(batch, ignore_index=True)
        frame["event"] += np.repeat(offsets, [len(f) for f in batch])
        stat_frames.append(_event_stats(frame, sum(batch_events)))
        if keep_hits:
            hit_batches.append(frame)
        batch.clear()
        batch_events.clear()

    try:
        with progress.stage(f"querying {len(chosen):,} parts"):
            with connect(source) as con:
                pools = _eligible_in_parts(con, source, chosen, joins, where)
        progress.begin_phase()
        for seen, (_, row) in enumerate(chosen.iterrows(), start=1):
            group, part = row["group"], row["part_key"]
            elig = pools.get((row["data_class"], part))
            if elig is None or elig.empty:
                continue
            idx = elig["local_idx"].to_numpy(np.int64)
            if apply_masks:
                keep = _fragment_ok(handles, spec, group, part, idx)
                elig, idx = elig[keep].reset_index(drop=True), idx[keep]
            if idx.size == 0:
                continue
            n_parts += 1
            for anchor in _anchors(rng, len(idx), int(row["n_blocks"]), block):
                sel = slice(anchor, min(anchor + block, len(idx)))
                frame = _event_identity(handles, spec, group, part,
                                        elig.iloc[sel])
                frame["block_id"] = len(ev_frames)
                ev_frames.append(frame)
                n_drawn += len(frame)
                if not with_hits:
                    continue
                block_hits = _hit_frame(handles, spec, group, part, idx[sel],
                                        sn_threshold, max_hits_per_read,
                                        max_gap_hits)
                n_hits += len(block_hits)
                batch.append(block_hits)
                batch_events.append(len(frame))
                if sum(batch_events) >= stats_batch_events:
                    flush()
                # Ticking per part is useless where a part is huge: `exp_full`
                # has 25 parts holding 3.7M events each, so the first line would
                # arrive after a twenty-fifth of the work.  Blocks are the unit
                # that is comparable across sources.
                progress.tick(f"block {len(ev_frames):,}/{n_blocks:,}  "
                              f"part {seen:,}/{len(chosen):,}  "
                              f"{n_drawn:,} events  {n_hits / 1e6:.1f}M hits",
                              done=len(ev_frames) / n_blocks)
        flush()
    finally:
        handles.close()
    progress.tick(f"read {n_parts:,} parts, {n_drawn:,} events, "
                  f"{n_hits / 1e6:.1f}M hits", force=True)

    with progress.stage("assembling"):
        events = pd.concat(ev_frames, ignore_index=True)
        if stat_frames:
            events = pd.concat(
                [events, pd.concat(stat_frames, ignore_index=True)], axis=1)
        if hit_batches:
            # `event` is batch-local so far; renumber into the merged table.
            offsets = np.cumsum([0] + [len(f) for f in stat_frames[:-1]])
            hits = pd.concat(hit_batches, ignore_index=True)
            hits["event"] += np.repeat(offsets, [len(f) for f in hit_batches])
        else:
            hits = pd.DataFrame()
    events.insert(0, "source", source)
    meta = dict(source=source, requested=n_events, drawn=len(events),
                block=block, seed=seed, n_parts=n_parts,
                n_blocks=len(ev_frames), sn_threshold=sn_threshold,
                drop_training=drop_training, apply_masks=apply_masks,
                extra=extra, eligible_total=int(sizes["n"].sum()),
                keep_hits=bool(hit_batches), n_hits=n_hits,
                elapsed_s=round(time.time() - progress.started, 1))
    return Sample(source=source, events=events, hits=hits, meta=meta)


def _event_identity(handles: _Handles, spec: SourceSpec, group: str, part: str,
                    elig: pd.DataFrame) -> pd.DataFrame:
    """Identity columns plus whatever truth or reconstruction the source has."""
    frame = elig.reset_index(drop=True).copy()
    frame.insert(0, "group", group)
    idx = frame["local_idx"].to_numpy(np.int64)
    if spec.has_prime and handles.has(spec.h5, group, "prime_prty"):
        prime = handles.dataset(spec.h5, group, "prime_prty", part)[:][idx]
        for col, name in enumerate(PRIME_PRTY_COLUMNS):
            frame[f"prime_{name}"] = prime[:, col]
    if spec.has_reco and handles.has(spec.h5, group, "reco_prty"):
        reco = handles.dataset(spec.h5, group, "reco_prty", part)[:][idx]
        for col, name in enumerate(columns_for(reco.shape[1])):
            frame[f"reco_{name}"] = reco[:, col]
    return frame


def _hit_frame(handles: _Handles, spec: SourceSpec, group: str, part: str,
               idx: np.ndarray, sn_threshold: float, max_hits: int,
               max_gap: int) -> pd.DataFrame:
    """Hits of one block, read through the shared chunk-aware block reader."""
    arrays = read_hit_block(
        {"data": handles.dataset(spec.h5, group, "raw", "data", part),
         "channel": handles.dataset(spec.h5, group, "raw", "channels", part),
         "prob": handles.dataset(spec.probs, group, "probs", part)},
        handles.dataset(spec.h5, group, "raw", "ev_starts", part)[:],
        idx, max_gap_hits=max_gap, max_hits_per_read=max_hits)
    frame = pd.DataFrame(arrays)
    frame["string"] = frame["channel"] // STRING_DIVISOR
    frame["is_sig"] = signal_mask(frame["prob"].to_numpy(), sn_threshold)
    return frame


def _event_stats(hits: pd.DataFrame, n_events: int) -> pd.DataFrame:
    """Per-event multiplicities and hit statistics, raw and signal.

    `std` is the pandas default (ddof=1); `ptp` is `max - min`.

    Aggregation is done in float64.  The hits are float32, and charge spans
    0.02 to 3000 p.e. in one event, so a float32 variance depends on the order
    the sum accumulates in: batching the same events differently moved `q_std`
    by 6e-8 relative.  That is harmless for a histogram but it would make the
    batch size -- a pure speed setting -- change the numbers, which is not a
    property worth having.
    """
    hits = hits.copy()
    hits[list(HIT_VARS)] = hits[list(HIT_VARS)].astype(np.float64)
    out = pd.DataFrame(index=pd.RangeIndex(n_events))
    for scope, subset in (("raw", hits), ("sig", hits[hits["is_sig"]])):
        grouped = subset.groupby("event")
        out[f"{scope}_n_hits"] = grouped.size().reindex(out.index, fill_value=0)
        for name, col in (("n_strings", "string"), ("n_channels", "channel")):
            out[f"{scope}_{name}"] = (grouped[col].nunique()
                                      .reindex(out.index, fill_value=0))
        agg = grouped[list(HIT_VARS)].agg(list(STATS))
        for var in HIT_VARS:
            for stat in STATS:
                out[f"{scope}_{var}_{stat}"] = agg[(var, stat)].reindex(out.index)
            out[f"{scope}_{var}_ptp"] = (out[f"{scope}_{var}_max"]
                                         - out[f"{scope}_{var}_min"])
    for stat in STATS:
        out[f"prob_{stat}"] = hits.groupby("event")["prob"].agg(stat)
    return out


def stat_columns(scope: str = "raw", stats=("mean", "std", "ptp", "min", "max"),
                 variables=HIT_VARS) -> list[str]:
    """Names of the per-event statistic columns, for feeding `panel`."""
    return [f"{scope}_{v}_{s}" for v in variables for s in stats]


def design_effect(sample: Sample, columns) -> pd.DataFrame:
    """What the block sampling actually cost, per column.

    Events inside a block are neighbours in the file, so they are not
    independent draws.  The design effect is how much larger the variance of a
    mean is than it would be for a simple random sample of the same size:

        deff = 1 + (m - 1) * rho

    with `m` the mean block size and `rho` the intraclass correlation, estimated
    from the between- and within-block variance.  `n_effective = n / deff` is the
    sample size the draw is really worth for that quantity.  A deff near 1 means
    the blocking is free; a large one means the column varies between blocks
    (between runs, between parts) more than inside them, and needs more anchors
    rather than a bigger block.
    """
    events = sample.events
    if "block_id" not in events:
        raise RuntimeError("this sample predates block_id; redraw with refresh=True")
    rows = []
    for column in columns:
        values = pd.to_numeric(events[column], errors="coerce")
        frame = pd.DataFrame({"v": values, "b": events["block_id"]}).dropna()
        grouped = frame.groupby("b")["v"]
        sizes, means = grouped.size(), grouped.mean()
        n, k = len(frame), len(sizes)
        if k < 2 or n <= k:
            continue
        m = sizes.mean()
        ms_between = (sizes * (means - frame["v"].mean()) ** 2).sum() / (k - 1)
        ms_within = ((frame["v"] - frame["b"].map(means)) ** 2).sum() / (n - k)
        rho = (ms_between - ms_within) / (ms_between + (m - 1) * ms_within)
        rho = float(np.clip(rho, 0.0, 1.0))
        rows.append(dict(column=column, block_size=round(m, 1),
                         icc=round(rho, 4), deff=round(1 + (m - 1) * rho, 1),
                         n=n, n_effective=int(n / (1 + (m - 1) * rho))))
    return pd.DataFrame(rows)


# ── Self-check ────────────────────────────────────────────────────────────────
def verify(sample: Sample, tol: int = 0) -> pd.DataFrame:
    """Check the addressing chain end to end, and fail loudly if it is broken.

    The hit counts in the prediction database were computed by a separate
    pipeline from the same probability file.  Recomputing them here from
    `(group, part_key, local_idx)` tests every link at once: the catalog key,
    the HDF5 row, the probability alignment and the threshold convention.  A
    mismatch means the sample is not the events it claims to be.
    """
    events = sample.events
    if "raw_n_hits" not in events:
        raise RuntimeError("draw(with_hits=False): nothing to verify")
    bad_hits = np.abs(events["sig_n_hits"] - events["db_n_sn_hits"]) > tol
    bad_strings = np.abs(events["sig_n_strings"]
                         - events["db_n_sn_strings"]) > tol
    report = pd.DataFrame({"check": ["n_sn_hits", "n_sn_strings"],
                           "events": [len(events)] * 2,
                           "mismatched": [int(bad_hits.sum()),
                                          int(bad_strings.sum())]})
    if bad_hits.any() or bad_strings.any():
        raise AssertionError(
            f"{sample.source}: recomputed hit counts disagree with the database "
            f"for {int(bad_hits.sum())}/{len(events)} events -- the addressing "
            f"is wrong, do not use this sample.\n{report}")
    return report


# ── Cache ─────────────────────────────────────────────────────────────────────
def save(sample: Sample, cache: Path | None = None) -> Path:
    """Write a sample to parquet, so a redraw is not needed for every figure."""
    out = Path(cache or paths.CACHE) / sample.source
    out.mkdir(parents=True, exist_ok=True)
    sample.events.to_parquet(out / "events.parquet")
    sample.hits.to_parquet(out / "hits.parquet")
    # JSON, not parquet: `meta` mixes ints, floats, strings, bools and None.
    # Written last, so its presence means the cache is complete.
    (out / "meta.json").write_text(json.dumps(sample.meta, indent=2, default=str))
    return out


def load(source: str, cache: Path | None = None) -> Sample:
    """Read back what `save` wrote."""
    out = Path(cache or paths.CACHE) / source
    return Sample(source=source,
                  events=pd.read_parquet(out / "events.parquet"),
                  hits=pd.read_parquet(out / "hits.parquet"),
                  meta=json.loads((out / "meta.json").read_text()))


#: Settings that define which events a sample contains.  A cache whose meta
#: differs from the request in any of these is a different sample, not a
#: cheaper version of the one asked for.
_IDENTITY_KEYS = ("requested", "block", "seed", "sn_threshold", "drop_training",
                  "apply_masks", "extra", "keep_hits")


def cache_mismatch(meta: dict, n_events: int, **kw) -> list[str]:
    """Which defining settings a cached sample disagrees with.  Empty if none."""
    wanted = dict(requested=n_events, block=64, seed=0,
                  sn_threshold=SN_THRESHOLD, drop_training=True,
                  apply_masks=True, extra=None, keep_hits=True)
    wanted.update({k: v for k, v in kw.items() if k in wanted})
    return [f"{key}: cached {meta.get(key)!r}, asked {wanted[key]!r}"
            for key in _IDENTITY_KEYS if meta.get(key) != wanted[key]]


def draw_or_load(source: str, n_events: int = 20_000, *, refresh: bool = False,
                 cache: Path | None = None, **kw) -> Sample:
    """Cached `draw`.  A cache miss draws, verifies and saves.

    A cache that was drawn with different settings raises rather than being
    returned.  Silently handing back a 20,000-event sample to a caller who
    asked for a million is the worst of the three options: redrawing without
    being told costs forty minutes, and returning the wrong thing costs a
    wrong result.
    """
    marker = Path(cache or paths.CACHE) / source / "meta.json"
    if not refresh and marker.exists():
        sample = load(source, cache)
        differences = cache_mismatch(sample.meta, n_events, **kw)
        if differences:
            raise ValueError(
                f"{source}: the cache in {marker.parent} was drawn with other "
                f"settings:\n  " + "\n  ".join(differences) +
                "\nPass refresh=True to redraw it, or cache=<other dir> to keep "
                "both.")
        return sample
    sample = draw(source, n_events, **kw)
    verify(sample)
    save(sample, cache)
    return sample


def draw_all(n_events: int = 20_000, *, refresh: bool = False, **kw
             ) -> dict[str, Sample]:
    """One sample per source, same settings, each verified before use."""
    out = {}
    for name in SOURCES:
        out[name] = draw_or_load(name, n_events, refresh=refresh, **kw)
        print(f"{name:<10} {out[name]!r}  "
              f"eligible={out[name].meta['eligible_total']:,}")
    return out
