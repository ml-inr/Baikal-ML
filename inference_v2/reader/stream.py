"""The reader itself: events, and a stream of chunks carrying hits.

Iteration follows parts, because a part is the structural unit of a source -- one
detector run, or one MC file, with its own `ev_starts`, `clusters_centers` and
reconstruction.  But a part cannot be the unit of memory: parts range from 202 events
(~1 MB of hits) in `mc_reco` to 3,730,680 events (10.65 GB) in `exp_full`, five orders
of magnitude.  So the stream keeps the part as the iteration unit while sizing chunks
by a byte budget: small parts are accumulated, large ones are cut into slabs.
"""
from __future__ import annotations

import itertools
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import parts as parts_module
from . import query
from .registry import Checkpoint, open_checkpoint
from .schema import BYTES_PER_HIT, SN_THRESHOLD
from .spec import SOURCES, Source, describe

#: Default worker processes for reading hits.  Measured on this machine, reading
#: scattered parts of `mc_merged` (two rounds, disjoint cold blocks each):
#:
#:     workers   ms/part   speedup   cores busy
#:           1      73.8     1.00x         0.09
#:           2      42.3     1.75x         0.18
#:           4      33.2     2.23x         0.26
#:           8      29.6     2.50x         0.33
#:          32      26.3     2.81x         0.76
#:
#: Four is where the curve stops paying: eight adds 12% for twice the processes.
#: The ceiling is the storage, not the CPU -- `/home` is a RAID of spinning disks
#: and a worker sits idle 90-98% of the time waiting for it, which is why 96 cores
#: buy nothing.  **Threads buy nothing at all**: measured at exactly 1.00x with a
#: shared handle and 1.00x with one handle per thread, because h5py serialises.
DEFAULT_WORKERS = 4

#: Do not start a pool for a read smaller than this many parts.  Forking four
#: workers and opening the HDF5 files in each costs about a second, which is more
#: than a short read takes in the first place.
PARALLEL_AFTER_PARTS = 16


@dataclass(frozen=True)
class Progress:
    """Where a read has got to.  Reported, never printed.

    The reader is a library: printing from it would corrupt logs, break tests and
    ignore that the caller may not be a terminal at all.  But only the reader knows
    the stage and the counts, so it hands them over and the caller decides what to
    show.  `Reader.count()` gives the totals to size a bar against.

    `stage` is one of:
        "querying"  the streaming query is starting -- nothing has arrived yet, and
                    on a wide selection this is 7-20 s of apparent silence;
        "training"  the training-exclusion table is being built from the NPY dataset
                    (about 3 s, once per read);
        "reading"   parts are being read;
        "done"      the stream is exhausted.
    """

    stage: str
    parts: int = 0
    events: int = 0
    hits: int = 0
    part_key: str | None = None


@dataclass(frozen=True)
class _Job:
    """One unit of HDF5 reading: a slab of one part.  Crosses to a worker, so it
    holds only picklable, small things -- never a file handle."""

    group: str
    part_key: str
    idx: np.ndarray
    sn_threshold: float
    with_hits: bool


@dataclass
class Chunk:
    """One batch of events with their hits, never crossing a part boundary blindly."""

    events: pd.DataFrame
    hits: pd.DataFrame
    parts: list[str]
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"<Chunk {len(self.events):,} events, {len(self.hits):,} hits, "
                f"{len(self.parts)} part(s), {self.meta.get('n_runs', 0)} run(s)>")


class Reader:
    """One source scored by one checkpoint.  Reads; never writes."""

    def __init__(self, source: str, checkpoint: str | Checkpoint, *,
                 sn_threshold: float = SN_THRESHOLD,
                 npy_dir: str | Path | None = None) -> None:
        if source not in SOURCES:
            raise KeyError(f"unknown source {source!r}; "
                           f"known: {sorted(SOURCES)}")
        self.spec: Source = SOURCES[source]
        self.checkpoint: Checkpoint = (
            checkpoint if isinstance(checkpoint, Checkpoint)
            else open_checkpoint(checkpoint))
        self.sn_threshold = sn_threshold
        if npy_dir is not None:
            # Explicit override for checkpoints whose training config is missing.
            object.__setattr__(self.checkpoint, "config", {
                "data": {"source_domain": {"npy_dir": str(npy_dir)},
                         "target_domain": {"npy_dir": str(npy_dir)}},
                "experiment": {"seed": 42}})
        # Fail now, not mid-stream, if this checkpoint never scored this source.
        self.checkpoint.database(source)

    def __repr__(self) -> str:
        return f"<Reader {self.spec.name} @ {self.checkpoint.name}>"

    def describe(self) -> str:
        return describe(self.spec.name)

    def part_map(self) -> pd.DataFrame:
        """Every part of this source with its event_fk range.  Built on demand.

        Streaming does not need it -- rows arrive already grouped by part -- so it is
        never built implicitly. That matters: for `mc_merged` the catalog query behind
        it takes 26.7 s, which would otherwise dominate every narrow read.
        """
        return parts_module.part_map(self.spec.name)

    def count(self, where=None) -> dict:
        """How many events and parts a selection covers, without reading hits.

        One aggregate query, seconds even on the 94M-row `mc_merged` database. Use
        it to size a progress bar, or to check a selection before paying for it.

        Masks are **not** counted: the `mc_reco` fragment cut lives in the
        probabilities file, not in SQL, so a masked read returns fewer events than
        this reports. The difference is about 12% for `mc_reco` and zero elsewhere.
        """
        connection = query.connect(self.checkpoint, self.spec.name)
        try:
            sql, tables = query.build_sql(connection, self.spec, self.checkpoint,
                                          where)
            for alias, frame in tables.items():
                connection.register(alias, frame)
            inner = sql.replace(" ORDER BY p.event_fk", "")
            events, parts = connection.execute(
                f"SELECT count(*), count(DISTINCT part_key) FROM ({inner})"
            ).fetchone()
        finally:
            connection.close()
        return {"events": int(events), "parts": int(parts)}

    # ── layer 1: events, no hits touched ─────────────────────────────────────
    def events(self, where=None, columns=None, masks=(), *,
               progress=None) -> pd.DataFrame:
        """Every selected event, with its HDF5-resident per-event values.

        No hit is read.  For the reco sources this is the only way to see the BARS
        reconstruction at all: their prediction databases hold nothing but
        `event_fk`, `score` and the two hit counts.
        """
        frames = []
        report = _reporter(progress)
        report(Progress("querying"))
        seen = 0
        with parts_module.Handles() as handles:
            for data_class, part_key, rows in query.iter_parts(
                    self.spec, self.checkpoint, where, columns):
                rows, _ = self._apply_masks(handles, rows, masks, part_key)
                seen += 1
                if rows.empty:
                    continue
                idx = rows["local_idx"].to_numpy(np.int64)
                extra = parts_module.event_scalars(
                    handles, self.spec, self._group_of(data_class), part_key, idx)
                frame = rows.reset_index(drop=True)
                for name, values in extra.items():
                    frame[name] = values
                frames.append(frame)
                report(Progress("reading", parts=seen,
                                events=sum(len(f) for f in frames),
                                part_key=part_key))
        report(Progress("done", parts=seen,
                        events=sum(len(f) for f in frames)))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out.insert(0, "source", self.spec.name)
        return out

    # ── layer 2: the stream ──────────────────────────────────────────────────
    def stream(self, where=None, columns=None, masks=(), *, budget_mb: int = 512,
               with_hits: bool = True, progress=None,
               n_workers: int = DEFAULT_WORKERS) -> Iterator[Chunk]:
        """Chunks of events and their hits, sized to a memory budget.

        `budget_mb` bounds the hits held at once (about 41 bytes per hit, measured).
        A part larger than the budget is cut into slabs; parts smaller than it are
        accumulated, because paying pandas' fixed per-frame cost 24,718 times for
        `mc_reco` would cost eleven minutes of pure overhead.

        `n_workers` reads parts in that many processes, worth about 2.2x -- see
        `DEFAULT_WORKERS` for the measured curve and why threads are not used.
        Chunks still arrive **in part order**, and the same selection returns the
        same events whatever the worker count; only the wall time changes.  Set 1
        to read in this process, which is faster for a handful of parts and is what
        a pool below `PARALLEL_AFTER_PARTS` falls back to anyway.

        The budget is shared between workers, not multiplied by them, so
        `budget_mb` means the same thing at any `n_workers`.
        """
        budget_hits = max(1, int(budget_mb * 1e6 / BYTES_PER_HIT))
        parallel = n_workers > 1
        if parallel:
            # Divided, not per worker: otherwise four workers would hold four
            # budgets and the parameter would quietly mean something else.
            budget_hits = max(1, budget_hits // n_workers)

        report = _reporter(progress)
        # Reported before the query runs: on a wide selection the wait for the first
        # row is 7-20 s, and a bar that only moves on the first chunk looks hung.
        report(Progress("querying"))
        done = {"parts": 0, "events": 0, "hits": 0}
        pending: list[Chunk] = []
        pending_hits = 0

        with parts_module.Handles() as handles:
            jobs = self._jobs(handles, where, columns, masks, budget_hits,
                              with_hits, done)
            for job, rows, applied, payload in self._run(jobs, handles, n_workers):
                chunk = self._assemble(job, rows, applied, payload)
                done["events"] += len(chunk.events)
                done["hits"] += len(chunk.hits)
                report(Progress("reading", part_key=job.part_key, **done))
                if not with_hits:
                    yield chunk
                    continue
                if pending_hits + len(chunk.hits) > budget_hits and pending:
                    yield _merge(pending)
                    pending, pending_hits = [], 0
                pending.append(chunk)
                pending_hits += len(chunk.hits)
                if pending_hits >= budget_hits:
                    yield _merge(pending)
                    pending, pending_hits = [], 0
            if pending:
                yield _merge(pending)
        report(Progress("done", **done))

    def _jobs(self, handles, where, columns, masks, budget_hits, with_hits, done):
        """Lazily turn the SQL stream into read jobs, one per slab of one part.

        Stays a generator: the SQL stream keeps producing while workers read, and
        nothing accumulates in the parent.
        """
        for data_class, part_key, rows in query.iter_parts(
                self.spec, self.checkpoint, where, columns):
            group = self._group_of(data_class)
            rows, applied = self._apply_masks(handles, rows, masks, part_key)
            if rows.empty:
                continue
            done["parts"] += 1
            for piece in self._slabs(rows, budget_hits):
                yield (_Job(group=group, part_key=part_key,
                            idx=piece["local_idx"].to_numpy(np.int64),
                            sn_threshold=self.sn_threshold, with_hits=with_hits),
                       piece, applied)

    def _run(self, jobs, handles, n_workers: int):
        """Execute read jobs, in this process or in a pool, order preserved.

        The pool is started only once enough parts have appeared.  A read of three
        parts would otherwise spend more time forking workers and opening files in
        each than doing the reading.
        """
        buffered: list = []
        remaining = iter(jobs)
        for job in remaining:
            buffered.append(job)
            if len(buffered) >= PARALLEL_AFTER_PARTS:
                break
        every = itertools.chain(buffered, remaining)

        if n_workers <= 1 or len(buffered) < PARALLEL_AFTER_PARTS:
            for job, rows, applied in every:
                yield job, rows, applied, _read_payload(handles, self.spec, job)
            return

        with ProcessPoolExecutor(n_workers, initializer=_worker_init,
                                 initargs=(self.spec,)) as pool:
            plain = ((job, (rows, applied)) for job, rows, applied in every)
            for job, extra, payload in _with_context(plain, pool, n_workers):
                rows, applied = extra
                yield job, rows, applied, payload

    def _assemble(self, job: _Job, rows: pd.DataFrame, applied: list[str],
                  payload) -> Chunk:
        """Build a chunk from the rows and whatever the read returned."""
        read, scalars = payload
        events = rows.reset_index(drop=True)
        for name, values in scalars.items():
            events[name] = values
        events.insert(0, "source", self.spec.name)
        if read is None:
            return Chunk(events=events, hits=pd.DataFrame(), parts=[job.part_key],
                         meta={"masks": applied, "n_runs": 0})
        return Chunk(events=events, hits=read.frame, parts=[job.part_key],
                     meta={"masks": applied, "n_runs": read.n_runs,
                           "hits_read": read.hits_read,
                           "hits_kept": read.hits_kept})

    # ── the addressing self-check ────────────────────────────────────────────
    def verify(self, chunk: Chunk, tol: int = 0) -> pd.DataFrame:
        """Recompute the stored hit counts from HDF5 and compare with the database.

        This tests the whole addressing chain at once -- catalog key, HDF5 row,
        probability-file alignment, threshold convention -- because the counts in
        the database were produced by a separate pipeline from the same
        probabilities.
        A mismatch means the chunk is not the events it claims to be.
        """
        if "n_sn_hits" not in chunk.events or chunk.hits.empty:
            raise RuntimeError("verify needs a chunk read with hits, from a "
                               "checkpoint whose predictions carry n_sn_hits")
        grouped = chunk.hits[chunk.hits["is_sig"]].groupby("event")
        mine_hits = grouped.size().reindex(range(len(chunk.events)), fill_value=0)
        mine_strings = (grouped["string"].nunique()
                        .reindex(range(len(chunk.events)), fill_value=0))
        bad_hits = np.abs(mine_hits.to_numpy()
                          - chunk.events["n_sn_hits"].to_numpy()) > tol
        bad_strings = np.abs(mine_strings.to_numpy()
                             - chunk.events["n_sn_strings"].to_numpy()) > tol
        report = pd.DataFrame({"check": ["n_sn_hits", "n_sn_strings"],
                               "events": [len(chunk.events)] * 2,
                               "mismatched": [int(bad_hits.sum()),
                                              int(bad_strings.sum())]})
        if bad_hits.any() or bad_strings.any():
            raise AssertionError(
                f"{self.spec.name}: recomputed hit counts disagree with the database "
                f"for {int(bad_hits.sum())}/{len(chunk.events)} events -- the "
                f"addressing is wrong, do not use this chunk.\n{report}")
        return report

    # ── internals ────────────────────────────────────────────────────────────
    def _group_of(self, data_class: str) -> str:
        from .spec import h5_group
        return h5_group(self.spec.name, data_class)

    def _apply_masks(self, handles, rows: pd.DataFrame, masks, part_key: str):
        """Masks that cannot be expressed in SQL, applied after selection."""
        applied: list[str] = []
        if "whole_events" in masks and self.spec.fragment_cut is not None:
            group = self._group_of(rows["data_class"].iloc[0])
            keep = parts_module.fragment_mask(
                handles, self.spec, group, part_key,
                rows["local_idx"].to_numpy(np.int64))
            rows = rows[keep]
            applied.append(f"whole_events (n_gt_sig_hits > {self.spec.fragment_cut})")
        return rows.reset_index(drop=True), applied

    def _slabs(self, rows: pd.DataFrame, budget_hits: int):
        """Cut one part's selected rows into pieces that fit the budget.

        Sized by an estimate of hits per event; the true count is only known after
        reading. 120 hits per event is generous for every source measured (81-89).
        """
        per_event = 120
        max_events = max(1, budget_hits // per_event)
        if len(rows) <= max_events:
            return [rows]
        return [rows.iloc[start:start + max_events]
                for start in range(0, len(rows), max_events)]



#: Per-process state.  Handles are opened once per worker and kept: resolving
#: `group/raw/data` afresh for every part costs 6.0 ms against 0.071 ms hoisted.
_WORKER: dict = {}


def _worker_init(spec: Source) -> None:
    _WORKER["spec"] = spec
    _WORKER["handles"] = parts_module.Handles()


def _worker_read(job: _Job):
    return _read_payload(_WORKER["handles"], _WORKER["spec"], job)


def _read_payload(handles, spec: Source, job: _Job):
    """The two HDF5 reads a part needs.  Runs in the parent or in a worker."""
    scalars = parts_module.event_scalars(handles, spec, job.group, job.part_key,
                                         job.idx)
    if not job.with_hits:
        return None, scalars
    read = parts_module.read_hits(handles, spec, job.group, job.part_key, job.idx,
                                  sn_threshold=job.sn_threshold)
    return read, scalars


def _with_context(jobs, pool: ProcessPoolExecutor, lookahead: int):
    """Run jobs through the pool, yielding results **in submission order**.

    Only `lookahead` reads are in flight at once.  Without that bound the pool
    races ahead of the consumer and buffers the whole selection in memory, which
    defeats the point of streaming.

    `extra` carries what the worker must not see -- the row frame and the applied
    masks stay in the parent, since the worker needs only the local indices and
    shipping the frame both ways would double the pickling for nothing.
    """
    inflight: deque = deque()
    for job, extra in jobs:
        inflight.append((job, extra, pool.submit(_worker_read, job)))
        if len(inflight) >= lookahead:
            done_job, done_extra, future = inflight.popleft()
            yield done_job, done_extra, future.result()
    while inflight:
        done_job, done_extra, future = inflight.popleft()
        yield done_job, done_extra, future.result()


def _reporter(progress):
    """Normalise the `progress=` argument into something always callable."""
    if progress is None:
        return lambda state: None
    if not callable(progress):
        raise TypeError("progress must be a callable taking a Progress, or None")
    return progress


def _merge(chunks: list[Chunk]) -> Chunk:
    """Join accumulated small parts into one chunk, renumbering hit ownership."""
    if len(chunks) == 1:
        return chunks[0]
    offsets = np.cumsum([0] + [len(c.events) for c in chunks[:-1]])
    hits = []
    for offset, chunk in zip(offsets, chunks):
        piece = chunk.hits.copy()
        piece["event"] = piece["event"].to_numpy() + offset
        hits.append(piece)
    meta = {"masks": chunks[0].meta.get("masks", []),
            "n_runs": sum(c.meta.get("n_runs", 0) for c in chunks),
            "hits_read": sum(c.meta.get("hits_read", 0) for c in chunks),
            "hits_kept": sum(c.meta.get("hits_kept", 0) for c in chunks)}
    return Chunk(events=pd.concat([c.events for c in chunks], ignore_index=True),
                 hits=pd.concat(hits, ignore_index=True),
                 parts=[p for c in chunks for p in c.parts],
                 meta=meta)
