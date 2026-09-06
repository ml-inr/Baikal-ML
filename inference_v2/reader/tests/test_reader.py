#!/usr/bin/env python3
"""Tests for `inference_v2.reader`.

These run against the real stores.  There is no fixture data in this project, and a
reader whose whole job is to address 900 GB of HDF5 correctly cannot be tested on
anything else -- a synthetic file would only test the synthesiser.  They take a few
minutes, and skip themselves when a store is missing.

The one that matters most is `addressing_matches_database`: it recomputes the stored
signal-hit counts from HDF5 through the reader's own addressing and compares them with
what a separate pipeline wrote into the prediction database.  That single check covers
the catalog key, the HDF5 row, the probability-file alignment and the threshold
convention at once, and it is what caught a `>=` that should have been `>`.

Usage:  python3 test_reader.py [--only NAME]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference_v2 import reader                                    # noqa: E402
from inference_v2.reader import parts, query, spec, training       # noqa: E402

CHECKPOINT = ("260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256"
              "@best_da_model")

#: Selections narrow enough to run in seconds, one part each where possible.
CASES = {
    "exp_reco": [reader.H8S3, reader.NOT_EXCLUDED,
                 "l.part_key = 'part_s2020_c04_r0074'"],
    "mc_merged": [reader.H8S3, reader.NOT_TRAINED, "l.part_key = 'part_10065'"],
    "mc_reco": [reader.H8S3, "e.data_class = 'nue2_2020'", "p.score > 0.99"],
}

#: Population sizes measured 2026-08-31.  A change means the reader or the stores
#: changed, and both are worth being told about.
ELIGIBLE = {
    "mc_merged": (25_071_978, [reader.H8S3, reader.NOT_TRAINED]),
    "exp_full": (1_049_948, [reader.H8S3, reader.NOT_DA_TARGET,
                             reader.NOT_EXCLUDED]),
    "mc_reco": (1_257_559, [reader.H8S3]),
    "exp_reco": (6_659_624, [reader.H8S3, reader.NOT_EXCLUDED]),
}


class Skip(Exception):
    """Raised when a store this check needs is not on disk."""


TESTS: list = []


def test(function):
    TESTS.append(function)
    return function


def need_stores(source: str) -> None:
    source_spec = spec.SOURCES[source]
    if not source_spec.h5.exists() or not source_spec.probs.exists():
        raise Skip(f"{source}: HDF5 stores not present")


def first_chunk(source: str):
    need_stores(source)
    handle = reader.open(source, checkpoint=CHECKPOINT)
    for chunk in handle.stream(where=CASES[source], budget_mb=64):
        return handle, chunk
    raise Skip(f"{source}: selection returned nothing")


# ── the schema is still the production's schema ──────────────────────────────
@test
def reco_columns_match_converter_config():
    widths = reader.check_against_converter_config(spec.ROOT)
    assert widths == {"exp_reco": 25, "mc_reco": 31}, widths


@test
def signal_mask_is_strict_and_float32():
    """A probability of exactly float32(0.8) is signal for `>=` and not for `>`.

    It turns up about once in 20,000 events, which is often enough to make the
    reader's counts disagree with the database if the convention slips.
    """
    assert not reader.signal_mask(np.array([np.float32(0.8)]), 0.8)[0]
    assert reader.signal_mask(np.array([np.float32(0.80001)]), 0.8)[0]


# ── addressing ───────────────────────────────────────────────────────────────
@test
def addressing_matches_database():
    for source in sorted(CASES):
        handle, chunk = first_chunk(source)
        report = handle.verify(chunk)
        assert report["mismatched"].sum() == 0, f"{source}\n{report}"


@test
def event_fk_contiguous_within_parts():
    """`part_map` refuses to build when this fails, so building it is the check."""
    for source in sorted(ELIGIBLE):
        need_stores(source)
        frame = parts.part_map(source)
        assert (frame.hi - frame.lo + 1 == frame.n).all(), source


@test
def eligible_population_sizes():
    checkpoint = reader.open_checkpoint(CHECKPOINT)
    for source, (expected, where) in sorted(ELIGIBLE.items()):
        need_stores(source)
        connection = query.connect(checkpoint, source)
        try:
            sql, tables = query.build_sql(connection, spec.SOURCES[source],
                                          checkpoint, where)
            for alias, frame in tables.items():
                connection.register(alias, frame)
            counted = connection.execute(
                f"SELECT count(*) FROM ({sql.replace(' ORDER BY p.event_fk', '')})"
            ).fetchone()[0]
        finally:
            connection.close()
        assert counted == expected, f"{source}: {counted:,} != {expected:,}"


@test
def count_matches_what_the_stream_delivers():
    """`count()` sizes a progress bar, so it must not lie about the total.

    Masks are the documented exception: the `mc_reco` fragment cut lives in the
    probabilities file and cannot be counted in SQL.
    """
    need_stores("exp_reco")
    handle = reader.open("exp_reco", checkpoint=CHECKPOINT)
    where = CASES["exp_reco"]
    total = handle.count(where=where)
    streamed = sum(len(c.events) for c in handle.stream(where=where, budget_mb=64))
    assert total["events"] == streamed, f"{total['events']} != {streamed}"
    assert total["parts"] >= 1


@test
def progress_is_reported_but_never_printed():
    """The reader hands the caller state; it must not decide how to show it."""
    need_stores("exp_reco")
    handle = reader.open("exp_reco", checkpoint=CHECKPOINT)
    seen = []
    for _ in handle.stream(where=CASES["exp_reco"], budget_mb=64,
                           progress=seen.append):
        pass
    stages = [state.stage for state in seen]
    assert stages[0] == "querying", stages[:3]
    assert stages[-1] == "done", stages[-3:]
    assert any(s == "reading" for s in stages)
    # counters only grow
    events = [s.events for s in seen]
    assert events == sorted(events), events[:10]


# ── reading strategy ─────────────────────────────────────────────────────────
@test
def gap_threshold_changes_cost_not_result():
    """Merging runs across small gaps is a cost decision and nothing more."""
    need_stores("exp_reco")
    source_spec = spec.SOURCES["exp_reco"]
    with parts.Handles() as handles:
        group = source_spec.groups[0]
        part = sorted(handles.group(source_spec.probs, group, "probs").keys())[3]
        starts = handles.dataset(source_spec.h5, group, "raw", "ev_starts",
                                 part=part)[:]
        idx = np.sort(np.random.default_rng(0).choice(len(starts) - 1, 300,
                                                      replace=False))
        wide = parts.read_hits(handles, source_spec, group, part, idx,
                               sn_threshold=0.8, max_gap_hits=10 ** 9)
        narrow = parts.read_hits(handles, source_spec, group, part, idx,
                                 sn_threshold=0.8, max_gap_hits=1)
    assert wide.n_runs == 1, wide.n_runs
    assert narrow.n_runs > 1, narrow.n_runs
    assert wide.hits_read > narrow.hits_read
    pd.testing.assert_frame_equal(wide.frame, narrow.frame)


@test
def budget_does_not_change_which_events_are_read():
    """`budget_mb` is a memory setting; it must not select different events."""
    need_stores("exp_reco")
    handle = reader.open("exp_reco", checkpoint=CHECKPOINT)

    def collect(budget: int) -> np.ndarray:
        frames = [c.events for c in handle.stream(where=CASES["exp_reco"],
                                                  budget_mb=budget)]
        return np.sort(pd.concat(frames, ignore_index=True)["event_fk"].to_numpy())

    assert np.array_equal(collect(16), collect(256))


@test
def workers_change_the_speed_and_nothing_else():
    """Reading in a pool must return exactly what reading here returns.

    Not approximately: the same events in the same order, and the same hit table.
    A parallel path that quietly reorders or drops rows would be worse than no
    parallel path, because every result built on it would be wrong in a way no
    plot would show.
    """
    need_stores("mc_merged")
    handle = reader.open("mc_merged", checkpoint=CHECKPOINT)
    # Enough parts that the pool actually starts (see PARALLEL_AFTER_PARTS).
    where = [reader.H8S3, reader.NOT_TRAINED, "e.data_class = 'nue2_2020'",
             "p.score > 0.6 AND p.score <= 0.8", "hash(l.part_key) % 8 = 0"]

    def collect(workers):
        events, hits = [], []
        for chunk in handle.stream(where=where, budget_mb=256,
                                   n_workers=workers):
            offset = sum(len(e) for e in events)
            piece = chunk.hits.copy()
            piece["event"] = piece["event"].to_numpy() + offset
            events.append(chunk.events)
            hits.append(piece)
        return (pd.concat(events, ignore_index=True),
                pd.concat(hits, ignore_index=True))

    serial_events, serial_hits = collect(1)
    pooled_events, pooled_hits = collect(4)
    assert len(serial_events) > 0, "selection returned nothing to compare"
    assert serial_events["event_fk"].equals(pooled_events["event_fk"])
    numeric = [c for c in serial_events.columns
               if serial_events[c].dtype.kind in "ifb"]
    for column in numeric:
        assert serial_events[column].equals(pooled_events[column]), column
    assert serial_hits.equals(pooled_hits)


# ── training exclusion ───────────────────────────────────────────────────────
@test
def training_exclusion_leaves_nothing_the_model_saw():
    """Self-consistent by construction.

    Deliberately not compared against the old `splits` table: that table was built
    from these same arrays, so agreement would prove nothing about either.
    """
    need_stores("mc_merged")
    checkpoint = reader.open_checkpoint(CHECKPOINT)
    identity = training.identity(checkpoint.training_dataset())
    seen = set(zip(identity["data_class"].astype(str),
                   identity["part_key"].astype(str),
                   identity["local_idx"].to_numpy()))
    handle = reader.open("mc_merged", checkpoint=CHECKPOINT)
    events = handle.events(where=CASES["mc_merged"])
    assert not events.empty, "selection returned nothing to check"
    leaked = [key for key in zip(events["data_class"], events["part_key"],
                                 events["local_idx"]) if key in seen]
    assert not leaked, f"{len(leaked)} training events survived NOT_TRAINED"


@test
def exact_training_half_has_the_configured_size():
    checkpoint = reader.open_checkpoint(CHECKPOINT)
    selection = checkpoint.training_selection()
    exact = training.identity(checkpoint.training_dataset(),
                              strictness="train", selection=selection)
    expected = int(selection["max_events"] * selection["train_split"])
    assert len(exact) == expected, f"{len(exact):,} != {expected:,}"


@test
def unresolved_training_raises_rather_than_skipping():
    """A checkpoint whose training is unknown must fail loudly, not read all."""
    unresolved = reader.unresolved()
    if unresolved.empty:
        raise Skip("every checkpoint resolves")
    checkpoint = reader.open_checkpoint(unresolved.iloc[0]["checkpoint"])
    try:
        checkpoint.training_dataset()
    except LookupError:
        return
    raise AssertionError("expected LookupError for an unresolved checkpoint")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run only checks whose name contains this")
    arguments = parser.parse_args()

    selected = [t for t in TESTS
                if not arguments.only or arguments.only in t.__name__]
    failures = skipped = 0
    for check in selected:
        started = time.time()
        try:
            check()
        except Skip as reason:
            print(f"  SKIP  {check.__name__}: {reason}")
            skipped += 1
        except Exception:
            print(f"  FAIL  {check.__name__}  ({time.time() - started:.1f}s)")
            traceback.print_exc()
            failures += 1
        else:
            print(f"  ok    {check.__name__}  ({time.time() - started:.1f}s)")
    print(f"\n{len(selected) - failures - skipped} passed, {failures} failed, "
          f"{skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
