"""Per-event signal-hit counts, read from the sig-noise probabilities file.

The prefilter scores **all** raw hits, so these counts never decide what gets scored.
They are stored beside the score so an analysis can apply the same quality cut the
nu-classifier uses — h8s3, at least 8 signal hits on at least 3 strings — without
re-deriving it from a different definition.

Two definitions live in the same file, and both are copied verbatim rather than
recomputed:

* ``n_sig_hits_{thr}`` / ``n_sig_strings_{thr}`` — hits the sig-noise model calls signal
  at that threshold, and the strings they touch. Experimental data has only these.
* ``n_gt_sig_hits`` / ``n_gt_sig_strings`` — the same counts from MC truth
  (``raw/labels != 0``). MC only.

Both are written by ``data_manager/nu_classifier_ds_builder/io.py``
(``read_part_metadata`` / ``_count_sig_hits_strings``), which is where the nu-classifier's
own stored counts come from — so an h8s3 cut means the same thing in both pipelines.
Verified 2026-09-07 by recomputing all three from ``raw/labels``, ``raw/channels`` and
``probs`` for one part of each MC class: hits, strings and the ``probs > 0.8`` counts all
matched the stored arrays exactly.

Coverage is partial and callers must handle its absence: sig-noise probabilities exist for
10,100 of the 20,004 ``mc_merged`` parts. A part with no entry yields ``None``, which the
prediction writer stores as NULL — "not known here", which a later ``n_sn_hits >= 8``
correctly drops instead of reading as a zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

#: The sig-noise threshold the pipeline is built around. The file also holds 0.5.
SN_THRESHOLD = 0.8


@dataclass(frozen=True)
class PartCounts:
    """Per-event counts for one part. Any field is None when the file has no such array."""

    n_sn_hits: Optional[np.ndarray] = None
    n_sn_strings: Optional[np.ndarray] = None
    n_gt_sig_hits: Optional[np.ndarray] = None
    n_gt_sig_strings: Optional[np.ndarray] = None

    @property
    def empty(self) -> bool:
        return all(getattr(self, f) is None for f in
                   ("n_sn_hits", "n_sn_strings", "n_gt_sig_hits", "n_gt_sig_strings"))

    def slice(self, start: int, stop: int) -> "PartCounts":
        """The same counts for events [start:stop) — for readers that walk a part in windows."""
        return PartCounts(**{
            name: (None if value is None else value[start:stop])
            for name, value in self.__dict__.items()
        })

    def take(self, index: np.ndarray) -> "PartCounts":
        """The same counts for a selection of events."""
        return PartCounts(**{
            name: (None if value is None else value[index])
            for name, value in self.__dict__.items()
        })


def probs_path_for(h5_path: str | Path, sn_tag: str = "k_nsol_labelneq0_da_hs128_k0p0001"
                   ) -> Path:
    """The probabilities file that belongs beside a source file, by naming convention."""
    h5_path = Path(h5_path)
    return h5_path.with_name(f"{h5_path.stem}_probs_{sn_tag}.h5")


def read_part_counts(probs_group: Optional[h5py.Group], part_key: str,
                     threshold: float = SN_THRESHOLD) -> PartCounts:
    """Counts for one part, or an empty PartCounts when the file does not cover it.

    ``threshold`` selects which stored sig-noise arrays to read. It is not applied here:
    the counts were made when the probabilities were written, and re-deriving them from
    ``probs`` would risk a different comparison convention than the one the pipeline uses
    (``>`` in float32 — see inference/reader/schema.py).
    """
    if probs_group is None:
        return PartCounts()

    def read(name: str) -> Optional[np.ndarray]:
        key = f"{name}/{part_key}/data"
        return probs_group[key][:].astype(np.int32) if key in probs_group else None

    return PartCounts(
        n_sn_hits=read(f"n_sig_hits_{threshold}"),
        n_sn_strings=read(f"n_sig_strings_{threshold}"),
        n_gt_sig_hits=read("n_gt_sig_hits"),
        n_gt_sig_strings=read("n_gt_sig_strings"),
    )


def check_lengths(counts: PartCounts, n_events: int, where: str) -> None:
    """Fail loudly if a counts array does not line up with the part it is meant to describe.

    A silent misalignment here would attach one event's hit counts to another's score, and
    nothing downstream could detect it.
    """
    for name, value in counts.__dict__.items():
        if value is not None and len(value) != n_events:
            raise ValueError(
                f"{where}: {name} has {len(value)} entries but the part holds "
                f"{n_events} events — the probabilities file does not match the source.")
