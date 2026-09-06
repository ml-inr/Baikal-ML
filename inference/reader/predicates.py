"""Named selection conditions, so common cuts are not retyped from memory.

`where=` accepts plain SQL strings and the objects here, mixed freely; everything is
joined with `AND`.  Nothing is applied by default -- a reader that silently cuts three
quarters of the data is how someone ends up believing they looked at everything.

A predicate is more than a string because some cuts are not expressible as one.
`NOT_TRAINED` needs a table registered into the connection and an anti-join; that is
what `resolve` returns alongside the SQL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:                                    # pragma: no cover
    from .registry import Checkpoint
    from .spec import Source


@dataclass
class Resolved:
    """What a predicate contributes to a query."""

    where: list[str] = field(default_factory=list)
    joins: list[str] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def merge(self, other: "Resolved") -> None:
        self.where += other.where
        self.joins += other.joins
        self.tables.update(other.tables)


@dataclass(frozen=True)
class Sql:
    """A plain SQL fragment over the query's aliases: `p`, `e`, `l`."""

    text: str
    label: str = ""

    def resolve(self, source: "Source", checkpoint: "Checkpoint") -> Resolved:
        return Resolved(where=[self.text])

    def __str__(self) -> str:
        return self.label or self.text


@dataclass(frozen=True)
class Excluded:
    """Drop the runs and clusters this source is known to have wrong.

    Source-dependent, so it cannot be a fixed string.  The evidence for each entry is
    in `spec.py` next to it.
    """

    def resolve(self, source: "Source", checkpoint: "Checkpoint") -> Resolved:
        where = []
        if source.bad_runs:
            listed = ", ".join(f"'{r}'" for r in source.bad_runs)
            where.append(f"l.part_key NOT IN ({listed})")
        if source.bad_clusters:
            listed = ", ".join(str(c) for c in source.bad_clusters)
            where.append(f"e.cluster NOT IN ({listed})")
        if not where:
            where = ["TRUE"]
        return Resolved(where=where)

    def __str__(self) -> str:
        return "NOT_EXCLUDED"


@dataclass(frozen=True)
class NotTrained:
    """Drop what this checkpoint was fitted on, via its NPY training dataset.

    `domain` is "source" (labelled MC) or "target" (unlabelled DA target).
    `strictness` is "all" (every event in the dataset, the safe default) or "train"
    (only the training half, reproducing the trainer's two seeded steps).

    Raises if the checkpoint's training dataset cannot be resolved.  It does not
    quietly skip the exclusion: an unfiltered result that looks filtered is worse
    than an error.
    """

    domain: str = "source"
    strictness: str = "all"

    def resolve(self, source: "Source", checkpoint: "Checkpoint") -> Resolved:
        from . import training

        npy_dir = checkpoint.training_dataset(self.domain)
        selection = (checkpoint.training_selection(self.domain)
                     if self.strictness == "train" else None)
        frame = training.identity(npy_dir, domain=self.domain,
                                  strictness=self.strictness, selection=selection)
        alias = f"trained_{self.domain}_{self.strictness}"
        on = [f"{alias}.part_key = l.part_key",
              f"{alias}.local_idx = l.local_idx"]
        if "data_class" in frame.columns:
            on.append(f"{alias}.data_class = e.data_class")
        # The frame is registered with string columns; DuckDB reads pandas categories
        # as VARCHAR, so the join keys line up with the catalog's own types.
        return Resolved(
            joins=[f"LEFT JOIN {alias} ON " + " AND ".join(on)],
            where=[f"{alias}.part_key IS NULL"],
            tables={alias: frame.assign(part_key=frame.part_key.astype(str))},
        )

    def __str__(self) -> str:
        return f"NOT_TRAINED({self.domain}, {self.strictness})"


#: The h8s3 quality cut used throughout the excess work.
H8S3 = Sql("p.n_sn_hits >= 8 AND p.n_sn_strings >= 3", "H8S3")

#: Everything the model was fitted on, conservatively.
NOT_TRAINED = NotTrained("source", "all")
#: Only what actually reached the training half.
NOT_TRAINED_EXACT = NotTrained("source", "train")
#: The unlabelled domain-adaptation target.
NOT_DA_TARGET = NotTrained("target", "all")
#: Runs and clusters with recorded faults.
NOT_EXCLUDED = Excluded()


def resolve_all(conditions, source: "Source",
                checkpoint: "Checkpoint") -> Resolved:
    """Turn a `where=` argument into SQL, joins and tables to register."""
    if conditions is None:
        conditions = []
    if isinstance(conditions, (str, Sql, Excluded, NotTrained)):
        conditions = [conditions]
    out = Resolved()
    for condition in conditions:
        if isinstance(condition, str):
            out.merge(Resolved(where=[condition]))
        else:
            out.merge(condition.resolve(source, checkpoint))
    return out
