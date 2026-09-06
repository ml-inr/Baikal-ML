"""What scored checkpoints exist, and what each was trained on.

A **checkpoint** here is one set of model weights scored against one or more
sources: a directory under `preds/` named `<experiment>@<checkpoint file>`, holding
one DuckDB per source plus a `run_info.json` recording how it was produced.

The word `run` is deliberately not used for this.  In this project a *run* is a
detector run -- it is a column of the catalog, a field of BARS, and part of every
`part_key` -- and it appears as the `run` column of `Reader.events()`.  Naming the
model side `run` too would put two different meanings in the same word in the same
library, in tables a reader looks at side by side.

The important job here is resolving a run back to the datasets its model was trained
on, because that is what lets an analysis drop biased events without depending on any
table someone wrote after the fact.  The chain is

    preds/<dir>/run_info.json -> "checkpoint" -> experiments/.../da_config.yaml
        -> data.source_domain.npy_dir   (labelled MC)
        -> data.target_domain.npy_dir   (unlabelled DA target)
        -> max_events, train_split, experiment.seed

Measured 2026-08-31: 33 of the 35 directories resolve.  The two that do not are
listed by `unresolved()`, and asking for a training exclusion on one of them raises
rather than silently skipping the exclusion.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .spec import ROOT

#: Directories scanned for run folders.  Add to this rather than passing paths
#: around: a run should be findable by name from anywhere.
PREDS_ROOTS = (
    ROOT / "inference_v2/nu_classifier/preds",
    ROOT / "inference_v2/prefilter/preds",
)


#: Database file name -> (source, threshold).  The threshold is in the file name of
#: every prediction database, which makes it the only per-source record that cannot
#: go missing -- see `_normalise_info` for why `run_info.json` often has lost it.
_DB_NAME = re.compile(r"^(?P<source>.+?)_(?:thr(?P<thr>[0-9p]+)|allhits)$")

#: Keys that belonged to one run in the pre-schema-2 flat `run_info.json`.
_FLAT_RUN_KEYS = {
    "source", "h5_path", "npy_dir", "threshold", "min_hits", "min_strings",
    "n_events_total", "sn_probs_source", "sn_batch_size", "elapsed_s", "written",
}


def _parse_database(path: Path) -> tuple[str, float | None]:
    """`mc_merged_thr0p8.duckdb` -> `("mc_merged", 0.8)`."""
    match = _DB_NAME.match(path.stem)
    if match is None:
        return path.stem, None
    threshold = match["thr"]
    return match["source"], (float(threshold.replace("p", ".")) if threshold
                             else None)


def _normalise_info(info: dict) -> dict:
    """Bring the two `run_info.json` layouts to the same shape.

    Schema 2 keeps one record per source under `runs`.  Everything written before it
    was a single flat dict that each scoring run overwrote, so 34 of the 35 run
    directories describe only the source that happened to be scored last -- the exact
    loss `inference_v2/shared/run_info.py` was written to stop.  Nothing can recover
    the missing records, but the one that survived should not be thrown away too.
    """
    if info.get("runs"):
        return info
    flat = {key: value for key, value in info.items() if key in _FLAT_RUN_KEYS}
    source = flat.get("source")
    return {**info, "runs": {source: flat} if source else {}}


@dataclass(frozen=True)
class Checkpoint:
    """One set of scored weights, and the training it came from."""

    name: str
    directory: Path
    checkpoint_path: str | None
    databases: dict[str, Path]          # source -> DuckDB path
    thresholds: dict[str, float | None]  # source -> sig-noise threshold
    info: dict                          # run_info.json, normalised to schema 2
    config: dict | None                 # the training config, if it resolved

    def database(self, source: str) -> Path:
        try:
            return self.databases[source]
        except KeyError:
            raise KeyError(
                f"checkpoint {self.name!r} has no predictions for source {source!r}; "
                f"it has {sorted(self.databases)}") from None

    def source_info(self, source: str) -> dict:
        """What `run_info.json` records for one source.

        Often `{}`: before schema 2 the file held a single flat record that every
        scoring run overwrote, so most directories describe only the source scored
        last.  Use `thresholds[source]` for the threshold -- that one is recoverable
        from the database file name and is never missing.
        """
        return self.info.get("runs", {}).get(source, {})

    def threshold(self, source: str) -> float | None:
        """Sig-noise threshold of one source's database, from its file name."""
        return self.thresholds.get(source)

    # ── training provenance ──────────────────────────────────────────────────
    def _domain(self, domain: str) -> dict:
        if self.config is None:
            raise LookupError(
                f"checkpoint {self.name!r}: cannot tell what it was trained on -- "
                f"{self._why_unresolved()}. Pass npy_dir=... explicitly if you know "
                f"the dataset.")
        return self.config.get("data", {}).get(domain, {})

    def _why_unresolved(self) -> str:
        if not (self.directory / "run_info.json").exists():
            return "no run_info.json in the directory"
        if not self.checkpoint_path:
            return "run_info.json records no checkpoint"
        return (f"no da_config.yaml beside the checkpoint "
                f"({Path(self.checkpoint_path).parent})")

    def training_dataset(self, domain: str = "source") -> Path:
        """Directory of the NPY dataset this run's model was trained on.

        `domain` is "source" (labelled MC) or "target" (unlabelled DA target).
        """
        key = {"source": "source_domain", "target": "target_domain"}[domain]
        npy_dir = self._domain(key).get("npy_dir")
        if not npy_dir:
            raise LookupError(
                f"checkpoint {self.name!r}: config has no {key}.npy_dir")
        return ROOT / npy_dir

    def training_selection(self, domain: str = "source") -> dict:
        """`max_events`, `train_split` and `seed` -- how the dataset was subsetted."""
        key = {"source": "source_domain", "target": "target_domain"}[domain]
        section = self._domain(key)
        return {"max_events": section.get("max_events"),
                "train_split": section.get("train_split", 0.85),
                "seed": self.config.get("experiment", {}).get("seed", 42)}


def _load_checkpoint(directory: Path) -> Checkpoint:
    databases: dict[str, Path] = {}
    thresholds: dict[str, float | None] = {}
    for path in sorted(directory.glob("*.duckdb")):
        source, threshold = _parse_database(path)
        databases[source] = path
        thresholds[source] = threshold
    info: dict = {}
    info_path = directory / "run_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except json.JSONDecodeError:
            info = {}
    checkpoint = info.get("checkpoint")
    if checkpoint in ("", "unknown"):
        checkpoint = None
    config = None
    if checkpoint:
        config_path = ROOT / Path(checkpoint).parent / "da_config.yaml"
        if not config_path.is_absolute():
            config_path = Path(checkpoint).parent / "da_config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
    return Checkpoint(name=directory.name, directory=directory,
                      checkpoint_path=checkpoint, databases=databases,
                      thresholds=thresholds, info=_normalise_info(info),
                      config=config)


def _all_checkpoints() -> dict[str, Checkpoint]:
    found: dict[str, Checkpoint] = {}
    for root in PREDS_ROOTS:
        if not root.exists():
            continue
        for directory in sorted(root.iterdir()):
            if directory.is_dir() and any(directory.glob("*.duckdb")):
                found[directory.name] = _load_checkpoint(directory)
    return found


def open_checkpoint(name: str) -> Checkpoint:
    """One checkpoint by directory name.  Substrings work when unambiguous."""
    known = _all_checkpoints()
    if name in known:
        return known[name]
    matches = [key for key in known if name in key]
    if len(matches) == 1:
        return known[matches[0]]
    if not matches:
        raise KeyError(f"no scored checkpoint matching {name!r}; "
                       f"call reader.checkpoints() to list them")
    raise KeyError(f"{name!r} matches {len(matches)} checkpoints: {matches[:5]}")


def checkpoints() -> pd.DataFrame:
    """Every scored checkpoint: sources, provenance, whether training resolves."""
    rows = []
    for name, run in _all_checkpoints().items():
        # The threshold comes from the database file names, which always carry it.
        # The sig-noise batch size exists only in `run_info.json`, and only schema 2
        # kept it per source -- older runs simply do not record it, and it is left
        # missing rather than guessed. That matters: the batch size decides which
        # hits the filter keeps, so an unknown one is a real gap, not a formality.
        seen = sorted({t for t in run.thresholds.values() if t is not None})
        batches = sorted({record.get("sn_batch_size")
                          for record in run.info.get("runs", {}).values()
                          if record.get("sn_batch_size") is not None})
        rows.append({
            "checkpoint": name,
            "sources": ", ".join(sorted(run.databases)),
            "threshold": seen[0] if len(seen) == 1 else (seen or None),
            "sn_batch": batches[0] if len(batches) == 1 else (batches or None),
            "info_sources": ", ".join(sorted(run.info.get("runs", {}))) or None,
            "training_resolves": run.config is not None,
            "npy_dir": (run.training_dataset().name if run.config else None),
        })
    return pd.DataFrame(rows).sort_values("checkpoint").reset_index(drop=True)


def unresolved() -> pd.DataFrame:
    """Checkpoints whose training datasets cannot be found, and why."""
    rows = [{"checkpoint": name, "reason": found._why_unresolved()}
            for name, found in _all_checkpoints().items()
            if found.config is None]
    return pd.DataFrame(rows)
