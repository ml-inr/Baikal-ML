"""Reading the source HDF5 files, and reassembling events split across clusters.

Layout facts this module relies on, all recorded in ``doc/hdf5_format.md``:

* the converter ran with ``split_multi: true``, so one physical event that lit
  several clusters appears as one **row per cluster**;
* in MC, ``ev_ids`` is ``{particle}_{part}_{root_entry}``, and the ROOT entry
  number is shared by the rows of one physical event -- that is the grouping key
  (``doc/mc_provenance.md`` section 4);
* in experiment, one part is one *(cluster, run)* pair, and ``header_prty`` is
  ``[season, cluster, run, event_id, sec, nsec]`` with
  ``event_id = sec * 1e9 + nsec`` taken from the cluster-controller clock.
  Cross-cluster grouping therefore has to go through wall-clock time, which is
  what stage 00 measures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import yaml

PART_RE = re.compile(r"part_s(?P<season>\d+)_c(?P<cluster>\d+)_r(?P<run>\d+)$")


@dataclass(frozen=True)
class Config:
    """Parsed ``config.yaml`` with every path already made absolute."""

    raw: dict
    root: Path
    here: Path

    def path(self, *keys: str) -> Path:
        node = self.raw["paths"]
        for key in keys:
            node = node[key]
        return (self.root / node).resolve()

    def __getitem__(self, key: str):
        return self.raw[key]


def load_config(here: Path) -> Config:
    here = Path(here).resolve()
    raw = yaml.safe_load((here / "config.yaml").read_text())
    root = (here / raw["paths"]["root"]).resolve()
    return Config(raw=raw, root=root, here=here)


def exp_parts(h5_path: Path) -> list[str]:
    with h5py.File(h5_path, "r") as handle:
        return sorted(handle["exp_full"]["header_prty"].keys())


def parse_exp_part(part: str) -> tuple[int, int, int]:
    """``part_s2020_c02_r0020`` -> ``(season, cluster, run)``."""
    match = PART_RE.match(part)
    if match is None:
        raise ValueError(f"unparsable experimental part name: {part}")
    return (int(match["season"]), int(match["cluster"]), int(match["run"]))


def exp_event_times(h5_path: Path, part: str) -> np.ndarray:
    """Absolute event times of one experimental part, nanoseconds, int64.

    Values come from ``header_prty`` column 3 (``sec * 1e9 + nsec``), the
    BJointHeader cluster-controller timestamp -- unique and monotonic per run.
    """
    with h5py.File(h5_path, "r") as handle:
        return handle["exp_full"]["header_prty"][part]["data"][:, 3].astype(np.int64)


def mc_root_entries(h5_path: Path, klass: str, part: str) -> np.ndarray:
    """ROOT entry number per row, the grouping key of a multi-cluster MC event."""
    with h5py.File(h5_path, "r") as handle:
        ids = handle[klass]["ev_ids"][part]["data"][:]
    return np.array([int(raw.decode().rsplit("_", 1)[1]) for raw in ids],
                    dtype=np.int64)


def mc_cluster_ids(h5_path: Path, klass: str, part: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        return handle[klass]["raw"]["cluster_ids"][part]["data"][:].astype(np.int16)


def mc_parts(h5_path: Path, klass: str) -> list[str]:
    with h5py.File(h5_path, "r") as handle:
        return sorted(handle[klass]["ev_ids"].keys())


def nearest_dt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every time in ``a``, the signed distance to the closest time in ``b``.

    Both inputs must be sorted ascending.  Returned in the units of the inputs.
    """
    if len(b) == 0:
        return np.full(len(a), np.iinfo(np.int64).max, dtype=np.int64)
    idx = np.searchsorted(b, a)
    left = b[np.clip(idx - 1, 0, len(b) - 1)]
    right = b[np.clip(idx, 0, len(b) - 1)]
    d_left, d_right = a - left, a - right
    take_left = np.abs(d_left) <= np.abs(d_right)
    return np.where(take_left, d_left, d_right)


def raw_hits_per_event(h5_path: Path, group: str, part: str) -> np.ndarray:
    """Number of raw hits of every event of a part, before the sig-noise filter.

    Read from ``raw/ev_starts``, which is a prefix-sum of hit counts, so this
    costs one small array per part instead of touching the hits themselves.
    Raw multiplicity is the honest noise-load measure: everything stored in the
    prediction database is computed *after* filtering and therefore cannot see
    the noise that was removed.
    """
    with h5py.File(h5_path, "r") as handle:
        starts = handle[group]["raw"]["ev_starts"][part]["data"][:]
    return np.diff(starts.astype(np.int64))
