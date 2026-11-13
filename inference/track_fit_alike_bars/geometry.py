"""OM positions for the isolation probability -- the geometry BARS itself used.

The fit never needs this: it works on the pulses an event happens to have, and the
HDF5 files carry a position for every hit.  A probability that a *silent* module
would fire does need it, and for those there is no position anywhere in the HDF5:
its layout stores only hit coordinates.  So the table comes from the ROOT files,
which carry BARS' own `BGeomTel.fOMs.fPosition`.

Three things this module decides, and all three are silent traps otherwise:

* **which frame.**  A track in the catalogue lives in the HDF5 frame
  (cluster-centred); `fPosition` is absolute.  `oms()` subtracts the cluster centre
  and hands back positions in the HDF5 frame, so a track read from `fits.duckdb`
  can be used as is.
* **which event.**  For experimental data the branch is not one table per run but
  one per event, `(3, n_events, 288)`, and it moves inside a run.  Measured on
  `c04_r0031` (3.04 M events, tables read at 12 points of the run): the per-event
  table sits 0.07-0.46 m (median per OM) from the run's median, worst OM 1.54 m.
  Over the first 3000 events alone the scatter is only millimetres -- the drift
  accumulates, so a short window measures nothing.  Between runs of one cluster it
  is up to 0.22 m.  At 1.5 m from a track the light model changes mu by tens of
  per cent, so the run snapshot is *not* a default: for experimental sources
  `oms()` insists on `local_idx=` (exact, one small read) and the snapshot is
  available under `snapshot=True`, for looking around, not for numbers.
* **which channels.**  `exp_full` numbers channels locally within a cluster and
  its 29 parts all have a ROOT file of their own; the MC sources number channels
  globally over 7 clusters and their geometry is static (scatter 0 m).

    from inference.track_fit_alike_bars import geometry

    om = geometry.oms("exp_full", "part_s2020_c04_r0031", local_idx=17)  # exact
    om = geometry.oms("mc_merged")                     # MC: one static table
    om = geometry.oms("exp_full", "part_s2020_c04_r0031", snapshot=True)
    om.xyz[:3], om.enabled is not None, om.cluster
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot

from . import masks
from ..reader.spec import SOURCES

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Raw experimental ROOT files, one run each, `s2020_c04_r0031.root` -- the
#: cluster's own `BGeomTel`, the geometry the 2020 production used.  The text
#: `bars/config/2020/static/*` is *not* it: measured up to 7.3 m off in z.
RAW_ROOT = REPO_ROOT / "data_manager/data/exp_root"

#: MC geometry is static and the same for every class; one sample file carries the
#: whole 2016-channel array under `ArrayConfig`.
MC_ROOT = REPO_ROOT / "data_manager/data/mc_root_samples/nuatm.root"

GEOMETRY = ("BGeomTel./BGeomTel.BGeomTel/BGeomTel.BGeomTel.fOMs/"
            "BGeomTel.BGeomTel.fOMs.fPosition")

#: Channels per cluster, the h5 and BARS convention alike (`ch // 288` = cluster).
CHANNELS_PER_CLUSTER = 288

#: How many events the run snapshot averages over, in how many spread windows.
SAMPLE_EVENTS = 1000
SAMPLE_WINDOWS = 4

#: Reading a span of per-event tables costs about a hundred microseconds per
#: entry, one entry at a time about sixty milliseconds.  A cluster of wanted
#: events is therefore read as a span when the span is not absurdly wider than
#: the cluster; `SPAN_CHUNK` bounds the memory of one read.
CLUSTER_GAP = 10_000
SPAN_MIN = 10_000
#: one entry at a time costs ~60 ms, a span of them ~50 us each, so a span pays
#: for itself up to a thousand entries per wanted event
SPAN_SLACK = 1000
SPAN_CHUNK = 30_000

_RAW = re.compile(r"s(\d+)_c(\d+)_r(\d+)\.root$")

_cache: dict[tuple, np.ndarray] = {}


@dataclass(frozen=True)
class Oms:
    """Positions of every optical module of one detector, in the HDF5 frame.

    `channel` is the id space of the source's HDF5 files (`exp_full`: 0..287 of
    that cluster; MC: 0..2015 global), which is also the space of the fit's input.
    `xyz` are metres in the hit frame: `h5 = om.xyz[ch]` for any fired channel
    (agreement with the hit positions is 2e-6 m).  `enabled` is the DQM channel
    mask for the runs whose mask is one array per run (`exp_full`, `exp`), and
    `None` for MC -- which BARS reconstructs without a mask, so `None` means
    "nothing is switched off", not "unknown" -- and for `exp_reco`, whose mask is
    per event and has to be handed to `pnonhit(enabled=...)` instead.
    """

    source: str
    part_key: str | None
    channel: np.ndarray
    xyz: np.ndarray
    enabled: np.ndarray | None
    cluster: int | None

    def __len__(self) -> int:
        return len(self.channel)

    def __repr__(self) -> str:
        mask = "no mask" if self.enabled is None else f"{int(self.enabled.sum())} on"
        return (f"<Oms {self.source} {self.part_key or '-'} "
                f"{len(self.channel)} channels, {mask}>")

    def usable(self) -> np.ndarray:
        """Boolean per channel: counted by the probability, nothing switched off."""
        if self.enabled is None:
            return np.ones(len(self.channel), dtype=bool)
        return self.enabled


def raw_root_path(part_key: str) -> Path:
    """Raw ROOT file of an experimental part, e.g. `s2020_c04_r0031.root`."""
    season, cluster, run = masks.part_run(part_key)
    return RAW_ROOT / f"s{season}_c{cluster:02d}_r{run:04d}.root"


def geometry_root_path(source: str, part_key: str) -> Path:
    """The ROOT file whose `BGeomTel` belongs to this part.

    `exp_full` is unpacked as 34 raw files, one per run it holds; `exp_reco` and
    `exp` span 370 runs and their raw files are not unpacked, but every one of
    those runs has its BARS output next to it (`masks.reco_root_path`), and that
    file carries the same per-event geometry.
    """
    if source == "exp_full":
        return raw_root_path(part_key)
    path = masks.reco_root_path(part_key)
    return path if path.exists() else raw_root_path(part_key)


def _event_table(tree, local_idx: int) -> np.ndarray:
    """One event's `(288, 3)`, absolute coordinates, from an open `Events` tree."""
    window = dict(entry_start=int(local_idx), entry_stop=int(local_idx) + 1)
    return np.asarray(ak.unzip(tree[GEOMETRY].array(**window)),
                      dtype=np.float64)[:, 0, :].T


def _read_exp_table(path: Path, local_idx: int | None) -> np.ndarray:
    """`(n_events, 288, 3)` or one event's `(288, 3)`, absolute coordinates."""
    if not path.exists():
        raise FileNotFoundError(
            f"нет ROOT-файла геометрии {path}: для этого рана он не распакован, "
            f"а геометрия данных пер-событийная и из h5 недоступна")
    with uproot.open(path) as file:
        tree = file["Events"]
        if local_idx is not None:
            return _event_table(tree, local_idx)
        n_events = int(tree.num_entries)
        span = max(1, SAMPLE_EVENTS // SAMPLE_WINDOWS)
        parts = []
        for start, stop in _spread_windows(n_events, span, SAMPLE_WINDOWS):
            table = np.asarray(ak.unzip(tree[GEOMETRY].array(entry_start=start,
                                                             entry_stop=stop)),
                               dtype=np.float64)
            parts.append(table.transpose(1, 2, 0))          # (n_ev, 288, 3)
        # the run's table: per-OM median over the sample, not the first event --
        # the first event is as arbitrary as any other, the median is not
        return np.median(np.concatenate(parts), axis=0)


def _spread_windows(n_events: int, span: int, count: int) -> list[tuple[int, int]]:
    """`count` disjoint windows of `span` events, spread through the run."""
    if n_events <= span * count:
        return [(0, n_events)]
    step = n_events // count
    return [(k * step, min(k * step + span, n_events)) for k in range(count)]


def _read_mc_table() -> np.ndarray:
    """`(2016, 3)` absolute MC positions, global channel order."""
    with uproot.open(MC_ROOT) as file:
        table = np.asarray(ak.unzip(file["ArrayConfig"][GEOMETRY].array()),
                           dtype=np.float64)
    return table[:, 0, :].T


def _centre_h5(source: str, part_key: str | None, cluster: int | None) -> np.ndarray:
    """Cluster centre(s) from the source's own HDF5, for the frame conversion."""
    path = SOURCES[source].h5
    with h5py.File(path, "r") as file:
        if source in ("exp_full", "exp", "exp_reco"):
            return np.asarray(file[f"{SOURCES[source].stem}/clusters_centers/"
                                   f"{part_key}/data"][:], dtype=np.float64)
        group = next(g for g in file.keys() if g.endswith("_2020"))
        return np.asarray(file[f"{group}/clusters_centers/data"][:], dtype=np.float64)


def oms(source: str, part_key: str | None = None, *,
        local_idx: int | None = None, snapshot: bool = False) -> Oms:
    """Every OM of the detector a part belongs to, in the part's hit frame.

    Experimental data needs `local_idx`: the detector table of a run is stored
    per event and drifts up to 1.5 m across it, which is not a detail at 22 m.
    `snapshot=True` gives the run's median table instead -- for exploring, not for
    numbers.  The MC sources have one static table and take neither.
    """
    # the MC table is common to every part of every class, so the part does not
    # take part in the key (and a caller looping over parts is not charged for it)
    mc = source not in ("exp_full", "exp", "exp_reco")
    if not mc and local_idx is None and not snapshot:
        raise ValueError(
            f"{source}: геометрия данных хранится на каждое событие и плывёт по рану "
            f"до ~1.5 м -- укажите local_idx= (точно) или snapshot=True (медиана рана, "
            f"приблизительно)")
    key = (source, None if mc else part_key, local_idx)
    if key not in _cache:
        if mc:
            # `part_key` is accepted and ignored: the MC detector is static and
            # the same for every class, and callers naturally have the key at hand
            _store(key, source, None, None, _read_mc_table())
        else:
            _store(key, source, part_key, local_idx,
                   _read_exp_table(geometry_root_path(source, part_key), local_idx))
    return _oms_of(key, source, part_key)


def prefill(source: str, part_key: str | None, local_idxs) -> None:
    """Read the per-event tables of several events in one file open.

    `oms(local_idx=...)` opens the file per call, which is fine for one event and
    ruinous in a loop over a frame; `attach` comes through here.  Reading one
    entry at a time costs ~60 ms (uproot parses the whole basket either way),
    while a contiguous span costs ~100 us per entry -- and the two differ by
    orders of magnitude, so a cluster of wanted events is read as a range and the
    needed entries are picked out of it.  Already cached events are skipped.
    """
    mc = source not in ("exp_full", "exp", "exp_reco")
    if mc:
        oms(source)
        return
    need = sorted({int(i) for i in local_idxs
                   if (source, part_key, int(i)) not in _cache})
    if not need:
        return
    path = geometry_root_path(source, part_key)
    if not path.exists():
        raise FileNotFoundError(f"нет ROOT-файла геометрии {path}")
    with uproot.open(path) as file:
        tree = file["Events"]
        for first, last, count in _clusters(need):
            span = last - first + 1
            if span <= max(SPAN_MIN, SPAN_SLACK * count):
                for local_idx, table in _read_span(tree, first, last, set(need)):
                    _store((source, part_key, local_idx), source, part_key,
                           local_idx, table)
            else:                                   # далеко разбросанные события
                for local_idx in need:
                    if first <= local_idx <= last:
                        _store((source, part_key, local_idx), source, part_key,
                               local_idx, _event_table(tree, local_idx))


def _clusters(need: list[int]) -> list[tuple[int, int, int]]:
    """`(first, last, count)` runs of wanted entries no further apart than `CLUSTER_GAP`."""
    out = []
    first = last = need[0]
    count = 1
    for i in need[1:]:
        if i - last > CLUSTER_GAP:
            out.append((first, last, count))
            first = i
            count = 0
        last = i
        count += 1
    out.append((first, last, count))
    return out


def _read_span(tree, first: int, last: int, need: set[int]):
    """Yield `(local_idx, (288, 3))` for the wanted entries of a span."""
    start = first
    while start <= last:
        stop = min(start + SPAN_CHUNK, last + 1)
        table = np.asarray(ak.unzip(tree[GEOMETRY].array(entry_start=start,
                                                         entry_stop=stop)),
                           dtype=np.float64)
        for i in range(start, stop):
            if i in need:
                yield i, table[:, i - start, :].T
        start = stop


def centre_of(source: str, part_key: str) -> np.ndarray:
    """Cluster centre of an experimental part, metres, in the absolute frame.

    `h5_frame = absolute - centre_of(...)`: what turns a track written by BARS
    (absolute) into the frame of the hits and of `Oms`.  The MC sources have one
    centre per cluster, not per part, and are not served here.
    """
    if source not in ("exp_full", "exp", "exp_reco"):
        raise ValueError(f"{source}: у МС центр свой на каждый кластер")
    _, cluster, _ = masks.part_run(part_key)
    return np.ravel(_centre_h5(source, part_key, cluster))


def _store(key, source: str, part_key: str | None, local_idx: int | None,
           absolute: np.ndarray) -> None:
    """Convert absolute positions to the h5 frame and put them in the cache."""
    if source in ("exp_full", "exp", "exp_reco"):
        if part_key is None:
            raise ValueError(f"{source}: нужен part_key, геометрия у каждого рана своя")
        _, cluster, _ = masks.part_run(part_key)
        xyz = absolute - _centre_h5(source, part_key, cluster)
        # `exp_reco` has a per-event mask (an id-keyed dict from `masks`); an Oms
        # holds one array per part, so it carries none there, and
        # `pnonhit(..., enabled=...)` takes the event's own mask instead
        enabled = (masks.dqm_masks(source, part_key)
                   if source in ("exp_full", "exp") else None)
    else:
        channel = np.arange(len(absolute), dtype=np.int64)
        xyz = absolute - _centre_h5(source, None, None)[
            channel // CHANNELS_PER_CLUSTER]
        enabled = None
        cluster = None
    _cache[key] = (np.ascontiguousarray(xyz, dtype=np.float64),
                   np.arange(len(xyz), dtype=np.int64),
                   None if enabled is None else np.asarray(enabled, dtype=bool),
                   cluster)


def _oms_of(key, source: str, part_key: str | None) -> Oms:
    xyz, channel, enabled, cluster = _cache[key]
    return Oms(source, part_key, channel, xyz, enabled, cluster)
