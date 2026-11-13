"""How much a fitted track expects the modules around it to fire.

`BARS::Muon::PHitChannel` gives P(>=1 p.e.) at one module for one track; this
module runs it over a whole detector and reduces the answer to the numbers BARS
keeps as `fPnonHit` (see `BARS::Muon::PnonHitProduct` and
`BMuonCandidate::GetPnonHitProd`): a track that passes through a region of
silence is judged by how likely that silence is.

Three deliberate differences from BARS, each one a choice rather than an
accident, and each one visible in the numbers:

* **the distance.**  BARS has two copies of the light model: the legacy fit path
  takes the perpendicular distance to the track line, the namespace copy
  (`PHitChannel`) the light path, perpendicular / sin(theta_c).  The light path is
  the physical one and is the default here (`distance="light"`);
  `distance="perpendicular"` is kept for comparing with BARS' own `fPnonHit`.
* **which modules count as empty.**  BARS excludes the channels its final track
  used, and a pulse the track dropped still counts as a silent module.  Nothing
  here knows the fit's hit set, so the caller passes the channels that *fired*:
  `empty = usable and not fired`.  For a pulse the fit dropped, BARS says "quiet"
  where this says "fired" -- the one place the two definitions part.
* **no sentinel.**  When nothing is silent within the radius BARS returns 1e-9,
  the worst value on the scale, which reads as "extremely suspicious" where the
  honest answer is "no evidence either way".  `pnonhit` is NaN there instead, and
  the caller decides.

    from inference.track_fit_alike_bars import isolation, geometry

    om = geometry.oms("exp_full", "part_s2020_c04_r0031")
    field = isolation.phit_field(track, om)            # P(>=1 p.e.) per OM
    iso = isolation.pnonhit(track, om, fired=channels) # pnonhit and counters

    tracks = reader.open_fits().attach(events)         # fit_* columns
    tracks = isolation.attach(tracks)                  # iso_* columns
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from numba import njit

from . import frame as frame_module
from . import geometry
from .geometry import Oms, oms                                   # noqa: F401
from .physics import (SIN_C, phit_light_path, phit_perpendicular,   # noqa: F401
                      phit_table, track_distance)

#: The window BARS uses for "modules around the track", metres.
RADIUS = 22.0

TRACK_FIELDS = ("ref_x", "ref_y", "ref_z", "dir_x", "dir_y", "dir_z")
ISO_FIELDS = ("pnonhit", "n_non_hit", "n_oms", "min_p_silent")


# ── the probability field over a detector ─────────────────────────────────────
@njit(cache=True)
def _distance_light(rx, ry, rz, dx, dy, dz, pos):
    out = np.empty(pos.shape[0])
    for i in range(pos.shape[0]):
        out[i] = track_distance(rx, ry, rz, dx, dy, dz,
                                pos[i, 0], pos[i, 1], pos[i, 2]) / SIN_C
    return out


@njit(cache=True)
def _distance_perp(rx, ry, rz, dx, dy, dz, pos):
    out = np.empty(pos.shape[0])
    for i in range(pos.shape[0]):
        out[i] = track_distance(rx, ry, rz, dx, dy, dz,
                                pos[i, 0], pos[i, 1], pos[i, 2])
    return out


@njit(cache=True)
def _field_light(rx, ry, rz, dx, dy, dz, pos, table):
    out = np.empty(pos.shape[0])
    for i in range(pos.shape[0]):
        out[i] = phit_light_path(rx, ry, rz, dx, dy, dz,
                                 pos[i, 0], pos[i, 1], pos[i, 2], table)
    return out


@njit(cache=True)
def _field_perp(rx, ry, rz, dx, dy, dz, pos, table, npe):
    out = np.empty(pos.shape[0])
    for i in range(pos.shape[0]):
        out[i] = phit_perpendicular(rx, ry, rz, dx, dy, dz,
                                    pos[i, 0], pos[i, 1], pos[i, 2], table, npe)
    return out


def _track(track) -> tuple[np.ndarray, np.ndarray]:
    """`(ref, direction)` out of a `TrackFit`, a mapping or a frame row."""

    def value(name: str) -> float:
        if isinstance(track, Mapping):
            return float(track[name])
        try:
            return float(getattr(track, name))
        except AttributeError:
            raise KeyError(
                f"у трека нет поля {name!r}: нужны {', '.join(TRACK_FIELDS)} "
                f"(TrackFit, строка кадра или словарь)") from None

    ref = np.array([value("ref_x"), value("ref_y"), value("ref_z")])
    direction = np.array([value("dir_x"), value("dir_y"), value("dir_z")])
    return ref, direction


def distances(track, om: Oms, *, distance: str = "light") -> np.ndarray:
    """Distance from every OM of `om` to the track line, metres.

    `light` counts the light path from the emission point (perpendicular /
    sin(theta_c)), `perpendicular` the distance to the line -- the two models
    BARS carries in `PHitChannel` and `GetPHitChan`.
    """
    ref, direction = _track(track)
    if distance == "light":
        return _distance_light(*ref, *direction, om.xyz)
    if distance == "perpendicular":
        return _distance_perp(*ref, *direction, om.xyz)
    raise ValueError(f"distance: {distance!r}, ожидалось 'light' или 'perpendicular'")


def phit_field(track, om: Oms, *, distance: str = "light",
               npe: int = 1) -> np.ndarray:
    """P(>= `npe` p.e.) at every OM of `om` for this track.

    One number per channel of `om`, in `om.channel` order, unconditional on
    whether that module fired.  `npe > 1` exists only for the perpendicular
    model: BARS' light-path copy is the `>= 1` form.
    """
    ref, direction = _track(track)
    table = phit_table()
    if distance == "light":
        if npe != 1:
            raise ValueError("световая модель BARS -- только P(>=1 p.e.); "
                             "для npe>1 используйте distance='perpendicular'")
        return _field_light(*ref, *direction, np.ascontiguousarray(om.xyz), table)
    if distance == "perpendicular":
        return _field_perp(*ref, *direction, np.ascontiguousarray(om.xyz),
                           table, int(npe))
    raise ValueError(f"distance: {distance!r}, ожидалось 'light' или 'perpendicular'")


# ── the event-level summary ───────────────────────────────────────────────────
@dataclass(frozen=True)
class Isolation:
    """What the track says about the silence around it.

    `pnonhit` is the geometric mean of P(silent) over the empty modules of the
    window -- BARS' `fPnonHit` up to the three differences in the module
    docstring.  `n_oms` counts every usable module in the window (BARS'
    `nOMs_22m`), `n_non_hit` the empty ones, and `min_p_silent` the smallest
    P(silent) among them: the one module the track most expected to fire.
    `pnonhit` is NaN when nothing is empty in the window -- no evidence, not the
    worst value.
    """

    pnonhit: float
    n_non_hit: int
    n_oms: int
    min_p_silent: float

    def __repr__(self) -> str:
        return (f"Isolation(pnonhit={self.pnonhit:.4g}, "
                f"n_non_hit={self.n_non_hit}/{self.n_oms}, "
                f"min_p_silent={self.min_p_silent:.3g})")


def pnonhit(track, om: Oms, fired, *, radius: float = RADIUS,
            distance: str = "light", enabled=None) -> Isolation:
    """`Isolation` for one event: `fired` are the channels with a pulse.

    A channel not in `fired` and not switched off is a silent module; a channel
    beyond `radius` of the track is not judged either way.  `enabled` overrides
    the mask of `om`, which the per-event DQM mask of a source like `exp_reco`
    needs (that mask differs from event to event and cannot live in an `Oms`).
    """
    ref, direction = _track(track)
    if np.isnan(ref).any() or np.isnan(direction).any():
        return Isolation(float("nan"), 0, 0, float("nan"))

    field = phit_field(track, om, distance=distance)
    dist = distances(track, om, distance=distance)
    fired_mask = np.zeros(len(om.channel), dtype=bool)
    if fired is not None and len(fired):
        fired_mask[np.isin(om.channel, np.asarray(fired, dtype=np.int64))] = True

    usable = om.usable() if enabled is None else np.asarray(enabled, dtype=bool)
    window = usable & (dist <= radius)
    empty = window & ~fired_mask
    n_oms = int(window.sum())
    if not empty.any():
        return Isolation(float("nan"), 0, n_oms, float("nan"))

    silent = 1.0 - field[empty]
    with np.errstate(divide="ignore"):
        logs = np.log10(silent)
    return Isolation(float(10 ** logs.mean()), int(empty.sum()), n_oms,
                     float(silent.min()))


# ── over a frame of events ────────────────────────────────────────────────────
def attach(events: pd.DataFrame, *, hits=None, fit_prefix: str = "fit_",
           prefix: str = "iso_") -> pd.DataFrame:
    """`events` with the isolation columns joined on, keeping its rows and order.

    `events` carries the addressing (`source`, `part_key`, `local_idx`, and
    `data_class` for the MC sources) and a fitted track per row, as
    `reader.open_fits().attach(ev)` leaves it (`fit_ref_x`, ...).  Pulses are read
    from the HDF5 of each part, once per part; `hits` skips that read when the
    caller already has them -- a mapping `local_idx -> iterable of channels`.

    A row whose fit did not converge keeps NaN in every column -- there is no
    track to judge, and that has to stay distinguishable from a *converged* event
    with nothing in the window, which reports `n_oms = 0` and `n_non_hit = 0`.
    """
    missing = [c for c in ("source", "part_key", "local_idx") if c not in events.columns]
    if missing:
        raise KeyError(f"attach: нет колонок {missing} -- нужен кадр с адресацией "
                       f"(как у reader.events())")
    fields = [fit_prefix + name for name in TRACK_FIELDS]
    if not all(c in events.columns for c in fields):
        raise KeyError(f"attach: нет колонок трека {fields[:3]}... -- "
                       f"сначала reader.open_fits().attach(events)")

    out = events.copy()
    for name, dtype in (("pnonhit", float), ("n_non_hit", float),
                        ("n_oms", float), ("min_p_silent", float)):
        out[prefix + name] = np.full(len(out), np.nan, dtype=dtype)

    tracks = events[[c for c in ("source", "part_key", "local_idx", "data_class")
                     if c in events.columns]]
    for (source, part_key), group in tracks.groupby(["source", "part_key"],
                                                    observed=True, sort=False):
        geometry.prefill(source, part_key, [int(v) for v in group.local_idx])
        if hits is None:
            data_class = getattr(group, "data_class",
                                 pd.Series(source, index=group.index))
            pulses = frame_module.read_event_hits(
                source, str(data_class.iloc[0]), part_key,
                np.sort(group.local_idx.to_numpy()))
            channels = dict(zip(np.sort(group.local_idx.to_numpy()),
                                [[np.asarray(p[5], dtype=np.int64)
                                  for p in event_pulses] for event_pulses in pulses]))
        else:
            channels = {int(k): list(v) for k, v in hits.items()}

        for idx, row in events.loc[group.index].iterrows():
            local_idx = int(row.local_idx)
            track = {name: row[fit_prefix + name] for name in TRACK_FIELDS}
            if not all(np.isfinite(value) for value in track.values()):
                continue                # нет трека: колонки остаются NaN
            om = geometry.oms(source, part_key, local_idx=local_idx)
            iso = pnonhit(track, om, channels.get(local_idx, ()))
            out.loc[idx, prefix + "pnonhit"] = iso.pnonhit
            out.loc[idx, prefix + "n_non_hit"] = iso.n_non_hit
            out.loc[idx, prefix + "n_oms"] = iso.n_oms
            out.loc[idx, prefix + "min_p_silent"] = iso.min_p_silent
    return out
