"""Self-checking script for the isolation tool (`geometry.py`, `isolation.py`).

    python inference/track_fit_alike_bars/tests/test_isolation.py

Each check anchors the new pieces to something outside them:

* the geometry to the hit positions of the HDF5 files, which the converters wrote
  from BARS' own ROOT, not from this loader -- and the per-event slice to its own
  event, the run snapshot to the millimetres of drift measured in the docs;
* the probability field to the single-OM primitives in `physics.py`, which are
  already anchored to the C++ (`PHitChannel`, `GetPHitChan`);
* the summary to a hand-written numpy reduction of the same field, so an
  off-by-one in the window or the empty set cannot hide.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference.track_fit_alike_bars import geometry, isolation  # noqa: E402
from inference.track_fit_alike_bars.frame import read_event_hits  # noqa: E402
from inference.track_fit_alike_bars.physics import (  # noqa: E402
    phit_light_path, phit_perpendicular, phit_table)

RESULTS: list[tuple[str, str]] = []


def check(name):
    def wrap(fn):
        try:
            note = fn()
            RESULTS.append((name, f"ok{'' if not note else '  -- ' + note}"))
        except SkipTest as e:
            RESULTS.append((name, f"skipped: {e}"))
        except AssertionError as e:
            RESULTS.append((name, f"FAILED: {e}"))
        return fn
    return wrap


class SkipTest(Exception):
    pass


def _pulses_of_first_ok(part: str, n: int = 200):
    """Hits of events of `part` -- whatever the first ones with enough pulses are."""
    hits = read_event_hits("exp_full", "exp_full", part, np.arange(n))
    for local_idx, pulses in enumerate(hits):
        if len(pulses[0]) >= 6:
            return local_idx, pulses
    raise SkipTest(f"{part}: нет событий с хитами")


# ── geometry ───────────────────────────────────────────────────────────────────
@check("geometry: OM positions equal the hit positions of the same channels (exp)")
def _():
    part = "part_s2020_c04_r0031"
    local_idx, pulses = _pulses_of_first_ok(part)
    om = geometry.oms("exp_full", part, local_idx=local_idx)
    ch = np.asarray(pulses[5], dtype=np.int64)
    hit = np.column_stack([pulses[2], pulses[3], pulses[4]])
    gap = np.abs(om.xyz[ch] - hit).max()
    # the per-event table is the one BARS used for this very event: the hits must
    # land in their modules to the micron
    assert gap < 1e-3, f"пер-событийная таблица: {gap:.2e} м"
    return f"{gap*1e6:.1f} мкм на {len(ch)} каналах (событие {local_idx})"


@check("geometry: OM positions equal the hit positions of the same channels (MC)")
def _():
    part, group = "part_1000", "muatm_2020"
    om = geometry.oms("mc_merged", part)
    hits = read_event_hits("mc_merged", group, part, np.arange(200))
    for pulses in hits:
        if len(pulses[0]) >= 6:
            break
    else:
        raise SkipTest("нет событий с хитами")
    ch = np.asarray(pulses[5], dtype=np.int64)
    hit = np.column_stack([pulses[2], pulses[3], pulses[4]])
    gap = np.abs(om.xyz[ch] - hit).max()
    assert gap < 1e-4, f"{gap:.2e} м: MC-геометрия статична, расхождения быть не должно"
    assert om.enabled is None, "у МС маски каналов нет"
    return f"{gap*1e6:.2f} мкм, {len(om)} каналов"


@check("geometry: a run snapshot is refused for data, and it is metres off")
def _():
    part = "part_s2020_c04_r0031"
    try:
        geometry.oms("exp_full", part)
    except ValueError:
        pass
    else:
        raise AssertionError("oms() без local_idx не отказался работать на данных")
    snapshot = geometry.oms("exp_full", part, snapshot=True)
    exact = geometry.oms("exp_full", part, local_idx=0)
    gap = np.linalg.norm(snapshot.xyz - exact.xyz, axis=1)
    # the drift over a 3.04 M-event run is metres (see the module docstring):
    # if this ever stops being true, the docstring needs the new number
    assert gap.max() > 0.05, f"снапшот ближе {gap.max()*1000:.0f} мм к событию 0"
    assert snapshot.enabled.sum() == exact.enabled.sum()
    return f"событие 0 отличается от медианы рана на {gap.max()*1000:.0f} мм"


# ── the probability field ──────────────────────────────────────────────────────
@check("field: equals the single-OM primitives elementwise")
def _():
    om = geometry.oms("exp_full", "part_s2020_c04_r0031", snapshot=True)
    rng = np.random.default_rng(7)
    table = phit_table()
    for _ in range(5):
        direction = rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        ref = rng.uniform(-100, 100, 3)
        track = dict(zip(isolation.TRACK_FIELDS, [*ref, *direction]))
        light = isolation.phit_field(track, om, distance="light")
        perp = isolation.phit_field(track, om, distance="perpendicular")
        for i in range(0, len(om), 37):
            p = om.xyz[i]
            a = phit_light_path(*ref, *direction, *p, table)
            b = phit_perpendicular(*ref, *direction, *p, table, 1)
            assert light[i] == a, (i, light[i], a)
            assert perp[i] == b, (i, perp[i], b)
    # границы: ОМ на линии трека и ОМ очень далеко
    track = dict(ref_x=0.0, ref_y=0.0, ref_z=0.0, dir_x=0.0, dir_y=0.0, dir_z=1.0)
    om_near = geometry.Oms("synthetic", None, np.array([0]),
                           np.array([[1e-9, 0.0, 10.0]]), None, None)
    om_far = geometry.Oms("synthetic", None, np.array([0]),
                          np.array([[900.0, 0.0, 10.0]]), None, None)
    near = isolation.phit_field(track, om_near)[0]
    far = isolation.phit_field(track, om_far)[0]
    assert near > 0.9, near
    assert far < 1e-20, far
    return f"на линии {near:.3f}, в 900 м {far:.1e}"


@check("summary: pnonhit equals a hand-written reduction, counters close")
def _():
    om = geometry.oms("exp_full", "part_s2020_c04_r0031", snapshot=True)
    rng = np.random.default_rng(11)
    hub = om.xyz[om.usable()].mean(axis=0)          # трек через сгусток ОМ
    track = dict(zip(isolation.TRACK_FIELDS, [*hub, 0.0, 0.0, 1.0]))
    fired = np.flatnonzero(rng.random(len(om)) < 0.15)

    iso = isolation.pnonhit(track, om, fired=fired)
    field = isolation.phit_field(track, om)
    dist = isolation.distances(track, om)
    usable = om.usable()
    window = usable & (dist <= isolation.RADIUS)
    empty = window & ~np.isin(om.channel, fired)
    assert iso.n_oms == int(window.sum()), (iso.n_oms, int(window.sum()))
    assert iso.n_non_hit == int(empty.sum()), (iso.n_non_hit, int(empty.sum()))
    if empty.any():
        expected = float(10 ** np.log10(1 - field[empty]).mean())
        assert abs(iso.pnonhit - expected) <= 1e-12 * max(expected, 1e-30), \
            (iso.pnonhit, expected)
        assert abs(iso.min_p_silent - float((1 - field[empty]).min())) < 1e-15
        note = (f"окно {iso.n_oms}, пустых {iso.n_non_hit}, "
                f"pnonhit {iso.pnonhit:.4f}")
    else:
        assert np.isnan(iso.pnonhit)
        note = "окно пусто"
    # всё окно помечено сработавшим -> пустых нет -> NaN, а не сентинел BARS 1e-9
    iso_all = isolation.pnonhit(track, om, fired=om.channel[window])
    assert np.isnan(iso_all.pnonhit) and iso_all.n_oms == iso.n_oms
    return note


@check("summary: approaching a group of OMs lowers pnonhit")
def _():
    om = geometry.oms("exp_full", "part_s2020_c04_r0031", snapshot=True)
    on = np.flatnonzero(om.usable())
    hub = om.xyz[on].mean(axis=0)                   # центр облака ОМ
    # два трека вдоль y: один в 60 м от центра (струны через 60 м), другой через него
    far = dict(ref_x=float(hub[0] + 60.0), ref_y=float(hub[1]), ref_z=float(hub[2]),
               dir_x=0.0, dir_y=1.0, dir_z=0.0)
    near = dict(far, ref_x=float(hub[0]))
    iso_far = isolation.pnonhit(far, om, fired=())
    iso_near = isolation.pnonhit(near, om, fired=())
    assert iso_near.n_non_hit > iso_far.n_non_hit, (iso_near, iso_far)
    assert iso_near.pnonhit < iso_far.pnonhit, (iso_near.pnonhit, iso_far.pnonhit)
    return (f"далеко {iso_far.pnonhit:.3f} ({iso_far.n_non_hit} пустых), "
            f"вплотную {iso_near.pnonhit:.3f} ({iso_near.n_non_hit})")


if __name__ == "__main__":
    width = max(len(n) for n, _ in RESULTS)
    for name, outcome in RESULTS:
        print(f"{name:<{width}}  {outcome}")
    failed = [n for n, o in RESULTS if o.startswith("FAILED")]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed or skipped")
    sys.exit(1 if failed else 0)
