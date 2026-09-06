#!/usr/bin/env python3
"""Tests for truth.py: analytic cases, then regression against measured numbers.

The analytic half checks the geometry and the propagation where the answer is known on paper.
The regression half re-derives, from `baikal_mc_merged.h5`, the facts this file's design rests
on -- if any of them stops holding, the companion file is built on a wrong premise and the
build must not proceed.

Checks that need the shower chain (s >= s_track_start, time >= 0) live in the build validation
instead, since they need the extracted npz.

Usage:  python3 test_truth.py [--h5 PATH]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import truth

MAIN_H5 = "/home/albert/Baikal2025/data_manager/data/h5datasets/baikal_mc_merged.h5"
PRODUCTIONS = ("nuatm_2020", "nue2_2020", "muatm_2020")
# Per-part values, which scatter around the file-wide averages (1.00 / 1.00 / 4.09). The
# check is that neutrino productions carry exactly one muon per row while muatm carries a
# bundle, not the precise mean of one part.
EXPECTED_MU_PER_ROW = {"nuatm_2020": (1.00, 1.00), "nue2_2020": (1.00, 1.00),
                       "muatm_2020": (3.0, 6.0)}

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def empty_chain(n_muons: int) -> truth.Chain:
    return truth.build_chain(np.zeros(0), np.zeros(0), np.zeros(n_muons + 1, dtype=np.int64))


def test_cylinder() -> None:
    print("cylinder crossing")
    radius, z_half = 60.0, 265.0
    centre = np.zeros((1, 3))

    # Straight down the axis: the caps set the interval, the radius never binds.
    ref = np.array([[0.0, 0.0, 0.0]])
    d = truth.direction_from_angles(np.array([np.pi]), np.array([0.0]))
    s_in, s_out = truth.cylinder_crossing(ref, d, centre, radius, z_half)
    check("vertical track through the axis spans 2 * z_half",
          np.isclose(s_out[0] - s_in[0], 2 * z_half), f"{s_out[0] - s_in[0]:.3f} m")

    # Horizontal through the centre: the radius sets the interval.
    d = truth.direction_from_angles(np.array([np.pi / 2]), np.array([0.0]))
    s_in, s_out = truth.cylinder_crossing(ref, d, centre, radius, z_half)
    check("horizontal track through the centre spans 2 * radius",
          np.isclose(s_out[0] - s_in[0], 2 * radius), f"{s_out[0] - s_in[0]:.3f} m")

    # Right radius, wrong height: the finite cylinder must reject it.
    ref = np.array([[0.0, 0.0, 400.0]])
    s_in, s_out = truth.cylinder_crossing(ref, d, centre, radius, z_half)
    check("horizontal track above the cap misses", bool(np.isnan(s_in[0])))

    # Beside the cylinder, parallel to the axis.
    ref = np.array([[200.0, 0.0, 0.0]])
    d = truth.direction_from_angles(np.array([np.pi]), np.array([0.0]))
    s_in, s_out = truth.cylinder_crossing(ref, d, centre, radius, z_half)
    check("vertical track outside the radius misses", bool(np.isnan(s_in[0])))

    # Nested volumes must nest: a crossing track enters the larger one first.
    ref = np.array([[10.0, 5.0, 0.0]])
    d = truth.direction_from_angles(np.array([2.0]), np.array([0.7]))
    a_in, a_out = truth.cylinder_crossing(ref, d, centre, 60.0, 265.0)
    b_in, b_out = truth.cylinder_crossing(ref, d, centre, 120.0, 325.0)
    check("the outer volume is entered first and left last",
          bool(b_in[0] <= a_in[0] and b_out[0] >= a_out[0]))


def test_propagation() -> None:
    print("propagation")
    chain = empty_chain(1)
    e_ref = np.array([1000.0])

    s = np.array([100.0])
    check("ionisation only is linear",
          np.isclose(truth.energy_at(chain, e_ref, s)[0],
                     1000.0 - truth.IONISATION_GEV_PER_M * 100.0))

    check("propagating backwards adds energy",
          truth.energy_at(chain, e_ref, np.array([-100.0]))[0] > 1000.0)

    check("energy at the reference point is the reference energy",
          np.isclose(truth.energy_at(chain, e_ref, np.array([0.0]))[0], 1000.0))

    # One 50 GeV shower at s = 10, evaluated on both sides of it.
    chain = truth.build_chain(np.array([10.0]), np.array([50.0]), np.array([0, 1]))
    before = truth.energy_at(chain, e_ref, np.array([9.0]))[0]
    after = truth.energy_at(chain, e_ref, np.array([11.0]))[0]
    check("a shower removes exactly its energy",
          np.isclose(before - after, 50.0 + 2 * truth.IONISATION_GEV_PER_M),
          f"drop {before - after:.4f} GeV across it")

    # Showers given out of order must be handled: the chain sorts them itself. At s = 11 the
    # showers at -20 and 10 have been crossed, the one at 40 has not -- and the one upstream
    # of the reference point adds energy rather than removing it.
    shuffled = truth.build_chain(np.array([40.0, -20.0, 10.0]), np.array([5.0, 7.0, 50.0]),
                                 np.array([0, 3]))
    got = truth.energy_at(shuffled, e_ref, np.array([11.0]))[0]
    expected = 1000.0 - truth.IONISATION_GEV_PER_M * 11.0 - 50.0
    check("an unsorted chain is sorted internally",
          np.isclose(got, expected), f"{got:.4f} vs {expected:.4f}")

    # Evaluated upstream of the shower at s = -20, that shower has NOT yet happened, so its
    # energy is added back: e_ref already has it subtracted.
    got = truth.energy_at(shuffled, e_ref, np.array([-30.0]))[0]
    expected = 1000.0 + truth.IONISATION_GEV_PER_M * 30.0 + 7.0
    check("a shower upstream of the reference point adds energy going backwards",
          np.isclose(got, expected), f"{got:.4f} vs {expected:.4f}")

    # Round trip: from the reference point to a point and back.
    e_far = truth.energy_at(shuffled, e_ref, np.array([100.0]))
    back = truth.energy_between(shuffled, e_far, np.array([100.0]), np.array([0.0]))
    check("forward then backward returns the reference energy",
          np.isclose(back[0], 1000.0, atol=1e-9), f"{back[0]:.12f}")

    # Several muons at once must not leak sums into each other.
    multi = truth.build_chain(np.array([5.0, 5.0]), np.array([100.0, 200.0]),
                              np.array([0, 1, 2]))
    got = truth.energy_at(multi, np.array([1000.0, 1000.0]), np.array([10.0, 10.0]))
    check("chains of different muons stay separate",
          np.allclose(got, [1000.0 - 2.4 - 100.0, 1000.0 - 2.4 - 200.0]), str(np.round(got, 3)))


def test_status() -> None:
    print("entry status")
    chain = empty_chain(4)
    e_ref = np.array([1000.0, 1000.0, 1.0, truth.SENTINEL_E_REF], dtype=np.float32)
    e_reg = np.array([0.0, 0.0, 0.0, 500.0])
    s_start = np.array([-500.0, 20.0, -500.0, 10.0])       # muon 1 begins inside the volume
    s_in = np.array([10.0, 10.0, 10.0, 5.0])
    s_out = np.array([50.0, 50.0, 50.0, 50.0])
    e, st = truth.entry_energy(chain, e_ref.astype(np.float64), s_start, e_reg, s_in, s_out)
    check("ordinary muon is propagated", st[0] == truth.STATUS_PROPAGATED)
    check("muon starting inside is flagged", st[1] == truth.STATUS_BORN_INSIDE)
    check("a muon that cannot reach the volume alive is flagged dead",
          st[2] == truth.STATUS_DIED_BEFORE and np.isnan(e[2]))
    check("a sentinel born inside gets its own code", st[3] == truth.STATUS_SENTINEL_INSIDE)

    nan = np.array([np.nan])
    e, st = truth.entry_energy(empty_chain(1), np.array([1000.0]), np.array([-100.0]),
                               np.array([0.0]), nan, nan)
    check("a missed volume is flagged and its energy is NaN",
          st[0] == truth.STATUS_NO_CROSSING and np.isnan(e[0]))


def test_against_main_file(path: str) -> None:
    print(f"regression against {path}")
    try:
        import h5py
        f = h5py.File(path, "r")
    except Exception as exc:
        print(f"  SKIP  cannot open the main file: {exc}")
        return

    for p in PRODUCTIONS:
        pt = sorted(f[f"{p}/ev_ids"].keys())[0]
        ind = f[f"{p}/muons_prty/individ/{pt}/data"][:]
        ms = f[f"{p}/muons_prty/mu_starts/{pt}/data"][:]
        d = truth.direction_from_angles(ind[:, 0], ind[:, 1])
        ref = ind[:, 2:5].astype(np.float64)
        e_ref, t_first = ind[:, 6], ind[:, 5]

        # The reference point is the closest approach to the origin of the global frame:
        # the position vector there is perpendicular to the track.
        s0 = np.abs(np.einsum("ij,ij->i", ref, d))
        check(f"{p}: reference point is the closest approach to the array origin",
              float(s0.max()) < 0.01, f"worst {s0.max() * 1000:.2f} mm over {len(s0):,} muons")

        # The sentinel means "not yet born at the reference point", so its start lies ahead.
        s_start = truth.track_start(t_first)
        sent = e_ref == np.float32(truth.SENTINEL_E_REF)
        if sent.any():
            check(f"{p}: every sentinel muon starts downstream of the reference point",
                  bool((s_start[sent] > 0).all()), f"{sent.sum():,} sentinels")
        # <= rather than <: a muon may begin exactly at the reference point, giving -0.0.
        check(f"{p}: no non-sentinel muon starts downstream",
              bool((s_start[~sent] <= 0).all()), f"{(~sent).sum():,} muons")

        got = ms[-1] / (len(ms) - 1)
        lo, hi = EXPECTED_MU_PER_ROW[p]
        check(f"{p}: muons per row", lo - 0.01 <= got <= hi + 0.01, f"{got:.2f}")

        # Angles are radians here and degrees in prime_prty -- a trap worth a standing test.
        check(f"{p}: muon angles are radians", float(np.abs(ind[:, 0]).max()) <= np.pi + 1e-6,
              f"max theta {ind[:, 0].max():.4f}")
    f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=MAIN_H5)
    args = ap.parse_args()

    test_cylinder()
    test_propagation()
    test_status()
    test_against_main_file(args.h5)

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: " + "; ".join(_failures))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
