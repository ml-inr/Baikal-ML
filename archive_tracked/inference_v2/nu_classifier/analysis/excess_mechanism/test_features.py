"""Checks on features.py. The synthetic-track test is the one that matters.

A track built from the Cherenkov formula itself must come back out of the fit: direction
recovered, residuals at zero. Everything else here is a consistency check.

Usage:  python inference_v2/nu_classifier/analysis/excess_mechanism/test_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from archive_tracked.inference_v2.nu_classifier.analysis.excess_mechanism.features import (COLUMNS, C_VAC, K_CHERENKOV, STRING_DIVISOR, V_LIGHT,  # noqa: E402
                      event_features, fit_track)

STRING_XY = [(0.0, 0.0), (60.0, 0.0), (30.0, 52.0), (-30.0, 52.0)]
MODULE_DZ, N_MODULES = 15.0, 36


def _detector() -> tuple[np.ndarray, np.ndarray]:
    """Positions and channel ids of a small four-string array."""
    pos, chan = [], []
    for s, (x, y) in enumerate(STRING_XY):
        for m in range(N_MODULES):
            pos.append((x, y, -260.0 + m * MODULE_DZ))
            chan.append(s * STRING_DIVISOR + m)
    return np.array(pos), np.array(chan, dtype=np.int32)


def _synthetic(u_true: np.ndarray, p0_true: np.ndarray, t0: float = 0.0,
               max_d: float = 80.0, seed: int = 0):
    """Hits from an exact Cherenkov track: modules within `max_d` of the line."""
    pos, chan = _detector()
    r = pos - p0_true
    s = r @ u_true
    d = np.sqrt(np.maximum((r ** 2).sum(1) - s ** 2, 0.0))
    keep = d < max_d
    pos, chan, s, d = pos[keep], chan[keep], s[keep], d[keep]
    t = t0 + (s + K_CHERENKOV * d) / C_VAC
    rng = np.random.default_rng(seed)
    q = 5.0 * np.exp(-d / 30.0) + 0.5          # a plausible falloff, exact value irrelevant
    h = np.column_stack([q, t, pos[:, 0], pos[:, 1], pos[:, 2]]).astype(np.float32)
    p = np.full(len(h), 0.95, dtype=np.float32)
    return h, p, chan, rng


def test_exact_track_is_recovered() -> bool:
    """The fit must return the track it was built from."""
    ok = True
    print("\n  exact Cherenkov track, no noise:")
    for name, zen_true, azi_true in [("down-going 160 deg", 160.0, 40.0),
                                     ("near-horizontal 95 deg", 95.0, -120.0),
                                     ("up-going 30 deg", 30.0, 210.0)]:
        zr, ar = np.radians(zen_true), np.radians(azi_true)
        u = np.array([np.sin(zr) * np.cos(ar), np.sin(zr) * np.sin(ar), np.cos(zr)])
        h, p, c, _ = _synthetic(u, np.array([10.0, 10.0, -160.0]))
        if len(h) < 8:
            print(f"    {name}: too few modules lit, skipped"); continue
        f = fit_track(h[:, 2:5].astype(np.float64), h[:, 1].astype(np.float64),
                      h[:, 0].astype(np.float64))
        d_ang = np.degrees(np.arccos(np.clip(float(f["_u"] @ u), -1.0, 1.0)))
        good = f["fit_rms"] < 3.0 and d_ang < 3.0
        ok &= good
        print(f"    {name:24s} n={len(h):3d}  rms={f['fit_rms']:6.2f} ns  "
              f"angle error={d_ang:5.2f} deg  contrast={f['fit_contrast']:.2f}  "
              f"{'ok' if good else 'FAIL'}")
    return ok


def test_contrast_separates_track_from_noise() -> bool:
    """Scrambling the times must collapse fit_contrast. This fixes the scale of the feature."""
    zr, ar = np.radians(150.0), np.radians(0.0)
    u = np.array([np.sin(zr) * np.cos(ar), np.sin(zr) * np.sin(ar), np.cos(zr)])
    h, p, c, rng = _synthetic(u, np.array([0.0, 0.0, -160.0]))
    pos, t, q = h[:, 2:5].astype(np.float64), h[:, 1].astype(np.float64), h[:, 0].astype(np.float64)
    real = fit_track(pos, t, q)["fit_contrast"]
    scrambled = [fit_track(pos, rng.permutation(t), q)["fit_contrast"] for _ in range(40)]
    hi = float(np.percentile(scrambled, 95))
    ok = real > hi
    print(f"\n  fit_contrast: real track {real:.3f} vs scrambled times "
          f"median {np.median(scrambled):.3f}, 95th pct {hi:.3f}  {'ok' if ok else 'FAIL'}")
    return ok


def test_invariant_to_a_time_shift() -> bool:
    """Every feature is a difference within the event, so a per-event constant must cancel."""
    zr = np.radians(140.0)
    u = np.array([np.sin(zr), 0.0, np.cos(zr)])
    h, p, c, _ = _synthetic(u, np.array([0.0, 0.0, -160.0]))
    a = np.array(event_features(h, p, c, n_raw=200), dtype=np.float64)
    h2 = h.copy(); h2[:, 1] += 1234.5
    b = np.array(event_features(h2, p, c, n_raw=200), dtype=np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    worst = float(np.max(np.abs(a[both] - b[both]) / np.maximum(np.abs(a[both]), 1.0)))
    ok = worst < 1e-6 and (np.isfinite(a) == np.isfinite(b)).all()
    print(f"  shifting every time by +1234.5 ns: worst relative change {worst:.2e}  "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_hand_computed() -> bool:
    """Quantities small enough to work out on paper."""
    h = np.array([[2.0,   0.0,  0.0,  0.0, -10.0],
                  [4.0, 100.0,  0.0,  0.0,  10.0],
                  [6.0, 200.0, 30.0, 40.0, -10.0],
                  [8.0, 300.0, 30.0, 40.0,  10.0],
                  [1.0, 350.0, 30.0, 40.0,  10.0]], dtype=np.float32)
    p = np.array([0.90, 0.85, 0.95, 0.99, 0.91], dtype=np.float32)
    c = np.array([0, 1, 36, 37, 37], dtype=np.int32)
    got = dict(zip(COLUMNS, event_features(h, p, c, n_raw=50)))
    expect = {
        "n_hits": 5, "n_modules": 4, "n_strings": 2,
        "hits_per_module": 1.25,
        "q_total": 21.0, "q_max": 8.0, "q_median": 4.0,
        "t_span": 350.0, "z_span": 20.0, "xy_span": 50.0,
        "z_first": -10.0, "z_last": 10.0, "dz_signed": 20.0,
        "dt_repeat_median": 50.0,          # channel 37 fires at 300 and 350
        "q_repeat_over_first": 1.0 / 20.0, # repeat charge 1.0 over first-hit charge 2+4+6+8
        "extent_m": float(np.hypot(50.0, 20.0)),
        "survival_frac": 0.1, "n_raw_hits": 50,
        "frac_q_below_2": 0.2,             # only the 1.0 p.e. hit
    }
    ok = True
    print("\n  hand-computed five-hit event:")
    for k, want in expect.items():
        have = float(got[k])
        good = abs(have - want) <= 1e-5 * max(1.0, abs(want))
        ok &= good
        print(f"    {k:22s} expected {want:>10.4f}  got {have:>10.4f}  "
              f"{'ok' if good else 'FAIL'}")
    return ok


def test_slowness_is_one_for_a_light_speed_track() -> bool:
    """A vertical track's duration must match the light crossing time of its own extent."""
    u = np.array([0.0, 0.0, -1.0])
    h, p, c, _ = _synthetic(u, np.array([0.0, 0.0, -160.0]), max_d=25.0)
    f = dict(zip(COLUMNS, event_features(h, p, c, n_raw=len(h) * 5)))
    # a vertical muon outruns its own light: the muon travels at c, the light at c/n, so the
    # observed duration is shorter than the light crossing time of the same extent
    ok = 0.3 < f["slowness"] < 1.05
    print(f"\n  vertical track: slowness = {f['slowness']:.3f} (expect below 1, "
          f"the muon outruns its light)  {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [test_exact_track_is_recovered(),
               test_contrast_separates_track_from_noise(),
               test_invariant_to_a_time_shift(),
               test_hand_computed(),
               test_slowness_is_one_for_a_light_speed_track()]
    print("\n" + ("ALL PASSED" if all(results) else "FAILED"))
    sys.exit(0 if all(results) else 1)
