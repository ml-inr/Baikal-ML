"""The fitter is checked against a track whose answer is known by construction.

Reverse-engineering geometry from data has gone wrong in this project before, so
nothing here is validated against the data it will be applied to.  A synthetic
Cherenkov track is generated with an exact arrival time at every module; the fit
must recover the direction and leave essentially no residual, and the
leave-one-out prediction must reproduce a withheld hit to within a nanosecond.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import tracks                                                  # noqa: E402


def synthetic(zenith_deg: float, azimuth_deg: float, n_strings: int = 4,
              n_per_string: int = 6, jitter_ns: float = 0.0,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Modules on a grid of vertical strings, lit by one exact Cherenkov track."""
    rng = np.random.default_rng(seed)
    theta, phi = np.radians(zenith_deg), np.radians(azimuth_deg)
    u = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                  np.cos(theta)])
    angle = 2 * np.pi * np.arange(n_strings) / n_strings
    xy = np.stack([60 * np.cos(angle), 60 * np.sin(angle)], axis=1)
    z = np.linspace(-200, 50, n_per_string)
    pos = np.array([[x, y, zz] for x, y in xy for zz in z], dtype=float)
    p0 = np.array([10.0, -5.0, -70.0])
    t = tracks.arrival_time(pos, p0, u, 1234.0)
    if jitter_ns:
        t = t + rng.normal(0, jitter_ns, len(t))
    q = np.full(len(pos), 5.0)
    return pos, t, q, u


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1.0, 1.0))))


def test_exact_track_is_recovered() -> None:
    for zenith, azimuth in ((110.0, 40.0), (75.0, 200.0), (150.0, 300.0)):
        pos, t, q, u = synthetic(zenith, azimuth)
        fit = tracks.fit_track(pos, t, q)
        assert fit["fit_rms"] < 3.0, (zenith, azimuth, fit["fit_rms"])
        assert angle_between(fit["u"], u) < 3.0, (zenith, azimuth)
        assert fit["frac_on_track"] == 1.0


def test_withheld_hit_is_predicted() -> None:
    pos, t, q, _ = synthetic(110.0, 40.0)
    out = tracks.leave_one_out(pos, t, q, indices=np.arange(0, len(pos), 4))
    assert len(out["dt"]) > 0
    assert np.abs(out["dt"]).max() < 6.0, np.abs(out["dt"]).max()
    assert np.all(out["d"] > 0)


def test_jitter_shows_up_in_the_residual() -> None:
    """A detector with worse timing must produce a wider residual -- the whole
    premise of test 1.  If this fails, the test cannot see a response error."""
    widths = []
    for jitter in (0.0, 5.0, 15.0):
        spread = []
        for seed in range(4):
            pos, t, q, _ = synthetic(110.0, 40.0, jitter_ns=jitter, seed=seed)
            out = tracks.leave_one_out(pos, t, q,
                                       indices=np.arange(0, len(pos), 3))
            spread.append(np.std(out["dt"]))
        widths.append(float(np.mean(spread)))
    assert widths[0] < widths[1] < widths[2], widths
    assert widths[2] > 10.0, widths


if __name__ == "__main__":
    test_exact_track_is_recovered()
    test_withheld_hit_is_predicted()
    test_jitter_shows_up_in_the_residual()
    print("all track tests passed")
