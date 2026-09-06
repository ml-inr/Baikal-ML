"""Checks on the time-versus-depth direction estimator.

The estimator has to survive two kinds of mistake. It could recover a slope that is simply
wrong — caught here by feeding tracks whose slope is known by construction. Or it could
return a confident number for an event that cannot support one, such as hits spread over no
depth at all, which is caught by requiring a refusal rather than a value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from build_direction import SPEED_OF_LIGHT_M_PER_NS, fit_time_against_depth


def synthetic_track(vertical_speed_m_per_ns: float, depths: np.ndarray) -> np.ndarray:
    """Times for light sweeping the array at a given vertical speed."""
    return depths / vertical_speed_m_per_ns


def test_recovers_a_known_slope() -> bool:
    depths = np.linspace(-200.0, 200.0, 40)
    ok = True

    # travelling downward: the shallowest hits happen first, so time falls as depth rises
    downward_times = synthetic_track(-SPEED_OF_LIGHT_M_PER_NS, depths)
    slope, quality, spread, speed = fit_time_against_depth(downward_times, depths)
    expected_slope = -1.0 / SPEED_OF_LIGHT_M_PER_NS
    slope_ok = abs(slope - expected_slope) < 1e-9
    quality_ok = abs(quality - 1.0) < 1e-9
    speed_ok = abs(speed - SPEED_OF_LIGHT_M_PER_NS) < 1e-9
    ok &= slope_ok and quality_ok and speed_ok
    print(f"  downward track: slope {slope:.4f} ns/m (expected {expected_slope:.4f})  "
          f"{'ok' if slope_ok else 'FAIL'}")
    print(f"  perfect line gives quality {quality:.6f}                      "
          f"{'ok' if quality_ok else 'FAIL'}")
    print(f"  implied speed {speed:.6f} m/ns equals c                     "
          f"{'ok' if speed_ok else 'FAIL'}")

    upward_times = synthetic_track(SPEED_OF_LIGHT_M_PER_NS, depths)
    upward_slope, _, _, _ = fit_time_against_depth(upward_times, depths)
    sign_flips = upward_slope > 0 > slope
    ok &= sign_flips
    print(f"  upward track flips the sign: {upward_slope:+.4f} vs {slope:+.4f}   "
          f"{'ok' if sign_flips else 'FAIL'}")
    return ok


def test_refuses_when_depth_carries_no_information() -> bool:
    """Hits on one horizontal plane cannot say anything about direction."""
    depths = np.full(20, 42.0)
    times = np.linspace(0.0, 100.0, 20)
    slope, quality, spread, speed = fit_time_against_depth(times, depths)
    refused = np.isnan(slope) and np.isnan(quality)
    zero_spread = spread == 0.0
    print(f"\n  flat event returns no slope                              "
          f"{'ok' if refused else 'FAIL'}")
    print(f"  and reports zero depth spread                             "
          f"{'ok' if zero_spread else 'FAIL'}")
    return refused and zero_spread


def test_noise_lowers_quality_without_moving_the_slope() -> bool:
    """Scatter must show up in the quality figure, not as a biased slope."""
    generator = np.random.default_rng(0)
    depths = np.linspace(-200.0, 200.0, 200)
    clean_times = synthetic_track(-SPEED_OF_LIGHT_M_PER_NS, depths)
    noisy_times = clean_times + generator.normal(0.0, 20.0, depths.size)

    clean_slope, clean_quality, _, _ = fit_time_against_depth(clean_times, depths)
    noisy_slope, noisy_quality, _, _ = fit_time_against_depth(noisy_times, depths)

    slope_stable = abs(noisy_slope - clean_slope) < 0.2
    quality_dropped = noisy_quality < clean_quality
    print(f"\n  noise leaves the slope near the truth: {noisy_slope:.3f} vs {clean_slope:.3f}  "
          f"{'ok' if slope_stable else 'FAIL'}")
    print(f"  and shows up as lower quality: {noisy_quality:.3f} < {clean_quality:.3f}      "
          f"{'ok' if quality_dropped else 'FAIL'}")
    return slope_stable and quality_dropped


def test_reports_the_slope_it_was_given() -> bool:
    """Even a physically impossible input must come back unchanged, not clipped.

    A real track sweeps depth at c*cos(theta), so |dt/dz| is never below 1/c and the implied
    speed never exceeds c — but that bound belongs to the events, not to the arithmetic. If
    the estimator quietly clamped, a genuinely impossible event would look ordinary instead
    of announcing that something upstream is wrong.
    """
    depths = np.linspace(-100.0, 100.0, 30)
    impossible_times = depths * 0.5 / SPEED_OF_LIGHT_M_PER_NS   # twice light speed in depth
    slope, _, _, speed = fit_time_against_depth(impossible_times, depths)
    faithful = abs(speed - 2 * SPEED_OF_LIGHT_M_PER_NS) < 1e-9
    print(f"\n  impossible input reported as-is: {speed:.3f} m/ns = 2c        "
          f"{'ok' if faithful else 'FAIL — estimator is clamping'}")
    print("  (the physical bound speed <= c is checked on real MC events, not here)")
    return faithful


if __name__ == "__main__":
    outcomes = [test_recovers_a_known_slope(),
                test_refuses_when_depth_carries_no_information(),
                test_noise_lowers_quality_without_moving_the_slope(),
                test_reports_the_slope_it_was_given()]
    print("\n" + ("ALL PASSED" if all(outcomes) else "FAILED"))
    sys.exit(0 if all(outcomes) else 1)
