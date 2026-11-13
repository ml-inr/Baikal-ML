#!/usr/bin/env python3
"""Checks for the metric core of `report_lib`, against brute force.

`_Cumulative` is where a quiet mistake would be least visible: it turns 21M scores into
every suppression, efficiency and working point in the report, and a wrong answer would
look like a plausible curve. So it is checked against the definition it implements,
computed the slow way, on samples small enough to compute both.

Three properties, over 36 configurations including heavy score ties and samples of 3:

* `pass_fraction(t)` equals `sum(w[score >= t]) / sum(w)` exactly;
* `threshold_for(target)` returns a cut whose OWN pass fraction respects the target
  (not one that overshoots by the weight of a tie block it landed inside), unless the
  target is finer than the sample can express;
* that cut is the loosest one that does: the next lower distinct score overshoots.

Ties are the case worth having a test for. A threshold cannot split events that share a
score, so a naive implementation reading the running cumulative alone names a threshold
whose real pass fraction is larger than asked. `_Cumulative` works from the distinct
scores for exactly this reason, and trials 2 and 5 below are what would catch a
regression.

Usage:  python inference/prefilter/analysis/report_260906_2310/test_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import _Cumulative                                    # noqa: E402

#: (n_events, decimals to round scores to — None for no ties)
CASES = [(20_000, None), (20_000, 2), (97, None), (5, None), (20_000, 1), (3, 1)]
SEEDS = range(6)
TARGETS = np.array([1.0, 0.5, 1e-1, 1e-2, 1e-3, 1e-6])
TOL = 1e-12


def check(n: int, decimals: int | None, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.random(n).astype(np.float32)
    if decimals is not None:
        scores = np.round(scores, decimals).astype(np.float32)
    weights = rng.exponential(1.0, n)
    cumulative = _Cumulative(scores, weights)
    total = weights.sum()

    probes = np.unique(np.concatenate(
        [rng.choice(scores, min(50, n)), [0.0, 1.0, scores.min(), scores.max()]]))
    brute = np.array([weights[scores >= t].sum() / total for t in probes])
    assert np.allclose(cumulative.pass_fraction(probes), brute), \
        f"pass_fraction disagrees with brute force (n={n}, decimals={decimals})"

    # The cuts the class offers must be the cuts it measures.
    assert np.allclose(cumulative.pass_fraction(cumulative.cut_scores),
                       cumulative.cut_fraction), "cut_fraction is not self-consistent"

    floor = cumulative.granularity_floor
    thresholds = cumulative.threshold_for(TARGETS)
    achieved = cumulative.pass_fraction(thresholds)
    for target, threshold, reached in zip(TARGETS, thresholds, achieved):
        assert reached <= target + TOL or reached <= floor + TOL, (
            f"threshold overshoots: target={target}, achieved={reached}, "
            f"granularity floor={floor} (n={n}, decimals={decimals})")
        lower = scores[scores < threshold]
        if lower.size and reached <= target + TOL:
            assert cumulative.pass_fraction(lower.max())[0] > target + TOL, (
                f"a looser cut also respects target={target} — threshold {threshold} "
                f"is not the loosest (n={n}, decimals={decimals})")


def main() -> int:
    for seed in SEEDS:
        for n, decimals in CASES:
            check(n, decimals, seed)
    print(f"{len(SEEDS) * len(CASES)} configurations passed: pass_fraction exact, "
          f"threshold is the loosest existing cut respecting the target, ties handled")

    # The unweighted case must reduce to plain counting, readably.
    plain = _Cumulative(np.array([0.1, 0.4, 0.4, 0.9]), np.ones(4))
    assert plain.pass_fraction(0.4)[0] == 0.75, plain.pass_fraction(0.4)
    assert plain.threshold_for(0.25)[0] == 0.9, plain.threshold_for(0.25)
    assert plain.threshold_for(1.0)[0] == 0.1, plain.threshold_for(1.0)
    # 0.5 is unreachable: the 0.4 block carries three of four events.
    assert plain.threshold_for(0.5)[0] == 0.9, plain.threshold_for(0.5)
    print("unweighted case reduces to counting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
