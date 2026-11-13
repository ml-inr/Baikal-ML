"""Self-checks for the non-neural event-level feature extractor.

Run from the repository root:
    python -m src.data.mu_energy_dataset.test_non_neural_features
"""

from __future__ import annotations

import numpy as np

from .non_neural_features import (
    FEATURE_NAMES,
    compute_non_neural_features,
    feature_groups,
    feature_names,
)


def _synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Event 0: three hits, Q is clipped at 100 for the last hit.
    # Event 1: two hits, all raw Q below the clip.
    features = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 1.0, 0.0, 0.0],
            [120.0, 2.0, 0.0, 1.0, 0.0],
            [50.0, 10.0, 10.0, 0.0, 100.0],
            [50.0, 20.0, 11.0, 0.0, 101.0],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 3, 5], dtype=np.int64)
    n_sig_hits = np.array([3, 2], dtype=np.int32)
    n_sig_strings = np.array([2, 1], dtype=np.int64)
    return features, offsets, n_sig_hits, n_sig_strings


def main() -> None:
    features, offsets, n_sig_hits, n_sig_strings = _synthetic()
    matrix = compute_non_neural_features(
        features,
        offsets,
        n_sig_strings,
        amp_clip=100.0,
        n_sig_hits=n_sig_hits,
    )

    assert matrix.shape == (2, len(FEATURE_NAMES)), matrix.shape
    assert np.isfinite(matrix).all()
    assert feature_names() == list(FEATURE_NAMES)
    groups = feature_groups()
    assert set(groups) == {"counts", "charge", "time", "geometry"}

    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    row0 = matrix[0]
    row1 = matrix[1]

    # Counts and clipped charge.
    assert row0[idx["n_hits"]] == 3.0
    assert row0[idx["sumQ"]] == 103.0
    assert row0[idx["maxQ"]] == 100.0
    assert row0[idx["n_clip"]] == 1.0
    assert np.isclose(row0[idx["frac_clip"]], 1.0 / 3.0)
    assert np.isclose(row0[idx["frac_Q_gt_50"]], 1.0 / 3.0)
    assert row1[idx["sumQ"]] == 100.0
    assert row1[idx["n_clip"]] == 0.0
    assert row1[idx["frac_clip"]] == 0.0

    # Time moments: event 0 has t=[0, 1, 2].
    assert np.isclose(row0[idx["mean_t"]], 1.0)
    assert np.isclose(row0[idx["std_t"]], np.sqrt(2.0 / 3.0))
    assert np.isclose(row0[idx["range_t"]], 2.0)

    # Geometry: event 1 is a line along x and z.
    assert np.isclose(row1[idx["mean_x"]], 10.5)
    assert np.isclose(row1[idx["range_z"]], 1.0)
    assert row1[idx["lambda1"]] >= row1[idx["lambda2"]] >= row1[idx["lambda3"]]
    assert 0.0 <= row1[idx["linearity"]] <= 1.0

    # n_sig_hits is a fatal consistency check, not a silent filter.
    try:
        compute_non_neural_features(
            features,
            offsets,
            n_sig_strings,
            amp_clip=100.0,
            n_sig_hits=np.array([2, 2], dtype=np.int32),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched n_sig_hits was not rejected")

    # A non-positive clip is invalid.
    try:
        compute_non_neural_features(
            features, offsets, n_sig_strings, amp_clip=0.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive amp_clip was not rejected")

    print(f"OK: {matrix.shape[1]} features, {matrix.shape[0]} events")
    print("per-group:", {k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    main()
