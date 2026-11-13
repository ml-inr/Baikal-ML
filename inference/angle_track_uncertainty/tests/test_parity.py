#!/usr/bin/env python3
"""Self-check for the angle/track/uncertainty integration.

Builds a synthetic source + SN-probs + catalog, then checks:

1. the released bundle and SN checkpoint have the documented SHA256 values;
2. the compact model and release-SN loaders work;
3. the streaming adapter (``predict.py``) agrees with the vendored one-pass CSV
   script on the same selected hits;
4. the adapter's conditional gate agrees with the vendored rethresholded script.

Run from the repository root:

    python inference/angle_track_uncertainty/tests/test_parity.py
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
BUNDLE_DIR = ROOT / "trained_models/angle_track_uncertainty/release_v3"
BUNDLE = BUNDLE_DIR / "angle_track_uncertainty.pt"
SN = BUNDLE_DIR / "signal_noise.ckpt"
EXPECTED_BUNDLE_SHA = "7cb8f77cfdb2a7cf021d090be5246b83f8a94c7513cc24c793ab019c67686f41"
EXPECTED_SN_SHA = "be9cc39b93dd6a57a69233faf6755278e3f202c56de3de7d1cc8d2d4a8b1e93d"

sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixture(directory: Path) -> tuple[Path, Path, Path]:
    source = directory / "synthetic_mc.h5"
    probs = directory / "synthetic_probs.h5"
    catalog = directory / "synthetic_catalog.duckdb"
    rng = np.random.default_rng(0)
    n_events, n_hits = 3, 15
    raw = rng.normal(size=(n_events * n_hits, 5)).astype(np.float32)
    raw[:, 0] = rng.uniform(1, 50, n_events * n_hits)
    channels = np.tile(
        np.array([0, 36, 1, 37, 2, 38, 3, 39, 4, 40, 5, 41, 6, 42, 7], dtype=np.int32),
        n_events,
    )
    ev_starts = np.arange(0, (n_events + 1) * n_hits, n_hits, dtype=np.int64)
    with h5py.File(source, "w") as h5:
        group = h5.create_group("muatm_2020")
        group.create_dataset("raw/data/part_0/data", data=raw)
        group.create_dataset("raw/ev_starts/part_0/data", data=ev_starts)
        group.create_dataset("raw/channels/part_0/data", data=channels)
        group.create_dataset(
            "ev_ids/part_0/data",
            data=np.array([b"muatm_0_0", b"muatm_0_1", b"muatm_0_2"]),
        )
    with h5py.File(probs, "w") as h5:
        group = h5.create_group("muatm_2020")
        # 0.99 on the first 10 hits, 0.60 on the last 5: the p>=0.50 gate
        # adds hits to every event.
        group.create_dataset(
            "probs/part_0/data",
            data=np.tile(
                np.array([0.99] * 10 + [0.60] * 5, dtype=np.float32), n_events
            ),
        )
    if catalog.exists():
        catalog.unlink()
    conn = duckdb.connect(str(catalog))
    conn.execute(
        "CREATE TABLE events(id BIGINT, source VARCHAR, data_class VARCHAR, "
        "season INTEGER, run VARCHAR, event_id BIGINT)"
    )
    conn.executemany(
        "INSERT INTO events VALUES (?,?,?,?,?,?)",
        [
            (1000 + index, "mc_merged", "muatm_2020", 2020, "part_0", index)
            for index in range(n_events)
        ],
    )
    conn.execute(
        "CREATE TABLE h5_locations(event_fk BIGINT, h5_path VARCHAR, "
        "part_key VARCHAR, local_idx INTEGER)"
    )
    conn.executemany(
        "INSERT INTO h5_locations VALUES (?,?,?,?)",
        [
            (1000 + index, str(source), "part_0", index)
            for index in range(n_events)
        ],
    )
    conn.close()
    return source, probs, catalog


def run(module: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ROOT,
        check=True,
    )


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="") as stream:
        return {row["event_id"]: row for row in csv.DictReader(stream)}


def main() -> None:
    if not BUNDLE.is_file() or not SN.is_file():
        print("SKIP: release weights are not present")
        return
    assert sha256(BUNDLE) == EXPECTED_BUNDLE_SHA, "bundle SHA256 mismatch"
    assert sha256(SN) == EXPECTED_SN_SHA, "SN checkpoint SHA256 mismatch"
    print("PASS bundle SHA256")

    from trained_models.angle_track_uncertainty.release_v3.model import load_bundle
    from trained_models.angle_track_uncertainty.release_v3.sn_model import (
        load_release_sn,
    )

    model, bundle = load_bundle(BUNDLE, device="cpu")
    x = torch.randn(2, 16, 5)
    mask = torch.ones(2, 16, dtype=torch.bool)
    with torch.inference_mode():
        output = model(x, mask)
    assert [tuple(item.shape) for item in output] == [(2, 3), (2, 3), (2,), (2,)]
    assert torch.allclose(output[0].norm(dim=1), torch.ones(2), atol=1e-5)
    print("PASS compact model load/forward")

    sn_model = load_release_sn(SN, device="cpu")
    with torch.inference_mode():
        probs = torch.sigmoid(sn_model(x, mask)[..., 1])
    assert torch.isfinite(probs).all() and probs.min() >= 0 and probs.max() <= 1
    print("PASS release SN load/forward")

    from inference.angle_track_uncertainty.common import (
        release_sn_probs,
        select_events,
    )

    class EchoModel(torch.nn.Module):
        """Return the first feature as the signal logit, ignoring batching."""

        def forward(self, values: torch.Tensor, event_mask: torch.Tensor):
            return torch.stack(
                [torch.zeros_like(values[..., 0]), values[..., 0]], dim=-1
            )

    n_events = 7
    lengths = np.array([5, 3, 8, 1, 6, 2, 4], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    flat = np.arange(1, starts[-1] + 1, dtype=np.float32)[:, None]
    raw = np.concatenate(
        [flat, np.zeros((flat.size, 4), dtype=np.float32)], axis=1
    )
    mean = np.zeros(5, dtype=np.float32)
    std = np.ones(5, dtype=np.float32)
    mapped = release_sn_probs(
        EchoModel(), raw, starts, mean, std, 2, torch.device("cpu")
    )
    expected = 1.0 / (1.0 + np.exp(-flat[:, 0]))
    assert np.allclose(mapped, expected, atol=1e-6), (mapped, expected)
    shifted = release_sn_probs(
        EchoModel(), raw, starts + 1000, mean, std, 2, torch.device("cpu")
    )
    assert np.allclose(shifted, expected, atol=1e-6), (shifted, expected)
    print("PASS release SN batch mapping")

    boundary_raw = np.zeros((2, 5), dtype=np.float32)
    boundary_channels = np.array([0, 36], dtype=np.int32)
    boundary_starts = np.array([0, 2], dtype=np.int64)
    boundary_probs = np.array([0.70, 0.70], dtype=np.float32)
    selected = select_events(
        boundary_raw, boundary_probs, boundary_channels, boundary_starts, None,
        threshold=0.70, min_hits=2, min_strings=2,
    )
    assert len(selected) == 1, selected
    print("PASS threshold boundary uses >=")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source, probs_h5, catalog = build_fixture(tmp)

        event_set_db = tmp / "event_set.duckdb"
        conn = duckdb.connect(str(event_set_db))
        conn.execute("CREATE TABLE predictions(event_fk BIGINT PRIMARY KEY)")
        conn.executemany("INSERT INTO predictions VALUES (?)", [(1000,), (1002,)])
        conn.close()
        from inference.angle_track_uncertainty.common import (
            event_set_indices,
            filter_event_set,
        )

        event_set, stats = event_set_indices(
            event_set_db, catalog, "mc_merged"
        )
        local_idx, event_fk = event_set[("muatm_2020", "part_0")]
        assert local_idx.tolist() == [0, 2], local_idx
        assert event_fk.tolist() == [1000, 1002], event_fk
        assert stats["n_event_fk_resolved"] == 2, stats
        kept, not_selected = filter_event_set(
            event_set[("muatm_2020", "part_0")],
            0,
            3,
            [
                {"index": 0},
                {"index": 1},
                {"index": 2},
            ],
        )
        assert [record["index"] for record in kept] == [0, 2], kept
        assert not_selected == 0, not_selected
        print("PASS event-set restriction")

        subset_preds = tmp / "subset_preds"
        run(
            "inference.angle_track_uncertainty.predict",
            "--source", str(source),
            "--source-name", "mc_merged",
            "--probs-h5", str(probs_h5),
            "--catalog", str(catalog),
            "--event-set-db", str(event_set_db),
            "--output-dir", str(subset_preds),
            "--device", "cpu",
            "--threads", "2",
        )
        subset_db = (
            subset_preds
            / "release_v3@angle_track_uncertainty"
            / "mc_merged_canonical_p070_track.duckdb"
        )
        conn = duckdb.connect(str(subset_db), read_only=True)
        subset = conn.execute(
            "SELECT event_fk, gate_applied, selected_threshold "
            "FROM predictions ORDER BY event_fk"
        ).df()
        conn.close()
        assert subset["event_fk"].tolist() == [1000, 1002], subset
        assert not subset["gate_applied"].any(), subset
        assert (subset["selected_threshold"] == 0.70).all(), subset
        print("PASS adapter event-set restriction")

        prepared_p070 = tmp / "selected_p070.h5"
        prepared_p050 = tmp / "selected_p050.h5"
        for prepared, threshold in ((prepared_p070, "0.70"), (prepared_p050, "0.50")):
            run(
                "inference.angle_track_uncertainty.prepare_selected_h5",
                "--source", str(source),
                "--source-name", "mc_merged",
                "--probs-h5", str(probs_h5),
                "--output", str(prepared),
                "--threshold", threshold,
                "--device", "cpu",
                "--threads", "2",
            )
            assert prepared.is_file(), "prepared HDF5 was not written"
        print("PASS prepared-HDF5 conversion")

        one_pass_csv = tmp / "one_pass.csv"
        run(
            "inference.angle_track_uncertainty.predict_one_pass",
            "--checkpoint", str(BUNDLE),
            "--data", str(prepared_p070),
            "--split", "test",
            "--output", str(one_pass_csv),
            "--device", "cpu",
            "--threads", "2",
        )
        one_pass = read_csv(one_pass_csv)
        assert len(one_pass) == 3, one_pass

        preds = tmp / "preds"
        run(
            "inference.angle_track_uncertainty.predict",
            "--source", str(source),
            "--source-name", "mc_merged",
            "--probs-h5", str(probs_h5),
            "--catalog", str(catalog),
            "--output-dir", str(preds),
            "--device", "cpu",
            "--threads", "2",
        )
        db_path = (
            preds
            / "release_v3@angle_track_uncertainty"
            / "mc_merged_canonical_p070_track.duckdb"
        )
        conn = duckdb.connect(str(db_path), read_only=True)
        rows = conn.execute(
            "SELECT event_id, sn_mode, theta_deg, phi_deg, direction_x, "
            "direction_y, direction_z, anchor_x_m, anchor_y_m, anchor_z_m, "
            "sigma_angle_deg, sigma_anchor_m, angle_r68_deg, "
            "uncertainty_calibration_scope FROM predictions ORDER BY event_id"
        ).df()
        conn.close()
        assert len(rows) == 3, rows
        assert (rows["sn_mode"] == "canonical").all()
        assert rows["angle_r68_deg"].isna().all(), rows
        assert (
            rows["uncertainty_calibration_scope"]
            == "canonical_sn_selection_unvalidated"
        ).all(), rows
        for row in rows.itertuples(index=False):
            reference = one_pass[row.event_id]
            for column in (
                "theta_deg", "phi_deg", "direction_x", "direction_y",
                "direction_z", "anchor_x_m", "anchor_y_m", "anchor_z_m",
                "sigma_angle_deg", "sigma_anchor_m",
            ):
                difference = abs(
                    float(reference[column]) - float(getattr(row, column))
                )
                assert difference < 1e-4, (
                    row.event_id, column, reference[column], getattr(row, column)
                )
        print("PASS adapter vs one-pass CSV (canonical, NULL radii)")

        gate_csv = tmp / "gate.csv"
        run(
            "inference.angle_track_uncertainty.predict_rethresholded",
            "--bundle", str(BUNDLE),
            "--data", str(prepared_p050),
            "--split", "test",
            "--threshold", "0.70",
            "--conditional-low-threshold", "0.50",
            "--gate-theta-max-deg", "179.9",
            "--max-added-photon-residual-ns", "1e9",
            "--output", str(gate_csv),
            "--device", "cpu",
            "--threads", "2",
        )
        gate = read_csv(gate_csv)
        assert all(row["selected_threshold"] == "0.5" for row in gate.values()), gate

        gate_preds = tmp / "gate_preds"
        run(
            "inference.angle_track_uncertainty.predict",
            "--source", str(source),
            "--source-name", "mc_merged",
            "--probs-h5", str(probs_h5),
            "--catalog", str(catalog),
            "--output-dir", str(gate_preds),
            "--device", "cpu",
            "--threads", "2",
            "--conditional-low-threshold", "0.50",
            "--gate-theta-max-deg", "179.9",
            "--max-added-photon-residual-ns", "1e9",
        )
        gate_db = (
            gate_preds
            / "release_v3@angle_track_uncertainty"
            / "mc_merged_canonical_p070_gate1000000000_track.duckdb"
        )
        conn = duckdb.connect(str(gate_db), read_only=True)
        gate_rows = conn.execute(
            "SELECT event_id, n_hits, selected_threshold, theta_deg, "
            "angle_r68_deg, gate_applied, uncertainty_calibration_scope "
            "FROM predictions ORDER BY event_id"
        ).df()
        conn.close()
        assert len(gate_rows) == 3, gate_rows
        for row in gate_rows.itertuples(index=False):
            reference = gate[row.event_id]
            assert float(row.selected_threshold) == 0.5
            assert int(row.n_hits) == int(reference["n_hits"])
            assert abs(float(row.theta_deg) - float(reference["theta_deg"])) < 1e-4
            assert bool(row.gate_applied)
            assert pd.isna(row.angle_r68_deg)
            assert row.uncertainty_calibration_scope == "p050_gate_unvalidated"
        print("PASS gate adapter vs rethresholded CSV (NULL radii)")

        # The release-SN mode must write a different DB, not overwrite the
        # canonical rows through INSERT OR IGNORE.
        release_preds = tmp / "release_preds"
        run(
            "inference.angle_track_uncertainty.predict",
            "--source", str(source),
            "--source-name", "mc_merged",
            "--sn-mode", "release",
            "--catalog", str(catalog),
            "--output-dir", str(release_preds),
            "--device", "cpu",
            "--threads", "2",
            "--sn-batch-size", "256",
        )
        release_db = (
            release_preds
            / "release_v3@angle_track_uncertainty"
            / "mc_merged_release_p070_track.duckdb"
        )
        assert release_db.is_file(), "release mode did not write its own DB"
        conn = duckdb.connect(str(release_db), read_only=True)
        release_rows = conn.execute(
            "SELECT sn_mode, angle_r68_deg, uncertainty_calibration_scope "
            "FROM predictions"
        ).df()
        conn.close()
        assert len(release_rows) > 0, release_rows
        assert (release_rows["sn_mode"] == "release").all()
        assert release_rows["angle_r68_deg"].notna().all(), release_rows
        assert (
            release_rows["uncertainty_calibration_scope"] == "p070_crossseason"
        ).all(), release_rows
        print("PASS release-SN DB separation and calibrated radii")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
