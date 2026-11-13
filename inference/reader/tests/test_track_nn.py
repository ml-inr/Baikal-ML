#!/usr/bin/env python3
"""Checks for ``inference.reader.track_nn`` against a real track-NN database.

The database is task-specific (exp predictions for one nu-classifier run), so the
test skips when it is absent.  It covers the two properties the notebooks rely
on: prefixed left join by ``event_fk`` and an explicit missing-row count.

Usage:  python3 test_track_nn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference import reader  # noqa: E402

DB = (ROOT / "inference/angle_track_uncertainty/preds/"
      "from_nu_261024_1202_muOnlyDA_h8s3/release_v3@angle_track_uncertainty/"
      "exp_full_canonical_p070_track.duckdb")


def main() -> None:
    if not DB.is_file():
        print("SKIP: track-NN predictions are not present")
        return

    track = reader.open_track_nn(DB)
    columns = track.columns()
    for required in (
        "event_fk", "theta_deg", "sigma_angle_deg", "n_hits",
        "gate_applied", "uncertainty_calibration_scope",
    ):
        assert required in columns, (required, columns)

    frame = track.frame(
        limit=3,
        columns=[
            "event_fk", "theta_deg", "n_hits",
            "gate_applied", "uncertainty_calibration_scope",
        ],
    )
    assert len(frame) == 3, frame
    assert np.isfinite(frame["theta_deg"]).all(), frame
    assert not frame["gate_applied"].any(), frame

    missing_fk = int(frame["event_fk"].max()) + 10**9
    events = pd.DataFrame({"event_fk": [*frame["event_fk"], missing_fk]})
    attached = track.attach(events, columns=["theta_deg", "n_hits"])
    assert list(attached.columns) == ["event_fk", "track_theta_deg", "track_n_hits"]
    assert attached["track_theta_deg"].notna().sum() == 3, attached
    assert attached["track_theta_deg"].isna().sum() == 1, attached

    coverage = track.coverage(events)
    assert int(coverage["n_events"].sum()) == 4, coverage
    assert int(coverage["n_matched"].sum()) == 3, coverage
    assert int(coverage["n_missing"].sum()) == 1, coverage

    summary = track.summary()
    assert int(summary["events"].sum()) == 3_353_716, summary
    print("PASS track-NN reader")


if __name__ == "__main__":
    main()
