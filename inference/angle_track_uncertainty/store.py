"""DuckDB storage for the released angle/track/uncertainty model.

Same storage contract as the other inference tasks::

    preds/<run>@<checkpoint>/<source>_<sn_mode>_<tag>.duckdb
    preds/<run>@<checkpoint>/run_info.json
    preds/prediction_history.csv

Predictions are keyed by ``event_fk`` from ``catalog_v2.duckdb`` and written with
``INSERT OR IGNORE``, so a run over an overlapping part subset is resumable.

Calibrated 68%/95% radii are only populated for unchanged release-SN p>=0.70 MC
rows on CPU with the release selection conventions. Canonical-SN, experimental,
gate and changed-threshold rows keep raw sigmas but NULL radii and a scope string
saying why.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from inference.shared.catalog_query import open_predictions_db

PREDICTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    event_fk                      BIGINT PRIMARY KEY,
    event_id                      VARCHAR,
    data_class                    VARCHAR,
    h5_part_key                   VARCHAR,
    h5_local_event_id             INTEGER,
    sn_mode                       VARCHAR NOT NULL,
    selection_op                  VARCHAR NOT NULL,
    gate_applied                  BOOLEAN NOT NULL,
    selected_threshold            FLOAT NOT NULL,
    n_raw_hits                    INTEGER,
    n_hits                        INTEGER NOT NULL,
    n_strings                     INTEGER NOT NULL,
    theta_deg                     FLOAT NOT NULL,
    phi_deg                       FLOAT NOT NULL,
    direction_x                   FLOAT NOT NULL,
    direction_y                   FLOAT NOT NULL,
    direction_z                   FLOAT NOT NULL,
    anchor_x_m                    FLOAT NOT NULL,
    anchor_y_m                    FLOAT NOT NULL,
    anchor_z_m                    FLOAT NOT NULL,
    sigma_angle_deg               FLOAT NOT NULL,
    angle_r68_deg                 FLOAT,
    angle_r95_deg                 FLOAT,
    sigma_anchor_m                FLOAT NOT NULL,
    anchor_r68_m                  FLOAT,
    anchor_r95_m                  FLOAT,
    track_point_scope             VARCHAR NOT NULL,
    uncertainty_calibration_scope VARCHAR NOT NULL
);
"""

COLUMNS = [
    "event_fk", "event_id", "data_class", "h5_part_key", "h5_local_event_id",
    "sn_mode", "selection_op", "gate_applied", "selected_threshold",
    "n_raw_hits", "n_hits", "n_strings",
    "theta_deg", "phi_deg",
    "direction_x", "direction_y", "direction_z",
    "anchor_x_m", "anchor_y_m", "anchor_z_m",
    "sigma_angle_deg", "angle_r68_deg", "angle_r95_deg",
    "sigma_anchor_m", "anchor_r68_m", "anchor_r95_m",
    "track_point_scope", "uncertainty_calibration_scope",
]

_FLOAT_COLUMNS = [
    "selected_threshold", "theta_deg", "phi_deg",
    "direction_x", "direction_y", "direction_z",
    "anchor_x_m", "anchor_y_m", "anchor_z_m",
    "sigma_angle_deg", "sigma_anchor_m",
]

#: Calibrated radii are NULL where the release does not claim calibration.
_NULLABLE_FLOAT_COLUMNS = [
    "angle_r68_deg", "angle_r95_deg", "anchor_r68_m", "anchor_r95_m",
]

_INT_COLUMNS = [
    "h5_local_event_id", "n_raw_hits", "n_hits", "n_strings",
]

_STRING_COLUMNS = [
    "event_id", "data_class", "h5_part_key", "sn_mode", "selection_op",
    "track_point_scope", "uncertainty_calibration_scope",
]


def open_angle_track_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """Open or create the angle/track predictions DB."""
    return open_predictions_db(str(db_path), PREDICTIONS_SCHEMA)


def append_predictions_angle_track(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    check_existing: bool = False,
    tol: float = 1e-4,
) -> tuple[int, int]:
    """Insert a chunk with INSERT OR IGNORE semantics; return (new, skipped)."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")
    d = df[COLUMNS].copy()
    d["event_fk"] = d["event_fk"].astype(np.int64)
    for column in _STRING_COLUMNS:
        d[column] = d[column].astype(str)
    d["gate_applied"] = d["gate_applied"].astype(bool)
    for column in _INT_COLUMNS:
        d[column] = d[column].astype(np.int64)
    for column in _FLOAT_COLUMNS:
        d[column] = d[column].astype(np.float32)
    # pandas nullable Float64 keeps missing radii as pd.NA, so DuckDB stores NULL
    # and SQL `IS NULL` works. A plain float cast would turn them into NaN.
    for column in _NULLABLE_FLOAT_COLUMNS:
        d[column] = pd.array(d[column], dtype="Float64")

    conn.register("_tmp_new", d)
    before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if check_existing:
        conflicts = conn.execute(
            """
            SELECT n.event_fk, n.theta_deg AS new_theta, p.theta_deg AS old_theta
            FROM _tmp_new n
            JOIN predictions p ON p.event_fk = n.event_fk
            WHERE ABS(n.theta_deg - p.theta_deg) > ?
               OR ABS(n.direction_x - p.direction_x) > ?
               OR ABS(n.direction_y - p.direction_y) > ?
               OR ABS(n.direction_z - p.direction_z) > ?
               OR ABS(n.sigma_angle_deg - p.sigma_angle_deg) > ?
            """,
            [tol, tol, tol, tol, tol],
        ).df()
        if len(conflicts):
            import logging

            logging.getLogger(__name__).warning(
                "check_existing: %s events differ in direction/sigma "
                "(max theta diff=%.2e)",
                len(conflicts),
                abs(conflicts["new_theta"] - conflicts["old_theta"]).max(),
            )
    placeholders = ", ".join(COLUMNS)
    conn.execute(
        f"INSERT OR IGNORE INTO predictions ({placeholders}) "
        f"SELECT {placeholders} FROM _tmp_new"
    )
    conn.unregister("_tmp_new")
    after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_inserted = after - before
    return n_inserted, len(d) - n_inserted
