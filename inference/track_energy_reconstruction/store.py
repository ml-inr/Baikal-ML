"""DuckDB storage for muon-energy checkpoint predictions.

Same storage contract as ``inference/nu_classifier/preds``:
    preds/<run>@<checkpoint>/<source>_<cuts>_thr<tag>.duckdb
    preds/<run>@<checkpoint>/run_info.json
    preds/prediction_history.csv

The table keeps the model prediction, the dataset truth and the split provenance
(train / val / test, flat-selection weight, whether the labels were used).  This
makes the database self-contained for analysis and keeps the train events from
silently mixing with the held-out ones.

Schema: predictions(event_fk BIGINT PK, pred_log10e, pred_energy_gev, pred_sigma_dex,
true_log10e, true_energy_gev, energy_status, particle_type, n_sig_hits, n_sig_strings,
theta, split, flat_weight, used_for_labels, h5_part_key, h5_local_event_id).

The prediction directory is shared by full-source runs with different ``--subset``
values; ``INSERT OR IGNORE`` keyed by ``event_fk`` lets a ``--subset test`` run and a
later ``--subset all`` run accumulate without duplicating events.  Older DBs that
predate the ``theta`` column are migrated in place by :func:`open_energy_db`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from inference.shared.catalog_query import open_predictions_db

PREDICTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    event_fk           BIGINT PRIMARY KEY,
    pred_log10e        FLOAT,
    pred_energy_gev    DOUBLE,
    pred_sigma_dex     FLOAT,
    true_log10e        FLOAT,
    true_energy_gev    DOUBLE,
    energy_status      INTEGER,
    particle_type      INTEGER,
    n_sig_hits         INTEGER,
    n_sig_strings      INTEGER,
    theta              FLOAT,
    split              VARCHAR,
    flat_weight        FLOAT,
    used_for_labels    BOOLEAN,
    h5_part_key        VARCHAR,
    h5_local_event_id  INTEGER
);
"""

_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_predictions_split ON predictions(split);"

COLUMNS = [
    "event_fk", "pred_log10e", "pred_energy_gev", "pred_sigma_dex", "true_log10e",
    "true_energy_gev", "energy_status", "particle_type", "n_sig_hits",
    "n_sig_strings", "theta", "split", "flat_weight", "used_for_labels",
    "h5_part_key", "h5_local_event_id",
]


def open_energy_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """Open or create the energy predictions DB with the table and split index."""
    conn = open_predictions_db(str(db_path), PREDICTIONS_SCHEMA)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info('predictions')").fetchall()
    }
    if "theta" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN theta FLOAT")
    if "pred_sigma_dex" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN pred_sigma_dex FLOAT")
    conn.execute(_INDEX_SQL)
    return conn


def append_predictions_energy(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    check_existing: bool = False,
    tol: float = 1e-4,
) -> tuple[int, int]:
    """Insert a chunk with INSERT OR IGNORE semantics; return (new, skipped)."""
    df = df.copy()
    # Older callers do not provide the optional uncertainty columns.
    for optional in ("theta", "pred_sigma_dex"):
        if optional not in df.columns:
            df[optional] = np.nan
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")
    d = df[COLUMNS].copy()
    d["event_fk"] = d["event_fk"].astype(np.int64)
    d["pred_log10e"] = d["pred_log10e"].astype(np.float32)
    d["true_log10e"] = d["true_log10e"].astype(np.float32)
    d["used_for_labels"] = d["used_for_labels"].astype(bool)
    for c in ("pred_energy_gev", "true_energy_gev"):
        d[c] = d[c].astype(np.float64)
    d["pred_sigma_dex"] = d["pred_sigma_dex"].astype(np.float32)
    for c in ("energy_status", "particle_type", "n_sig_hits", "n_sig_strings",
              "h5_local_event_id"):
        d[c] = d[c].astype(np.int64)
    d["theta"] = d["theta"].astype(np.float32)
    d["flat_weight"] = d["flat_weight"].astype(np.float32)

    conn.register("_tmp_new", d)
    before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if check_existing:
        conflicts = conn.execute("""
            SELECT n.event_fk, n.pred_log10e AS new_pred, p.pred_log10e AS old_pred
            FROM _tmp_new n
            JOIN predictions p ON p.event_fk = n.event_fk
            WHERE ABS(n.pred_log10e - p.pred_log10e) > ?
        """, [tol]).df()
        if len(conflicts):
            logger_msg = (
                f"check_existing: {len(conflicts)} events have differing "
                f"pred_log10e (max diff="
                f"{abs(conflicts['new_pred'] - conflicts['old_pred']).max():.2e})"
            )
            import logging
            logging.getLogger(__name__).warning(logger_msg)
    placeholders = ", ".join(COLUMNS)
    conn.execute(f"""
        INSERT OR IGNORE INTO predictions ({placeholders})
        SELECT {placeholders} FROM _tmp_new
    """)
    conn.unregister("_tmp_new")
    after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_inserted = after - before
    return n_inserted, len(d) - n_inserted
