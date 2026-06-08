"""DuckDB catalog lookup and prediction storage helpers."""

import logging
from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maps mc_reco h5 group names → catalog data_class values
MC_RECO_PTYPE_TO_DATA_CLASS = {
    "muatm":        "muatm_2020",
    "nuatm_conv":   "nuatm_conv_2020",
    "nuatm_prompt": "nuatm_prompt_2020",
    "nue2":         "nue2_2020",
}

NU_CLASSIFIER_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    event_fk     BIGINT  PRIMARY KEY,
    score        FLOAT,
    n_sn_hits    INTEGER,
    n_sn_strings INTEGER
)
"""

PREFILTER_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    event_fk  BIGINT PRIMARY KEY,
    score     FLOAT,
    n_hits    INTEGER
)
"""

EMBEDDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    event_fk  BIGINT PRIMARY KEY,
    embedding FLOAT[]
)
"""


def open_predictions_db(db_path: str, schema_sql: str) -> duckdb.DuckDBPyConnection:
    """Open (or create) a predictions DuckDB file and ensure table exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    conn.execute(schema_sql)
    return conn


def get_event_fks_mc(
    catalog_path: str,
    source: str,
    data_classes: np.ndarray,
    seasons: np.ndarray,
    runs: np.ndarray,
    event_ids: np.ndarray,
) -> pd.DataFrame:
    """Batch catalog lookup for MC sources (mc_merged or mc_reco).

    Args:
        source: 'mc_merged' or 'mc_reco'
        data_classes: per-event catalog data_class strings
        seasons: per-event season integers
        runs: per-event part_key strings
        event_ids: per-event local 0-based index within part

    Returns:
        DataFrame with columns: query_idx, event_fk
        Only rows that matched are returned; missing events are absent.
    """
    n = len(data_classes)
    df_q = pd.DataFrame({
        "query_idx":  np.arange(n, dtype=np.int64),
        "data_class": data_classes,
        "season":     seasons.astype(np.int32),
        "run":        runs,
        "event_id":   event_ids.astype(np.int64),
    })
    conn = duckdb.connect(catalog_path, read_only=True)
    conn.register("_tmp_q", df_q)
    result = conn.execute(f"""
        SELECT q.query_idx, e.id AS event_fk
        FROM _tmp_q q
        JOIN events e
            ON  e.source     = '{source}'
            AND e.data_class = q.data_class
            AND e.season     = q.season
            AND e.run        = q.run
            AND e.event_id   = q.event_id
    """).df()
    conn.unregister("_tmp_q")
    conn.close()
    return result


def get_event_fks_exp(
    catalog_path: str,
    source: str,
    seasons: np.ndarray,
    clusters: np.ndarray,
    runs: np.ndarray,
    event_ids: np.ndarray,
) -> pd.DataFrame:
    """Batch catalog lookup for exp sources (exp or exp_reco).

    Args:
        source: 'exp' or 'exp_reco'
        seasons: from header_prty[:, 0]
        clusters: from header_prty[:, 1]
        runs: str(int(header_prty[:, 2])) — run number as string
        event_ids: header_prty[:, 3] — event_id_in_run

    Returns:
        DataFrame with columns: query_idx, event_fk
    """
    n = len(seasons)
    df_q = pd.DataFrame({
        "query_idx": np.arange(n, dtype=np.int64),
        "season":    seasons.astype(np.int32),
        "cluster":   clusters.astype(np.int32),
        "run":       runs,
        "event_id":  event_ids.astype(np.int64),
    })
    conn = duckdb.connect(catalog_path, read_only=True)
    conn.register("_tmp_q", df_q)
    result = conn.execute(f"""
        SELECT q.query_idx, e.id AS event_fk
        FROM _tmp_q q
        JOIN events e
            ON  e.source   = '{source}'
            AND e.season   = q.season
            AND e.cluster  = q.cluster
            AND e.run      = q.run
            AND e.event_id = q.event_id
    """).df()
    conn.unregister("_tmp_q")
    conn.close()
    return result


def append_predictions_nu_classifier(
    conn: duckdb.DuckDBPyConnection,
    event_fks: np.ndarray,
    scores: np.ndarray,
    n_sn_hits: np.ndarray,
    n_sn_strings: np.ndarray,
    check_existing: bool = False,
    tol: float = 1e-5,
) -> tuple[int, int]:
    """Insert predictions with INSERT OR IGNORE semantics.

    Args:
        conn: Open predictions DuckDB connection.
        event_fks: (N,) int64 — catalog event_fk values.
        scores: (N,) float32 — model sigmoid scores.
        n_sn_hits: (N,) int32 — sig-noise filtered hit count.
        n_sn_strings: (N,) int32 — sig-noise filtered string count.
        check_existing: If True, warn when skipped events have differing scores.
        tol: Score comparison tolerance for check_existing.

    Returns:
        (n_inserted, n_skipped)
    """
    df_new = pd.DataFrame({
        "event_fk":    event_fks.astype(np.int64),
        "score":       scores.astype(np.float32),
        "n_sn_hits":   n_sn_hits.astype(np.int32),
        "n_sn_strings": n_sn_strings.astype(np.int32),
    })
    conn.register("_tmp_new", df_new)

    count_before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    if check_existing:
        conflicts = conn.execute("""
            SELECT n.event_fk, n.score AS new_score, p.score AS old_score
            FROM _tmp_new n
            JOIN predictions p ON p.event_fk = n.event_fk
            WHERE ABS(n.score - p.score) > {tol}
        """.replace("{tol}", str(tol))).df()
        if len(conflicts) > 0:
            logger.warning(
                f"check_existing: {len(conflicts)} events have differing scores "
                f"(max diff={abs(conflicts['new_score'] - conflicts['old_score']).max():.6f})"
            )

    conn.execute("""
        INSERT OR IGNORE INTO predictions
        SELECT event_fk, score, n_sn_hits, n_sn_strings FROM _tmp_new
    """)
    conn.unregister("_tmp_new")

    count_after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_inserted = count_after - count_before
    n_skipped  = len(event_fks) - n_inserted
    return n_inserted, n_skipped


def append_embeddings(
    conn: duckdb.DuckDBPyConnection,
    event_fks: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[int, int]:
    """Insert encoder-level embeddings with INSERT OR IGNORE semantics.

    Args:
        conn: Open predictions DuckDB connection (must have embeddings table).
        event_fks:  (N,) int64
        embeddings: (N, d_model) float32

    Returns:
        (n_inserted, n_skipped)
    """
    df_new = pd.DataFrame({
        "event_fk":  event_fks.astype(np.int64),
        "embedding": [row.tolist() for row in embeddings.astype(np.float32)],
    })
    conn.register("_tmp_emb", df_new)

    count_before = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.execute("INSERT OR IGNORE INTO embeddings SELECT event_fk, embedding FROM _tmp_emb")
    conn.unregister("_tmp_emb")

    count_after  = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    n_inserted   = count_after - count_before
    n_skipped    = len(event_fks) - n_inserted
    return n_inserted, n_skipped


def append_predictions_prefilter(
    conn: duckdb.DuckDBPyConnection,
    event_fks: np.ndarray,
    scores: np.ndarray,
    n_hits: np.ndarray,
    check_existing: bool = False,
    tol: float = 1e-5,
) -> tuple[int, int]:
    """Insert prefilter predictions with INSERT OR IGNORE semantics.

    Returns:
        (n_inserted, n_skipped)
    """
    df_new = pd.DataFrame({
        "event_fk": event_fks.astype(np.int64),
        "score":    scores.astype(np.float32),
        "n_hits":   n_hits.astype(np.int32),
    })
    conn.register("_tmp_new", df_new)

    count_before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    if check_existing:
        conflicts = conn.execute("""
            SELECT n.event_fk, n.score AS new_score, p.score AS old_score
            FROM _tmp_new n
            JOIN predictions p ON p.event_fk = n.event_fk
            WHERE ABS(n.score - p.score) > {tol}
        """.replace("{tol}", str(tol))).df()
        if len(conflicts) > 0:
            logger.warning(
                f"check_existing: {len(conflicts)} events have differing scores "
                f"(max diff={abs(conflicts['new_score'] - conflicts['old_score']).max():.6f})"
            )

    conn.execute("""
        INSERT OR IGNORE INTO predictions
        SELECT event_fk, score, n_hits FROM _tmp_new
    """)
    conn.unregister("_tmp_new")

    count_after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_inserted  = count_after - count_before
    n_skipped   = len(event_fks) - n_inserted
    return n_inserted, n_skipped
