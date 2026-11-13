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
    event_fk        BIGINT  PRIMARY KEY,
    score           FLOAT,
    n_sn_hits       INTEGER,
    n_sn_strings    INTEGER,
    -- Provenance of the event, written at scoring time from the NPY back-links
    -- (part_key + local event id), so no later pass over the database is needed to
    -- tell which events the model was fitted on. Same names and meaning as in the
    -- `splits` table: `used_for_labels` = its labels were used in training (MC),
    -- `was_da_target` = it was fed to the domain discriminator as an unlabelled
    -- target (data). Both default to FALSE, so callers that do not mark anything
    -- are unaffected.
    used_for_labels BOOLEAN DEFAULT FALSE,
    was_da_target   BOOLEAN DEFAULT FALSE
)
"""

# The prefilter scores ALL raw hits — the hit counts below are analysis metadata, never
# a selection. n_sn_* count the hits the sig-noise model calls signal at threshold 0.8 and
# the strings they touch; n_gt_sig_* are the same counts from MC truth (`raw/labels != 0`).
# Both are copied from the probabilities file, which is where the nu-classifier's own
# counts come from, so an h8s3 cut means the same thing in both pipelines. They are NULL
# where the probabilities file has no entry for the part: sig-noise probabilities exist for
# 10,100 of the 20,004 mc_merged parts, and MC truth does not exist for experimental data.
PREFILTER_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    event_fk         BIGINT PRIMARY KEY,
    score            FLOAT,
    n_hits           INTEGER,
    n_sn_hits        INTEGER,
    n_sn_strings     INTEGER,
    n_gt_sig_hits    INTEGER,
    n_gt_sig_strings INTEGER
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
    used_for_labels: Optional[np.ndarray] = None,
    was_da_target: Optional[np.ndarray] = None,
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
        used_for_labels: (N,) bool — the event's labels were used in training (MC).
        was_da_target: (N,) bool — the event was a domain-adaptation target (data).

    Returns:
        (n_inserted, n_skipped)
    """
    n = len(event_fks)
    df_new = pd.DataFrame({
        "event_fk":    event_fks.astype(np.int64),
        "score":       scores.astype(np.float32),
        "n_sn_hits":   n_sn_hits.astype(np.int32),
        "n_sn_strings": n_sn_strings.astype(np.int32),
        "used_for_labels": (np.zeros(n, dtype=bool) if used_for_labels is None
                            else np.asarray(used_for_labels, dtype=bool)),
        "was_da_target": (np.zeros(n, dtype=bool) if was_da_target is None
                          else np.asarray(was_da_target, dtype=bool)),
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
            (event_fk, score, n_sn_hits, n_sn_strings, used_for_labels, was_da_target)
        SELECT event_fk, score, n_sn_hits, n_sn_strings, used_for_labels, was_da_target
        FROM _tmp_new
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


def _nullable_counts(values: Optional[np.ndarray], n: int) -> pd.Series:
    """Counts as a nullable integer column.

    `None` means the whole batch has no such count. A float array means some events have
    one and some do not — NaN marks the gaps, which happens when one flushed buffer mixes
    parts the probabilities file covers with parts it does not. Both become NULL rather
    than a sentinel: a sentinel would be a number, and every later `>= 8` would silently
    treat it as a real count that failed the cut instead of an unknown one.
    """
    if values is None:
        return pd.Series([pd.NA] * n, dtype="Int32")
    values = np.asarray(values)
    if values.dtype.kind == "f":
        missing = np.isnan(values)
        out = pd.Series(np.where(missing, 0, values).astype(np.int32), dtype="Int32")
        out[missing] = pd.NA
        return out
    return pd.Series(values.astype(np.int32), dtype="Int32")


def append_predictions_prefilter(
    conn: duckdb.DuckDBPyConnection,
    event_fks: np.ndarray,
    scores: np.ndarray,
    n_hits: np.ndarray,
    n_sn_hits: Optional[np.ndarray] = None,
    n_sn_strings: Optional[np.ndarray] = None,
    n_gt_sig_hits: Optional[np.ndarray] = None,
    n_gt_sig_strings: Optional[np.ndarray] = None,
    check_existing: bool = False,
    tol: float = 1e-5,
) -> tuple[int, int]:
    """Insert prefilter predictions with INSERT OR IGNORE semantics.

    The four count arguments are optional and land as NULL when absent: the sig-noise
    counts have no source for a part the probabilities file never covered, and MC truth
    has none for experimental data. NULL says "not known here", which is what a later
    `WHERE n_sn_hits >= 8` should act on — it drops those rows rather than treating an
    unknown as a zero.

    Returns:
        (n_inserted, n_skipped)
    """
    n = len(event_fks)
    df_new = pd.DataFrame({
        "event_fk":         event_fks.astype(np.int64),
        "score":            scores.astype(np.float32),
        "n_hits":           n_hits.astype(np.int32),
        "n_sn_hits":        _nullable_counts(n_sn_hits, n),
        "n_sn_strings":     _nullable_counts(n_sn_strings, n),
        "n_gt_sig_hits":    _nullable_counts(n_gt_sig_hits, n),
        "n_gt_sig_strings": _nullable_counts(n_gt_sig_strings, n),
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
        SELECT event_fk, score, n_hits, n_sn_hits, n_sn_strings,
               n_gt_sig_hits, n_gt_sig_strings
        FROM _tmp_new
    """)
    conn.unregister("_tmp_new")

    count_after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_inserted  = count_after - count_before
    n_skipped   = len(event_fks) - n_inserted
    return n_inserted, n_skipped
