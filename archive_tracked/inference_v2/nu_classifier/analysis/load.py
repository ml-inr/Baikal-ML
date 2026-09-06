"""Load nu-classifier predictions and join with catalog for analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CATALOG = "data_manager/catalog_v2.duckdb"


def load_training_parts(npy_dir: str) -> Set[str]:
    """Return the set of h5 part keys used in the NPY training dataset.

    Used to exclude training events from mc_merged prediction DBs, which may
    contain both training events (written by predict_npy.py) and out-of-training
    events (written by predict_mc.py --npy-dir-to-exclude) in the same table.
    """
    parts = np.load(Path(npy_dir) / "h5_part_keys.npy", allow_pickle=True)
    return set(parts.tolist())


def _db_path_for(checkpoint_dir: str, source: str, thr: float) -> Path:
    thr_tag = str(thr).replace(".", "p")
    return Path(checkpoint_dir) / f"{source}_thr{thr_tag}.duckdb"


def load_preds(
    checkpoint_dir: str,
    source: str,
    thr: float = 0.8,
    catalog_path: str = DEFAULT_CATALOG,
    cuts: Optional[Dict] = None,
    exclude_parts: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Load predictions DuckDB and JOIN with catalog events table.

    Args:
        checkpoint_dir: Path to preds/{checkpoint_name}/ directory.
        source: 'mc_merged' | 'mc_reco' | 'exp_reco' | 'exp'
        thr: Sig-noise threshold used during prediction (for filename lookup).
        catalog_path: Path to catalog_v2.duckdb.
        cuts: Optional filters applied after join, e.g.:
            {'min_sn_hits': 8, 'min_sn_strings': 2, 'score_min': 0.5}

    Returns:
        DataFrame with columns:
            event_fk, score, n_sn_hits, n_sn_strings,
            data_class, season, cluster, run, event_id
    """
    db_path = _db_path_for(checkpoint_dir, source, thr)
    if not db_path.exists():
        raise FileNotFoundError(f"Predictions DB not found: {db_path}")
    if not Path(catalog_path).exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    conn = duckdb.connect()
    conn.execute(f"ATTACH '{db_path}' AS p (READ_ONLY)")
    conn.execute(f"ATTACH '{catalog_path}' AS cat (READ_ONLY)")

    where_clauses = []
    if cuts:
        if "min_sn_hits" in cuts:
            where_clauses.append(f"p.predictions.n_sn_hits >= {cuts['min_sn_hits']}")
        if "min_sn_strings" in cuts:
            where_clauses.append(f"p.predictions.n_sn_strings >= {cuts['min_sn_strings']}")
        if "score_min" in cuts:
            where_clauses.append(f"p.predictions.score >= {cuts['score_min']}")
        if "score_max" in cuts:
            where_clauses.append(f"p.predictions.score <= {cuts['score_max']}")

    if exclude_parts and source == "mc_merged":
        conn.register("_excl_parts", pd.DataFrame({"pk": list(exclude_parts)}))
        where_clauses.append("e.run NOT IN (SELECT pk FROM _excl_parts)")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    df = conn.execute(f"""
        SELECT
            p.predictions.event_fk,
            p.predictions.score,
            p.predictions.n_sn_hits,
            p.predictions.n_sn_strings,
            e.data_class,
            e.season,
            e.cluster,
            e.run,
            e.event_id
        FROM p.predictions
        JOIN cat.events e ON e.id = p.predictions.event_fk
        {where_sql}
    """).df()

    conn.close()
    logger.info(f"Loaded {len(df):,} predictions from {db_path.name}  (source={source})")
    return df


def load_moe_preds(
    checkpoint_dirs: List[str],
    source: str,
    thr: float = 0.8,
    catalog_path: str = DEFAULT_CATALOG,
    cuts: Optional[Dict] = None,
    exclude_parts: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Inner-join predictions from multiple checkpoints and return averaged scores.

    Only events present in ALL models are included (inner join on event_fk).
    Catalog join and quality cuts are applied in the same query.

    Args:
        checkpoint_dirs: Paths to preds/{checkpoint_name}/ directories.
        source: 'mc_merged' | 'mc_reco' | 'exp' | 'exp_reco'
        thr: Sig-noise threshold tag used in DB filename.
        catalog_path: Path to catalog_v2.duckdb.
        cuts: Optional quality cuts, e.g. {'min_sn_hits': 8, 'min_sn_strings': 2}

    Returns:
        DataFrame with same schema as load_preds() — score is the mean across models.
    """
    if not checkpoint_dirs:
        raise ValueError("checkpoint_dirs must be non-empty")

    db_paths = [_db_path_for(d, source, thr) for d in checkpoint_dirs]
    for p in db_paths:
        if not p.exists():
            raise FileNotFoundError(f"Predictions DB not found: {p}")
    if not Path(catalog_path).exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    n = len(db_paths)
    conn = duckdb.connect()
    for i, db_path in enumerate(db_paths):
        conn.execute(f"ATTACH '{db_path}' AS p{i} (READ_ONLY)")
    conn.execute(f"ATTACH '{catalog_path}' AS cat (READ_ONLY)")

    score_sum = " + ".join(f"p{i}.predictions.score" for i in range(n))
    inner_joins = "\n        ".join(
        f"INNER JOIN p{i}.predictions USING (event_fk)" for i in range(1, n)
    )

    where_clauses = []
    if cuts:
        if "min_sn_hits" in cuts:
            where_clauses.append(f"p0.predictions.n_sn_hits >= {cuts['min_sn_hits']}")
        if "min_sn_strings" in cuts:
            where_clauses.append(f"p0.predictions.n_sn_strings >= {cuts['min_sn_strings']}")

    if exclude_parts and source == "mc_merged":
        conn.register("_excl_parts", pd.DataFrame({"pk": list(exclude_parts)}))
        where_clauses.append("e.run NOT IN (SELECT pk FROM _excl_parts)")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    df = conn.execute(f"""
        SELECT
            p0.predictions.event_fk,
            ({score_sum}) / {n}          AS score,
            p0.predictions.n_sn_hits,
            p0.predictions.n_sn_strings,
            e.data_class,
            e.season,
            e.cluster,
            e.run,
            e.event_id
        FROM p0.predictions
        {inner_joins}
        JOIN cat.events e ON e.id = p0.predictions.event_fk
        {where_sql}
    """).df()

    conn.close()
    logger.info(
        f"MoE ({n} models, source={source}): {len(df):,} inner-joined events"
        + (f" after cuts {cuts}" if cuts else "")
    )
    return df


def compare_checkpoints(
    checkpoint_dirs: List[str],
    source: str,
    thr: float = 0.8,
    catalog_path: str = DEFAULT_CATALOG,
    cuts: Optional[Dict] = None,
) -> pd.DataFrame:
    """Load predictions from multiple checkpoints and tag each with checkpoint name.

    Returns:
        DataFrame with all columns from load_preds plus 'checkpoint' column.
    """
    frames = []
    for ckpt_dir in checkpoint_dirs:
        df = load_preds(ckpt_dir, source, thr=thr, catalog_path=catalog_path, cuts=cuts)
        df["checkpoint"] = Path(ckpt_dir).name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_history(preds_dir: str) -> pd.DataFrame:
    """Load the prediction_history.csv for a task."""
    csv_path = Path(preds_dir) / "prediction_history.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)
