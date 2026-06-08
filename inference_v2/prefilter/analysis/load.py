"""Load prefilter predictions and join with catalog for analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CATALOG = "data_manager/catalog_v2.duckdb"


def load_preds(
    checkpoint_dir: str,
    source: str,
    catalog_path: str = DEFAULT_CATALOG,
    cuts: Optional[Dict] = None,
) -> pd.DataFrame:
    """Load prefilter predictions and JOIN with catalog.

    Args:
        checkpoint_dir: Path to preds/{checkpoint_name}/ directory.
        source: 'mc_merged' | 'mc_reco' | 'exp_reco' | 'exp'
        catalog_path: Path to catalog_v2.duckdb.
        cuts: Optional filters, e.g. {'score_min': 0.5, 'n_hits_min': 10}

    Returns:
        DataFrame: event_fk, score, n_hits, data_class, season, cluster, run, event_id
    """
    db_path = Path(checkpoint_dir) / f"{source}_allhits.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(f"Predictions DB not found: {db_path}")

    conn = duckdb.connect()
    conn.execute(f"ATTACH '{db_path}' AS p (READ_ONLY)")
    conn.execute(f"ATTACH '{catalog_path}' AS cat (READ_ONLY)")

    where_clauses = []
    if cuts:
        if "score_min" in cuts:
            where_clauses.append(f"p.predictions.score >= {cuts['score_min']}")
        if "score_max" in cuts:
            where_clauses.append(f"p.predictions.score <= {cuts['score_max']}")
        if "n_hits_min" in cuts:
            where_clauses.append(f"p.predictions.n_hits >= {cuts['n_hits_min']}")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    df = conn.execute(f"""
        SELECT
            p.predictions.event_fk,
            p.predictions.score,
            p.predictions.n_hits,
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
    logger.info(f"Loaded {len(df):,} predictions from {db_path.name}")
    return df


def compare_checkpoints(
    checkpoint_dirs: List[str],
    source: str,
    catalog_path: str = DEFAULT_CATALOG,
    cuts: Optional[Dict] = None,
) -> pd.DataFrame:
    """Load predictions from multiple checkpoints, tag with checkpoint name."""
    frames = []
    for ckpt_dir in checkpoint_dirs:
        df = load_preds(ckpt_dir, source, catalog_path=catalog_path, cuts=cuts)
        df["checkpoint"] = Path(ckpt_dir).name
        frames.append(df)
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def load_history(preds_dir: str) -> pd.DataFrame:
    """Load the prediction_history.csv for the prefilter task."""
    import pandas as pd
    csv_path = Path(preds_dir) / "prediction_history.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)
