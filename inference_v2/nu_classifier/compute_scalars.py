"""Per-event scalars derived from the sig-noise-filtered hits, into the predictions DB.

The prediction DuckDBs hold scores and embeddings but no hits, because hits are cheap to
re-read from the source HDF5 (0.63 ms/event) and expensive to duplicate. What is *not*
cheap is recomputing the same handful of derived quantities on every analysis pass, so they
are computed once per model and stored beside the scores as a `scalars` table.

Most definitions come from `analysis/exp_excess_investigation/build_dataset.py:event_scalars`
and must stay identical to it — the excess study was built on those numbers, and silently
redefining them would make old and new results incomparable.

`n_sig_strings` is the exception, and deliberately so: that builder counts strings by
rounding cluster-centred (x, y) to a grid, which can merge two strings that sit at the same
position in different clusters. The pipeline itself has always counted them by channel id
(`io.py:_count_sig_hits_strings`), and that is what defines the h8s3 selection, so this
follows the pipeline rather than the builder.

`test_scalars.py` checks both claims: the shared quantities against the builder, and
`n_sig_strings` against the pipeline's own function.

Hits are selected exactly as the scoring run selected them: `prob > threshold` from the
same probs file. Order does not matter here — every scalar is order-invariant.

Usage:
    python inference_v2/nu_classifier/compute_scalars.py \\
        --db     inference_v2/nu_classifier/preds/<ckpt>/mc_merged_thr0p8.duckdb \\
        --h5     data_manager/data/h5datasets/baikal_mc_merged.h5 \\
        --probs  data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_....h5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import duckdb
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data_manager.nu_classifier_ds_builder.io import STRING_DIVISOR

logger = logging.getLogger(__name__)

# Strings are identified by channel id, exactly as data_manager/nu_classifier_ds_builder/
# io.py:_count_sig_hits_strings does — that function produced n_sig_strings in the probs
# files and n_sn_strings in the predictions, i.e. the h8s3 selection itself.
#
# The obvious-looking alternative, rounding (x, y) to a grid, is wrong in principle: the
# stored coordinates are cluster-centred, so two strings in different clusters can share an
# (x, y) and be merged into one. Channel ids are global and cannot collide. Measured on
# 101,777 MC events the two agree in every case (events do not span clusters), but the
# coordinate version is right only by accident, and three analysis scripts had copied it.
Q_CLIP = 100.0        # p.e.; the bound the classifier itself applies (_AMP_CLIP_Q100)

SCALARS_SCHEMA = """
CREATE TABLE IF NOT EXISTS scalars (
    event_fk      BIGINT PRIMARY KEY,
    n_sig_hits    INTEGER,
    n_sig_strings INTEGER,
    n_channels    INTEGER,
    q_mean        DOUBLE,
    q_std         DOUBLE,
    q_total       DOUBLE,
    q_max         DOUBLE,
    q_frac_max    DOUBLE,
    r_vert        DOUBLE,
    t_span        DOUBLE,
    t_span_core   DOUBLE,
    z_span        DOUBLE,
    xy_span       DOUBLE,
    x_c           DOUBLE,
    y_c           DOUBLE,
    z_c           DOUBLE,
    prob_mean     DOUBLE,
    prob_min      DOUBLE
)
"""

COLUMNS = ["n_sig_hits", "n_sig_strings", "n_channels", "q_mean", "q_std", "q_total",
           "q_max", "q_frac_max", "r_vert", "t_span", "t_span_core", "z_span", "xy_span",
           "x_c", "y_c", "z_c", "prob_mean", "prob_min"]


def event_scalars(h: np.ndarray, p: np.ndarray, c: np.ndarray) -> tuple:
    """One event's scalars. `h` = (n, 5) raw hits, `p` = probs, `c` = channel ids.

    Fourteen of the values reproduce `event_scalars` in the excess-investigation builder
    exactly, including the 100 p.e. charge clip and the 1e-3 guard in r_vert. The fifteenth,
    `n_sig_strings`, deliberately does not: it is taken from channel ids like the rest of the
    pipeline, not from rounded coordinates as that builder does.
    """
    q, t, x, y, z = h[:, 0], h[:, 1], h[:, 2], h[:, 3], h[:, 4]
    qc = np.clip(q, 0, Q_CLIP)
    sx, sy, sz = x.std(), y.std(), z.std()
    q_total, q_max = float(qc.sum()), float(qc.max())
    t_lo, t_hi = np.percentile(t, [5, 95])
    return (
        len(h),
        int(np.unique(c // STRING_DIVISOR).size),
        int(np.unique(c).size),
        float(qc.mean()), float(qc.std()), q_total, q_max,
        q_max / q_total if q_total > 0 else 0.0,
        float(sz / (np.sqrt(sx * sx + sy * sy) + 1e-3)),
        float(t.max() - t.min()), float(t_hi - t_lo),
        float(z.max() - z.min()),
        float(np.hypot(x.max() - x.min(), y.max() - y.min())),
        float(x.mean()), float(y.mean()), float(z.mean()),
        float(p.mean()), float(p.min()),
    )


def _parts_to_do(conn, catalog: str, resume: bool) -> list[tuple[str, str, int]]:
    """(data_class, part_key, n_events) still needing scalars, largest classes first."""
    conn.execute(f"ATTACH '{catalog}' AS cat (READ_ONLY)")
    done = set()
    if resume:
        rows = conn.execute("""
            SELECT DISTINCT l.part_key, e.data_class
            FROM scalars s
            JOIN cat.h5_locations l ON l.event_fk = s.event_fk
            JOIN cat.events e       ON e.id       = s.event_fk
        """).fetchall()
        done = {(dc, pk) for pk, dc in rows}
        logger.info(f"  {len(done)} parts already have scalars — skipping them")

    rows = conn.execute("""
        SELECT e.data_class, l.part_key, count(*)
        FROM predictions p
        JOIN cat.h5_locations l ON l.event_fk = p.event_fk
        JOIN cat.events e       ON e.id       = p.event_fk
        GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall()
    return [(dc, pk, n) for dc, pk, n in rows if (dc, pk) not in done]


def compute(db: str, h5_path: str, probs_path: str, catalog: str, threshold: float,
            group: Optional[str], resume: bool, limit_parts: Optional[int]) -> None:
    conn = duckdb.connect(db)
    conn.execute(SCALARS_SCHEMA)

    parts = _parts_to_do(conn, catalog, resume)
    if limit_parts:
        parts = parts[:limit_parts]
    total_ev = sum(n for _, _, n in parts)
    logger.info(f"{len(parts):,} parts, {total_ev:,} events to process")

    insert = f"INSERT OR IGNORE INTO scalars SELECT * FROM _tmp"
    t0, done_ev = time.time(), 0
    _RDCC = dict(rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=1_000_003)

    with h5py.File(h5_path, "r", **_RDCC) as src, h5py.File(probs_path, "r", **_RDCC) as pf:
        for i, (data_class, part_key, n_ev) in enumerate(parts, 1):
            grp = group or data_class
            rows = conn.execute("""
                SELECT l.local_idx, p.event_fk
                FROM predictions p
                JOIN cat.h5_locations l ON l.event_fk = p.event_fk
                JOIN cat.events e       ON e.id       = p.event_fk
                WHERE l.part_key = ? AND e.data_class = ?
            """, [part_key, data_class]).fetchall()
            if not rows:
                continue
            idx = np.array([r[0] for r in rows], dtype=np.int64)
            fks = np.array([r[1] for r in rows], dtype=np.int64)
            order = np.argsort(idx)          # sequential reads through the part
            idx, fks = idx[order], fks[order]

            ev   = src[f"{grp}/raw/ev_starts/{part_key}/data"][:].astype(np.int64)
            data = src[f"{grp}/raw/data/{part_key}/data"][:].astype(np.float32)
            chan = src[f"{grp}/raw/channels/{part_key}/data"][:].astype(np.int32)
            prob = pf[f"{grp}/probs/{part_key}/data"][:].astype(np.float32)

            out_fk, out_rows = [], []
            for local_idx, fk in zip(idx, fks):
                s, e = int(ev[local_idx]), int(ev[local_idx + 1])
                m = prob[s:e] > threshold
                if not m.any():
                    continue
                out_fk.append(int(fk))
                out_rows.append(event_scalars(data[s:e][m], prob[s:e][m], chan[s:e][m]))

            if out_rows:
                import pandas as pd
                df = pd.DataFrame(out_rows, columns=COLUMNS)
                df.insert(0, "event_fk", out_fk)
                conn.register("_tmp", df)
                conn.execute(insert)
                conn.unregister("_tmp")

            done_ev += n_ev
            if i % 25 == 0 or i == len(parts):
                el = time.time() - t0
                eta = el / done_ev * (total_ev - done_ev) if done_ev else 0
                logger.info(f"  [{i:>6}/{len(parts)}] {data_class}/{part_key}  "
                            f"{done_ev:,}/{total_ev:,} events  "
                            f"{done_ev/el:,.0f} ev/s  eta ~{eta/60:.0f}m")

    n = conn.execute("SELECT count(*) FROM scalars").fetchone()[0]
    logger.info(f"\nDone in {(time.time()-t0)/60:.1f}m — scalars rows in DB: {n:,}")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="predictions .duckdb to add `scalars` to")
    ap.add_argument("--h5", required=True, help="source HDF5 with raw hits and channels")
    ap.add_argument("--probs", required=True, help="sig-noise probs HDF5 used for scoring")
    ap.add_argument("--catalog", default="data_manager/catalog_v2.duckdb")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--group", default=None,
                    help="h5 top-level group; defaults to each event's data_class (MC)")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute parts that already have scalars")
    ap.add_argument("--limit-parts", type=int, default=None, help="for smoke tests")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    compute(a.db, a.h5, a.probs, a.catalog, a.threshold, a.group,
            not a.no_resume, a.limit_parts)


if __name__ == "__main__":
    main()
