"""Compute the group-structure feature table for a stratified sample of the eight groups.

Reads group membership from the prediction DuckDBs, the HDF5 location of each event from the
catalog, and the hits from HDF5 + the sig-noise probs.

**Why it is organised by part and not by group.** `raw/data` is chunked (43215, 1) -- one
column, 43215 rows. Reading one event's 40 rows therefore decompresses five chunks, ~216,000
values for 200 wanted: measured at 42 ms per event. Whole-part reads are the only sensible
access pattern, so each part is read exactly once and every selected event of every group is
taken from it in that pass.

Sampling is deterministic: `ORDER BY hash(event_fk) LIMIT n`, never `USING SAMPLE`, whose
reservoir depends on the order a parallel scan returns rows in and so differs between runs.

The sampled groups (2, 4, 5, 7) are drawn from the parts the exhaustive groups (1, 3, 6, 8)
already force us to read. For muatm that is 5,373 of 10,100 parts; whether a part contains at
least one high-scoring event among its ~2,300 is close to random, but it is a restriction and
is recorded as one.

Usage:
    nohup python inference_v2/nu_classifier/analysis/excess_mechanism/build_features.py \
        > build_features.log 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))

from features import COLUMNS, event_features  # noqa: E402

MODEL = "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
PREDS = ROOT / "inference_v2/nu_classifier/preds" / MODEL
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"
H5 = {"mc": ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5",
      "exp": ROOT / "data_manager/data/h5datasets/exp_full.h5"}
PROBS = {"mc": ROOT / ("data_manager/data/h5datasets/"
                       "baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"),
         "exp": ROOT / ("data_manager/data/h5datasets/"
                        "exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5")}

THRESHOLD = 0.8      # sig-noise cut, the pipeline's convention
XI = 0.8             # classifier score cut defining the groups
QUALITY = "p.n_sn_hits >= 8 AND p.n_sn_strings >= 3"

# group id -> (source, h5 group, score side, quota or None for "take all")
GROUPS = {
    1: ("mc", "nuatm_2020", "<",  None),
    2: ("mc", "nuatm_2020", ">=", 100_000),
    3: ("mc", "nue2_2020",  "<",  None),
    4: ("mc", "nue2_2020",  ">=", 100_000),
    5: ("mc", "muatm_2020", "<",  200_000),
    6: ("mc", "muatm_2020", ">=", None),
    7: ("exp", "exp_full",  "<",  200_000),
    8: ("exp", "exp_full",  ">=", None),
}
RDCC = dict(rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=2_000_003)

logger = logging.getLogger(__name__)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS features (
    source VARCHAR, event_fk BIGINT, group_id TINYINT,
    part_key VARCHAR, cluster VARCHAR,
    {", ".join(f"{c} DOUBLE" for c in COLUMNS)},
    PRIMARY KEY (source, event_fk)
)
"""


def _cluster_of(part_key: str) -> str:
    bits = [b for b in part_key.split("_") if b.startswith("c") and b[1:].isdigit()]
    return bits[0] if bits else "mc"


def _connect() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(PREDS / "mc_merged_thr0p8.duckdb"), read_only=True)
    conn.execute(f"ATTACH '{PREDS / 'exp_full_thr0p8.duckdb'}' AS exp (READ_ONLY)")
    conn.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    return conn


def _where(source: str, h5_group: str, side: str) -> str:
    if source == "mc":
        base = f"{QUALITY} AND NOT s.used_for_labels AND s.data_class = '{h5_group}'"
    else:
        base = f"{QUALITY} AND NOT s.excluded"
    return f"{base} AND p.score {side} {XI}"


def build_selection(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """(source, event_fk, part_key, local_idx, group_id) for every group."""
    frames = []
    must_read: dict[str, set[str]] = {}

    for gid, (source, h5_group, side, quota) in sorted(GROUPS.items()):
        if quota is not None:
            continue
        prefix = "" if source == "mc" else "exp."
        df = conn.execute(f"""
            SELECT '{source}' AS source, p.event_fk, l.part_key, l.local_idx,
                   {gid}::TINYINT AS group_id
            FROM {prefix}predictions p JOIN {prefix}splits s USING (event_fk)
            JOIN cat.h5_locations l ON l.event_fk = p.event_fk
            WHERE {_where(source, h5_group, side)}
            ORDER BY hash(p.event_fk)
        """).df()
        frames.append(df)
        must_read.setdefault(h5_group, set()).update(df.part_key.unique().tolist())
        logger.info(f"group {gid}: {len(df):,} events over "
                    f"{df.part_key.nunique():,} parts (exhaustive)")

    for gid, (source, h5_group, side, quota) in sorted(GROUPS.items()):
        if quota is None:
            continue
        prefix = "" if source == "mc" else "exp."
        parts = sorted(must_read.get(h5_group, set()))
        conn.execute("CREATE OR REPLACE TEMP TABLE _parts (part_key VARCHAR)")
        conn.executemany("INSERT INTO _parts VALUES (?)", [(p,) for p in parts])
        df = conn.execute(f"""
            SELECT '{source}' AS source, p.event_fk, l.part_key, l.local_idx,
                   {gid}::TINYINT AS group_id
            FROM {prefix}predictions p JOIN {prefix}splits s USING (event_fk)
            JOIN cat.h5_locations l ON l.event_fk = p.event_fk
            JOIN _parts pk ON pk.part_key = l.part_key
            WHERE {_where(source, h5_group, side)}
            ORDER BY hash(p.event_fk) LIMIT {quota}
        """).df()
        frames.append(df)
        logger.info(f"group {gid}: {len(df):,} events over "
                    f"{df.part_key.nunique():,} parts (quota {quota:,})")

    return pd.concat(frames, ignore_index=True)


def process_part(task: tuple) -> tuple[str, str, list, list, list]:
    """One part: read once, compute features for every selected event in it."""
    source, h5_group, part, idx, fks, gids = task
    rows, out_fk, out_gid = [], [], []
    with h5py.File(H5[source], "r", **RDCC) as src, \
         h5py.File(PROBS[source], "r", **RDCC) as pf:
        ev = src[f"{h5_group}/raw/ev_starts/{part}/data"][:].astype(np.int64)
        data = src[f"{h5_group}/raw/data/{part}/data"][:].astype(np.float32)
        chan = src[f"{h5_group}/raw/channels/{part}/data"][:].astype(np.int32)
        prob = pf[f"{h5_group}/probs/{part}/data"][:].astype(np.float32)
    order = np.argsort(idx)
    for k in order:
        s, e = int(ev[idx[k]]), int(ev[idx[k] + 1])
        m = prob[s:e] > THRESHOLD
        if m.sum() < 5:
            continue
        rows.append(event_features(data[s:e][m], prob[s:e][m], chan[s:e][m], n_raw=e - s))
        out_fk.append(int(fks[k]))
        out_gid.append(int(gids[k]))
    return source, part, out_fk, out_gid, rows


def run(out_db: Path, workers: int, limit_parts: int | None) -> None:
    conn = _connect()
    sel = build_selection(conn)
    conn.close()
    logger.info(f"{len(sel):,} events selected in total")

    out = duckdb.connect(str(out_db))
    out.execute(SCHEMA)
    # keyed by (h5_group, part_key), matching the task key -- part names repeat across MC
    # classes, so part_key alone would skip a part that was only done for a different class
    _h5_of = {gid: h5g for gid, (_, h5g, _, _) in GROUPS.items()}
    done_parts = {(_h5_of[g], pk) for g, pk in out.execute(
        "SELECT DISTINCT group_id, part_key FROM features").fetchall()}
    if done_parts:
        logger.info(f"resuming: {len(done_parts):,} parts already done")

    # The h5 group must be carried per row, not looked up per part name. Part names are NOT
    # unique across MC classes -- part_1271, part_1216 and others exist in two data_classes at
    # once (verified in the catalog) -- so a (source, part_key) key silently merges two
    # different parts and applies one class's local_idx to the other class's arrays. That is
    # what raised "index 24036 is out of bounds for axis 0 with size 23554" and killed the
    # first full run.
    h5_of_group = {gid: h5_group for gid, (_, h5_group, _, _) in GROUPS.items()}
    sel = sel.assign(h5_group=sel.group_id.map(h5_of_group))

    tasks = []
    for (source, h5_group, part), sub in sel.groupby(
            ["source", "h5_group", "part_key"], sort=True):
        if (h5_group, part) in done_parts:
            continue
        tasks.append((source, h5_group, part,
                      sub.local_idx.values, sub.event_fk.values, sub.group_id.values))
    if limit_parts:
        tasks = tasks[:limit_parts]
    # biggest first: the exp parts are ~90x the size of an MC part, so starting them early
    # keeps the tail short
    tasks.sort(key=lambda t: (t[0] != "exp", -len(t[3])))
    logger.info(f"{len(tasks):,} parts to read, {workers} workers")

    t0, written, n_done = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_part, t): t[2] for t in tasks}
        for fut in as_completed(futures):
            source, part, fks, gids, rows = fut.result()
            n_done += 1
            if rows:
                df = pd.DataFrame(rows, columns=COLUMNS)
                df.insert(0, "cluster", _cluster_of(part))
                df.insert(0, "part_key", part)
                df.insert(0, "group_id", gids)
                df.insert(0, "event_fk", fks)
                df.insert(0, "source", source)
                out.register("_tmp", df)
                out.execute("INSERT OR IGNORE INTO features SELECT * FROM _tmp")
                out.unregister("_tmp")
                written += len(df)
            if n_done % 200 == 0 or n_done == len(tasks):
                el = time.time() - t0
                eta = el / n_done * (len(tasks) - n_done)
                logger.info(f"  {n_done:,}/{len(tasks):,} parts  {written:,} events  "
                            f"{written/max(el,1):.0f} ev/s  eta {eta/60:.0f} min")

    n = out.execute("SELECT count(*) FROM features").fetchone()[0]
    by_group = out.execute("SELECT group_id, count(*) FROM features "
                           "GROUP BY 1 ORDER BY 1").fetchall()
    logger.info(f"done in {(time.time()-t0)/60:.1f} min — {n:,} rows")
    for gid, cnt in by_group:
        logger.info(f"  group {gid}: {cnt:,}")
    out.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "features.duckdb"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-parts", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    run(Path(a.out), a.workers, a.limit_parts)


if __name__ == "__main__":
    main()
