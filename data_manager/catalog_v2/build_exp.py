"""Populate the event catalog (DuckDB v2) from an exp-family HDF5 file.

Handles both:
  exp.h5      — top-level group 'exp',       no header_prty,
                event_id = sequential local index within part
  exp_reco.h5 — top-level group 'exp_reco',  has header_prty,
                event_id = physics event_id_in_run from header_prty[:, 3]

Usage:
    python -m data_manager.catalog_v2.build_exp \\
        --h5-path  data_manager/data/h5datasets/exp_reco.h5 \\
        --source   exp_reco \\
        [--root-dir data_manager/data/exp_reco_root] \\
        [--catalog  data_manager/catalog_v2.duckdb]
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xxhash
from tqdm.auto import tqdm

_H5_RDCC_NBYTES = 32 * 1024 * 1024

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_manager.catalog_v2.schema import open_catalog, create_schema
from data_manager.catalog_v2._parse import (
    parse_exp_part_key,
    exp_root_filename,
    exp_reco_root_filename,
)

logger = logging.getLogger(__name__)

CATALOG_V2_PATH = Path(__file__).resolve().parents[1] / "catalog_v2.duckdb"

_EV_COLS   = ["source", "data_class", "season", "cluster", "run", "event_id", "feature_hash"]
_H5_COLS   = ["event_fk", "h5_path", "part_key", "local_idx"]
_ROOT_COLS = ["event_fk", "root_path", "local_idx"]

_COMMIT_EVERY = 50


def _channel_hash(channels: np.ndarray) -> str:
    return xxhash.xxh64(channels.astype(np.int32).tobytes()).hexdigest()


def _find_root_file(root_dir: Path | None, source: str, season: int, cluster: int, run: int) -> str | None:
    if root_dir is None or not root_dir.is_dir():
        return None
    if source == "exp_reco":
        candidate = root_dir / "exp_reco_files" / exp_reco_root_filename(season, cluster, run)
    else:
        candidate = root_dir / exp_root_filename(season, cluster, run)
    return str(candidate.resolve()) if candidate.exists() else None


# ── Process + write one part ───────────────────────────────────────────────

def _process_and_write(
    conn,
    h5_path_str: str,
    top_key: str,
    source: str,
    part_key: str,
    has_header: bool,
    root_path: str | None,
) -> tuple[int, int]:
    season, cluster, run = parse_exp_part_key(part_key)

    with h5py.File(h5_path_str, "r", rdcc_nbytes=_H5_RDCC_NBYTES) as f:
        grp = f[top_key]
        ev_starts    = grp["raw"]["ev_starts"][part_key]["data"][:].astype(np.int64)
        channels_raw = grp["raw"]["channels"][part_key]["data"][:].astype(np.int32)
        if has_header:
            header_prty   = grp["header_prty"][part_key]["data"][:]
            ev_ids_in_run = header_prty[:, 3].astype(int)
        else:
            ev_ids_in_run = None

    n_events = len(ev_starts) - 1
    if ev_ids_in_run is None:
        ev_ids_in_run = np.arange(n_events, dtype=int)

    event_rows = []
    h5_rows    = []
    root_rows  = []

    for i in range(n_events):
        chs   = channels_raw[ev_starts[i]: ev_starts[i + 1]]
        fhash = _channel_hash(chs)
        event_rows.append((source, source, season, cluster, str(run), int(ev_ids_in_run[i]), fhash))
        h5_rows.append((h5_path_str, part_key, i))
        if root_path is not None:
            root_rows.append((root_path, None))

    # -- events ---------------------------------------------------------------
    df_ev = pd.DataFrame(event_rows, columns=_EV_COLS)
    conn.register("_tmp_ev", df_ev)
    rows = conn.execute("""
        INSERT INTO events (source, data_class, season, cluster, run, event_id, feature_hash)
        SELECT source, data_class, season, cluster, run, event_id, feature_hash
        FROM _tmp_ev
        RETURNING id, event_id
    """).fetchall()
    conn.unregister("_tmp_ev")

    id_map  = {r[1]: r[0] for r in rows}
    all_fks = [id_map[int(ev_ids_in_run[i])] for i in range(n_events)]

    # -- h5_locations ---------------------------------------------------------
    df_h5 = pd.DataFrame(
        [(all_fks[i], h5_rows[i][0], h5_rows[i][1], h5_rows[i][2]) for i in range(n_events)],
        columns=_H5_COLS,
    )
    conn.register("_tmp_h5", df_h5)
    conn.execute("""
        INSERT INTO h5_locations (event_fk, h5_path, part_key, local_idx)
        SELECT event_fk, h5_path, part_key, local_idx FROM _tmp_h5
    """)
    conn.unregister("_tmp_h5")

    # -- root_locations -------------------------------------------------------
    if root_rows:
        df_root = pd.DataFrame(
            [(all_fks[i], root_rows[i][0], root_rows[i][1]) for i in range(n_events)],
            columns=_ROOT_COLS,
        )
        conn.register("_tmp_root", df_root)
        conn.execute("""
            INSERT INTO root_locations (event_fk, root_path, local_idx)
            SELECT event_fk, root_path, local_idx FROM _tmp_root
        """)
        conn.unregister("_tmp_root")

    return n_events, n_events


# ── Build ──────────────────────────────────────────────────────────────────

def build(
    h5_path: Path,
    source: str,
    catalog_path: Path,
    root_dir: Path | None = None,
) -> None:
    h5_path      = Path(h5_path).resolve()
    catalog_path = Path(catalog_path)

    conn = open_catalog(catalog_path)
    create_schema(conn)

    logger.info(f"Source  : {source}")
    logger.info(f"H5 file : {h5_path}")
    logger.info(f"Catalog : {catalog_path}")

    with h5py.File(h5_path, "r") as h5:
        top_key    = list(h5.keys())[0]
        has_header = "header_prty" in h5[top_key]
        part_keys  = sorted(h5[top_key]["raw"]["ev_starts"].keys())

    logger.info(f"H5 top key: '{top_key}' | header_prty: {has_header}")
    logger.info(f"Parts   : {len(part_keys)}")

    # Remove any existing data for this source before rebuilding
    n_existing = conn.execute(
        "SELECT COUNT(*) FROM events WHERE source=?", [source]
    ).fetchone()[0]
    if n_existing:
        logger.info(f"Removing {n_existing:,} existing {source} events...")
        conn.execute(
            "DELETE FROM h5_locations WHERE event_fk IN (SELECT id FROM events WHERE source=?)",
            [source],
        )
        conn.execute("DELETE FROM events WHERE source=?", [source])
        conn.commit()

    h5_path_str  = str(h5_path)
    total_events = 0
    total_new    = 0

    conn.execute("BEGIN")
    for part_idx, part_key in enumerate(tqdm(part_keys, desc="parts", unit="part")):
        scr = parse_exp_part_key(part_key)
        root_path = _find_root_file(root_dir, source, *scr)

        n_ev, n_new = _process_and_write(
            conn, h5_path_str, top_key, source, part_key, has_header, root_path,
        )
        total_events += n_ev
        total_new    += n_new

        if (part_idx + 1) % _COMMIT_EVERY == 0:
            conn.execute("COMMIT")
            conn.execute("BEGIN")
            gc.collect()

    conn.execute("COMMIT")
    logger.info(f"Done. {total_events:,} events processed, {total_new:,} new h5_locations.")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate catalog_v2 from an exp-family H5 file")
    parser.add_argument("--h5-path",  required=True)
    parser.add_argument("--source",   required=True, help="e.g. 'exp' or 'exp_reco'")
    parser.add_argument("--root-dir", default=None,  help="Directory with local ROOT files")
    parser.add_argument("--catalog",  default=None,  help="Path to catalog_v2.duckdb")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    build(
        h5_path      = Path(args.h5_path),
        source       = args.source,
        catalog_path = Path(args.catalog) if args.catalog else CATALOG_V2_PATH,
        root_dir     = Path(args.root_dir) if args.root_dir else None,
    )


if __name__ == "__main__":
    main()
