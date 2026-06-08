"""Populate the event catalog (DuckDB v2) from baikal_mc_merged.h5.

Structure per particle type:
  {ptype}/raw/data/{pk}/data          (n_hits, 5) float32
  {ptype}/raw/ev_starts/{pk}/data     (n_events+1,) int64
  {ptype}/raw/channels/{pk}/data      (n_hits,) int32
  {ptype}/raw/cluster_ids/{pk}/data   (n_events,) int32

Catalog mapping:
  source     = 'mc_merged'
  data_class = particle type  (e.g. 'muatm_2020')
  season     = year extracted from data_class
  cluster    = cluster_ids[event_idx]
  run        = part_key string as-is  (e.g. 'part_1000', 'part_1000_0')
  event_id   = local 0-based index within part
  feature_hash = xxh64 of int32 channel sequence (order-sensitive)

Usage:
    python -m data_manager.catalog_v2.build_mc \\
        --h5-path  data_manager/data/h5datasets/baikal_mc_merged.h5 \\
        [--particles muatm_2020 nuatm_2020 nue2_2020] \\
        [--no-hash] \\
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
from data_manager.catalog_v2.schema import open_catalog, create_schema, create_hash_index

CATALOG_V2_PATH = Path(__file__).resolve().parents[1] / "catalog_v2.duckdb"

logger = logging.getLogger(__name__)

_COMMIT_EVERY = 50

_EV_COLS = ["source", "data_class", "season", "cluster", "run", "event_id", "feature_hash"]
_H5_COLS = ["event_fk", "h5_path", "part_key", "local_idx"]


def _channel_hash(channels: np.ndarray) -> str:
    return xxhash.xxh64(channels.astype(np.int32).tobytes()).hexdigest()


def _season_from_data_class(data_class: str) -> int:
    return int(data_class.rsplit("_", 1)[-1])


# ── Index management for bulk load ────────────────────────────────────────────

def _drop_indexes_for_bulk(conn) -> None:
    """Drop all indexes (and UNIQUE constraint if present) before bulk insert.

    MC always deletes before inserting, so no uniqueness enforcement is needed
    during the insert loop.  Indexes are rebuilt once at the end.
    """
    for idx in ("idx_events_key", "idx_events_hash", "idx_h5_event"):
        conn.execute(f"DROP INDEX IF EXISTS {idx}")
    # Drop inline UNIQUE constraint from CREATE TABLE if still present
    rows = conn.execute(
        "SELECT constraint_name FROM duckdb_constraints() "
        "WHERE table_name='events' AND constraint_type='UNIQUE'"
    ).fetchall()
    for (name,) in rows:
        try:
            conn.execute(f'ALTER TABLE events DROP CONSTRAINT "{name}"')
            logger.info(f"Dropped UNIQUE constraint: {name}")
        except Exception as exc:
            logger.warning(f"Could not drop constraint {name}: {exc}")
    conn.commit()


def _recreate_indexes(conn, compute_hash: bool) -> None:
    """Rebuild lookup indexes after bulk insert."""
    logger.info("Building idx_events_key...")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_key "
        "ON events(source, data_class, season, cluster, run, event_id)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_h5_event ON h5_locations(event_fk)")
    if compute_hash:
        logger.info("Building idx_events_hash...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_hash ON events(feature_hash)")
    conn.commit()


# ── Process + write one part ───────────────────────────────────────────────

def _process_and_write(
    conn,
    h5: h5py.File,
    h5_path_str: str,
    ptype: str,
    part_key: str,
    season: int,
    compute_hash: bool,
) -> int:
    """Insert one part; return number of events inserted."""
    grp         = h5[ptype]
    ev_starts   = grp["raw"]["ev_starts"][part_key]["data"][:].astype(np.int64)
    cluster_ids = grp["raw"]["cluster_ids"][part_key]["data"][:].astype(np.int32)

    if compute_hash:
        channels_raw = grp["raw"]["channels"][part_key]["data"][:].astype(np.int32)

    n_events   = len(ev_starts) - 1
    event_rows = []

    for i in range(n_events):
        fhash = _channel_hash(channels_raw[ev_starts[i]: ev_starts[i + 1]]) if compute_hash else None
        event_rows.append(("mc_merged", ptype, season, int(cluster_ids[i]), part_key, i, fhash))

    # -- events: plain INSERT with RETURNING to get auto-generated IDs --------
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
    all_fks = [id_map[i] for i in range(n_events)]

    # -- h5_locations: plain INSERT --------------------------------------------
    df_h5 = pd.DataFrame(
        [(all_fks[i], h5_path_str, part_key, i) for i in range(n_events)],
        columns=_H5_COLS,
    )
    conn.register("_tmp_h5", df_h5)
    conn.execute("""
        INSERT INTO h5_locations (event_fk, h5_path, part_key, local_idx)
        SELECT event_fk, h5_path, part_key, local_idx FROM _tmp_h5
    """)
    conn.unregister("_tmp_h5")

    return n_events


# ── Build ──────────────────────────────────────────────────────────────────

def build(
    h5_path: Path,
    catalog_path: Path,
    particles: list[str] | None = None,
    compute_hash: bool = True,
) -> None:
    h5_path      = Path(h5_path).resolve()
    catalog_path = Path(catalog_path)

    conn = open_catalog(catalog_path)
    create_schema(conn)

    logger.info(f"H5 file : {h5_path}")
    logger.info(f"Catalog : {catalog_path}")
    logger.info(f"Hashing : {'yes' if compute_hash else 'no (NULL placeholders)'}")

    with h5py.File(h5_path, "r") as h5:
        all_ptypes = list(h5.keys())
        ptypes     = [p for p in all_ptypes if particles is None or p in particles]
        part_map   = {p: sorted(h5[p]["raw"]["ev_starts"].keys()) for p in ptypes}

    logger.info(f"Particle types: {ptypes}")

    # Remove any partial MC data for the particles being rebuilt.
    # Scoped to ptypes so other already-complete MC particles are untouched.
    placeholders = ", ".join(["?" for _ in ptypes])
    n_existing = conn.execute(
        f"SELECT COUNT(*) FROM events WHERE source='mc_merged' AND data_class IN ({placeholders})",
        ptypes,
    ).fetchone()[0]
    if n_existing:
        logger.info(f"Removing {n_existing:,} partial MC events from previous run...")
        conn.execute(
            f"DELETE FROM h5_locations WHERE event_fk IN "
            f"(SELECT id FROM events WHERE source='mc_merged' AND data_class IN ({placeholders}))",
            ptypes,
        )
        conn.execute(
            f"DELETE FROM events WHERE source='mc_merged' AND data_class IN ({placeholders})",
            ptypes,
        )
        conn.commit()

    # Drop indexes so bulk insert isn't O(N log N) per-row constraint checking
    _drop_indexes_for_bulk(conn)

    h5_path_str  = str(h5_path)
    total_events = 0

    with h5py.File(h5_path, "r", rdcc_nbytes=_H5_RDCC_NBYTES) as h5:
        for ptype in ptypes:
            season    = _season_from_data_class(ptype)
            part_keys = part_map[ptype]
            logger.info(f"  {ptype}: {len(part_keys)} parts  season={season}")

            ptype_events = 0

            conn.execute("BEGIN")
            for part_idx, part_key in enumerate(tqdm(part_keys, desc=ptype, unit="part")):
                n_ev = _process_and_write(
                    conn, h5, h5_path_str, ptype, part_key, season, compute_hash,
                )
                ptype_events += n_ev

                if (part_idx + 1) % _COMMIT_EVERY == 0:
                    conn.execute("COMMIT")
                    conn.execute("BEGIN")
                    if ptype_events % 2_000_000 == 0:
                        gc.collect()

            conn.execute("COMMIT")
            logger.info(f"  {ptype} done: {ptype_events:,} events.")
            total_events += ptype_events

    # Rebuild indexes and UNIQUE constraint in one efficient pass
    _recreate_indexes(conn, compute_hash)
    logger.info(f"Done. {total_events:,} total MC events.")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate catalog_v2 from baikal_mc_merged.h5")
    parser.add_argument("--h5-path",   required=True)
    parser.add_argument("--particles", nargs="*", default=None)
    parser.add_argument("--no-hash",   action="store_true",
                        help="Skip hashing; store NULL in feature_hash (faster)")
    parser.add_argument("--catalog",   default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    build(
        h5_path      = Path(args.h5_path),
        catalog_path = Path(args.catalog) if args.catalog else CATALOG_V2_PATH,
        particles    = args.particles,
        compute_hash = not args.no_hash,
    )


if __name__ == "__main__":
    main()
