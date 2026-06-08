"""Populate the event catalog (DuckDB v2) from baikal_mc_reco.h5.

Structure per particle type:
  {ptype}/raw/data/{pk}/data          (n_hits, 5) float32
  {ptype}/raw/ev_starts/{pk}/data     (n_events+1,) int32
  {ptype}/raw/channels/{pk}/data      (n_hits,) int32
  {ptype}/raw/cluster_ids/{pk}/data   (n_events,) int32
  {ptype}/reco_prty/{pk}/data         (n_events, 31) float32
  {ptype}/prime_prty/{pk}/data        (n_events, 6) float32
  {ptype}/ev_ids/{pk}/data            (n_events,) bytes  -- links back to original ROOT event

Catalog mapping:
  source     = 'mc_reco'
  data_class = particle type  (e.g. 'muatm', 'nuatm_conv', 'nuatm_prompt', 'nue2')
  season     = year extracted from part_key  (e.g. 'part_2020_cl1_...' → 2020)
  cluster    = cluster_ids[event_idx]
  run        = part_key string as-is
  event_id   = local 0-based index within part
  feature_hash = xxh64 of int32 channel sequence (order-sensitive)

ev_ids (e.g. b'muatm_2020_cl1_run10000_scl_nu_MC_s19-21_42') link each reco event
back to the original ROOT file event; readable from h5 via h5_locations.

Usage:
    python -m data_manager.catalog_v2.build_mc_reco \\
        --h5-path  data_manager/data/h5datasets/baikal_mc_reco.h5 \\
        [--particles muatm nuatm_conv nuatm_prompt nue2] \\
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
from data_manager.catalog_v2.schema import open_catalog, create_schema

CATALOG_V2_PATH = Path(__file__).resolve().parents[1] / "catalog_v2.duckdb"

logger = logging.getLogger(__name__)

_COMMIT_EVERY = 50

_EV_COLS = ["source", "data_class", "season", "cluster", "run", "event_id", "feature_hash"]
_H5_COLS = ["event_fk", "h5_path", "part_key", "local_idx"]


def _channel_hash(channels: np.ndarray) -> str:
    return xxhash.xxh64(channels.astype(np.int32).tobytes()).hexdigest()


def _season_from_part_key(part_key: str) -> int:
    # 'part_2020_cl1_run10000_scl_nu_MC_s19-21' → 2020
    return int(part_key.split("_")[1])



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
        event_rows.append((
            "mc_reco", f"{ptype}_{season}", season,
            int(cluster_ids[i]), part_key, i, fhash,
        ))

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

    # -- h5_locations: plain INSERT -------------------------------------------
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

    # Derive season-suffixed data_class names used in the catalog
    season_map   = {p: _season_from_part_key(part_map[p][0]) for p in ptypes}
    data_classes = [f"{p}_{season_map[p]}" for p in ptypes]

    logger.info(f"Particle types: {data_classes}")

    # Remove any partial mc_reco data for the particles being rebuilt
    placeholders = ", ".join(["?" for _ in data_classes])
    n_existing = conn.execute(
        f"SELECT COUNT(*) FROM events WHERE source='mc_reco' AND data_class IN ({placeholders})",
        data_classes,
    ).fetchone()[0]
    if n_existing:
        logger.info(f"Removing {n_existing:,} partial mc_reco events from previous run...")
        conn.execute(
            f"DELETE FROM h5_locations WHERE event_fk IN "
            f"(SELECT id FROM events WHERE source='mc_reco' AND data_class IN ({placeholders}))",
            data_classes,
        )
        conn.execute(
            f"DELETE FROM events WHERE source='mc_reco' AND data_class IN ({placeholders})",
            data_classes,
        )
        conn.commit()

    h5_path_str  = str(h5_path)
    total_events = 0

    with h5py.File(h5_path, "r", rdcc_nbytes=_H5_RDCC_NBYTES) as h5:
        for ptype in ptypes:
            part_keys = part_map[ptype]
            # All parts for a given ptype share the same season (encoded in part key)
            season = season_map[ptype]
            logger.info(f"  {ptype}_{season}: {len(part_keys)} parts")

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

    logger.info(f"Done. {total_events:,} total mc_reco events.")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate catalog_v2 from baikal_mc_reco.h5")
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
