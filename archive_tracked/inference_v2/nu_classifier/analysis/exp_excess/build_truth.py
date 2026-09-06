"""Copy per-event Monte Carlo truth into the predictions database.

Zenith is what separates a neutrino-induced event from an atmospheric muon, and the event
weight is what turns simulated counts into expected rates. Both live in the source HDF5 as
`prime_prty`; joining them by event_fk once is cheaper than re-reading that file in every
analysis step.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from archive_tracked.inference_v2.nu_classifier.analysis.exp_excess.build_splits import load_config

logger = logging.getLogger(__name__)

# Column layout of prime_prty, from doc/hdf5_format.md
ZENITH_DEGREES = 0
AZIMUTH_DEGREES = 1
PRIMARY_ENERGY_GEV = 2
EVENT_WEIGHT = 5

TRUTH_TABLE = """
CREATE TABLE IF NOT EXISTS truth (
    event_fk           BIGINT PRIMARY KEY,
    zenith_deg         DOUBLE,
    azimuth_deg        DOUBLE,
    primary_energy_gev DOUBLE,
    event_weight       DOUBLE
)
"""


def read_truth_for_part(source: h5py.File, data_class: str, part_key: str,
                        local_event_ids: np.ndarray) -> np.ndarray:
    primary_properties = source[f"{data_class}/prime_prty/{part_key}/data"]
    return primary_properties[:][local_event_ids]


def build_truth(database_path: str, source_h5_path: str) -> None:
    connection = duckdb.connect(database_path)
    connection.execute("DROP TABLE IF EXISTS truth")
    connection.execute(TRUTH_TABLE)

    parts = connection.execute("""
        SELECT DISTINCT data_class, part_key FROM splits ORDER BY data_class, part_key
    """).fetchall()
    logger.info(f"  {len(parts):,} parts to read")

    with h5py.File(source_h5_path, "r") as source:
        for index, (data_class, part_key) in enumerate(parts, 1):
            events = connection.execute("""
                SELECT event_fk, local_idx FROM splits
                WHERE data_class = ? AND part_key = ? ORDER BY local_idx
            """, [data_class, part_key]).df()
            if events.empty:
                continue

            properties = read_truth_for_part(
                source, data_class, part_key, events.local_idx.to_numpy())
            connection.register("part_truth", pd.DataFrame({
                "event_fk": events.event_fk.to_numpy(),
                "zenith_deg": properties[:, ZENITH_DEGREES].astype("float64"),
                "azimuth_deg": properties[:, AZIMUTH_DEGREES].astype("float64"),
                "primary_energy_gev": properties[:, PRIMARY_ENERGY_GEV].astype("float64"),
                "event_weight": properties[:, EVENT_WEIGHT].astype("float64"),
            }))
            connection.execute("INSERT OR IGNORE INTO truth SELECT * FROM part_truth")
            connection.unregister("part_truth")

            if index % 500 == 0 or index == len(parts):
                logger.info(f"  [{index:>6}/{len(parts)}] {data_class}/{part_key}")

    stored = connection.execute("SELECT count(*) FROM truth").fetchone()[0]
    logger.info(f"  truth rows: {stored:,}")
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--source-h5",
                        default="data_manager/data/h5datasets/baikal_mc_merged.h5")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

    config = load_config(Path(arguments.config))
    logger.info("Monte Carlo truth:")
    build_truth(str(PROJECT_ROOT / config["predictions"]["mc"]),
                str(PROJECT_ROOT / arguments.source_h5))


if __name__ == "__main__":
    main()
