"""Copy each experimental event's absolute timestamp into the predictions database.

Bursts of light — bioluminescence, electronics glitches — are visible only in time, and the
analysis so far aggregated whole runs, where a flash lasting minutes inside a 22-hour run
washes out entirely. `header_prty` carries the timestamp already; this puts it next to the
scores so a burst test is a join rather than a re-read of 52 GB.
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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from archive_tracked.inference_v2.nu_classifier.analysis.exp_excess.build_splits import load_config

logger = logging.getLogger(__name__)

# header_prty columns, from data_manager/root2h5/root2h5_exp_full.py
TIMESTAMP_NS = 3

EVENT_TIME_TABLE = """
CREATE TABLE IF NOT EXISTS event_time (
    event_fk     BIGINT PRIMARY KEY,
    timestamp_ns BIGINT,
    seconds_into_run DOUBLE
)
"""


def build_event_time(database_path: str, source_h5_path: str, group: str) -> None:
    connection = duckdb.connect(database_path)
    connection.execute("DROP TABLE IF EXISTS event_time")
    connection.execute(EVENT_TIME_TABLE)

    parts = [row[0] for row in connection.execute(
        "SELECT DISTINCT part_key FROM splits ORDER BY part_key").fetchall()]
    logger.info(f"  {len(parts)} runs")

    with h5py.File(source_h5_path, "r") as source:
        for part_key in parts:
            events = connection.execute(
                "SELECT event_fk, local_idx FROM splits WHERE part_key = ? ORDER BY local_idx",
                [part_key]).df()
            if events.empty:
                continue
            timestamps = source[f"{group}/header_prty/{part_key}/data"][:, TIMESTAMP_NS]
            selected = timestamps[events.local_idx.to_numpy()]
            connection.register("part_time", pd.DataFrame({
                "event_fk": events.event_fk.to_numpy(),
                "timestamp_ns": selected.astype("int64"),
                "seconds_into_run": (selected - timestamps.min()) / 1e9,
            }))
            connection.execute("INSERT OR IGNORE INTO event_time SELECT * FROM part_time")
            connection.unregister("part_time")
            span_hours = (timestamps.max() - timestamps.min()) / 1e9 / 3600
            logger.info(f"  {part_key}: {len(events):,} events over {span_hours:.1f} h")

    stored = connection.execute("SELECT count(*) FROM event_time").fetchone()[0]
    logger.info(f"  event_time rows: {stored:,}")
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--source-h5", default="data_manager/data/h5datasets/exp_full.h5")
    parser.add_argument("--group", default="exp_full")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    config = load_config(Path(arguments.config))
    build_event_time(str(PROJECT_ROOT / config["predictions"]["exp"]),
                     str(PROJECT_ROOT / arguments.source_h5), arguments.group)


if __name__ == "__main__":
    main()
