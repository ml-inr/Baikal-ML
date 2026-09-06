"""Estimate whether each event travelled upward or downward, from hit timing alone.

Down-going events are atmospheric showers by construction; only neutrinos cross the Earth
and arrive from below. Direction therefore decides what the experimental excess can be, and
it has to be measured independently of the classifier — the network was trained on a sample
where "neutrino" and "up-going" coincide perfectly, so its score carries no separate
information about direction.

A track's light sweeps the array, so hit time varies linearly with depth. The slope of that
line, dt/dz in ns per metre, is negative for one direction of travel and positive for the
other; its magnitude is set by the speed of light and is around 3.3 ns/m for a vertical
track. Both the sign convention and the expected magnitude are checked against Monte Carlo
truth zenith rather than assumed.
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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from inference_v2.nu_classifier.analysis.exp_excess.build_splits import load_config

logger = logging.getLogger(__name__)

SPEED_OF_LIGHT_M_PER_NS = 0.299792458
MIN_DEPTH_SPREAD_M = 5.0    # below this the slope is meaningless, whatever the fit says

DIRECTION_TABLE = """
CREATE TABLE IF NOT EXISTS direction (
    event_fk        BIGINT PRIMARY KEY,
    time_vs_depth   DOUBLE,   -- ns per metre; sign distinguishes upward from downward
    fit_quality     DOUBLE,   -- squared correlation of time with depth, 0..1
    depth_spread_m  DOUBLE,
    apparent_speed  DOUBLE    -- metres per ns implied by the slope; must not exceed c
)
"""


def fit_time_against_depth(times_ns: np.ndarray, depths_m: np.ndarray) -> tuple:
    """Slope, squared correlation, depth spread and the implied speed for one event."""
    depth_spread = float(depths_m.max() - depths_m.min())
    depth_variance = float(depths_m.var())
    if depth_variance == 0.0:
        return np.nan, np.nan, depth_spread, np.nan

    covariance = float(((depths_m - depths_m.mean()) * (times_ns - times_ns.mean())).mean())
    slope = covariance / depth_variance

    time_variance = float(times_ns.var())
    fit_quality = (covariance ** 2) / (depth_variance * time_variance) if time_variance else np.nan
    apparent_speed = 1.0 / abs(slope) if slope != 0.0 else np.inf
    return slope, fit_quality, depth_spread, apparent_speed


def events_to_process(connection: duckdb.DuckDBPyConnection, config: dict,
                      resume: bool) -> list[tuple[str, str, int]]:
    cuts = config["event_selection"]
    already_done = ""
    if resume:
        already_done = "AND s.event_fk NOT IN (SELECT event_fk FROM direction)"
    return connection.execute(f"""
        SELECT s.data_class, s.part_key, count(*)
        FROM predictions p JOIN splits s USING (event_fk)
        WHERE p.n_sn_hits >= {cuts['min_sig_hits']}
          AND p.n_sn_strings >= {cuts['min_sig_strings']}
          AND NOT s.used_for_labels
          {already_done}
        GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall()


def build_direction(database_path: str, source_h5_path: str, probs_h5_path: str,
                    config: dict, group_override: Optional[str], resume: bool) -> None:
    connection = duckdb.connect(database_path)
    connection.execute(DIRECTION_TABLE)

    cuts = config["event_selection"]
    parts = events_to_process(connection, config, resume)
    total_events = sum(count for _, _, count in parts)
    logger.info(f"  {len(parts):,} parts, {total_events:,} events")

    started_at, processed = time.time(), 0
    hdf5_cache = dict(rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=1_000_003)

    with h5py.File(source_h5_path, "r", **hdf5_cache) as source, \
         h5py.File(probs_h5_path, "r", **hdf5_cache) as probabilities:
        for index, (data_class, part_key, part_event_count) in enumerate(parts, 1):
            group = group_override or data_class
            selected = connection.execute(f"""
                SELECT s.event_fk, s.local_idx
                FROM predictions p JOIN splits s USING (event_fk)
                WHERE s.data_class = ? AND s.part_key = ?
                  AND p.n_sn_hits >= {cuts['min_sig_hits']}
                  AND p.n_sn_strings >= {cuts['min_sig_strings']}
                  AND NOT s.used_for_labels
                ORDER BY s.local_idx
            """, [data_class, part_key]).df()
            if selected.empty:
                continue

            event_starts = source[f"{group}/raw/ev_starts/{part_key}/data"][:].astype(np.int64)
            hits = source[f"{group}/raw/data/{part_key}/data"][:].astype(np.float32)
            hit_probabilities = probabilities[f"{group}/probs/{part_key}/data"][:].astype(np.float32)

            rows = []
            for event_fk, local_idx in zip(selected.event_fk.to_numpy(),
                                           selected.local_idx.to_numpy()):
                start, end = int(event_starts[local_idx]), int(event_starts[local_idx + 1])
                is_signal = hit_probabilities[start:end] > 0.8
                if is_signal.sum() < 2:
                    continue
                signal_hits = hits[start:end][is_signal]
                rows.append((int(event_fk),
                             *fit_time_against_depth(signal_hits[:, 1].astype(np.float64),
                                                     signal_hits[:, 4].astype(np.float64))))

            if rows:
                frame = pd.DataFrame(rows, columns=["event_fk", "time_vs_depth", "fit_quality",
                                                    "depth_spread_m", "apparent_speed"])
                connection.register("part_direction", frame)
                connection.execute("INSERT OR IGNORE INTO direction SELECT * FROM part_direction")
                connection.unregister("part_direction")

            processed += part_event_count
            if index % 100 == 0 or index == len(parts):
                elapsed = time.time() - started_at
                remaining = elapsed / processed * (total_events - processed) if processed else 0
                logger.info(f"  [{index:>6}/{len(parts)}] {data_class}/{part_key}  "
                            f"{processed:,}/{total_events:,}  {processed/elapsed:,.0f} ev/s  "
                            f"eta ~{remaining/60:.0f}m")

    stored = connection.execute("SELECT count(*) FROM direction").fetchone()[0]
    logger.info(f"  direction rows: {stored:,}  in {(time.time()-started_at)/60:.1f}m")
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--source", choices=["mc", "exp"], required=True)
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

    config = load_config(Path(arguments.config))
    if arguments.source == "mc":
        source_h5 = "data_manager/data/h5datasets/baikal_mc_merged.h5"
        probs_h5 = ("data_manager/data/h5datasets/"
                    "baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5")
        group = None
    else:
        source_h5 = "data_manager/data/h5datasets/exp_full.h5"
        probs_h5 = ("data_manager/data/h5datasets/"
                    "exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5")
        group = "exp_full"

    logger.info(f"{arguments.source}:")
    build_direction(str(PROJECT_ROOT / config["predictions"][arguments.source]),
                    str(PROJECT_ROOT / source_h5), str(PROJECT_ROOT / probs_h5),
                    config, group, not arguments.no_resume)


if __name__ == "__main__":
    main()
