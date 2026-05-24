"""
Build event catalog from experimental data file (exp.h5).

Similar to build_catalog.py but adapted for experimental data:
- No prime_prty (no MC truth: energy, theta, phi, weight)
- No signal/noise labels (all labels are 0 for real data)
- Part names are run identifiers (e.g. part_s2020_c01_r0027), stored as strings
- Single top-level group: 'exp'

String number is derived from channel_id // 36 (STRING_DIVISOR).

Usage:
    python data_manager/build_catalog_exp.py
    python data_manager/build_catalog_exp.py --h5 /path/to/exp.h5 --output-dir /path/to/catalogs
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
import h5py
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_H5_PATH = Path(__file__).parent / "h5datasets" / "exp.h5"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "catalogs_exp"

STRING_DIVISOR = 36


def get_part_names(particle_group: h5py.Group) -> List[str]:
    """Get sorted list of part_* names from ev_starts subgroup."""
    return sorted(particle_group["raw/ev_starts"].keys())


def process_part(
    particle_group: h5py.Group,
    part_name: str,
) -> Optional[dict]:
    """Extract per-event metadata from a single part.

    Returns dict of numpy arrays, one entry per column.
    """
    ev_starts = particle_group[f"raw/ev_starts/{part_name}/data"][:]
    n_events = len(ev_starts) - 1
    if n_events == 0:
        return None

    # Hit counts per event
    hit_start = ev_starts[:-1]
    hit_end = ev_starts[1:]
    n_hits = (hit_end - hit_start).astype(np.int32)

    # Per-hit arrays
    #channels = particle_group[f"raw/channels/{part_name}/data"][:]
    z_coords = particle_group[f"raw/data/{part_name}/data"][:, 4]

    # --- Vectorized per-event computations ---

    # Mean z per event
    z_sums = np.add.reduceat(z_coords, hit_start)
    mean_z = (z_sums / n_hits).astype(np.float32)

    # Number of unique strings (all hits) — pre-computed in h5
    num_un_strings = particle_group[f"raw/num_un_strings/{part_name}/data"][:]

    # # Cluster ID per event
    # cluster_ids = particle_group[f"raw/cluster_ids/{part_name}/data"][:]

    # Event IDs (byte strings)
    ev_ids = particle_group[f"ev_ids/{part_name}/data"][:]

    return {
        "event_id": ev_ids.astype("S30"),
        #"cluster_id": cluster_ids.astype(np.int32),
        "n_hits": n_hits,
        "n_unique_strings": num_un_strings.astype(np.int32),
        "mean_z": mean_z,
        "h5_part": np.full(n_events, part_name, dtype="U30"),
        "hit_start_idx": hit_start.astype(np.int64),
        "hit_end_idx": hit_end.astype(np.int64),
    }


def build_exp_catalog(
    h5_path: Path,
    output_dir: Path,
) -> Optional[Path]:
    """Build and save a Parquet catalog for experimental data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "exp.parquet"

    logger.info(f"Processing experimental data: {h5_path}")
    t0 = time.time()

    with h5py.File(h5_path, "r", rdcc_nbytes=0) as f:
        if "exp" not in f:
            logger.error("'exp' group not found in file")
            return None

        group = f["exp"]
        part_names = get_part_names(group)
        logger.info(f"  Found {len(part_names)} parts (runs)")

        # --- Pass 1: count total events ---
        part_event_counts = []
        for part_name in part_names:
            n_ev = len(group[f"raw/ev_starts/{part_name}/data"]) - 1
            part_event_counts.append(n_ev)
        n_total = sum(part_event_counts)

        if n_total == 0:
            logger.warning("No events found")
            return None

        logger.info(f"  Total events: {n_total:,}, pre-allocating arrays")

        # Pre-allocate output arrays
        event_id = np.empty(n_total, dtype="S30")
        #cluster_id = np.empty(n_total, dtype=np.int32)
        n_hits_arr = np.empty(n_total, dtype=np.int32)
        n_unique_strings = np.empty(n_total, dtype=np.int32)
        mean_z_arr = np.empty(n_total, dtype=np.float32)
        h5_part = np.empty(n_total, dtype="U30")
        hit_start_idx = np.empty(n_total, dtype=np.int64)
        hit_end_idx = np.empty(n_total, dtype=np.int64)

        # --- Pass 2: fill arrays in-place ---
        offset = 0
        for i, part_name in tqdm(
            enumerate(part_names), total=len(part_names), desc="Processing parts"
        ):
            n_ev = part_event_counts[i]
            if n_ev == 0:
                continue

            part_data = process_part(group, part_name)
            if part_data is None:
                continue

            sl = slice(offset, offset + n_ev)
            event_id[sl] = part_data["event_id"]
            #cluster_id[sl] = part_data["cluster_id"]
            n_hits_arr[sl] = part_data["n_hits"]
            n_unique_strings[sl] = part_data["n_unique_strings"]
            mean_z_arr[sl] = part_data["mean_z"]
            h5_part[sl] = part_data["h5_part"]
            hit_start_idx[sl] = part_data["hit_start_idx"]
            hit_end_idx[sl] = part_data["hit_end_idx"]
            offset += n_ev

    # Trim in case some parts had 0 events
    n_total = offset

    df = pl.DataFrame({
        "event_id": event_id[:n_total],
        #"cluster_id": cluster_id[:n_total],
        "n_hits": n_hits_arr[:n_total],
        "n_unique_strings": n_unique_strings[:n_total],
        "mean_z": mean_z_arr[:n_total],
        "h5_part": h5_part[:n_total],
        "hit_start_idx": hit_start_idx[:n_total],
        "hit_end_idx": hit_end_idx[:n_total],
    })
    df.write_parquet(output_path)

    elapsed = time.time() - t0
    logger.info(
        f"  Saved {n_total:,} events to {output_path} "
        f"({output_path.stat().st_size / 1e6:.1f} MB, {elapsed:.1f}s)"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build event catalog from experimental HDF5 file")
    parser.add_argument(
        "--h5", type=Path, default=DEFAULT_H5_PATH,
        help=f"Path to experimental HDF5 file (default: {DEFAULT_H5_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for catalog file (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not args.h5.exists():
        logger.error(f"HDF5 file not found: {args.h5}")
        return

    # Save source path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_path.txt").write_text(str(args.h5.resolve()) + "\n")

    build_exp_catalog(args.h5, args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
