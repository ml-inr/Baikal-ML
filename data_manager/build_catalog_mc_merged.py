"""
Build per-particle-type event catalogs from baikal_mc_merged.h5.

Scans all events and extracts metadata into Parquet files (one per particle type),
readable with polars. No full hit arrays are loaded — only ev_starts, labels,
channels, cluster_ids, raw/data (z coordinate), num_un_strings, prime_prty,
and ev_ids are read per part.

String number is derived from channel_id // 36 (STRING_DIVISOR).

Usage:
    python data_manager/build_catalog.py
    python data_manager/build_catalog.py --particles muatm_2020 nue2_2020
    python data_manager/build_catalog.py --h5 /path/to/file.h5 --output-dir /path/to/catalogs
    python data_manager/build_catalog.py --particles nuatm_2019 nue2_2019 muatm_2019 nue2_2020 muatm_2020

Warning:
    takes about 60GB RAM for 'muatm_2020'!
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

DEFAULT_H5_PATH = Path("/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5")
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "h5_catalogs" / "catalogs_mc_merged"

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
    labels = particle_group[f"raw/labels/{part_name}/data"][:]
    channels = particle_group[f"raw/channels/{part_name}/data"][:]
    z_coords = particle_group[f"raw/data/{part_name}/data"][:, 4]

    # --- Vectorized per-event computations ---

    # Signal hit count per event: sum of (label != 0) per segment
    signal_mask = labels != 0  # bool array for masking
    is_signal_int = signal_mask.astype(np.int32)  # int array for reduceat
    n_signal = np.add.reduceat(is_signal_int, hit_start)

    # Mean z per event
    z_sums = np.add.reduceat(z_coords, hit_start)
    mean_z = (z_sums / n_hits).astype(np.float32)

    # Unique signal strings per event:
    # Assign each signal hit its string ID, noise hits get -1 (sentinel).
    # Within each event segment, count unique non-negative values.
    string_ids = np.where(signal_mask, channels // STRING_DIVISOR, -1)

    # Event index per hit (for grouping)
    event_idx = np.repeat(np.arange(n_events), n_hits)

    # Keep only signal hits, then count unique (event_idx, string_id) pairs
    if signal_mask.any():
        sig_events = event_idx[signal_mask]
        sig_strings = string_ids[signal_mask]

        # Encode (event_idx, string_id) as single int, then count unique per event
        combined = sig_events.astype(np.int64) * 1000 + sig_strings.astype(np.int64)
        sort_idx = combined.argsort()
        combined_sorted = combined[sort_idx]
        # A new unique pair occurs where consecutive values differ
        unique_mask = np.empty(len(combined_sorted), dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = combined_sorted[1:] != combined_sorted[:-1]
        # Recover event indices from sorted order
        unique_events = sig_events[sort_idx[unique_mask]]
        n_unique_signal_strings = np.bincount(unique_events, minlength=n_events).astype(np.int32)
    else:
        n_unique_signal_strings = np.zeros(n_events, dtype=np.int32)

    # Number of unique strings (all hits)
    num_un_strings = particle_group[f"raw/num_un_strings/{part_name}/data"][:]

    # Cluster ID per event
    cluster_ids = particle_group[f"raw/cluster_ids/{part_name}/data"][:]

    # Event IDs (byte strings)
    ev_ids = particle_group[f"ev_ids/{part_name}/data"][:]

    # Primary particle properties (MC only)
    prime_prty = particle_group[f"prime_prty/{part_name}/data"][:]

    return {
        "event_id": ev_ids.astype("S25"),
        "cluster_id": cluster_ids.astype(np.int32),
        "n_hits": n_hits,
        "n_signal_hits": n_signal,
        "n_unique_strings": num_un_strings.astype(np.int32),
        "n_unique_signal_strings": n_unique_signal_strings,
        "mean_z": mean_z,
        "energy": prime_prty[:, 2].astype(np.float32),
        "theta": prime_prty[:, 0].astype(np.float32),
        "phi": prime_prty[:, 1].astype(np.float32),
        "weight": prime_prty[:, 5].astype(np.float32),
        "h5_part_num": np.full(n_events, int(part_name.split("_")[1]), dtype=np.int32),
        "hit_start_idx": hit_start.astype(np.int64),
        "hit_end_idx": hit_end.astype(np.int64),
    }


def build_particle_catalog(
    h5_path: Path,
    particle_type: str,
    output_dir: Path,
) -> Optional[Path]:
    """Build and save a Parquet catalog for one particle type."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{particle_type}.parquet"

    logger.info(f"Processing particle type: {particle_type}")
    t0 = time.time()

    with h5py.File(h5_path, "r", rdcc_nbytes=0) as f:
        if particle_type not in f:
            logger.warning(f"Particle type '{particle_type}' not found in {h5_path}")
            return None

        group = f[particle_type]
        part_names = get_part_names(group)
        logger.info(f"  Found {len(part_names)} parts")

        # --- Pass 1: count total events per part to pre-allocate ---
        part_event_counts = []
        for part_name in part_names:
            n_ev = len(group[f"raw/ev_starts/{part_name}/data"]) - 1
            part_event_counts.append(n_ev)
        n_total = sum(part_event_counts)

        if n_total == 0:
            logger.warning(f"No events found for {particle_type}")
            return None

        logger.info(f"  Total events: {n_total:,}, pre-allocating arrays")

        # Pre-allocate output arrays
        # Event IDs are byte strings like b'muatm_12586_0'.
        # Store as bytes (S25) — 1 byte/char instead of 4 bytes/char for Unicode.
        event_id = np.empty(n_total, dtype="S25")
        cluster_id = np.empty(n_total, dtype=np.int32)
        n_hits_arr = np.empty(n_total, dtype=np.int32)
        n_signal_hits = np.empty(n_total, dtype=np.int32)
        n_unique_strings = np.empty(n_total, dtype=np.int32)
        n_unique_signal_strings = np.empty(n_total, dtype=np.int32)
        mean_z_arr = np.empty(n_total, dtype=np.float32)
        energy = np.empty(n_total, dtype=np.float32)
        theta = np.empty(n_total, dtype=np.float32)
        phi = np.empty(n_total, dtype=np.float32)
        weight = np.empty(n_total, dtype=np.float32)
        h5_part_num = np.empty(n_total, dtype=np.int32)
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
            cluster_id[sl] = part_data["cluster_id"]
            n_hits_arr[sl] = part_data["n_hits"]
            n_signal_hits[sl] = part_data["n_signal_hits"]
            n_unique_strings[sl] = part_data["n_unique_strings"]
            n_unique_signal_strings[sl] = part_data["n_unique_signal_strings"]
            mean_z_arr[sl] = part_data["mean_z"]
            energy[sl] = part_data["energy"]
            theta[sl] = part_data["theta"]
            phi[sl] = part_data["phi"]
            weight[sl] = part_data["weight"]
            h5_part_num[sl] = part_data["h5_part_num"]
            hit_start_idx[sl] = part_data["hit_start_idx"]
            hit_end_idx[sl] = part_data["hit_end_idx"]
            offset += n_ev

    # Trim in case some parts had 0 events
    n_total = offset

    df = pl.DataFrame({
        "event_id": event_id[:n_total],
        "cluster_id": cluster_id[:n_total],
        "n_hits": n_hits_arr[:n_total],
        "n_signal_hits": n_signal_hits[:n_total],
        "n_unique_strings": n_unique_strings[:n_total],
        "n_unique_signal_strings": n_unique_signal_strings[:n_total],
        "mean_z": mean_z_arr[:n_total],
        "energy": energy[:n_total],
        "theta": theta[:n_total],
        "phi": phi[:n_total],
        "weight": weight[:n_total],
        "h5_part_num": h5_part_num[:n_total],
        "hit_start_idx": hit_start_idx[:n_total],
        "hit_end_idx": hit_end_idx[:n_total],
    })
    df.write_parquet(output_path)

    elapsed = time.time() - t0
    logger.info(
        f"  Saved {n_total:,} events to {output_path} "
        f"({output_path.stat().st_size / 1e9:.2f} GB, {elapsed:.1f}s)"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build event catalogs from MC HDF5 file")
    parser.add_argument(
        "--h5", type=Path, default=DEFAULT_H5_PATH,
        help=f"Path to merged HDF5 file (default: {DEFAULT_H5_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for catalog files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--particles", nargs="+", default=None,
        help="Particle types to process (default: all found in file)",
    )
    args = parser.parse_args()

    if not args.h5.exists():
        logger.error(f"HDF5 file not found: {args.h5}")
        return

    # Determine particle types
    if args.particles:
        particle_types = args.particles
    else:
        with h5py.File(args.h5, "r") as f:
            particle_types = sorted(f.keys())
        logger.info(f"Found particle types: {particle_types}")

    # Save source path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_path.txt").write_text(str(args.h5.resolve()) + "\n")

    # Build catalog for each
    for pt in particle_types:
        build_particle_catalog(args.h5, pt, args.output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
