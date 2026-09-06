"""
Build event catalog from experimental reco data file (exp_reco.h5).

Combines exp-style structure (string part names, no MC truth) with:
- Signal/noise labels from ScanfitMask (reco hit mask, stored as raw/labels)
- Per-event reco parameters from reco_prty (analogous to prime_prty in MC)

reco_prty columns (13 fields):
  [thetaRec, phiRec, thetaErr, phiErr, funcValue, timeChi2,
   chargeTerm, LLFit, nHits, nStrings, nOMs, pathLength, timeXYZRec]

String number is derived from channel_id // 36 (STRING_DIVISOR).

Usage:
    python data_manager/build_catalog_exp_reco.py
    python data_manager/build_catalog_exp_reco.py --h5 /path/to/exp_reco.h5 --output-dir /path/to/catalogs
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

DEFAULT_H5_PATH = Path(__file__).parent / "data" / "h5datasets" / "exp_reco.h5"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "h5_catalogs/catalogs_exp_reco"

STRING_DIVISOR = 36


def get_part_names(particle_group: h5py.Group) -> List[str]:
    """Get sorted list of part_* names from ev_starts subgroup."""
    return sorted(particle_group["raw/ev_starts"].keys())


def process_part(
    particle_group: h5py.Group,
    part_name: str,
) -> Optional[dict]:
    """Extract per-event metadata from a single part."""
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

    # Signal hit count per event (label != 0 = signal from ScanfitMask)
    signal_mask = labels != 0
    is_signal_int = signal_mask.astype(np.int32)
    n_signal = np.add.reduceat(is_signal_int, hit_start)

    # Mean z per event
    z_sums = np.add.reduceat(z_coords, hit_start)
    mean_z = (z_sums / n_hits).astype(np.float32)

    # Unique signal strings per event
    string_ids = np.where(signal_mask, channels // STRING_DIVISOR, -1)
    event_idx = np.repeat(np.arange(n_events), n_hits)

    if signal_mask.any():
        sig_events = event_idx[signal_mask]
        sig_strings = string_ids[signal_mask]

        combined = sig_events.astype(np.int64) * 1000 + sig_strings.astype(np.int64)
        sort_idx = combined.argsort()
        combined_sorted = combined[sort_idx]

        unique_mask = np.empty(len(combined_sorted), dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = combined_sorted[1:] != combined_sorted[:-1]

        unique_events = sig_events[sort_idx[unique_mask]]
        n_unique_signal_strings = np.bincount(unique_events, minlength=n_events).astype(np.int32)
    else:
        n_unique_signal_strings = np.zeros(n_events, dtype=np.int32)

    # Number of unique strings (all hits)
    num_un_strings = particle_group[f"raw/num_un_strings/{part_name}/data"][:]

    # # Cluster ID per event
    # cluster_ids = particle_group[f"raw/cluster_ids/{part_name}/data"][:]

    # Event IDs (byte strings)
    ev_ids = particle_group[f"ev_ids/{part_name}/data"][:]

    # Reco per-event properties
    # Columns: [thetaRec, phiRec, thetaErr, phiErr, funcValue, timeChi2,
    #           chargeTerm, LLFit, nHits, nStrings, nOMs, pathLength, timeXYZRec]
    reco_prty = particle_group[f"reco_prty/{part_name}/data"][:]

    return {
        "event_id": ev_ids.astype("S30"),
        #"cluster_id": cluster_ids.astype(np.int32),
        "n_hits": n_hits,
        "n_signal_hits": n_signal,
        "n_unique_strings": num_un_strings.astype(np.int32),
        "n_unique_signal_strings": n_unique_signal_strings,
        "mean_z": mean_z,
        "reco_theta": reco_prty[:, 0].astype(np.float32),
        "reco_phi": reco_prty[:, 1].astype(np.float32),
        "reco_theta_err": reco_prty[:, 2].astype(np.float32),
        "reco_phi_err": reco_prty[:, 3].astype(np.float32),
        "reco_func_value": reco_prty[:, 4].astype(np.float32),
        "reco_time_chi2": reco_prty[:, 5].astype(np.float32),
        "reco_charge_term": reco_prty[:, 6].astype(np.float32),
        "reco_ll_fit": reco_prty[:, 7].astype(np.float32),
        "reco_n_hits": reco_prty[:, 8].astype(np.float32),
        "reco_n_strings": reco_prty[:, 9].astype(np.float32),
        "reco_n_oms": reco_prty[:, 10].astype(np.float32),
        "reco_path_length": reco_prty[:, 11].astype(np.float32),
        "reco_time_xyz": reco_prty[:, 12].astype(np.float32),
        "h5_part": np.full(n_events, part_name, dtype="U30"),
        "hit_start_idx": hit_start.astype(np.int64),
        "hit_end_idx": hit_end.astype(np.int64),
    }


def build_exp_reco_catalog(
    h5_path: Path,
    output_dir: Path,
) -> Optional[Path]:
    """Build and save a Parquet catalog for experimental reco data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "exp_reco.parquet"

    logger.info(f"Processing experimental reco data: {h5_path}")
    t0 = time.time()

    with h5py.File(h5_path, "r", rdcc_nbytes=0) as f:
        if "exp_reco" not in f:
            logger.error("'exp' group not found in file")
            return None

        group = f["exp_reco"]
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
        n_signal_hits = np.empty(n_total, dtype=np.int32)
        n_unique_strings = np.empty(n_total, dtype=np.int32)
        n_unique_signal_strings = np.empty(n_total, dtype=np.int32)
        mean_z_arr = np.empty(n_total, dtype=np.float32)
        reco_theta = np.empty(n_total, dtype=np.float32)
        reco_phi = np.empty(n_total, dtype=np.float32)
        reco_theta_err = np.empty(n_total, dtype=np.float32)
        reco_phi_err = np.empty(n_total, dtype=np.float32)
        reco_func_value = np.empty(n_total, dtype=np.float32)
        reco_time_chi2 = np.empty(n_total, dtype=np.float32)
        reco_charge_term = np.empty(n_total, dtype=np.float32)
        reco_ll_fit = np.empty(n_total, dtype=np.float32)
        reco_n_hits = np.empty(n_total, dtype=np.float32)
        reco_n_strings = np.empty(n_total, dtype=np.float32)
        reco_n_oms = np.empty(n_total, dtype=np.float32)
        reco_path_length = np.empty(n_total, dtype=np.float32)
        reco_time_xyz = np.empty(n_total, dtype=np.float32)
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
            n_signal_hits[sl] = part_data["n_signal_hits"]
            n_unique_strings[sl] = part_data["n_unique_strings"]
            n_unique_signal_strings[sl] = part_data["n_unique_signal_strings"]
            mean_z_arr[sl] = part_data["mean_z"]
            reco_theta[sl] = part_data["reco_theta"]
            reco_phi[sl] = part_data["reco_phi"]
            reco_theta_err[sl] = part_data["reco_theta_err"]
            reco_phi_err[sl] = part_data["reco_phi_err"]
            reco_func_value[sl] = part_data["reco_func_value"]
            reco_time_chi2[sl] = part_data["reco_time_chi2"]
            reco_charge_term[sl] = part_data["reco_charge_term"]
            reco_ll_fit[sl] = part_data["reco_ll_fit"]
            reco_n_hits[sl] = part_data["reco_n_hits"]
            reco_n_strings[sl] = part_data["reco_n_strings"]
            reco_n_oms[sl] = part_data["reco_n_oms"]
            reco_path_length[sl] = part_data["reco_path_length"]
            reco_time_xyz[sl] = part_data["reco_time_xyz"]
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
        "n_signal_hits": n_signal_hits[:n_total],
        "n_unique_strings": n_unique_strings[:n_total],
        "n_unique_signal_strings": n_unique_signal_strings[:n_total],
        "mean_z": mean_z_arr[:n_total],
        "reco_theta": reco_theta[:n_total],
        "reco_phi": reco_phi[:n_total],
        "reco_theta_err": reco_theta_err[:n_total],
        "reco_phi_err": reco_phi_err[:n_total],
        "reco_func_value": reco_func_value[:n_total],
        "reco_time_chi2": reco_time_chi2[:n_total],
        "reco_charge_term": reco_charge_term[:n_total],
        "reco_ll_fit": reco_ll_fit[:n_total],
        "reco_n_hits": reco_n_hits[:n_total],
        "reco_n_strings": reco_n_strings[:n_total],
        "reco_n_oms": reco_n_oms[:n_total],
        "reco_path_length": reco_path_length[:n_total],
        "reco_time_xyz": reco_time_xyz[:n_total],
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
    parser = argparse.ArgumentParser(description="Build event catalog from experimental reco HDF5 file")
    parser.add_argument(
        "--h5", type=Path, default=DEFAULT_H5_PATH,
        help=f"Path to experimental reco HDF5 file (default: {DEFAULT_H5_PATH})",
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

    build_exp_reco_catalog(args.h5, args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
