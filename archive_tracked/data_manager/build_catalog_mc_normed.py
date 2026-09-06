"""
Build per-regime event catalogs from baikal_2020_sig-noise_mid-eq_normed.h5.

Produces one Parquet file per regime (train, val, test) with the same columns
as build_catalog.py output for baikal_mc_merged, so catalogs can be cross-referenced.

Columns: event_id, cluster_id, n_hits, n_signal_hits, n_unique_strings,
         n_unique_signal_strings, mean_z, energy, theta, phi, weight,
         hit_start_idx, hit_end_idx

cluster_id is recovered from channels data: channels[first_hit_of_event] // CLUSTER_DIVISOR.

Usage:
    python data_manager/build_catalog_normed.py
    python data_manager/build_catalog_normed.py --regimes train val
    python data_manager/build_catalog_normed.py --h5 /path/to/file.h5 --output-dir /path/to/catalogs

Warning:
    train regime has ~19M events / ~1.4B hits. Per-hit arrays are loaded in chunks
    to limit memory usage. Expect ~20-30 GB RAM peak for train.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm
import h5py
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_H5_PATH = Path(
    "/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "catalogs_mc_signoise_normed"

STRING_DIVISOR = 36
CLUSTER_DIVISOR = 288

# Process hit arrays in chunks of this many events to limit RAM usage.
CHUNK_SIZE = 500_000


def build_regime_catalog(
    h5_path: Path,
    regime: str,
    output_dir: Path,
) -> Optional[Path]:
    """Build and save a Parquet catalog for one regime (train/val/test)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{regime}.parquet"

    logger.info(f"Processing regime: {regime}")
    t0 = time.time()

    with h5py.File(h5_path, "r", rdcc_nbytes=0) as f:
        if regime not in f:
            logger.warning(f"Regime '{regime}' not found in {h5_path}")
            return None

        grp = f[regime]

        # --- Read per-event arrays (small, load fully) ---
        ev_starts = grp["ev_starts/data"][:]
        n_events = len(ev_starts) - 1
        logger.info(f"  {n_events:,} events")

        if n_events == 0:
            logger.warning(f"No events in {regime}")
            return None

        hit_start = ev_starts[:-1]
        hit_end = ev_starts[1:]
        n_hits = (hit_end - hit_start).astype(np.int32)

        ev_ids = grp["ev_ids/data"][:]
        num_un_strings = grp["num_un_strings/data"][:].astype(np.int32)
        prime_prty = grp["prime_prty/data"][:]

        # --- Pre-allocate per-event output arrays ---
        cluster_id = np.empty(n_events, dtype=np.int32)
        n_signal_hits = np.empty(n_events, dtype=np.int32)
        n_unique_signal_strings = np.empty(n_events, dtype=np.int32)
        mean_z = np.empty(n_events, dtype=np.float32)

        # --- Per-hit arrays: process in chunks of events ---
        # Datasets for chunked reading
        ds_labels = grp["labels/data"]
        ds_channels = grp["channels/data"]
        ds_data = grp["data/data"]

        n_chunks = (n_events + CHUNK_SIZE - 1) // CHUNK_SIZE
        for chunk_idx in tqdm(range(n_chunks), desc=f"  {regime} chunks"):
            ev_lo = chunk_idx * CHUNK_SIZE
            ev_hi = min((chunk_idx + 1) * CHUNK_SIZE, n_events)
            ev_sl = slice(ev_lo, ev_hi)
            n_ev_chunk = ev_hi - ev_lo

            # Hit range for this chunk of events
            h_lo = int(hit_start[ev_lo])
            h_hi = int(hit_end[ev_hi - 1])

            # Read per-hit arrays for this chunk only
            labels_chunk = ds_labels[h_lo:h_hi]
            channels_chunk = ds_channels[h_lo:h_hi]
            z_chunk = ds_data[h_lo:h_hi, 4]

            # Local hit boundaries (relative to chunk start)
            local_start = (hit_start[ev_sl] - h_lo).astype(np.int64)
            local_n_hits = n_hits[ev_sl]

            # Cluster ID: channel of first hit per event // CLUSTER_DIVISOR
            cluster_id[ev_sl] = channels_chunk[local_start] // CLUSTER_DIVISOR

            # Signal hit count
            signal_mask = labels_chunk != 0
            is_signal_int = signal_mask.astype(np.int32)
            n_signal_hits[ev_sl] = np.add.reduceat(is_signal_int, local_start)

            # Mean z
            z_sums = np.add.reduceat(z_chunk, local_start)
            mean_z[ev_sl] = (z_sums / local_n_hits).astype(np.float32)

            # Unique signal strings per event
            string_ids = np.where(signal_mask, channels_chunk // STRING_DIVISOR, -1)
            event_idx = np.repeat(np.arange(n_ev_chunk), local_n_hits)

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
                n_unique_signal_strings[ev_sl] = np.bincount(
                    unique_events, minlength=n_ev_chunk
                ).astype(np.int32)
            else:
                n_unique_signal_strings[ev_sl] = 0

    # --- Build DataFrame ---
    df = pl.DataFrame({
        "event_id": ev_ids.astype("S25"),
        "cluster_id": cluster_id,
        "n_hits": n_hits,
        "n_signal_hits": n_signal_hits,
        "n_unique_strings": num_un_strings,
        "n_unique_signal_strings": n_unique_signal_strings,
        "mean_z": mean_z,
        "energy": prime_prty[:, 2].astype(np.float32),
        "theta": prime_prty[:, 0].astype(np.float32),
        "phi": prime_prty[:, 1].astype(np.float32),
        "weight": prime_prty[:, 5].astype(np.float32),
        "hit_start_idx": hit_start.astype(np.int64),
        "hit_end_idx": hit_end.astype(np.int64),
    })
    df.write_parquet(output_path)

    elapsed = time.time() - t0
    logger.info(
        f"  Saved {n_events:,} events to {output_path} "
        f"({output_path.stat().st_size / 1e6:.1f} MB, {elapsed:.1f}s)"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build event catalogs from normed HDF5 file"
    )
    parser.add_argument(
        "--h5", type=Path, default=DEFAULT_H5_PATH,
        help=f"Path to normed HDF5 file (default: {DEFAULT_H5_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for catalog files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--regimes", nargs="+", default=None,
        help="Regimes to process (default: all found in file, excluding norm_param)",
    )
    args = parser.parse_args()

    if not args.h5.exists():
        logger.error(f"HDF5 file not found: {args.h5}")
        return

    # Determine regimes
    if args.regimes:
        regimes = args.regimes
    else:
        with h5py.File(args.h5, "r") as f:
            regimes = [k for k in f.keys() if k != "norm_param"]
        regimes = sorted(regimes)
        logger.info(f"Found regimes: {regimes}")

    # Save source path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_path.txt").write_text(str(args.h5.resolve()) + "\n")

    # Extract and save normalization parameters
    norm_path = args.output_dir / "norm_params.yaml"
    with h5py.File(args.h5, "r") as f:
        if "norm_param" in f:
            means = f["norm_param/mean"][:].tolist()
            stds = f["norm_param/std"][:].tolist()
            norm_data = {
                "means": means,
                "stds": stds,
                "feature_names": ["amplitude", "time", "x", "y", "z"],
            }
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with open(norm_path, "w") as nf:
                yaml.dump(norm_data, nf, default_flow_style=False)
            logger.info(f"  Saved normalization params to {norm_path}")

    for regime in regimes:
        build_regime_catalog(args.h5, regime, args.output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
