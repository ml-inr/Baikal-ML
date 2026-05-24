"""Main pipeline: read HDF5 → soft labels → balance → shuffle → write .npy.

Two-pass approach for low memory usage:
    Pass 1: read only metadata (ev_starts, labels, channels) — ~500 MB RAM.
    Pass 2: write selected events' features via memory-mapped .npy.

Usage:
    python -m data_manager.prefilter_npy_ds_builder.build \
        --config data_manager/prefilter_npy_ds_builder/default_config.yaml
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import yaml

from .io import read_all_metadata, write_selected_features
from .soft_labels import compute_soft_label
from .balance import balance_classes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARTICLE_MAP = {
    "muatm": "muatm_2020",
    "nuatm": "nuatm_2020",
    "nue2": "nue2_2020",
}
PARTICLE_ENCODE = {
    "muatm_2020": 0,
    "nuatm_2020": 1,
    "nue2_2020": 2,
}
NEUTRINO_TYPES = {"nuatm_2020", "nue2_2020"}

DEFAULT_H5_PATH = "/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_prefilter_npy(cfg: dict) -> None:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    max_hits = cfg.get("max_hits")
    do_balance = cfg.get("balance", False)
    h5_path = cfg.get("h5_path", DEFAULT_H5_PATH)

    soft_cfg = cfg.get("soft_labels", {})
    soft_mode = soft_cfg.get("mode", "linear")
    h_saturate = soft_cfg.get("h_saturate", 15)
    steepness = soft_cfg.get("steepness")

    with open(cfg["parts_json"]) as f:
        parts_dict = json.load(f)
    logger.info(f"Parts JSON: {cfg['parts_json']}")
    for k, v in parts_dict.items():
        logger.info(f"  {k}: {len(v)} parts")

    # ---- Pass 1: Read metadata only (no hit features) ----
    logger.info(f"\n{'='*60}")
    logger.info("Pass 1: Reading event metadata from HDF5")
    logger.info(f"{'='*60}")

    t_total = time.time()
    with h5py.File(h5_path, "r") as h5:
        meta = read_all_metadata(h5, parts_dict, PARTICLE_MAP, PARTICLE_ENCODE)

    n_hits_arr = meta["n_hits"]
    n_signal_hits = meta["n_signal_hits"]
    n_signal_strings = meta["n_signal_strings"]
    particle_types = meta["particle_types"]
    locations = meta["locations"]
    n_events = len(n_hits_arr)

    elapsed = time.time() - t_total
    logger.info(
        f"\nMetadata read: {n_events:,} events in {elapsed:.0f}s "
        f"(RAM ≈ {n_events * 20 / 1e6:.0f} MB)"
    )

    # ---- Compute labels ----
    logger.info(f"\n{'='*60}")
    logger.info("Computing labels")
    logger.info(f"{'='*60}")

    is_neutrino = np.isin(
        particle_types,
        [PARTICLE_ENCODE[t] for t in NEUTRINO_TYPES],
    )
    labels = np.zeros(n_events, dtype=np.float32)

    if soft_mode == "hard":
        # All neutrino events get label 1.0 regardless of signal hits
        labels[is_neutrino] = 1.0
        mu_split_threshold = None  # no meaningful split for balancing
        logger.info("  Mode: hard — all neutrino events labeled 1.0")
    else:
        labels[is_neutrino] = compute_soft_label(
            n_signal_hits[is_neutrino],
            is_neutrino=True,
            mode=soft_mode,
            h_saturate=h_saturate,
            steepness=steepness,
        )
        mu_split_threshold = h_saturate // 2
        logger.info(f"  Mode: {soft_mode}, h_saturate={h_saturate}")
        logger.info(
            f"  Neutrino labels: mean={labels[is_neutrino].mean():.3f}, "
            f"median={np.median(labels[is_neutrino]):.3f}"
        )

    logger.info(f"  Labels >= 0.5: {(labels >= 0.5).sum():,} / {n_events:,}")

    # ---- Balance (optional) ----
    if do_balance:
        logger.info(f"\n{'='*60}")
        logger.info("Class balancing")
        logger.info(f"{'='*60}")

        rng = np.random.RandomState(seed)
        selected = balance_classes(
            labels, particle_types, n_signal_hits,
            rng=rng,
            mu_split_hits_threshold=mu_split_threshold,
        )
    else:
        selected = np.arange(n_events)

    # ---- Shuffle ----
    logger.info(f"\n{'='*60}")
    logger.info("Shuffling events")
    logger.info(f"{'='*60}")

    rng = np.random.RandomState(seed + 1 if seed is not None else None)
    rng.shuffle(selected)
    n_selected = len(selected)
    logger.info(f"  {n_selected:,} events after selection + shuffle")

    # ---- Build output offsets (with optional max_hits truncation) ----
    lengths = n_hits_arr[selected].astype(np.int64)
    if max_hits is not None:
        lengths = np.minimum(lengths, max_hits)
    offsets = np.zeros(n_selected + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    # ---- Pass 2: Write features via mmap ----
    logger.info(f"\n{'='*60}")
    logger.info(f"Pass 2: Writing features to {output_dir}")
    logger.info(f"{'='*60}")

    features_path = str(output_dir / "features.npy")
    write_selected_features(
        h5_path=h5_path,
        locations=locations,
        selected_idx=selected,
        offsets=offsets,
        output_path=features_path,
        max_hits=max_hits,
    )

    # ---- Write per-event arrays ----
    logger.info(f"\nWriting per-event arrays...")

    def save(name: str, arr: np.ndarray) -> None:
        path = output_dir / name
        np.save(path, arr)
        size_mb = arr.nbytes / 1e6
        logger.info(f"  {name}: {arr.shape} {arr.dtype} ({size_mb:.1f} MB)")

    save("offsets.npy", offsets)
    save("labels.npy", labels[selected])
    save("signal_hits.npy", n_signal_hits[selected])
    save("signal_strings.npy", n_signal_strings[selected])
    save("particle_types.npy", particle_types[selected])

    npy_names = [
        "features.npy", "offsets.npy", "labels.npy",
        "signal_hits.npy", "signal_strings.npy", "particle_types.npy",
    ]
    total_bytes = sum((output_dir / f).stat().st_size for f in npy_names)
    logger.info(f"\nTotal: {total_bytes / 1e9:.2f} GB, {n_selected:,} events")

    # ---- Save metadata ----
    info = {
        "n_events": n_selected,
        "n_hits": int(offsets[-1]),
        "total_bytes": total_bytes,
        "particle_encode": PARTICLE_ENCODE,
        "parts": {k: len(v) for k, v in parts_dict.items()},
    }
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"  Metadata: {info_path}")

    logger.info(f"\nDone in {time.time() - t_total:.0f}s total.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build flat .npy training arrays for prefilter model"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Save config copy alongside the data
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "build_config.yaml")

    build_prefilter_npy(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
