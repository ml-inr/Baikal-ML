"""Main pipeline: HDF5 → sig-noise filtering → balance → shuffle → write .npy.

Two-pass approach:
    Pass 1: load raw/data per part, run sig-noise model, compute per-event
            n_sig_hits and n_sig_strings. Keep only scalar metadata in RAM.
    Apply event cuts (n_sig_hits >= min_hits, n_sig_strings >= min_strings).
    Balance classes: 2n muatm vs n nuatm + n nue2.
    Pass 2: re-run model on selected parts, write prob-filtered hit features
            and per-hit probabilities to memory-mapped .npy files.

Usage:
    python -m data_manager.nu_classifier_ds_builder \
        --config data_manager/nu_classifier_ds_builder/default_config.yaml
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

from .io import read_all_metadata, write_selected_features
from .balance import balance_classes, PARTICLE_ENCODE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARTICLE_MAP = {
    "muatm": "muatm_2020",
    "nuatm": "nuatm_2020",
    "nue2":  "nue2_2020",
}
NEUTRINO_TYPES = {"nuatm_2020", "nue2_2020"}

DEFAULT_H5_PATH = "data_manager/data/h5datasets/baikal_mc_merged.h5"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_nu_classifier_npy(cfg: dict) -> None:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed      = cfg.get("seed", 42)
    h5_path   = cfg.get("h5_path", DEFAULT_H5_PATH)
    threshold = float(cfg.get("sig_noise_threshold", 0.5))

    cuts_cfg    = cfg.get("event_cuts", {})
    min_hits    = int(cuts_cfg.get("min_hits", 8))
    min_strings = int(cuts_cfg.get("min_strings", 2))

    model_cfg   = cfg.get("sig_noise_model", {})
    model_device = model_cfg.get("device", "auto")
    batch_size  = int(model_cfg.get("batch_size", 128))

    with open(cfg["parts_json"]) as f:
        parts_dict = json.load(f)
    logger.info(f"Parts JSON: {cfg['parts_json']}")
    for k, v in parts_dict.items():
        logger.info(f"  {k}: {len(v)} parts")

    # ── Load sig-noise model ──────────────────────────────────────────────
    logger.info("\nLoading sig-noise model...")
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
        load_model,
    )
    model, _, device = load_model(device=model_device)
    logger.info(f"  Model on {device}, threshold={threshold}")

    t_total = time.time()

    # ── Pass 1: metadata ──────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Pass 1: running sig-noise model, collecting metadata")
    logger.info(f"{'='*60}")

    with h5py.File(h5_path, "r") as h5:
        meta = read_all_metadata(
            h5_file=h5,
            parts_dict=parts_dict,
            particle_map=PARTICLE_MAP,
            particle_encode=PARTICLE_ENCODE,
            model=model,
            batch_size=batch_size,
            device=device,
            threshold=threshold,
        )

    n_sig_hits      = meta["n_sig_hits"]
    n_sig_strings   = meta["n_sig_strings"]
    n_gt_sig_hits   = meta["n_gt_sig_hits"]
    n_gt_sig_strings = meta["n_gt_sig_strings"]
    particle_types  = meta["particle_types"]
    locations       = meta["locations"]
    n_events      = len(n_sig_hits)

    logger.info(f"\nPass 1 done: {n_events:,} events total")

    # ── Event cuts ────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Applying cuts: n_sig_hits >= {min_hits}, n_sig_strings >= {min_strings}")
    logger.info(f"{'='*60}")

    cut_mask = (n_sig_hits >= min_hits) & (n_sig_strings >= min_strings)
    cut_idx  = np.where(cut_mask)[0]

    for ptype_full, code in PARTICLE_ENCODE.items():
        before = int((particle_types == code).sum())
        after  = int((particle_types[cut_idx] == code).sum())
        logger.info(f"  {ptype_full}: {before:,} → {after:,} ({100*after/max(before,1):.1f}%)")

    # Work on the cut subset from here on
    particle_types_cut = particle_types[cut_idx]
    n_cut = len(cut_idx)
    logger.info(f"  Total surviving: {n_cut:,} / {n_events:,}")

    # ── Balance ───────────────────────────────────────────────────────────
    rng = np.random.RandomState(seed)
    do_balance = bool(cfg.get("balance", True))
    logger.info(f"\n{'='*60}")
    logger.info(f"Class balancing: {'enabled' if do_balance else 'disabled'}")
    logger.info(f"{'='*60}")

    if do_balance:
        selected_in_cut = balance_classes(particle_types_cut, rng)
        selected = cut_idx[selected_in_cut]
    else:
        selected = cut_idx

    # ── Shuffle ───────────────────────────────────────────────────────────
    rng2 = np.random.RandomState(seed + 1)
    rng2.shuffle(selected)
    n_selected = len(selected)
    logger.info(f"\n{n_selected:,} events after {'balance + ' if do_balance else ''}shuffle")

    # ── Extract h5 back-links for selected events ─────────────────────────
    sel_part_keys    = np.array([locations[i].part_key          for i in selected], dtype=object)
    sel_h5_local_ids = np.array([locations[i].event_idx_in_part for i in selected], dtype=np.int32)

    # ── Build output offsets (n_sig_hits after cuts, per selected event) ──
    sig_lengths = n_sig_hits[selected].astype(np.int64)
    offsets = np.zeros(n_selected + 1, dtype=np.int64)
    np.cumsum(sig_lengths, out=offsets[1:])

    # ── Pass 2: write filtered features + probs ───────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Pass 2: writing filtered features to {output_dir}")
    logger.info(f"{'='*60}")

    write_selected_features(
        h5_path=h5_path,
        locations=locations,
        selected_idx=selected,
        offsets=offsets,
        features_path=str(output_dir / "features.npy"),
        probs_path=str(output_dir / "probs.npy"),
        model=model,
        batch_size=batch_size,
        device=device,
        threshold=threshold,
    )

    # ── Write per-event arrays ────────────────────────────────────────────
    logger.info("\nWriting per-event arrays...")

    def save(name: str, arr: np.ndarray) -> None:
        path = output_dir / name
        np.save(path, arr)
        logger.info(f"  {name}: {arr.shape} {arr.dtype} ({arr.nbytes / 1e6:.1f} MB)")

    sel_particle_types = particle_types[selected]
    sel_labels = np.isin(
        sel_particle_types,
        [PARTICLE_ENCODE[t] for t in NEUTRINO_TYPES],
    ).astype(np.float32)

    save("offsets.npy",           offsets)
    save("labels.npy",            sel_labels)
    save("n_sig_hits.npy",        n_sig_hits[selected].astype(np.int32))
    save("n_sig_strings.npy",     n_sig_strings[selected].astype(np.int32))
    save("n_gt_sig_hits.npy",     n_gt_sig_hits[selected].astype(np.int32))
    save("n_gt_sig_strings.npy",  n_gt_sig_strings[selected].astype(np.int32))
    save("particle_types.npy",    sel_particle_types)
    save("h5_part_keys.npy",      sel_part_keys)
    save("h5_local_event_ids.npy", sel_h5_local_ids)

    npy_names = [
        "features.npy", "probs.npy", "offsets.npy", "labels.npy",
        "n_sig_hits.npy", "n_sig_strings.npy",
        "n_gt_sig_hits.npy", "n_gt_sig_strings.npy", "particle_types.npy",
        "h5_part_keys.npy", "h5_local_event_ids.npy",
    ]
    total_bytes = sum((output_dir / f).stat().st_size for f in npy_names)

    # ── Dataset info JSON ─────────────────────────────────────────────────
    info = {
        "n_events":          n_selected,
        "n_sig_hits_total":  int(offsets[-1]),
        "total_bytes":       total_bytes,
        "sig_noise_threshold": threshold,
        "event_cuts":        {"min_hits": min_hits, "min_strings": min_strings},
        "h5_source":         cfg.get("h5_source", "mc_merged"),
        "particle_encode":   PARTICLE_ENCODE,
        "parts":             {k: len(v) for k, v in parts_dict.items()},
        "label_counts":      {
            "neutrino": int(sel_labels.sum()),
            "muatm":    int((sel_labels == 0).sum()),
        },
    }
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"\nTotal: {total_bytes / 1e9:.2f} GB, {n_selected:,} events")
    logger.info(f"Done in {time.time() - t_total:.0f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build flat .npy training arrays for nu-classifier model"
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "build_config.yaml")

    build_nu_classifier_npy(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
