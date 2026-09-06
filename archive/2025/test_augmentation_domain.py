"""Test: compare source and target train loader outputs under augmentation.

Checks whether augmentation introduces systematic feature differences between
MC (source) and exp (target) that would make domains artificially separable.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data.prefilter_npy_dataset import (
    PrefilterNpyDataset,
    ExpDataset,
    create_prefilter_npy_dataloader,
)

# --- Config (match baseline) ---
CONFIG = {
    "npy_dir": "/home2/albert/Baikal/datasets/baikal_mc2020_prefilter/",
    "exp_h5": "data_manager/data/h5datasets/exp.h5",
    "max_hits": 500,
    "max_events": 5000,
    "batch_size": 256,
    "seed": 42,
    "normalization": None,  # skip norm to see raw augmented values
    "augmentation": {
        "noise_std": None, #[0.05, 5.0, 1.0, 1.0, 2.0],
        "rotation_enabled": True,
    },
}

FEATURE_NAMES = ["amplitude", "time", "x", "y", "z"]

def get_feature_stats(loader, n_batches=10, label=""):
    means = []
    stds = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        feat = batch["features"]   # (B, L, 5)
        mask = batch["mask"]       # (B, L)
        # Flatten to (N_hits, 5) using mask
        flat = feat[mask]          # (N_valid_hits, 5)
        means.append(flat.mean(dim=0))
        stds.append(flat.std(dim=0))

    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)
    print(f"\n{label} feature stats (mean over {n_batches} batches):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:12s}: mean={mean[i]:8.3f}  std={std[i]:8.3f}")
    return mean, std


def main():
    print("Loading source dataset (MC NPY, muons only)...")
    src_ds = PrefilterNpyDataset(
        npy_dir=CONFIG["npy_dir"],
        max_hits=CONFIG["max_hits"],
        max_events=CONFIG["max_events"],
        seed=CONFIG["seed"],
    )
    src_train, _ = src_ds.split(0.85, seed=CONFIG["seed"])
    # Filter to muatm only (particle_type == 0)
    mu_mask = src_train.particle_types[src_train._indices] == 0
    src_train._indices = src_train._indices[mu_mask]
    print(f"  Muon-only source train: {len(src_train):,} events")

    print("Loading target dataset (exp.h5)...")
    tgt_ds = ExpDataset(
        h5_path=CONFIG["exp_h5"],
        max_hits=CONFIG["max_hits"],
        max_events=CONFIG["max_events"],
        seed=CONFIG["seed"],
    )
    tgt_train, _ = tgt_ds.split(0.85, seed=CONFIG["seed"])

    # --- With augmentation ---
    src_aug_loader = create_prefilter_npy_dataloader(
        src_train, CONFIG["batch_size"], shuffle=True,
        normalization_config=CONFIG["normalization"],
        augmentation_config=CONFIG["augmentation"],
        shuffle_batch=True, device="cpu",
    )
    tgt_aug_loader = create_prefilter_npy_dataloader(
        tgt_train, CONFIG["batch_size"], shuffle=True,
        normalization_config=CONFIG["normalization"],
        augmentation_config=CONFIG["augmentation"],
        shuffle_batch=True, device="cpu",
    )

    # --- Without augmentation ---
    src_noaug_loader = create_prefilter_npy_dataloader(
        src_train, CONFIG["batch_size"], shuffle=True,
        normalization_config=CONFIG["normalization"],
        augmentation_config=None,
        shuffle_batch=False, device="cpu",
    )
    tgt_noaug_loader = create_prefilter_npy_dataloader(
        tgt_train, CONFIG["batch_size"], shuffle=True,
        normalization_config=CONFIG["normalization"],
        augmentation_config=None,
        shuffle_batch=False, device="cpu",
    )

    N = 20
    src_mean_aug, src_std_aug   = get_feature_stats(src_aug_loader,   N, "SOURCE  (aug)")
    tgt_mean_aug, tgt_std_aug   = get_feature_stats(tgt_aug_loader,   N, "TARGET  (aug)")
    src_mean_raw, src_std_raw   = get_feature_stats(src_noaug_loader, N, "SOURCE  (raw)")
    tgt_mean_raw, tgt_std_raw   = get_feature_stats(tgt_noaug_loader, N, "TARGET  (raw)")

    print("\n--- Mean difference (source - target) ---")
    print(f"{'feature':12s}  {'aug':>10s}  {'raw':>10s}  {'aug/raw ratio':>14s}")
    for i, name in enumerate(FEATURE_NAMES):
        diff_aug = (src_mean_aug[i] - tgt_mean_aug[i]).abs().item()
        diff_raw = (src_mean_raw[i] - tgt_mean_raw[i]).abs().item()
        ratio = diff_aug / (diff_raw + 1e-8)
        print(f"  {name:12s}  {diff_aug:10.4f}  {diff_raw:10.4f}  {ratio:14.2f}x")

    print("\n--- Std difference (source - target) ---")
    print(f"{'feature':12s}  {'aug':>10s}  {'raw':>10s}  {'aug/raw ratio':>14s}")
    for i, name in enumerate(FEATURE_NAMES):
        diff_aug = (src_std_aug[i] - tgt_std_aug[i]).abs().item()
        diff_raw = (src_std_raw[i] - tgt_std_raw[i]).abs().item()
        ratio = diff_aug / (diff_raw + 1e-8)
        print(f"  {name:12s}  {diff_aug:10.4f}  {diff_raw:10.4f}  {ratio:14.2f}x")


if __name__ == "__main__":
    main()
