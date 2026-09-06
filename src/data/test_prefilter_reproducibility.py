"""
Test training reproducibility with PrefilterDataset.

Runs 3 training steps twice with the same seed and verifies
that losses and model weights are identical.

Usage:
    python src/data/test_prefilter_reproducibility.py
"""

import torch
import numpy as np
from torch.utils.data import random_split
from src.data.prefilter_dataset import (
    PrefilterDataset, create_prefilter_dataloader,
)
from src.models.base_models import create_model
from src.training.metrics import binary_cross_entropy_with_logits_weighted

SEED = 42
DEVICE = "cuda:0"
H5_PATH = "/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5"
CATALOG_DIR = "data_manager/h5_catalogs/catalogs_mc_merged"

MODEL_CONFIG = {
    "type": "numu",
    "feature_extractor": {
        "input_dim": 5, "d_model": 32, "num_heads": 2,
        "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0,
        "pooling": "cls", "use_positional_encoding": True,
        "max_seq_len": 500,
    },
    "classifier": {
        "hidden_dims": [16], "dropout": 0.0, "use_batch_norm": False,
    },
}

NORM_CONFIG = {"means": [0.0] * 5, "stds": [1.0] * 5}
AUG_CONFIG = {
    "noise_std": [0.05, 5.0, 1.0, 1.0, 2.0],
    "rotation_enabled": True,
}


def run_training(run_id: int):
    """Run 3 training steps and return losses + weight checksum."""
    # Seed everything
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataset (lazy loading)
    ds = PrefilterDataset(
        h5_path=H5_PATH, catalog_dir=CATALOG_DIR,
        particle_types=["muatm_2020", "nue2_2020"],
        neutrino_types=["nue2_2020"],
        events_per_particle={"muatm_2020": 64, "nue2_2020": 64},
        h_min=5, max_hits=200,
        soft_label_config={"mode": "linear", "h_saturate": 15},
        shuffle_events=True, device=DEVICE, seed=SEED,
        part_cache_size=4,
    )

    train_ds, val_ds = random_split(
        ds, [96, 32],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_dl = create_prefilter_dataloader(
        train_ds, batch_size=32, shuffle=True,
        normalization_config=NORM_CONFIG,
        augmentation_config=AUG_CONFIG,
        shuffle_batch=True, num_workers=0,
    )

    # Model (re-seed before creation for identical init weights)
    torch.manual_seed(SEED)
    model = create_model(MODEL_CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Train 3 steps
    model.train()
    losses = []
    for step, batch in enumerate(train_dl):
        if step >= 3:
            break
        logits = model(batch)
        loss = binary_cross_entropy_with_logits_weighted(
            logits, batch["labels"].float(),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Collect model weights checksum
    w_sum = sum(p.sum().item() for p in model.parameters())
    return losses, w_sum


if __name__ == "__main__":
    print("Run 1...")
    losses1, w1 = run_training(1)
    print(f"  Losses: {losses1}")
    print(f"  Weight sum: {w1:.10f}")

    print("Run 2...")
    losses2, w2 = run_training(2)
    print(f"  Losses: {losses2}")
    print(f"  Weight sum: {w2:.10f}")

    match = (losses1 == losses2) and (w1 == w2)
    print(f"\nReproducible: {match}")
    if not match:
        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            print(f"  Step {i}: {l1} vs {l2}, diff={abs(l1 - l2):.2e}")
        print(f"  Weight diff: {abs(w1 - w2):.2e}")
