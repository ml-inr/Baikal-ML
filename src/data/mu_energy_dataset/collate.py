"""Collate function for :class:`MuEnergyNpyDataset` batches.

Pads variable-length hit sequences, builds the padding mask, applies optional train-time
augmentation (rotation around z and z-flip), normalizes the input features with the MC
statistics and optionally z-normalizes the regression target.  Returns CPU tensors; the
trainer moves them to the device.  The target is returned both normalized (``targets``) and
in raw dex (``targets_dex``) so the trainer can keep its losses in z units and its metrics
in dex.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def mu_energy_collate_fn(
    batch: List[Dict[str, Any]],
    normalization_config: Optional[Dict[str, List[float]]] = None,
    target_normalization: Optional[Dict[str, float]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    shuffle_batch: bool = False,
) -> Dict[str, torch.Tensor]:
    if shuffle_batch:
        batch = list(batch)
        np.random.shuffle(batch)

    features_list = [item["features"] for item in batch]
    targets_dex = torch.stack([item["target"] for item in batch])
    energies = torch.stack([item["energy"] for item in batch])
    weights = torch.stack([item["weight"] for item in batch])
    lengths = torch.stack([item["lengths"] for item in batch])
    original_lengths = torch.stack([item["original_lengths"] for item in batch])
    signal_hit_counts = torch.tensor(
        [item["signal_hit_count"] for item in batch], dtype=torch.long
    )
    signal_string_counts = torch.tensor(
        [item["signal_string_count"] for item in batch], dtype=torch.long
    )
    particle_types = torch.tensor(
        [item["particle_type"] for item in batch], dtype=torch.long
    )
    theta = torch.stack([item["theta"] for item in batch])
    event_fks = torch.tensor(
        [item["event_fk"] for item in batch], dtype=torch.int64
    )
    local_event_ids = torch.tensor(
        [item["local_event_id"] for item in batch], dtype=torch.int64
    )
    energy_status = torch.tensor(
        [item["energy_status"] for item in batch], dtype=torch.long
    )
    part_keys = [item["part_key"] for item in batch]

    batch_size = len(features_list)
    max_len = int(lengths.max().item())
    feature_dim = features_list[0].shape[1]
    padded = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    for i, feat in enumerate(features_list):
        padded[i, :feat.shape[0]] = feat

    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    # --- Train-time augmentation (before feature normalization) ---
    if augmentation_config is not None:
        if augmentation_config.get("rotation_enabled", False):
            angles = torch.rand(batch_size) * 2 * torch.pi
            cos_a, sin_a = torch.cos(angles), torch.sin(angles)
            x = padded[:, :, 2].clone()
            y = padded[:, :, 3].clone()
            x_rot = cos_a[:, None] * x - sin_a[:, None] * y
            y_rot = sin_a[:, None] * x + cos_a[:, None] * y
            padded[:, :, 2] = torch.where(mask, x_rot, padded[:, :, 2])
            padded[:, :, 3] = torch.where(mask, y_rot, padded[:, :, 3])

        if augmentation_config.get("zflip_enabled", False):
            p_flip = float(augmentation_config.get("zflip_prob", 0.5))
            flips = torch.rand(batch_size) < p_flip
            z = padded[:, :, 4]
            padded[:, :, 4] = torch.where(
                mask & flips[:, None], -z, z
            )

        # Truncated-Gaussian noise on the hit features.  ``noise_std`` is in raw
        # feature units (Q in PE, t in ns, x/y/z in m), like the nu-classifier
        # collate.  ``noise_truncate_sigma`` resamples the |z| > k tails, so the
        # result is a true truncated normal; None/0 keeps the plain Gaussian.
        # When noise is enabled the hits are re-sorted by time, as in the
        # nu-classifier augmentation.  Without noise_std this branch is a no-op,
        # so existing configurations are byte-for-byte unchanged.
        noise_std = augmentation_config.get("noise_std")
        if noise_std is not None:
            ns = torch.tensor(noise_std, dtype=torch.float32)
            if ns.numel() != feature_dim:
                raise ValueError(
                    f"noise_std has {ns.numel()} entries, expected {feature_dim}"
                )
            noise = torch.randn_like(padded)
            trunc = augmentation_config.get("noise_truncate_sigma")
            if trunc is not None and float(trunc) > 0:
                k = float(trunc)
                for _ in range(10):
                    bad = noise.abs() > k
                    if not bool(bad.any()):
                        break
                    noise[bad] = torch.randn(int(bad.sum()))
                noise = noise.clamp(-k, k)
            noise = noise * ns
            padded = torch.where(
                mask.unsqueeze(-1), padded + noise, padded
            )

            time_for_sort = padded[:, :, 1].masked_fill(~mask, float("inf"))
            sort_idx = time_for_sort.argsort(dim=1)
            sort_idx_exp = sort_idx.unsqueeze(-1).expand(-1, -1, feature_dim)
            padded = padded.gather(dim=1, index=sort_idx_exp)
            mask = mask.gather(dim=1, index=sort_idx)

    # --- Target z-normalization; kept along with the raw dex target ---
    targets = targets_dex
    if target_normalization is not None and target_normalization.get(
        "enabled", True
    ):
        mean = float(target_normalization["mean"])
        std = float(target_normalization["std"])
        targets = (targets_dex - mean) / std

    if normalization_config is not None:
        means = torch.tensor(
            normalization_config["means"], dtype=torch.float32
        )
        stds = torch.tensor(
            normalization_config["stds"], dtype=torch.float32
        )
        padded = torch.where(
            mask.unsqueeze(-1),
            (padded - means) / (stds + 1e-8),
            padded,
        )

    return {
        "features": padded,
        "targets": targets,
        "targets_dex": targets_dex,
        "energies": energies,
        "weights": weights,
        "lengths": lengths,
        "original_lengths": original_lengths,
        "mask": mask,
        "signal_hit_counts": signal_hit_counts,
        "signal_string_counts": signal_string_counts,
        "particle_types": particle_types,
        "theta": theta,
        "event_fks": event_fks,
        "local_event_ids": local_event_ids,
        "energy_status": energy_status,
        "part_keys": part_keys,
    }
