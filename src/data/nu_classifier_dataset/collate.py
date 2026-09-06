"""Collate function for NuClassifierNpyDataset / NuClassifierExpNpyDataset batches.

Handles padding, augmentation (rotation + noise), and normalization.
Standalone function — no dataset instance required.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def nu_classifier_collate_fn(
    batch: List[Dict[str, Any]],
    max_hits: Optional[int] = None,
    normalization_config: Optional[Dict[str, List[float]]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    shuffle_batch: bool = True,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Collate variable-length events into a padded batch.

    Args:
        batch: List of dicts from ``NuClassifierNpyDataset.__getitem__``.
        max_hits: Extra truncation limit (usually already applied in dataset).
        normalization_config: ``{"means": [...], "stds": [...]}`` per feature.
        augmentation_config: ``{"noise_std": [...], "rotation_enabled": bool}``.
        shuffle_batch: Shuffle events within the batch.
        device: Target device for tensors.

    Returns:
        Dict with keys: features, labels, lengths, original_lengths, mask,
        signal_hit_counts, signal_string_counts, particle_types.
    """
    if shuffle_batch:
        batch = list(batch)
        np.random.shuffle(batch)

    features_list        = [item["features"] for item in batch]
    labels               = torch.stack([item["labels"] for item in batch])
    lengths              = torch.stack([item["lengths"] for item in batch])
    original_lengths     = torch.stack([item["original_lengths"] for item in batch])
    signal_hit_counts    = torch.tensor(
        [item["signal_hit_count"] for item in batch], dtype=torch.long
    )
    signal_string_counts = torch.tensor(
        [item["signal_string_count"] for item in batch], dtype=torch.long
    )
    particle_types       = torch.tensor(
        [item["particle_type"] for item in batch], dtype=torch.long
    )
    # GT zenith theta (deg) for the horizon-aware loss; present only for datasets
    # that carry theta.npy (MC source). NaN elsewhere → horizon weight w=0.
    # Per-item (not just batch[0]): fine-tuning mixes MC items (with theta) and
    # exp-BG items (without) in one batch, so fall back to NaN per missing item.
    theta                = torch.stack([
        item["theta"] if "theta" in item
        else torch.tensor(float("nan"), dtype=torch.float32)
        for item in batch
    ])

    # --- Afterpulse injection (before padding; source-only via aug config) ---
    # Simulates spurious hardware afterpulses that leaked through the sig-noise
    # filter into the signal-hit set: extra hit(s) at a random OM, random time,
    # charge Q~U(q_lo, q_hi). MC has no afterpulses; exp does — so this is passed
    # only to the MC (source) loader. Injected here so lengths/mask/sort account
    # for it downstream.
    #
    # Per-EVENT sequential draw of 1..max_afterpulses: roll for the 1st (prob p);
    # only if it fired, roll for the 2nd; etc. So P(n hits)=p^n·(1-p) (capped),
    # giving P(1)>P(2)>P(3) — brighter multi-afterpulse events are rarer.
    _ap = (augmentation_config or {}).get("afterpulse")
    if _ap and _ap.get("enabled", False) and _ap.get("om_pool") is not None:
        _prob = float(_ap.get("prob", 0.0))
        _max_ap = int(_ap.get("max_afterpulses", 3))
        _q_lo, _q_hi = _ap.get("q_range", [5.0, 100.0])
        _om = np.asarray(_ap["om_pool"], dtype=np.float32)
        if _prob > 0.0 and _max_ap > 0 and len(_om) > 0:
            _new = []
            for feat in features_list:
                # sequential rolls: stop at first miss or when max reached
                n_ap = 0
                while n_ap < _max_ap and feat.shape[0] > 0 and np.random.random() < _prob:
                    n_ap += 1
                if n_ap > 0:
                    fd  = feat.shape[1]
                    t   = feat[:, 1]
                    tlo, thi = float(t.min()), float(t.max())
                    aps = torch.zeros(n_ap, fd, dtype=feat.dtype)
                    for j in range(n_ap):
                        aps[j, 0] = float(np.random.uniform(_q_lo, _q_hi))    # Q
                        aps[j, 1] = float(np.random.uniform(tlo, thi)) if thi > tlo else tlo
                        aps[j, 2:5] = torch.from_numpy(_om[np.random.randint(len(_om))])
                        if fd >= 6:
                            aps[j, 5] = 1.0    # sig-noise prob: afterpulse "passed"
                    feat = torch.cat([feat, aps], dim=0)
                _new.append(feat)
            features_list = _new
            # lengths must include the injected hit(s) (was n_hits from the dataset)
            lengths = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)

    batch_size  = len(features_list)
    max_len     = int(lengths.max().item())
    feature_dim = features_list[0].shape[1]

    padded = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    for i, feat in enumerate(features_list):
        seq_len = feat.shape[0]
        padded[i, :seq_len] = feat

    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    padded               = padded.to(device)
    mask                 = mask.to(device)
    labels               = labels.to(device)
    lengths              = lengths.to(device)
    original_lengths     = original_lengths.to(device)
    signal_hit_counts    = signal_hit_counts.to(device)
    signal_string_counts = signal_string_counts.to(device)
    particle_types       = particle_types.to(device)
    theta                = theta.to(device)

    # --- Augmentation ---
    if augmentation_config is not None:
        if augmentation_config.get("rotation_enabled", False):
            angles = torch.rand(batch_size, device=device) * 2 * torch.pi
            cos_a  = torch.cos(angles)
            sin_a  = torch.sin(angles)
            x      = padded[:, :, 2].clone()
            y      = padded[:, :, 3].clone()
            x_rot  = cos_a[:, None] * x - sin_a[:, None] * y
            y_rot  = sin_a[:, None] * x + cos_a[:, None] * y
            padded[:, :, 2] = torch.where(mask, x_rot, padded[:, :, 2])
            padded[:, :, 3] = torch.where(mask, y_rot, padded[:, :, 3])

        noise_std_list = augmentation_config.get("noise_std")
        if noise_std_list is not None:
            noise_std = torch.tensor(
                noise_std_list, dtype=torch.float32, device=device
            )
            noise  = torch.randn_like(padded) * noise_std
            padded = torch.where(mask.unsqueeze(-1), padded + noise, padded)

            time_for_sort = padded[:, :, 1].masked_fill(~mask, float("inf"))
            sort_idx      = time_for_sort.argsort(dim=1)
            sort_idx_exp  = sort_idx.unsqueeze(-1).expand(-1, -1, feature_dim)
            padded        = padded.gather(dim=1, index=sort_idx_exp)
            mask          = mask.gather(dim=1, index=sort_idx)

    # --- Normalization ---
    if normalization_config is not None:
        means = torch.tensor(
            normalization_config["means"], dtype=torch.float32, device=device
        )
        stds  = torch.tensor(
            normalization_config["stds"], dtype=torch.float32, device=device
        )
        padded = torch.where(
            mask.unsqueeze(-1),
            (padded - means) / (stds + 1e-8),
            padded,
        )

    return {
        "features":            padded,
        "labels":              labels,
        "lengths":             lengths,
        "original_lengths":    original_lengths,
        "mask":                mask,
        "signal_hit_counts":   signal_hit_counts,
        "signal_string_counts": signal_string_counts,
        "particle_types":      particle_types,
        "theta":               theta,
    }
