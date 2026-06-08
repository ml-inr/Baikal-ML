"""Model loading and batched inference utilities.

Migrated from inference/shared_utils.py — load_model() and predict_scores().
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Load trained base model from a DA checkpoint.

    Returns:
        (model, normalization_config, train_config)
    """
    import sys
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.models.base_models import create_model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    # finetuned checkpoints store the fine-tuning YAML in "config" (no "model" key);
    # fall back to "pretrained_config" which holds the original DA training config
    if "model" not in config:
        config = ckpt["pretrained_config"]
    norm_config = ckpt["normalization_config"]

    model = create_model(config["model"])
    model.load_state_dict(ckpt["base_model_state_dict"])

    raw_clip = config["model"].get("amp_clip")
    if raw_clip is not None and norm_config is not None:
        mean_amp = norm_config["means"][0]
        std_amp = norm_config["stds"][0]
        model.amp_clip = (raw_clip - mean_amp) / std_amp

    model.to(device)
    model.eval()

    logger.info(
        f"Loaded model from epoch {ckpt['epoch']} "
        f"(best_metric={ckpt.get('best_metric', '?'):.4f})"
    )
    return model, norm_config, config


def load_sn_model(device: str = "auto") -> Tuple[nn.Module, Any, str]:
    """Load the canonical sig-noise model.

    Returns:
        (sn_model, sn_config, resolved_device)
    """
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
        load_model as _load,
    )
    sn_model, sn_config, resolved_dev = _load(device=device)
    logger.info(f"Sig-noise model loaded on {resolved_dev}")
    return sn_model, sn_config, resolved_dev


@torch.no_grad()
def predict_scores_and_embeddings(
    model: nn.Module,
    features_list: List[np.ndarray],
    normalization_config: Dict[str, Any],
    batch_size: int = 512,
    max_hits: Optional[int] = 500,
    device: str = "cpu",
    feats_with_probs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched inference, return (scores, embeddings).

    Splits the forward pass at the feature_extractor/classifier boundary.

    Returns:
        scores:     (n_events,)          float32 sigmoid probabilities
        embeddings: (n_events, d_model)  float32 encoder-level feature vectors
    """
    if not (hasattr(model, "feature_extractor") and hasattr(model, "classifier")):
        raise AttributeError(
            "Model must expose feature_extractor and classifier attributes; "
            "only NuMuClassifierModel is supported."
        )

    if feats_with_probs:
        normalization_config = {
            "means": normalization_config["means"] + [0.75],
            "stds":  normalization_config["stds"]  + [0.25],
        }
        features_num = 6
    else:
        features_num = 5

    means = torch.tensor(normalization_config["means"], dtype=torch.float32, device=device)
    stds  = torch.tensor(normalization_config["stds"],  dtype=torch.float32, device=device)

    all_scores:     List[np.ndarray] = []
    all_embeddings: List[np.ndarray] = []
    n = len(features_list)

    for start in range(0, n, batch_size):
        batch_feats = features_list[start : start + batch_size]
        b = len(batch_feats)

        lengths = []
        truncated = []
        for feat in batch_feats:
            nh = len(feat)
            if max_hits is not None and nh > max_hits:
                truncated.append(feat[:max_hits])
                lengths.append(max_hits)
            else:
                truncated.append(feat)
                lengths.append(nh)

        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        max_len   = int(lengths_t.max().item())

        padded = torch.zeros(b, max_len, features_num, dtype=torch.float32, device=device)
        for i, feat in enumerate(truncated):
            padded[i, :len(feat)] = torch.from_numpy(feat)

        mask = torch.arange(max_len, device=device)[None, :] < lengths_t[:, None]
        padded = torch.where(mask.unsqueeze(-1), (padded - means) / (stds + 1e-8), padded)

        batch_dict: Dict[str, torch.Tensor] = {
            "features": padded, "lengths": lengths_t, "mask": mask
        }
        if hasattr(model, "_clip_amplitude"):
            batch_dict = model._clip_amplitude(batch_dict, getattr(model, "amp_clip", None))

        emb    = model.feature_extractor(
            sequences=batch_dict["features"],
            lengths=batch_dict["lengths"],
            mask=batch_dict["mask"],
        )
        logits = model.classifier(emb)

        all_scores.append(torch.sigmoid(logits).cpu().numpy().flatten())
        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_scores), np.concatenate(all_embeddings, axis=0)


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    features_list: List[np.ndarray],
    normalization_config: Dict[str, Any],
    batch_size: int = 512,
    max_hits: Optional[int] = 500,
    device: str = "cpu",
    feats_with_probs: bool = False,
    with_tqdm: bool = True,
) -> np.ndarray:
    """Run batched inference and return sigmoid scores.

    Args:
        model: Trained base model (eval mode).
        features_list: List of (n_hits, n_feats) arrays per event.
        normalization_config: {"means": [...], "stds": [...]}.
        batch_size: Inference batch size.
        max_hits: Truncate events longer than this.
        device: torch device string.
        feats_with_probs: If True, append sig-noise prob as 6th feature.
        with_tqdm: Show progress bar.

    Returns:
        (n_events,) float32 array of sigmoid probabilities.
    """
    if feats_with_probs:
        normalization_config = {
            "means": normalization_config["means"] + [0.75],
            "stds":  normalization_config["stds"]  + [0.25],
        }
        features_num = 6
    else:
        features_num = 5

    means = torch.tensor(normalization_config["means"], dtype=torch.float32, device=device)
    stds  = torch.tensor(normalization_config["stds"],  dtype=torch.float32, device=device)

    all_scores: List[np.ndarray] = []
    n = len(features_list)
    iterator = range(0, n, batch_size)
    if with_tqdm:
        iterator = tqdm(iterator, desc="Predicting", unit="batch",
                        total=(n + batch_size - 1) // batch_size)

    for start in iterator:
        batch_feats = features_list[start : start + batch_size]
        b = len(batch_feats)

        lengths = []
        truncated = []
        for feat in batch_feats:
            nh = len(feat)
            if max_hits is not None and nh > max_hits:
                truncated.append(feat[:max_hits])
                lengths.append(max_hits)
            else:
                truncated.append(feat)
                lengths.append(nh)

        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        max_len   = int(lengths_t.max().item())

        padded = torch.zeros(b, max_len, features_num, dtype=torch.float32, device=device)
        for i, feat in enumerate(truncated):
            padded[i, :len(feat)] = torch.from_numpy(feat)

        mask = torch.arange(max_len, device=device)[None, :] < lengths_t[:, None]
        padded = torch.where(mask.unsqueeze(-1), (padded - means) / (stds + 1e-8), padded)

        logits = model({"features": padded, "lengths": lengths_t, "mask": mask})
        all_scores.append(torch.sigmoid(logits).cpu().numpy().flatten())

    return np.concatenate(all_scores)
