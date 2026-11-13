"""Transformer regressor for muon energy at the cluster entry volume.

The encoder is the same :class:`AttentionFeatureExtractor` used by the prefilter and the
nu-classifier; the head is a small MLP with a single output, interpreted as ``log10(E/GeV)``.
The amplitude channel is clipped in normalized space exactly as in the classifier, so the
same Q = 100 PE convention applies.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.base_models import AttentionFeatureExtractor

logger = logging.getLogger(__name__)

_AMP_CLIP_Q100: float = 98.0 / 60.0


class RegressionHead(nn.Module):
    """MLP with one scalar output; no sigmoid — the target is unbounded log10(E)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims or [64])
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class MuEnergyRegressor(nn.Module):
    """Attention encoder + scalar regression head."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.amp_clip: Optional[float] = config.get("amp_clip", _AMP_CLIP_Q100)

        fe = config.get("feature_extractor", {})
        self.feature_extractor = AttentionFeatureExtractor(
            input_dim=fe.get("input_dim", 5),
            d_model=fe.get("d_model", 128),
            num_heads=fe.get("num_heads", 4),
            num_layers=fe.get("num_layers", 4),
            dim_feedforward=fe.get("dim_feedforward", 256),
            dropout=fe.get("dropout", 0.1),
            pooling=fe.get("pooling", "cls"),
            use_positional_encoding=fe.get("use_positional_encoding", True),
            max_seq_len=fe.get("max_seq_len", 1000),
            latent_dim=fe.get("latent_dim"),
        )
        reg = config.get("regressor", {})
        self.regressor = RegressionHead(
            input_dim=self.feature_extractor.feature_dim,
            hidden_dims=reg.get("hidden_dims", [64]),
            dropout=reg.get("dropout", 0.1),
        )
        hcfg = config.get("heteroscedastic", {}) or {}
        self.heteroscedastic = bool(hcfg.get("enabled", False))
        self.variance_head: Optional[RegressionHead] = None
        if self.heteroscedastic:
            self.variance_head = RegressionHead(
                input_dim=self.feature_extractor.feature_dim,
                hidden_dims=hcfg.get("hidden_dims", [64]),
                dropout=hcfg.get("dropout", 0.1),
            )
        logger.info(
            "Created MuEnergyRegressor with %s parameters%s",
            f"{self.count_parameters():,}",
            " (heteroscedastic head)" if self.heteroscedastic else "",
        )

    def _clip_amplitude(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.amp_clip is None:
            return batch
        features = batch["features"].clone()
        amp = features[:, :, 0]
        features[:, :, 0] = torch.where(
            batch["mask"], amp.clamp(max=self.amp_clip), amp
        )
        return {**batch, "features": features}

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._clip_amplitude(batch)
        features = self.feature_extractor(
            sequences=batch["features"],
            lengths=batch["lengths"],
            mask=batch["mask"],
        )
        return self.regressor(features)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._clip_amplitude(batch)
        return self.feature_extractor(
            sequences=batch["features"],
            lengths=batch["lengths"],
            mask=batch["mask"],
        )

    def get_feature_representation(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self._encode(batch)

    def forward_with_variance(
        self,
        batch: Dict[str, torch.Tensor],
        detach_sigma_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and log-variance (both in the normalized target units).

        ``detach_sigma_features`` is used by the detached-sigma training variant:
        the variance head still sees the encoded features, but its loss cannot
        update the encoder.  The mean head keeps the normal graph.
        """
        if not self.heteroscedastic or self.variance_head is None:
            raise RuntimeError("forward_with_variance called on a deterministic model")
        features = self._encode(batch)
        sigma_features = features.detach() if detach_sigma_features else features
        return self.regressor(features), self.variance_head(sigma_features)

    def initialize_variance_head(self, log_var: float) -> None:
        """Set the last variance-head layer to a constant log-variance.

        The mean head is untouched.  Standard practice for starting an NLL
        objective: the initial variance is the expected residual variance, so the
        first NLL is finite and the variance head does not have to discover the
        scale from random outputs.
        """
        if self.variance_head is None:
            raise RuntimeError("model has no variance head")
        last = self.variance_head.net[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.fill_(float(log_var))


def create_mu_energy_model(config: Dict) -> MuEnergyRegressor:
    return MuEnergyRegressor(config)
