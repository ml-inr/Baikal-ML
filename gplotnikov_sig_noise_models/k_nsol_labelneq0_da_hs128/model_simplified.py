"""Simplified inference-only model for signal/noise hit classification.

Matches the encoder + main_head from train_config_mc_2020.yaml.
Domain-adaptation components (GRL, domain classifier) are omitted —
the checkpoint is loaded with strict=False so those keys are silently ignored.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """Transformer encoder returning (head_output, hidden_states)."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int,
        dim_feedforward_size: int,
        n_heads: int,
        out_size: int,
        dropout_p: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.dropout_p = dropout_p

        self.first_layer = nn.Linear(in_features, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """mask: (B, L) bool, True = valid hit."""
        x = self.first_layer(x)
        hidden_states = self.enc(x, src_key_padding_mask=(~mask).float())
        return self.head(hidden_states), hidden_states


class SigNoiseModel(nn.Module):
    """Encoder + per-hit classification head.  Returns (B, L, out_size) logits."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int,
        dim_feedforward_size: int,
        n_heads: int,
        out_size: int,
        dropout_p: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_features=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dim_feedforward_size=dim_feedforward_size,
            n_heads=n_heads,
            out_size=out_size,
            dropout_p=dropout_p,
        )
        self.main_head = nn.Linear(hidden_size, out_size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """mask: (B, L) bool, True = valid hit.  Returns (B, L, out_size)."""
        _, hidden_states = self.encoder(x, mask)
        return self.main_head(hidden_states)
