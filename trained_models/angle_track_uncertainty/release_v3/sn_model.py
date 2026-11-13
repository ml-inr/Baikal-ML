"""Release sig-noise classifier used by the angle/track release.

Copied verbatim from G. Plotnikov's Baikal-ML commit
15392372e4dce61b9c8c7245fb7fffe7c8f0a30f:

    pytorch_training/models/layers.py
    pytorch_training/models/encoder.py

This is the frozen hit classifier shipped in release_v3/signal_noise.ckpt. It is
NOT the canonical sig-noise model used elsewhere in this repository (different
checkpoint and normalization). Keep the float padding mask: it is part of the
hit-selection definition.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from trained_models.angle_track_uncertainty.release_v3 import provenance

# ── from models/layers.py ──────────────────────────────────────────────
class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for domain adaptation."""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    """Wrapper module for gradient reversal."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


# Backward compatibility alias
GradientReversalLayer = GradientReversalFunction

# ── from models/encoder.py ─────────────────────────────────────────────
class BatchNorm1dTranspose(nn.BatchNorm1d):
    """BatchNorm that handles (B, N, C) input by transposing."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class TransformerEncoderLayerBN(nn.TransformerEncoderLayer):
    """Transformer encoder layer with BatchNorm instead of LayerNorm."""

    def __init__(self, d_model: int, *args, **kwargs) -> None:
        super().__init__(d_model, *args, **kwargs)
        self.norm1 = BatchNorm1dTranspose(d_model)
        self.norm2 = BatchNorm1dTranspose(d_model)


class EncoderTwoHead(nn.Module):
    """Shared transformer trunk that splits into two task-specific branches.

    - shared body: ``num_shared_layers`` transformer encoder layers
    - classification branch: ``num_cls_layers`` extra transformer layers + linear head (2 logits)
    - t_res regression branch: ``num_tres_layers`` extra transformer layers + linear head (1 scalar)

    Forward returns (B, N, cls_out_size + tres_out_size) where the last dim is
    [cls_logit_0, cls_logit_1, tres_pred] (matching the existing multi-task interface).
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_shared_layers: int,
        num_cls_layers: int,
        num_tres_layers: int,
        dim_feedforward_size: int,
        n_heads: int,
        cls_out_size: int = 2,
        tres_out_size: int = 1,
        dropout_p: float = 0.0,
        use_batch_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.out_size = cls_out_size + tres_out_size
        self.dropout_p = dropout_p

        self.first_layer = nn.Linear(in_features, hidden_size)
        encoder_layer_cls = (
            TransformerEncoderLayerBN if use_batch_norm else nn.TransformerEncoderLayer
        )

        def _make_stack(n: int) -> nn.Module:
            if n <= 0:
                return nn.Identity()
            layer = encoder_layer_cls(
                hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
            )
            return nn.TransformerEncoder(layer, n)

        self.shared = _make_stack(num_shared_layers)
        self.cls_branch = _make_stack(num_cls_layers)
        self.tres_branch = _make_stack(num_tres_layers)

        self.cls_head = nn.Linear(hidden_size, cls_out_size, bias=False)
        self.tres_head = nn.Linear(hidden_size, tres_out_size)

    @staticmethod
    def _apply_stack(stack: nn.Module, x: Tensor, key_pad: Tensor) -> Tensor:
        if isinstance(stack, nn.TransformerEncoder):
            return stack(x, src_key_padding_mask=key_pad)
        return stack(x)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        key_pad = (~mask).float()
        x = self.first_layer(x)
        x = self._apply_stack(self.shared, x, key_pad)
        cls_x = self._apply_stack(self.cls_branch, x, key_pad)
        tres_x = self._apply_stack(self.tres_branch, x, key_pad)
        cls_y = self.cls_head(cls_x)
        tres_y = self.tres_head(tres_x)
        return torch.cat([cls_y, tres_y], dim=-1)

def load_release_sn(path: str, device: str = "cpu") -> EncoderTwoHead:
    """Load the frozen release SN checkpoint.

    The released file has a deeper ``tres_head.0/2`` (128->256->1) than the plain
    ``Linear`` in ``EncoderTwoHead``. Upstream loads with ``strict=False`` and
    ignores only the t_res branch, which is irrelevant to the signal probability.
    Every other key must match exactly.
    """
    path = str(path)
    provenance.verify_sha256(path, provenance.SN_SHA256, "release SN checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = EncoderTwoHead(
        in_features=5,
        hidden_size=128,
        num_shared_layers=5,
        num_cls_layers=0,
        num_tres_layers=0,
        dim_feedforward_size=512,
        n_heads=1,
        cls_out_size=2,
        tres_out_size=1,
        dropout_p=0.0,
    )
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if (
        missing != provenance.ALLOWED_SN_MISSING
        or unexpected != provenance.ALLOWED_SN_UNEXPECTED
    ):
        raise RuntimeError(
            "Release SN checkpoint mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    n_parameters = sum(p.numel() for p in model.parameters())
    if n_parameters != provenance.SN_PARAMETERS:
        raise ValueError(
            f"Release SN model has {n_parameters} parameters, expected "
            f"{provenance.SN_PARAMETERS}"
        )
    # The t_res branch is not part of the release's hit-selection path. Zero the
    # unused plain head so the third output is deterministic instead of random.
    with torch.no_grad():
        model.tres_head.weight.zero_()
        model.tres_head.bias.zero_()
    model.to(device).eval()
    return model
