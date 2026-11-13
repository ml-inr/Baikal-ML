"""Inference-only model for the released angle/track/uncertainty bundle.

Copied verbatim (class/function bodies) from G. Plotnikov's Baikal-ML commit
15392372e4dce61b9c8c7245fb7fffe7c8f0a30f:

    pytorch_training/scripts/angle_track_uncertainty/train_angle_signal.py
    pytorch_training/scripts/angle_track_uncertainty/train_track_anchor_v3.py
    pytorch_training/scripts/angle_track_uncertainty/train_track_uncertainty_v2.py
    pytorch_training/scripts/angle_track_uncertainty/photon_residual_upgrade.py
    pytorch_training/scripts/angle_track_uncertainty/photon_uncertainty_bundle.py

Only the inference-relevant pieces are kept; training loops, datasets and CLIs
are not. Do not edit the class bodies: the state_dict key layout must match the
released checkpoint.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from trained_models.angle_track_uncertainty.release_v3 import provenance

# ── from train_angle_signal.py ─────────────────────────────────────────
def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def engineered_features(x: Tensor, mask: Tensor) -> Tensor:
    mean = masked_mean(x, mask)
    centered = x[..., 1:5] - mean[:, None, 1:5]
    radius = torch.linalg.vector_norm(centered[..., 1:4], dim=-1, keepdim=True)
    return torch.cat((x, centered, radius), dim=-1)

class SetTransformerDirection(nn.Module):
    def __init__(
        self,
        hidden: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        feature_mode: str,
    ) -> None:
        super().__init__()
        self.feature_mode = feature_mode
        in_features = 10 if feature_mode == "engineered" else 5
        self.embed = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.cls = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(hidden))
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: Tensor, mask: Tensor, return_features: bool = False):
        if self.feature_mode == "engineered":
            x = engineered_features(x, mask)
        h = self.embed(x)
        cls = self.cls.expand(len(h), -1, -1)
        h = torch.cat((cls, h), dim=1)
        full_mask = torch.cat(
            (torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask), dim=1
        )
        h = self.encoder(h, src_key_padding_mask=~full_mask)
        pooled = torch.cat((h[:, 0], masked_mean(h[:, 1:], mask)), dim=1)
        direction = torch.nn.functional.normalize(self.head(pooled), dim=1)
        return (direction, pooled) if return_features else direction

# ── from train_track_anchor_v3.py ─────────────────────────────────────
class DirectionTrackAnchor(nn.Module):
    def __init__(
        self, backbone: SetTransformerDirection, hidden: int, dropout: float, scale_m: float
    ):
        super().__init__()
        self.backbone = backbone
        self.scale_m = scale_m
        self.point_head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        direction, pooled = self.backbone(x, mask, return_features=True)
        raw_offset = self.point_head(pooled) * self.scale_m
        projection_direction = direction.detach()
        offset = raw_offset - projection_direction * torch.sum(
            raw_offset * projection_direction, dim=1, keepdim=True
        )
        return direction, offset

# ── from train_track_uncertainty_v2.py ────────────────────────────────
class TrackUncertainty(nn.Module):
    def __init__(self, base: DirectionTrackAnchor, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.base = base
        # pooled encoder (2*256), direction, offset, log hits, time spread,
        # and three spatial spreads. No truth or particle label enters the head.
        width = 2 * hidden + 3 + 3 + 1 + 1 + 3
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias[:] = torch.tensor([math.log(3.0), math.log(8.0)])

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x: Tensor, mask: Tensor):
        with torch.no_grad():
            direction, pooled = self.base.backbone(x, mask, return_features=True)
            raw_offset = self.base.point_head(pooled) * self.base.scale_m
            offset = raw_offset - direction * (raw_offset * direction).sum(1, keepdim=True)
            direction = direction.float()
            offset = offset.float()
            pooled = pooled.float()
            count = mask.sum(1).float().clamp_min(1)
            weights = mask.unsqueeze(-1).float()
            mean = (x.float() * weights).sum(1) / count[:, None]
            spread = torch.sqrt(
                (((x.float() - mean[:, None]) ** 2) * weights).sum(1) / count[:, None] + 1e-6
            )
            features = torch.cat(
                [
                    pooled,
                    direction,
                    offset / self.base.scale_m,
                    torch.log(count[:, None]),
                    spread[:, 1:2],
                    spread[:, 2:5],
                ],
                dim=1,
            )
        raw = self.head(features)
        log_angle = raw[:, 0].clamp(math.log(0.1), math.log(90.0))
        log_line = raw[:, 1].clamp(math.log(0.1), math.log(200.0))
        return direction, offset, log_angle, log_line

# ── from photon_residual_upgrade.py ───────────────────────────────────
class PhotonResidualUpgrade(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        feature_mean: Tensor,
        feature_std: Tensor,
        *,
        hidden: int = 64,
        refractive_index: float = 1.36,
        direction_scale: float = 0.30,
        point_scale_m: float = 20.0,
    ) -> None:
        super().__init__()
        if refractive_index <= 1 or direction_scale <= 0 or point_scale_m <= 0:
            raise ValueError("Invalid physical model or correction scale")
        self.base = base
        self.base.requires_grad_(False)
        self.refractive_index = refractive_index
        self.direction_scale = direction_scale
        self.point_scale_m = point_scale_m
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean).float().view(1, 1, 5))
        self.register_buffer("feature_std", torch.as_tensor(feature_std).float().view(1, 1, 5))
        self.hit_head = nn.Sequential(
            nn.Linear(7, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.hit_head[-1].weight)
        nn.init.zeros_(self.hit_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def residual_votes(
        self, x: Tensor, mask: Tensor, base_direction: Tensor, base_point: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.autocast("cuda", enabled=False):
            raw = x[..., :5].float() * self.feature_std + self.feature_mean
            u = base_direction.float()
            p = base_point.float()
            weights = mask.float()
            counts = weights.sum(dim=1).clamp_min(1.0)
            positions = raw[..., 2:5]
            centroid = (positions * weights[..., None]).sum(dim=1) / counts[:, None]
            delta = positions - centroid[:, None] - p[:, None]
            along = (delta * u[:, None]).sum(dim=-1)
            transverse = delta - along[..., None] * u[:, None]
            distance = torch.linalg.vector_norm(transverse, dim=-1).clamp_min(0.1)
            expected_ns = (along + (self.refractive_index**2 - 1.0) ** 0.5 * distance) / 0.299792458
            residual = raw[..., 1] - expected_ns
            centered_time = residual.masked_fill(~mask, float("nan")).nanmedian(dim=1).values
            centered = (residual - centered_time[:, None]).masked_fill(~mask, 0.0)
            mad = centered.abs().masked_fill(~mask, float("nan")).nanmedian(dim=1).values
            features = torch.stack(
                (
                    (along / 100.0).clamp(-5, 5),
                    (distance / 100.0).clamp(0, 5),
                    (centered / 50.0).clamp(-10, 10),
                    torch.log1p(raw[..., 0].clamp_min(0)) / 5.0,
                    (mad[:, None] / 50.0).expand_as(along).clamp(0, 10),
                    (counts[:, None] / 30.0).expand_as(along).clamp(0, 10),
                    u[:, None, 2].expand_as(along),
                ),
                dim=-1,
            )
            coefficients = torch.tanh(self.hit_head(features).float())
            basis = transverse / distance[..., None]
            longitudinal = (along / 100.0).clamp(-5, 5)
            direction_weight = coefficients[..., 0] + coefficients[..., 1] * longitudinal
            point_weight = coefficients[..., 2] + coefficients[..., 3] * longitudinal
            direction_vote = (direction_weight[..., None] * basis * weights[..., None]).sum(
                dim=1
            ) / counts[:, None]
            point_vote = (point_weight[..., None] * basis * weights[..., None]).sum(dim=1) / counts[
                :, None
            ]
            return direction_vote, point_vote, mad

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        base_direction, base_point = self.base(x, mask)
        direction_vote, point_vote, _mad = self.residual_votes(x, mask, base_direction, base_point)
        direction = torch.nn.functional.normalize(
            base_direction.float() + self.direction_scale * direction_vote, dim=1
        )
        point = base_point.float() + self.point_scale_m * point_vote
        point = point - direction * (point * direction).sum(dim=1, keepdim=True)
        return direction, point

# ── bundle construction ─────────────────────────────────────────────────
def build_base_model(config: dict) -> TrackUncertainty:
    """Build the v4 direction/point/uncertainty base (bundle format 1)."""
    backbone = SetTransformerDirection(
        config["hidden"],
        config["layers"],
        config["heads"],
        config["ff_mult"],
        config["dropout"],
        config["feature_mode"],
    )
    base = DirectionTrackAnchor(
        backbone, config["hidden"], config["dropout"], config["anchor_scale_m"]
    )
    return TrackUncertainty(base, config["hidden"], config["uncertainty_dropout"])


# Name used inside PhotonCorrectedUncertainty (upstream module alias).
build_v4_model = build_base_model


# ── from photon_uncertainty_bundle.py ──────────────────────────────────
class PhotonCorrectedUncertainty(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.base_model = build_v4_model(config["base_model_config"])
        photon = config["photon"]
        # Only residual_votes is called. Identity avoids storing a second copy
        # of the same backbone in the single-file bundle.
        self.photon_votes = PhotonResidualUpgrade(
            nn.Identity(),
            torch.tensor(photon["feature_mean"], dtype=torch.float32),
            torch.tensor(photon["feature_std"], dtype=torch.float32),
            hidden=photon["hidden"],
            refractive_index=photon["refractive_index"],
            direction_scale=photon["direction_scale"],
            point_scale_m=photon["point_scale_m"],
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.base.eval()
        self.photon_votes.eval()
        return self

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        base_direction, base_point, log_angle, log_line = self.base_model(x, mask)
        direction_vote, point_vote, _ = self.photon_votes.residual_votes(
            x, mask, base_direction, base_point
        )
        direction = F.normalize(
            base_direction.float() + self.photon_votes.direction_scale * direction_vote,
            dim=1,
        )
        point = base_point.float() + self.photon_votes.point_scale_m * point_vote
        point = point - direction * (point * direction).sum(dim=1, keepdim=True)
        return direction, point, log_angle, log_line

def build_model(config: dict) -> PhotonCorrectedUncertainty:
    """Build the released format-2 model."""
    return PhotonCorrectedUncertainty(config)


def load_bundle(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load the released format-2 bundle, verifying its provenance.

    Checks the file SHA256, bundle format/type, state-dict size, preprocessing
    and calibration against the values recorded in ``provenance.py`` before
    strict-loading the weights. Returns ``(model, bundle)``.
    """
    path = Path(path)
    provenance.verify_sha256(
        path, provenance.BUNDLE_SHA256, "angle/track bundle"
    )
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if (
        bundle.get("format_version") != provenance.BUNDLE_FORMAT_VERSION
        or bundle.get("model_type") != provenance.BUNDLE_MODEL_TYPE
    ):
        raise ValueError(
            f"Unsupported bundle: format_version={bundle.get('format_version')!r}, "
            f"model_type={bundle.get('model_type')!r}"
        )
    state = bundle.get("state_dict", {})
    if len(state) != provenance.BUNDLE_STATE_ENTRIES:
        raise ValueError(
            f"Bundle state_dict has {len(state)} entries, expected "
            f"{provenance.BUNDLE_STATE_ENTRIES}"
        )
    prep = bundle.get("preprocessing", {})
    if int(prep.get("max_seq_len", -1)) != provenance.EXPECTED_MAX_SEQ_LEN:
        raise ValueError("Unexpected max_seq_len in the bundle")
    if bool(prep.get("center_time", True)):
        raise ValueError("The released bundle must not center time")
    if list(prep.get("feature_order", [])) != provenance.EXPECTED_FEATURE_ORDER:
        raise ValueError("Unexpected feature order in the bundle")
    mean = np.asarray(prep.get("input_norm_mean", []), dtype=np.float32)
    std = np.asarray(prep.get("input_norm_std", []), dtype=np.float32)
    if not np.allclose(mean, np.asarray(provenance.EXPECTED_NORM_MEAN, np.float32),
                       rtol=0, atol=1e-6):
        raise ValueError("Bundle input_norm_mean differs from provenance")
    if not np.allclose(std, np.asarray(provenance.EXPECTED_NORM_STD, np.float32),
                       rtol=0, atol=1e-6):
        raise ValueError("Bundle input_norm_std differs from provenance")
    calibration = bundle.get("calibration", {})
    for key, levels in provenance.EXPECTED_CALIBRATION.items():
        for level, expected in levels.items():
            actual = float(calibration.get(key, {}).get(level, np.nan))
            if not np.isclose(actual, expected, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Bundle calibration {key}/{level} differs from provenance"
                )

    model = build_model(bundle["model_config"])
    model.load_state_dict(state, strict=True)
    n_parameters = sum(p.numel() for p in model.parameters())
    if n_parameters != provenance.BUNDLE_PARAMETERS:
        raise ValueError(
            f"Bundle model has {n_parameters} parameters, expected "
            f"{provenance.BUNDLE_PARAMETERS}"
        )
    model.to(device).eval()
    model.requires_grad_(False)
    return model, bundle
