#!/usr/bin/env python3
"""Toy surrogate-DA study with distributional dirt and on-the-fly augmentation.

Design (locked 2026-10-24)
--------------------------
Five disjoint roles, each drawn from MNIST originals split before any
augmentation:

    role          form    dirt                       colour   pool
    source-main   1 digit N(2.75, 0.25) [-2sigma]     no       train A
    source-DA     2 digit same                        no       train B
    source-test   1 digit same                        no       train C
    target-DA     2 digit N(3.00, 0.25) [-2sigma]     option   test A
    final-target  1 digit same as target-DA           option   test B

The dirt level is a truncated normal, not a fixed number.  The other
augmentations are drawn on the fly per sample in every batch: rotation
+-15 deg, independent stretch 1 +- 0.15, translation +-2 px, shear +-5 deg.
Contrast is also a truncated normal: source N(1.0, 0.08), target
N(1.0, 0.15), so the target varies a little more.  The colour option tints
20% of the black background pixels with a per-image random hue of amplitude
0.25; the no-colour option keeps the target grey.

Two-digit images are built from two dirtied digits: after the dirt squeeze
the black padding is cropped, the two digits are concatenated, and the
result is resized to 28x28.  This keeps the final width equal to the
one-digit images and keeps the per-digit scale close to the one-digit
source.

Evaluation uses K=3 fixed draws per image (same draws for baseline and DA);
training draws fresh augmentations every batch.  The colour run and the
no-colour run are separate; the encoder is always 3-channel, with grey
source images in both.

Usage:
    python toy_da.py --epochs 10 --seeds 0 1 2 3 4 --color \\
        --out runs/color_l02.json
    python toy_da.py --epochs 10 --seeds 0 1 2 3 4 \\
        --out runs/nocolor_l02.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "runs" / "results.json"

SOURCE_DIRT = (2.75, 0.25)
TARGET_DIRT = (3.00, 0.25)
SOURCE_CONTRAST = (1.00, 0.08)
TARGET_CONTRAST = (1.00, 0.15)
TRUNC_SIGMA = 2.0
ROTATION_DEG = 15.0
STRETCH = 0.15
TRANSLATE_PX = 2.0
SHEAR_DEG = 5.0
COLOR_BLACK_FRAC = 0.20
COLOR_TINT = 0.25
N_CLASSES = 10


# --------------------------------------------------------------------------
# data and augmentation
# --------------------------------------------------------------------------
def load_mnist(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor, torch.Tensor]:
    """MNIST as float tensors in [0, 1], shaped (N, 1, 28, 28)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml("mnist_784", version=1, as_frame=False,
                         parser="auto", data_home=str(data_dir))
    x = bunch.data.reshape(-1, 28, 28).astype(np.float32) / 255.0
    y = np.asarray(bunch.target, dtype=np.int64)
    return (torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y),
            torch.from_numpy(x[60000:]).unsqueeze(1),
            torch.from_numpy(y[60000:]))


def truncated_normal(n: int, mean: float, sigma: float,
                     generator: torch.Generator,
                     device: torch.device) -> torch.Tensor:
    """Truncated normal on [mean - 2 sigma, mean + 2 sigma] by rejection."""
    low = mean - TRUNC_SIGMA * sigma
    high = mean + TRUNC_SIGMA * sigma
    out = torch.empty(n, device=device)
    pending = torch.arange(n, device=device)
    while pending.numel():
        values = mean + sigma * torch.randn(
            pending.numel(), generator=generator, device=device)
        ok = (values >= low) & (values <= high)
        out[pending[ok]] = values[ok]
        pending = pending[~ok]
    return out


def dirt_width(strengths: torch.Tensor) -> torch.Tensor:
    """Width of the squeezed digit before it is padded back to 28."""
    return (28.0 * (1.0 - 0.15 * strengths)).round().long().clamp(4, 28)


def corrupt_batch(x: torch.Tensor, strengths: torch.Tensor,
                  generator: torch.Generator) -> torch.Tensor:
    """Squeeze + blur + noise + salt + gain, one strength per image."""
    n = x.shape[0]
    widths = dirt_width(strengths)
    y = torch.zeros_like(x)
    for width in widths.unique().tolist():
        selected = widths == width
        squeezed = F.interpolate(
            x[selected], size=(28, int(width)), mode="bilinear",
            align_corners=False)
        pad = 28 - int(width)
        y[selected] = F.pad(squeezed, (pad // 2, pad - pad // 2, 0, 0))

    kernel = torch.tensor([[1.0, 2.0, 1.0],
                           [2.0, 4.0, 2.0],
                           [1.0, 2.0, 1.0]],
                          device=x.device).view(1, 1, 3, 3)
    y = F.conv2d(y, kernel / kernel.sum(), padding=1)

    scale = strengths.view(-1, 1, 1, 1)
    noise = torch.randn(y.shape, generator=generator, device=x.device)
    y = y + noise * (0.08 * scale)
    salt = torch.rand(y.shape, generator=generator, device=x.device)
    y = torch.where(salt < 0.012 * scale, torch.ones_like(y), y)
    gain = 1.0 + (torch.rand(n, 1, 1, 1, generator=generator,
                             device=x.device) - 0.5) * (0.30 * scale)
    y = (y * gain).clamp_(0.0, 1.0)
    zero = (strengths <= 0).view(-1, 1, 1, 1)
    return torch.where(zero, x, y)


def build_two_digit(first: torch.Tensor, second: torch.Tensor,
                    strengths: torch.Tensor,
                    generator: torch.Generator) -> torch.Tensor:
    """Dirt each digit, crop its black padding, concatenate, resize to 28."""
    n = first.shape[0]
    left = corrupt_batch(first, strengths, generator)
    right = corrupt_batch(second, strengths, generator)
    widths = dirt_width(strengths)
    out = torch.empty(n, 1, 28, 28, device=first.device)
    for width in widths.unique().tolist():
        selected = widths == width
        pad = (28 - int(width)) // 2
        a = left[selected, :, :, pad:pad + int(width)]
        b = right[selected, :, :, pad:pad + int(width)]
        joined = torch.cat([a, b], dim=3)
        out[selected] = F.interpolate(
            joined, size=(28, 28), mode="bilinear", align_corners=False)
    return out


def uniform(n: int, low: float, high: float, generator: torch.Generator,
            device: torch.device) -> torch.Tensor:
    return low + (high - low) * torch.rand(
        n, generator=generator, device=device)


def augment_geometry(x: torch.Tensor, generator: torch.Generator,
                     rotation: float = ROTATION_DEG,
                     stretch: float = STRETCH,
                     translate: float = TRANSLATE_PX,
                     shear: float = SHEAR_DEG) -> torch.Tensor:
    """Batched rotation, independent stretch, translation and shear."""
    n = x.shape[0]
    device = x.device
    angle = uniform(n, -rotation, rotation, generator, device)
    scale_x = uniform(n, 1.0 - stretch, 1.0 + stretch, generator, device)
    scale_y = uniform(n, 1.0 - stretch, 1.0 + stretch, generator, device)
    shear_x = torch.tan(uniform(n, -shear, shear, generator, device)
                        * math.pi / 180.0)
    shear_y = torch.tan(uniform(n, -shear, shear, generator, device)
                        * math.pi / 180.0)
    shift_x = uniform(n, -translate, translate, generator, device) * 2.0 / 28.0
    shift_y = uniform(n, -translate, translate, generator, device) * 2.0 / 28.0

    radians = angle * math.pi / 180.0
    cos, sin = torch.cos(radians), torch.sin(radians)
    rotate = torch.zeros(n, 3, 3, device=device)
    rotate[:, 0, 0] = cos
    rotate[:, 0, 1] = -sin
    rotate[:, 1, 0] = sin
    rotate[:, 1, 1] = cos
    rotate[:, 2, 2] = 1.0

    scale = torch.zeros(n, 3, 3, device=device)
    scale[:, 0, 0] = scale_x
    scale[:, 1, 1] = scale_y
    scale[:, 2, 2] = 1.0

    shear_m = torch.zeros(n, 3, 3, device=device)
    shear_m[:, 0, 0] = 1.0
    shear_m[:, 0, 1] = shear_x
    shear_m[:, 1, 0] = shear_y
    shear_m[:, 1, 1] = 1.0
    shear_m[:, 2, 2] = 1.0

    shift = torch.zeros(n, 3, 3, device=device)
    shift[:, 0, 0] = 1.0
    shift[:, 1, 1] = 1.0
    shift[:, 2, 2] = 1.0
    shift[:, 0, 2] = shift_x
    shift[:, 1, 2] = shift_y

    forward = shift @ shear_m @ scale @ rotate
    theta = torch.linalg.inv(forward)[:, :2, :]
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def apply_contrast(x: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * factors.view(-1, 1, 1, 1) + mean).clamp_(0.0, 1.0)


def gray_rgb(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(1, 3, 1, 1)


def colorize(x: torch.Tensor, generator: torch.Generator,
             black_frac: float = COLOR_BLACK_FRAC,
             tint: float = COLOR_TINT, mode: str = "background",
             alpha: float = 0.30) -> torch.Tensor:
    """Target colour.

    ``background`` tints 20% of the black pixels (the old cue).  ``global``
    multiplies every RGB channel by a per-image gain in [1-alpha, 1+alpha],
    so the digit itself is coloured; this is the stronger domain feature.
    """
    n = x.shape[0]
    if mode == "global":
        rgb = x.repeat(1, 3, 1, 1)
        gain = 1.0 + alpha * (torch.rand((n, 3, 1, 1), generator=generator,
                                         device=x.device) * 2.0 - 1.0)
        return (rgb * gain).clamp_(0.0, 1.0)
    rgb = x.repeat(1, 3, 1, 1)
    black = x < 0.1
    mask = black & (torch.rand(x.shape, generator=generator,
                               device=x.device) < black_frac)
    hue = (torch.rand((n, 3, 1, 1), generator=generator,
                      device=x.device) * 2.0 - 1.0) * tint
    return (rgb + mask.to(rgb.dtype) * hue).clamp_(0.0, 1.0)


def dirt_params(target: bool, args: argparse.Namespace) -> tuple[float, float]:
    mean = args.target_dirt if target else args.source_dirt
    return mean, args.dirt_sigma


def contrast_sigma(target: bool, args: argparse.Namespace) -> float:
    return (args.contrast_target_sigma if target
            else args.contrast_source_sigma)


def contrast_mean(target: bool, args: argparse.Namespace) -> float:
    return (getattr(args, "contrast_target_mean", 1.0) if target
            else getattr(args, "contrast_source_mean", 1.0))


def prepare_1digit_base(clean: torch.Tensor, strengths: torch.Tensor, *,
                        target: bool, args: argparse.Namespace,
                        generator: torch.Generator) -> torch.Tensor:
    x = corrupt_batch(clean, strengths, generator)
    if getattr(args, "augment", True):
        x = augment_geometry(x, generator)
    if getattr(args, "contrast", True):
        factors = truncated_normal(
            len(x), contrast_mean(target, args),
            contrast_sigma(target, args), generator, x.device)
        x = apply_contrast(x, factors)
    return x


def prepare_2digit_base(first: torch.Tensor, second: torch.Tensor,
                        strengths: torch.Tensor, *, target: bool,
                        args: argparse.Namespace,
                        generator: torch.Generator) -> torch.Tensor:
    x = build_two_digit(first, second, strengths, generator)
    if getattr(args, "augment", True):
        x = augment_geometry(x, generator)
    if getattr(args, "contrast", True):
        factors = truncated_normal(
            len(x), contrast_mean(target, args),
            contrast_sigma(target, args), generator, x.device)
        x = apply_contrast(x, factors)
    return x


def finish_domain_g(x: torch.Tensor, *, target: bool, color: bool,
                    generator: torch.Generator,
                    color_mode: str = "background",
                    color_alpha: float = 0.30) -> torch.Tensor:
    if target and color:
        return colorize(x, generator, mode=color_mode, alpha=color_alpha)
    return gray_rgb(x)


# --------------------------------------------------------------------------
# eval sets
# --------------------------------------------------------------------------
@dataclass
class EvalSet:
    """K fixed evaluation draws: (K, N, 3, 28, 28)."""
    x: torch.Tensor
    y: torch.Tensor | None = None


def build_eval_1digit(clean: torch.Tensor, labels: torch.Tensor, *,
                      target: bool, color: bool, seed: int, draws: int,
                      args: argparse.Namespace,
                      device: torch.device) -> EvalSet:
    chunks = []
    for k in range(draws):
        generator = torch.Generator(device=device).manual_seed(seed + k)
        strengths = truncated_normal(
            len(clean), *dirt_params(target, args), generator, device)
        base = prepare_1digit_base(clean, strengths, target=target,
                                   args=args, generator=generator)
        chunks.append(finish_domain_g(
            base, target=target, color=color, generator=generator,
            color_mode=getattr(args, "color_mode", "background"),
            color_alpha=getattr(args, "color_alpha", 0.30)))
    return EvalSet(torch.stack(chunks), labels)


def build_eval_1digit_pair(clean: torch.Tensor, labels: torch.Tensor, *,
                           seed: int, draws: int,
                           args: argparse.Namespace,
                           device: torch.device) -> tuple[EvalSet, EvalSet]:
    """Coloured and grey versions of the same target draws."""
    colours, greys = [], []
    for k in range(draws):
        generator = torch.Generator(device=device).manual_seed(seed + k)
        strengths = truncated_normal(
            len(clean), *dirt_params(True, args), generator, device)
        base = prepare_1digit_base(clean, strengths, target=True,
                                   args=args, generator=generator)
        greys.append(gray_rgb(base))
        colours.append(colorize(
            base, generator,
            mode=getattr(args, "color_mode", "background"),
            alpha=getattr(args, "color_alpha", 0.30)))
    return (EvalSet(torch.stack(colours), labels),
            EvalSet(torch.stack(greys), labels))


def build_eval_2digit(pool: torch.Tensor, *, target: bool, color: bool,
                      seed: int, draws: int, args: argparse.Namespace,
                      device: torch.device) -> EvalSet:
    pair_generator = torch.Generator(device=device).manual_seed(seed)
    second = torch.randint(0, len(pool), (len(pool),),
                           generator=pair_generator, device=device)
    first = torch.arange(len(pool), device=device)
    chunks = []
    for k in range(draws):
        generator = torch.Generator(device=device).manual_seed(seed + 100 + k)
        strengths = truncated_normal(
            len(pool), *dirt_params(target, args), generator, device)
        base = prepare_2digit_base(pool[first], pool[second], strengths,
                                   target=target, args=args,
                                   generator=generator)
        chunks.append(finish_domain_g(
            base, target=target, color=color, generator=generator,
            color_mode=getattr(args, "color_mode", "background"),
            color_alpha=getattr(args, "color_alpha", 0.30)))
    return EvalSet(torch.stack(chunks))


@dataclass
class Data:
    task_x: torch.Tensor
    task_y: torch.Tensor
    source_x: torch.Tensor
    target_x: torch.Tensor
    source_test: EvalSet
    final_target: EvalSet
    final_target_grey: EvalSet | None
    probe1: tuple[EvalSet, EvalSet]
    probe2: tuple[EvalSet, EvalSet]
    probe2_train: tuple[EvalSet, EvalSet]
    probe_train_test: tuple[EvalSet, EvalSet]
    probe_test_train: tuple[EvalSet, EvalSet]
    probe_color: tuple[EvalSet, EvalSet] | None


def build_data(args: argparse.Namespace,
               device: torch.device) -> Data:
    train_x, train_y, test_x, test_y = load_mnist(args.data_dir)
    if (args.n_task + args.n_source + args.n_source_test > len(train_x)
            or args.n_target_train + args.n_target_test > len(test_x)):
        raise ValueError("split exceeds the MNIST pools")

    task_x = train_x[:args.n_task].to(device)
    task_y = train_y[:args.n_task].to(device)
    source_x = train_x[args.n_task:args.n_task + args.n_source].to(device)
    s0 = args.n_task + args.n_source
    source_test_x = train_x[s0:s0 + args.n_source_test].to(device)
    source_test_y = train_y[s0:s0 + args.n_source_test].to(device)
    target_x = test_x[:args.n_target_train].to(device)
    t0 = args.n_target_train
    target_test_x = test_x[t0:t0 + args.n_target_test].to(device)
    target_test_y = test_y[t0:t0 + args.n_target_test].to(device)

    source_test = build_eval_1digit(
        source_test_x, source_test_y, target=False, color=False,
        seed=args.eval_seed, draws=args.eval_draws, args=args, device=device)
    if getattr(args, "color", False):
        final_target, final_target_grey = build_eval_1digit_pair(
            target_test_x, target_test_y, seed=args.eval_seed + 10,
            draws=args.eval_draws, args=args, device=device)
    else:
        final_target = build_eval_1digit(
            target_test_x, target_test_y, target=True, color=False,
            seed=args.eval_seed + 10, draws=args.eval_draws, args=args,
            device=device)
        final_target_grey = None

    source_test_2d = build_eval_2digit(
        source_test_x, target=False, color=False, seed=args.eval_seed + 20,
        draws=args.eval_draws, args=args, device=device)
    target_test_2d = build_eval_2digit(
        target_test_x, target=True,
        color=getattr(args, "color", False),
        seed=args.eval_seed + 30, draws=args.eval_draws, args=args,
        device=device)
    source_train_2d = build_eval_2digit(
        source_x, target=False, color=False, seed=args.eval_seed + 40,
        draws=args.eval_draws, args=args, device=device)
    target_train_2d = build_eval_2digit(
        target_x, target=True, color=getattr(args, "color", False),
        seed=args.eval_seed + 50,
        draws=args.eval_draws, args=args, device=device)

    probe1 = (source_test, final_target)
    probe2 = (source_test_2d, target_test_2d)
    probe2_train = (source_train_2d, target_train_2d)
    probe_train_test = (source_train_2d, target_test_2d)
    probe_test_train = (source_test_2d, target_train_2d)
    probe_color = ((final_target_grey, final_target)
                   if final_target_grey is not None else None)
    return Data(task_x, task_y, source_x, target_x, source_test,
                final_target, final_target_grey, probe1, probe2,
                probe2_train, probe_train_test, probe_test_train,
                probe_color)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.lambd * grad, None


def grl(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


class Encoder(nn.Module):
    """Small CNN for 28x28 3-channel inputs; returns a flat feature."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.out_dim = 64 * 3 * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).flatten(1)


class Classifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128,
                 dropout: float = 0.0, norm: bool = False) -> None:
        super().__init__()
        layers = ([nn.BatchNorm1d(in_dim, track_running_stats=False)]
                  if norm else [])
        if hidden <= 0:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(),
                       nn.Dropout(dropout),
                       nn.Linear(hidden, hidden), nn.ReLU(),
                       nn.Dropout(dropout),
                       nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


# --------------------------------------------------------------------------
# meters
# --------------------------------------------------------------------------
def accuracy(encoder: Encoder, classifier: Classifier,
             x: torch.Tensor, y: torch.Tensor,
             batch: int = 1024) -> float:
    encoder.eval()
    classifier.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = classifier(encoder(x[start:start + batch]))
            correct += int((logits.argmax(1) == y[start:start + batch]).sum())
    return correct / len(x)


def features(encoder: Encoder, x: torch.Tensor,
             batch: int = 1024) -> np.ndarray:
    encoder.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunks.append(encoder(x[start:start + batch]).cpu().numpy())
    return np.concatenate(chunks)


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.RandomState(seed).permutation(n)
    half = n // 2
    return order[:half], order[half:]


def subsample_eval(eval_set: EvalSet, max_n: int, seed: int) -> EvalSet:
    """Keep the same image indices across all K draws."""
    n = len(eval_set.x[0])
    if max_n <= 0 or n <= max_n:
        return eval_set
    generator = torch.Generator(device=eval_set.x.device).manual_seed(seed)
    index = torch.randperm(n, generator=generator,
                           device=eval_set.x.device)[:max_n]
    labels = eval_set.y[index] if eval_set.y is not None else None
    return EvalSet(eval_set.x[:, index], labels)


def _probe_features(encoder: Encoder, pos: torch.Tensor,
                    neg: torch.Tensor) -> tuple[list[np.ndarray],
                                                list[np.ndarray]]:
    return ([features(encoder, pos[k]) for k in range(len(pos))],
            [features(encoder, neg[k]) for k in range(len(neg))])


def _stack(lists: list[np.ndarray], index: np.ndarray) -> np.ndarray:
    return np.concatenate([values[index] for values in lists])


def probe_auc_draws(encoder: Encoder, pos: torch.Tensor,
                    neg: torch.Tensor, seed: int) -> float:
    """Fresh logistic, split by original image, pooled over the K draws."""
    pos_f, neg_f = _probe_features(encoder, pos, neg)
    tr_p, te_p = _split(len(pos_f[0]), seed)
    tr_n, te_n = _split(len(neg_f[0]), seed + 1)
    x_tr = np.concatenate([_stack(pos_f, tr_p), _stack(neg_f, tr_n)])
    y_tr = np.r_[np.zeros(len(tr_p) * len(pos_f)),
                 np.ones(len(tr_n) * len(neg_f))]
    x_te = np.concatenate([_stack(pos_f, te_p), _stack(neg_f, te_n)])
    y_te = np.r_[np.zeros(len(te_p) * len(pos_f)),
                 np.ones(len(te_n) * len(neg_f))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))


def probe_mlp_draws(encoder: Encoder, pos: torch.Tensor,
                    neg: torch.Tensor, seed: int) -> float:
    pos_f, neg_f = _probe_features(encoder, pos, neg)
    tr_p, te_p = _split(len(pos_f[0]), seed)
    tr_n, te_n = _split(len(neg_f[0]), seed + 1)
    x_tr = np.concatenate([_stack(pos_f, tr_p), _stack(neg_f, tr_n)])
    y_tr = np.r_[np.zeros(len(tr_p) * len(pos_f)),
                 np.ones(len(tr_n) * len(neg_f))]
    x_te = np.concatenate([_stack(pos_f, te_p), _stack(neg_f, te_n)])
    y_te = np.r_[np.zeros(len(te_p) * len(pos_f)),
                 np.ones(len(te_n) * len(neg_f))]
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000,
                      early_stopping=True, random_state=seed))
    clf.fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))


def probe_knn_draws(encoder: Encoder, pos: torch.Tensor,
                    neg: torch.Tensor, seed: int) -> float:
    pos_f, neg_f = _probe_features(encoder, pos, neg)
    tr_p, te_p = _split(len(pos_f[0]), seed)
    tr_n, te_n = _split(len(neg_f[0]), seed + 1)
    x_tr = np.concatenate([_stack(pos_f, tr_p), _stack(neg_f, tr_n)])
    y_tr = np.r_[np.zeros(len(tr_p) * len(pos_f)),
                 np.ones(len(tr_n) * len(neg_f))]
    x_te = np.concatenate([_stack(pos_f, te_p), _stack(neg_f, te_n)])
    y_te = np.r_[np.zeros(len(te_p) * len(pos_f)),
                 np.ones(len(te_n) * len(neg_f))]
    clf = make_pipeline(StandardScaler(),
                        KNeighborsClassifier(n_neighbors=15))
    clf.fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))


def probe_color_draws(encoder: Encoder, grey: torch.Tensor,
                      colour: torch.Tensor, seed: int) -> float:
    """Paired probe: same image grey vs coloured, split by image."""
    grey_f = [features(encoder, grey[k]) for k in range(len(grey))]
    colour_f = [features(encoder, colour[k]) for k in range(len(colour))]
    index = len(grey_f[0])
    tr, te = _split(index, seed)
    x_tr = np.concatenate([_stack(grey_f, tr), _stack(colour_f, tr)])
    y_tr = np.r_[np.zeros(len(tr) * len(grey_f)),
                 np.ones(len(tr) * len(colour_f))]
    x_te = np.concatenate([_stack(grey_f, te), _stack(colour_f, te)])
    y_te = np.r_[np.zeros(len(te) * len(grey_f)),
                 np.ones(len(te) * len(colour_f))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))


def accuracy_draws(encoder: Encoder, classifier: Classifier,
                   eval_set: EvalSet) -> tuple[float, float]:
    values = [accuracy(encoder, classifier, eval_set.x[k], eval_set.y)
              for k in range(len(eval_set.x))]
    return float(np.mean(values)), float(np.std(values))


def softmax_scores(encoder: Encoder, classifier: Classifier,
                   x: torch.Tensor, y: torch.Tensor,
                   batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    classifier.eval()
    maxp, truep = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = classifier(encoder(x[start:start + batch]))
            probs = torch.softmax(logits, dim=1)
            maxp.append(probs.max(1).values.cpu().numpy())
            truep.append(probs.gather(
                1, y[start:start + batch, None]).squeeze(1).cpu().numpy())
    return np.concatenate(maxp), np.concatenate(truep)


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a)
    b = np.sort(b)
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def tail_metrics(source_scores: np.ndarray,
                 target_scores: np.ndarray) -> dict:
    p99 = float(np.quantile(source_scores, 0.99))
    return {
        "source_mean": float(source_scores.mean()),
        "target_mean": float(target_scores.mean()),
        "ks": ks_stat(source_scores, target_scores),
        "source_p99": p99,
        "target_frac_above_source_p99": float(
            (target_scores > p99).mean()),
        "target_p90": float(np.quantile(target_scores, 0.90)),
        "target_p99": float(np.quantile(target_scores, 0.99)),
    }


def score_tails_draws(encoder: Encoder, classifier: Classifier,
                      source: EvalSet, target: EvalSet) -> dict:
    rows = []
    for k in range(len(source.x)):
        smax, strue = softmax_scores(encoder, classifier,
                                     source.x[k], source.y)
        tmax, ttrue = softmax_scores(encoder, classifier,
                                     target.x[k], target.y)
        rows.append({"maxp": tail_metrics(smax, tmax),
                     "truep": tail_metrics(strue, ttrue)})
    out = {}
    for kind in rows[0]:
        out[kind] = {key: float(np.mean([row[kind][key] for row in rows]))
                     for key in rows[0][kind]}
    return out


def critic_accuracy(encoder: Encoder, domain: DomainDiscriminator,
                    source: torch.Tensor, target: torch.Tensor) -> float:
    encoder.eval()
    domain.eval()
    with torch.no_grad():
        feats = torch.cat([encoder(source), encoder(target)])
        logits = domain(feats)
        labels = torch.cat([torch.zeros(len(source), device=logits.device),
                            torch.ones(len(target), device=logits.device)])
        return float(((logits > 0).float() == labels).float().mean())


def critic_accuracy_draws(encoder: Encoder, domain: DomainDiscriminator,
                          source: EvalSet, target: EvalSet) -> float:
    values = [critic_accuracy(encoder, domain, source.x[k], target.x[k])
              for k in range(len(source.x))]
    return float(np.mean(values))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def sample_task_batch(data: Data, indices: torch.Tensor,
                      args: argparse.Namespace,
                      generator: torch.Generator) -> tuple[torch.Tensor,
                                                           torch.Tensor]:
    strengths = truncated_normal(
        len(indices), *dirt_params(False, args), generator,
        data.task_x.device)
    base = prepare_1digit_base(data.task_x[indices], strengths, target=False,
                               args=args, generator=generator)
    x = finish_domain_g(base, target=False, color=False, generator=generator)
    return x, data.task_y[indices]


def sample_domain_batch(pool: torch.Tensor, indices: torch.Tensor, *,
                        target: bool, args: argparse.Namespace,
                        generator: torch.Generator) -> torch.Tensor:
    n = len(indices)
    second = torch.randint(0, len(pool), (n,), generator=generator,
                           device=pool.device)
    strengths = truncated_normal(
        n, *dirt_params(target, args), generator, pool.device)
    base = prepare_2digit_base(pool[indices], pool[second], strengths,
                               target=target, args=args,
                               generator=generator)
    return finish_domain_g(
        base, target=target, color=getattr(args, "color", False),
        generator=generator,
        color_mode=getattr(args, "color_mode", "background"),
        color_alpha=getattr(args, "color_alpha", 0.30))


def _epoch_probes(encoder: Encoder, classifier: Classifier, data: Data,
                  args: argparse.Namespace) -> dict:
    source_acc, source_spread = accuracy_draws(
        encoder, classifier, data.source_test)
    target_acc, target_spread = accuracy_draws(
        encoder, classifier, data.final_target)
    if getattr(args, "acc_only", False):
        nan = float("nan")
        return {"source": source_acc, "source_spread": source_spread,
                "target": target_acc, "target_spread": target_spread,
                "probe2": nan, "probe2_train": nan, "probe1": nan}
    return {
        "source": source_acc,
        "source_spread": source_spread,
        "target": target_acc,
        "target_spread": target_spread,
        "probe2": probe_auc_draws(encoder, data.probe2[0].x[:1],
                                  data.probe2[1].x[:1], args.probe_seed),
        "probe2_train": probe_auc_draws(encoder,
                                        data.probe2_train[0].x[:1],
                                        data.probe2_train[1].x[:1],
                                        args.probe_seed),
        "probe1": probe_auc_draws(encoder, data.probe1[0].x[:1],
                                  data.probe1[1].x[:1], args.probe_seed),
    }


def train_baseline(data: Data, args: argparse.Namespace,
                   seed: int) -> dict:
    torch.manual_seed(seed)
    device = data.task_x.device
    encoder = Encoder().to(device)
    classifier = Classifier(encoder.out_dim, N_CLASSES).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(seed)

    probe2_history = []
    probe2_train_history = []
    probe1_history = []
    source_history = []
    target_history = []
    for _ in range(args.epochs):
        permutation = torch.randperm(len(data.task_x), generator=generator)
        encoder.train()
        classifier.train()
        for start in range(0, len(data.task_x), args.batch_size):
            idx = permutation[start:start + args.batch_size]
            x, y = sample_task_batch(data, idx, args, generator)
            logits = classifier(encoder(x))
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = _epoch_probes(encoder, classifier, data, args)
        probe2_history.append(metrics["probe2"])
        probe2_train_history.append(metrics["probe2_train"])
        probe1_history.append(metrics["probe1"])
        source_history.append(metrics["source"])
        target_history.append(metrics["target"])

    nl2 = (subsample_eval(data.probe2[0], args.nonlinear_max,
                          args.probe_seed + 7),
           subsample_eval(data.probe2[1], args.nonlinear_max,
                          args.probe_seed + 7))
    nl1 = (subsample_eval(data.probe1[0], args.nonlinear_max,
                          args.probe_seed + 8),
           subsample_eval(data.probe1[1], args.nonlinear_max,
                          args.probe_seed + 8))
    final_probe2 = probe_auc_draws(encoder, data.probe2[0].x,
                                   data.probe2[1].x, args.probe_seed)
    final_probe2_train = probe_auc_draws(encoder, data.probe2_train[0].x,
                                         data.probe2_train[1].x,
                                         args.probe_seed)
    final_probe1 = probe_auc_draws(encoder, data.probe1[0].x,
                                   data.probe1[1].x, args.probe_seed)
    result = {
        "source": source_history[-1],
        "target": target_history[-1],
        "probe2": final_probe2,
        "probe2_train": final_probe2_train,
        "probe1": final_probe1,
        "probe2_history": probe2_history,
        "probe2_train_history": probe2_train_history,
        "probe1_history": probe1_history,
        "source_history": source_history,
        "target_history": target_history,
    }
    if not getattr(args, "acc_only", False):
        result.update({
            "probe_train_test": probe_auc_draws(
                encoder, data.probe_train_test[0].x,
                data.probe_train_test[1].x, args.probe_seed),
            "probe_test_train": probe_auc_draws(
                encoder, data.probe_test_train[0].x,
                data.probe_test_train[1].x, args.probe_seed),
            "probe2_mlp": probe_mlp_draws(
                encoder, nl2[0].x, nl2[1].x, args.probe_seed),
            "probe2_knn": probe_knn_draws(
                encoder, nl2[0].x, nl2[1].x, args.probe_seed),
            "probe1_mlp": probe_mlp_draws(
                encoder, nl1[0].x, nl1[1].x, args.probe_seed),
            "probe1_knn": probe_knn_draws(
                encoder, nl1[0].x, nl1[1].x, args.probe_seed),
            "score_tails": score_tails_draws(
                encoder, classifier, data.source_test, data.final_target),
        })
        if data.probe_color is not None:
            result["probe_color"] = probe_color_draws(
                encoder, data.probe_color[0].x, data.probe_color[1].x,
                args.probe_seed)
    return result


def train_surrogate_da(data: Data, args: argparse.Namespace,
                       seed: int) -> dict:
    torch.manual_seed(seed)
    device = data.task_x.device
    encoder = Encoder().to(device)
    classifier = Classifier(encoder.out_dim, N_CLASSES).to(device)
    domain = DomainDiscriminator(encoder.out_dim, args.critic_hidden,
                                 args.critic_dropout,
                                 args.critic_norm).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3)
    critic_opt = torch.optim.Adam(domain.parameters(), lr=args.critic_lr)
    generator = torch.Generator(device=device).manual_seed(seed)

    critic_history = []
    critic_train_history = []
    critic_loss_history = []
    da_coef_history = []
    probe2_history = []
    probe2_train_history = []
    probe1_history = []
    source_history = []
    target_history = []
    n_batches = (len(data.task_x) + args.batch_size - 1) // args.batch_size

    for epoch in range(args.epochs):
        epoch_train_acc = []
        epoch_loss = []
        epoch_coef = []
        permutation = torch.randperm(len(data.task_x), generator=generator)
        encoder.train()
        classifier.train()
        domain.train()
        for batch_idx, start in enumerate(
                range(0, len(data.task_x), args.batch_size)):
            progress = epoch + batch_idx / n_batches
            if progress < args.da_warmup_epochs:
                da_coef = 0.0
            elif args.da_ramp_epochs > 0:
                da_coef = args.da_lambda * min(
                    1.0, (progress - args.da_warmup_epochs)
                    / args.da_ramp_epochs)
            else:
                da_coef = args.da_lambda
            epoch_coef.append(da_coef)

            idx = permutation[start:start + args.batch_size]
            x, y = sample_task_batch(data, idx, args, generator)
            task_loss = F.cross_entropy(classifier(encoder(x)), y)

            for step in range(args.critic_steps):
                if args.critic_fresh or step == 0:
                    sidx = torch.randint(0, len(data.source_x),
                                         (args.critic_batch,),
                                         generator=generator, device=device)
                    tidx = torch.randint(0, len(data.target_x),
                                         (args.critic_batch,),
                                         generator=generator, device=device)
                    source = sample_domain_batch(
                        data.source_x, sidx, target=False, args=args,
                        generator=generator)
                    target = sample_domain_batch(
                        data.target_x, tidx, target=True, args=args,
                        generator=generator)
                    with torch.no_grad():
                        feat_s = encoder(source)
                        feat_t = encoder(target)
                dom_logits = domain(torch.cat([feat_s, feat_t]))
                dom_labels = torch.cat([
                    torch.zeros(len(feat_s), device=device),
                    torch.ones(len(feat_t), device=device)])
                if args.shuffle_domain:
                    dom_labels = dom_labels[torch.randperm(
                        len(dom_labels), generator=generator, device=device)]
                critic_loss = F.binary_cross_entropy_with_logits(
                    dom_logits, dom_labels)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                epoch_loss.append(critic_loss.detach().item())

            with torch.no_grad():
                domain.eval()
                logits = domain(torch.cat([feat_s, feat_t]))
                labels = torch.cat([
                    torch.zeros(len(feat_s), device=device),
                    torch.ones(len(feat_t), device=device)])
                epoch_train_acc.append(
                    float(((logits > 0).float() == labels).float().mean()))
                domain.train()

            feat = torch.cat([encoder(source), encoder(target)])
            dom_logits = domain(grl(feat, args.grl_lambda))
            dom_labels = torch.cat([
                torch.zeros(len(source), device=device),
                torch.ones(len(target), device=device)])
            if args.shuffle_domain:
                dom_labels = dom_labels[torch.randperm(
                    len(dom_labels), generator=generator, device=device)]
            domain_loss = F.binary_cross_entropy_with_logits(
                dom_logits, dom_labels)
            loss = task_loss + da_coef * domain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = _epoch_probes(encoder, classifier, data, args)
        critic_history.append(critic_accuracy_draws(
            encoder, domain, data.probe2[0], data.probe2[1]))
        critic_train_history.append(float(np.mean(epoch_train_acc)))
        critic_loss_history.append(float(np.mean(epoch_loss)))
        da_coef_history.append(float(np.mean(epoch_coef)))
        probe2_history.append(metrics["probe2"])
        probe2_train_history.append(metrics["probe2_train"])
        probe1_history.append(metrics["probe1"])
        source_history.append(metrics["source"])
        target_history.append(metrics["target"])

    nl2 = (subsample_eval(data.probe2[0], args.nonlinear_max,
                          args.probe_seed + 7),
           subsample_eval(data.probe2[1], args.nonlinear_max,
                          args.probe_seed + 7))
    nl1 = (subsample_eval(data.probe1[0], args.nonlinear_max,
                          args.probe_seed + 8),
           subsample_eval(data.probe1[1], args.nonlinear_max,
                          args.probe_seed + 8))
    final_probe2 = probe_auc_draws(encoder, data.probe2[0].x,
                                   data.probe2[1].x, args.probe_seed)
    final_probe2_train = probe_auc_draws(encoder, data.probe2_train[0].x,
                                         data.probe2_train[1].x,
                                         args.probe_seed)
    final_probe1 = probe_auc_draws(encoder, data.probe1[0].x,
                                   data.probe1[1].x, args.probe_seed)
    result = {
        "source": source_history[-1],
        "target": target_history[-1],
        "probe2": final_probe2,
        "probe2_train": final_probe2_train,
        "probe1": final_probe1,
        "probe2_history": probe2_history,
        "probe2_train_history": probe2_train_history,
        "probe1_history": probe1_history,
        "source_history": source_history,
        "target_history": target_history,
        "critic_history": critic_history,
        "critic_final": critic_history[-1],
        "critic_train_history": critic_train_history,
        "critic_train_final": critic_train_history[-1],
        "critic_loss_history": critic_loss_history,
        "da_coef_history": da_coef_history,
    }
    if not getattr(args, "acc_only", False):
        result.update({
            "probe_train_test": probe_auc_draws(
                encoder, data.probe_train_test[0].x,
                data.probe_train_test[1].x, args.probe_seed),
            "probe_test_train": probe_auc_draws(
                encoder, data.probe_test_train[0].x,
                data.probe_test_train[1].x, args.probe_seed),
            "probe2_mlp": probe_mlp_draws(
                encoder, nl2[0].x, nl2[1].x, args.probe_seed),
            "probe2_knn": probe_knn_draws(
                encoder, nl2[0].x, nl2[1].x, args.probe_seed),
            "probe1_mlp": probe_mlp_draws(
                encoder, nl1[0].x, nl1[1].x, args.probe_seed),
            "probe1_knn": probe_knn_draws(
                encoder, nl1[0].x, nl1[1].x, args.probe_seed),
            "score_tails": score_tails_draws(
                encoder, classifier, data.source_test, data.final_target),
        })
        if data.probe_color is not None:
            result["probe_color"] = probe_color_draws(
                encoder, data.probe_color[0].x, data.probe_color[1].x,
                args.probe_seed)
    return result


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-task", type=int, default=12000)
    parser.add_argument("--n-source", type=int, default=12000)
    parser.add_argument("--n-source-test", type=int, default=4000)
    parser.add_argument("--n-target-train", type=int, default=6000)
    parser.add_argument("--n-target-test", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--source-dirt", type=float, default=SOURCE_DIRT[0])
    parser.add_argument("--target-dirt", type=float, default=TARGET_DIRT[0])
    parser.add_argument("--dirt-sigma", type=float, default=SOURCE_DIRT[1])
    parser.add_argument("--contrast-source-sigma", type=float,
                        default=SOURCE_CONTRAST[1])
    parser.add_argument("--contrast-target-sigma", type=float,
                        default=TARGET_CONTRAST[1])
    parser.add_argument("--contrast-source-mean", type=float, default=1.0)
    parser.add_argument("--contrast-target-mean", type=float, default=1.0)
    parser.add_argument("--color", action="store_true",
                        help="apply the target colour feature")
    parser.add_argument("--color-mode", choices=("background", "global"),
                        default="background",
                        help="background: 20%% of black pixels; "
                             "global: per-image channel gain on the digit")
    parser.add_argument("--color-alpha", type=float, default=0.30)
    parser.add_argument("--no-augment", action="store_true",
                        help="disable rotation/stretch/translation/shear")
    parser.add_argument("--no-contrast", action="store_true",
                        help="disable the contrast draw")
    parser.add_argument("--da-lambda", type=float, default=0.2)
    parser.add_argument("--da-warmup-epochs", type=float, default=1.0)
    parser.add_argument("--da-ramp-epochs", type=float, default=0.0)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--critic-batch", type=int, default=512)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--critic-hidden", type=int, default=128)
    parser.add_argument("--critic-dropout", type=float, default=0.0)
    parser.add_argument("--critic-norm", action="store_true")
    parser.add_argument("--critic-fresh", action="store_true")
    parser.add_argument("--shuffle-domain", action="store_true")
    parser.add_argument("--eval-draws", type=int, default=3)
    parser.add_argument("--nonlinear-max", type=int, default=2000,
                        help="images per domain for the MLP/kNN probes")
    parser.add_argument("--baseline-only", action="store_true",
                        help="skip the DA branch (calibration)")
    parser.add_argument("--acc-only", action="store_true",
                        help="calibration: skip probes, report accuracy only")
    parser.add_argument("--eval-seed", type=int, default=20261101)
    parser.add_argument("--probe-seed", type=int, default=20261102)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.augment = not args.no_augment
    args.contrast = not args.no_contrast

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    print(f"device={device} threads={args.threads} color={args.color} "
          f"augment={args.augment} contrast={args.contrast}")

    data = build_data(args, device)
    print(f"split: task={len(data.task_x)} source={len(data.source_x)} "
          f"target={len(data.target_x)} "
          f"source-test={len(data.source_test.y)} "
          f"final-target={len(data.final_target.y)} "
          f"eval-draws={args.eval_draws}")
    print(f"dirt: source N({args.source_dirt:g},{args.dirt_sigma:g}) "
          f"target N({args.target_dirt:g},{args.dirt_sigma:g}); "
          f"contrast sigma {args.contrast_source_sigma:g}/"
          f"{args.contrast_target_sigma:g}")

    runs = []
    started = time.time()
    for seed in args.seeds:
        baseline = train_baseline(data, args, seed)
        if args.baseline_only:
            runs.append({"seed": seed, "baseline": baseline})
            print(f"seed {seed}: baseline source={baseline['source']:.4f} "
                  f"target={baseline['target']:.4f} "
                  f"probe1={baseline['probe1']:.3f} "
                  f"probe2={baseline['probe2']:.3f}")
            continue
        da = train_surrogate_da(data, args, seed)
        runs.append({"seed": seed, "baseline": baseline,
                     "surrogate_da": da})
        print(f"seed {seed}: baseline source={baseline['source']:.4f} "
              f"target={baseline['target']:.4f} "
              f"probe1={baseline['probe1']:.3f} "
              f"probe2={baseline['probe2']:.3f} | "
              f"DA source={da['source']:.4f} target={da['target']:.4f} "
              f"probe1={da['probe1']:.3f} probe2={da['probe2']:.3f} "
              f"critic={da['critic_final']:.3f}")

    summary = {}
    metrics = ("source", "target", "probe2", "probe2_train", "probe1",
               "probe_train_test", "probe_test_train", "probe2_mlp",
               "probe2_knn", "probe1_mlp", "probe1_knn", "probe_color")
    scenarios = (("baseline",) if args.baseline_only
                 else ("baseline", "surrogate_da"))
    for scenario in scenarios:
        for metric in metrics:
            values = np.array([r[scenario].get(metric, np.nan)
                               for r in runs])
            summary[f"{scenario}_{metric}"] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)),
            }
    for scenario in (() if args.acc_only else scenarios):
        for kind in ("maxp", "truep"):
            for key in ("ks", "target_frac_above_source_p99",
                        "source_mean", "target_mean"):
                values = np.array(
                    [r[scenario]["score_tails"][kind][key] for r in runs])
                summary[f"{scenario}_score_{kind}_{key}"] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
    if not args.baseline_only:
        values = np.array([r["surrogate_da"]["critic_final"] for r in runs])
        summary["surrogate_da_critic_final"] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }

    config = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()}
    result = {"config": config, "runs": runs, "summary": summary,
              "elapsed_s": time.time() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))

    print("\nscenario        source            target            "
          "probe1 AUC        probe2 AUC")
    for scenario in scenarios:
        c = summary[f"{scenario}_source"]
        d = summary[f"{scenario}_target"]
        p1 = summary[f"{scenario}_probe1"]
        p2 = summary[f"{scenario}_probe2"]
        print(f"{scenario:14s} {c['mean']:.4f}±{c['std']:.4f}   "
              f"{d['mean']:.4f}±{d['std']:.4f}   "
              f"{p1['mean']:.3f}±{p1['std']:.3f}   "
              f"{p2['mean']:.3f}±{p2['std']:.3f}")
    if args.color:
        for scenario in scenarios:
            pc = summary[f"{scenario}_probe_color"]
            print(f"{scenario:14s} paired colour probe: "
                  f"{pc['mean']:.3f}±{pc['std']:.3f}")
    print(f"\nresult -> {args.out}  ({result['elapsed_s']:.0f} s)")


if __name__ == "__main__":
    main()
