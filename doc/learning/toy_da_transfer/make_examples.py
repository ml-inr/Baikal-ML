#!/usr/bin/env python3
"""Render example images for the distributional-dirt toy study.

Three PNGs go under ``figures/``:

* ``classes.png``  -- the ten task classes clean and at source dirt 2.75.
* ``domains.png``  -- source-main, source-DA, target-DA (grey/coloured),
  final-target (grey/coloured), four examples each.
* ``construction.png`` -- the 2-digit pipeline (clean pair -> dirtied ->
  cropped -> concatenated -> resized), the colour/no-colour comparison, and
  the dirt/contrast histograms.

The examples use the same helpers as ``toy_da.py`` and fixed seeds.  The
training draws fresh augmentations every batch; these are representative
draws, not a fixed training batch.

Usage:
    python make_examples.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

import toy_da as toy

ROOT = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def make_args(color: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        augment=True, contrast=True, color=color,
        source_dirt=toy.SOURCE_DIRT[0], target_dirt=toy.TARGET_DIRT[0],
        dirt_sigma=toy.SOURCE_DIRT[1],
        contrast_source_sigma=toy.SOURCE_CONTRAST[1],
        contrast_target_sigma=0.45,
        color_mode="global",
        color_alpha=0.90,
    )


def split_pools():
    train_x, train_y, test_x, test_y = toy.load_mnist(toy.DEFAULT_DATA)
    task_x, task_y = train_x[:12000], train_y[:12000]
    source_x = train_x[12000:24000]
    source_test_x = train_x[24000:28000]
    target_x = test_x[:6000]
    target_test_x = test_x[6000:10000]
    return task_x, task_y, source_x, source_test_x, target_x, target_test_x


def show(ax, image: torch.Tensor) -> None:
    array = image.detach().cpu()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 3:
        ax.imshow(array.permute(1, 2, 0).numpy(), vmin=0.0, vmax=1.0,
                  interpolation="nearest", aspect="auto")
    else:
        ax.imshow(array.squeeze().numpy(), cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.7")


def one_digit(pool, mean, *, target, color, seed, n=4):
    args = make_args(color)
    generator = torch.Generator().manual_seed(seed)
    strengths = torch.full((n,), mean)
    base = toy.prepare_1digit_base(
        pool[:n], strengths, target=target, args=args, generator=generator)
    return (toy.colorize(base, generator, mode=args.color_mode,
                         alpha=args.color_alpha) if (target and color)
            else toy.gray_rgb(base))


def two_digit(pool, mean, *, target, color, seed, n=4):
    args = make_args(color)
    generator = torch.Generator().manual_seed(seed)
    strengths = torch.full((n,), mean)
    base = toy.prepare_2digit_base(
        pool[:n], pool[n:2 * n], strengths, target=target, args=args,
        generator=generator)
    return (toy.colorize(base, generator, mode=args.color_mode,
                         alpha=args.color_alpha) if (target and color)
            else toy.gray_rgb(base))


def figure_classes(task_x, task_y, out: Path) -> None:
    args = make_args(False)
    generator = torch.Generator().manual_seed(20261110)
    fig, axes = plt.subplots(2, 10, figsize=(15.0, 3.8))
    for digit in range(10):
        index = int((task_y == digit).nonzero()[0])
        clean = task_x[index:index + 1]
        dirty = toy.prepare_1digit_base(
            clean, torch.tensor([2.75]), target=False, args=args,
            generator=generator)
        show(axes[0, digit], toy.gray_rgb(clean))
        show(axes[1, digit], toy.gray_rgb(dirty))
        axes[0, digit].set_title(str(digit), fontsize=10)
    axes[0, 0].set_ylabel("чисто", fontsize=10, labelpad=6)
    axes[1, 0].set_ylabel("source 2.75\n+ augment", fontsize=9, labelpad=6)
    fig.suptitle("Классы задачи: 1-значные цифры, source main task",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def figure_domains(task_x, source_x, target_x, target_test_x,
                   out: Path) -> None:
    rows = [
        ("source-main\n1 цифра, 2.75", one_digit(
            task_x[100:], 2.75, target=False, color=False, seed=20261111)),
        ("source-DA\n2 цифры, 2.75", two_digit(
            source_x[100:], 2.75, target=False, color=False,
            seed=20261112)),
        ("target-DA\n2 цифры, 3.0, серый", two_digit(
            target_x[100:], 3.0, target=True, color=False, seed=20261113)),
        ("target-DA\n2 цифры, 3.0, цвет", two_digit(
            target_x[100:], 3.0, target=True, color=True, seed=20261113)),
        ("final-target\n1 цифра, 3.0, серый", one_digit(
            target_test_x[100:], 3.0, target=True, color=False,
            seed=20261114)),
        ("final-target\n1 цифра, 3.0, цвет", one_digit(
            target_test_x[100:], 3.0, target=True, color=True,
            seed=20261114)),
    ]
    fig, axes = plt.subplots(6, 4, figsize=(11.0, 12.0))
    for row, (label, images) in enumerate(rows):
        for column in range(4):
            show(axes[row, column], images[column])
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(label, fontsize=9, labelpad=6)
    fig.suptitle("Домены новой постановки (примеры, медиана грязи)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def figure_construction(target_x, out: Path) -> None:
    args = make_args(False)
    generator = torch.Generator().manual_seed(20261115)
    strengths = torch.tensor([3.0])
    first = target_x[0:1]
    second = target_x[1:2]
    left = toy.corrupt_batch(first, strengths, generator)
    right = toy.corrupt_batch(second, strengths, generator)
    width = int(toy.dirt_width(strengths)[0])
    pad = (28 - width) // 2
    left_crop = left[0, :, :, pad:pad + width]
    right_crop = right[0, :, :, pad:pad + width]
    joined = torch.cat([left_crop, right_crop], dim=2)
    resized = F.interpolate(joined.unsqueeze(0), size=(28, 28),
                            mode="bilinear", align_corners=False)

    fig = plt.figure(figsize=(13.0, 8.0))
    grid = fig.add_gridspec(3, 8, height_ratios=[1.0, 1.0, 1.25],
                            hspace=0.45, wspace=0.25,
                            left=0.05, right=0.98, top=0.90, bottom=0.06)
    steps = [
        ("чистая пара", torch.cat([first, second], dim=3)),
        ("левый digit\nпосле грязи", left),
        ("правый digit\nпосле грязи", right),
        ("левый кроп", left_crop.unsqueeze(0)),
        ("правый кроп", right_crop.unsqueeze(0)),
        ("склейка", joined.unsqueeze(0)),
        ("resize 28×28", resized),
    ]
    for column, (title, image) in enumerate(steps):
        ax = fig.add_subplot(grid[0, column:column + 1])
        show(ax, image)
        ax.set_title(title, fontsize=8)
    # the 8th column of the top row is empty; use it for the width note
    fig.text(0.93, 0.80, f"w={width} px\n→ 28 px", fontsize=9,
             va="center", ha="center")

    colour = toy.colorize(
        resized, torch.Generator().manual_seed(20261116),
        mode=args.color_mode, alpha=args.color_alpha)
    grey = toy.gray_rgb(resized)
    deviation = (colour[0] - grey[0]).abs().amax(dim=0).numpy()
    for column, (title, image) in enumerate([
            ("серый target", grey), ("цвет global α=0.9", colour)]):
        ax = fig.add_subplot(grid[1, column * 2:(column + 1) * 2])
        show(ax, image)
        ax.set_title(title, fontsize=9)
    ax = fig.add_subplot(grid[1, 4:6])
    im = ax.imshow(deviation, cmap="magma", vmin=0.0, vmax=0.3,
                   interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("|цвет − серый|", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    hist_ax = fig.add_subplot(grid[2, 0:3])
    generator = torch.Generator().manual_seed(20261117)
    source = toy.truncated_normal(
        20000, *toy.SOURCE_DIRT, generator, torch.device("cpu")).numpy()
    target = toy.truncated_normal(
        20000, *toy.TARGET_DIRT, generator, torch.device("cpu")).numpy()
    hist_ax.hist(source, bins=50, alpha=0.55, density=True,
                 label="source N(2.75, 0.25)")
    hist_ax.hist(target, bins=50, alpha=0.55, density=True,
                 label="target N(3.0, 0.25)")
    hist_ax.set_xlabel("dirt level")
    hist_ax.set_ylabel("density")
    hist_ax.legend(fontsize=8)
    hist_ax.set_title("Распределения грязи (усечение ±2σ)", fontsize=9)

    contrast_ax = fig.add_subplot(grid[2, 3:6])
    generator = torch.Generator().manual_seed(20261118)
    source_c = toy.truncated_normal(
        20000, 1.0, args.contrast_source_sigma, generator,
        torch.device("cpu")).numpy()
    target_c = toy.truncated_normal(
        20000, 1.0, args.contrast_target_sigma, generator,
        torch.device("cpu")).numpy()
    contrast_ax.hist(source_c, bins=50, alpha=0.55, density=True,
                     label="source N(1, 0.08)")
    contrast_ax.hist(target_c, bins=50, alpha=0.55, density=True,
                     label="target N(1, 0.45)")
    contrast_ax.set_xlabel("contrast factor")
    contrast_ax.set_ylabel("density")
    contrast_ax.legend(fontsize=8)
    contrast_ax.set_title("Контраст: source σ=0.08, target σ=0.45", fontsize=9)

    fig.suptitle("2-значный кадр: ширина 28, глобальный цвет α=0.9", fontsize=13)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["savefig.dpi"] = args.dpi

    task_x, task_y, source_x, source_test_x, target_x, target_test_x = \
        split_pools()
    figure_classes(task_x, task_y, args.out_dir / "classes.png")
    figure_domains(task_x, source_x, target_x, target_test_x,
                   args.out_dir / "domains.png")
    figure_construction(target_x, args.out_dir / "construction.png")
    print(f"figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
