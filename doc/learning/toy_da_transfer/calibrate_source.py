#!/usr/bin/env python3
"""Four-model calibration of the dirt/colour shift.

The shift is accepted only if both checks hold:

    A  trained on source, tested on source-test and final-target
    B  trained on target, tested on final-target
    A0 trained on clean source, tested on clean source-test
    B0 trained on clean target, tested on clean target-test

    check 1: A(target) < A(source)
             the target is harder for a source-only model;
    check 2: B(target) >= A(source) + (B0(target) - A0(source))
             the target-trained model keeps at least the clean
             dataset-realisation difference; the raw
             B(target) >= A(source) is reported as well.

A0/B0 are needed because the source and target roles are drawn from
different MNIST pools (train vs test).  Without them a one or two point
difference between A(source) and B(target) cannot be attributed to the
domain shift.

Usage:
    python calibrate_source.py --source-dirt 2.5 --target-dirt 3.25 \\
        --epochs 5 --seed 0 --out runs/calib_s2.5_t3.25.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import toy_da as toy

ROOT = Path(__file__).resolve().parent


def make_args(args: argparse.Namespace, clean: bool) -> argparse.Namespace:
    return argparse.Namespace(
        augment=not clean,
        contrast=not clean,
        color=(args.color and not clean),
        source_dirt=0.0 if clean else args.source_dirt,
        target_dirt=0.0 if clean else args.target_dirt,
        dirt_sigma=0.0 if clean else args.dirt_sigma,
        contrast_source_sigma=args.contrast_source_sigma,
        contrast_target_sigma=args.contrast_target_sigma,
        contrast_source_mean=getattr(args, "contrast_source_mean", 1.0),
        contrast_target_mean=getattr(args, "contrast_target_mean", 1.0),
        color_mode=getattr(args, "color_mode", "background"),
        color_alpha=getattr(args, "color_alpha", 0.30),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


def train_supervised(train_x: torch.Tensor, train_y: torch.Tensor, *,
                     target: bool, evals: dict[str, toy.EvalSet],
                     args: argparse.Namespace, seed: int,
                     device: torch.device) -> dict:
    torch.manual_seed(seed)
    encoder = toy.Encoder().to(device)
    classifier = toy.Classifier(encoder.out_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(seed)
    history = {name: [] for name in evals}

    for _ in range(args.epochs):
        permutation = torch.randperm(len(train_x), generator=generator)
        encoder.train()
        classifier.train()
        for start in range(0, len(train_x), args.batch_size):
            index = permutation[start:start + args.batch_size]
            strengths = toy.truncated_normal(
                len(index), *toy.dirt_params(target, args), generator,
                device)
            base = toy.prepare_1digit_base(
                train_x[index], strengths, target=target, args=args,
                generator=generator)
            x = toy.finish_domain_g(
                base, target=target, color=(args.color and target),
                generator=generator,
                color_mode=getattr(args, "color_mode", "background"),
                color_alpha=getattr(args, "color_alpha", 0.30))
            loss = torch.nn.functional.cross_entropy(
                classifier(encoder(x)), train_y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for name, eval_set in evals.items():
            accuracy, _ = toy.accuracy_draws(encoder, classifier, eval_set)
            history[name].append(accuracy)

    return {
        name: {
            "final": values[-1],
            "last2": float(np.mean(values[-2:])),
            "history": values,
        }
        for name, values in history.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=toy.DEFAULT_DATA)
    parser.add_argument("--source-dirt", type=float, default=2.75)
    parser.add_argument("--target-dirt", type=float, default=3.00)
    parser.add_argument("--dirt-sigma", type=float, default=0.25)
    parser.add_argument("--contrast-source-sigma", type=float, default=0.08)
    parser.add_argument("--contrast-target-sigma", type=float, default=0.15)
    parser.add_argument("--contrast-source-mean", type=float, default=1.0)
    parser.add_argument("--contrast-target-mean", type=float, default=1.0)
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--color-mode", choices=("background", "global"),
                        default="background")
    parser.add_argument("--color-alpha", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-draws", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=20261101)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-source-train", type=int, default=12000)
    parser.add_argument("--n-target-train", type=int, default=6000)
    parser.add_argument("--n-final-target", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs" / "calibrate_domains.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    started = time.time()

    train_x, train_y, test_x, test_y = toy.load_mnist(args.data_dir)
    task_x = train_x[:args.n_source_train].to(device)
    task_y = train_y[:args.n_source_train].to(device)
    source_test_x = train_x[24000:28000].to(device)
    source_test_y = train_y[24000:28000].to(device)
    t0 = args.n_target_train
    target_train_x = test_x[:t0].to(device)
    target_train_y = test_y[:t0].to(device)
    target_test_x = test_x[t0:t0 + args.n_final_target].to(device)
    target_test_y = test_y[t0:t0 + args.n_final_target].to(device)

    dirty = make_args(args, clean=False)
    clean = make_args(args, clean=True)

    source_test = toy.build_eval_1digit(
        source_test_x, source_test_y, target=False, color=False,
        seed=args.eval_seed, draws=args.eval_draws, args=dirty,
        device=device)
    final_target = toy.build_eval_1digit(
        target_test_x, target_test_y, target=True, color=args.color,
        seed=args.eval_seed + 10, draws=args.eval_draws, args=dirty,
        device=device)
    clean_source_test = toy.build_eval_1digit(
        source_test_x, source_test_y, target=False, color=False,
        seed=args.eval_seed + 20, draws=args.eval_draws, args=clean,
        device=device)
    clean_target_test = toy.build_eval_1digit(
        target_test_x, target_test_y, target=True, color=False,
        seed=args.eval_seed + 30, draws=args.eval_draws, args=clean,
        device=device)

    print(f"dirt source {args.source_dirt:g} target {args.target_dirt:g} "
          f"sigma {args.dirt_sigma:g}; color={args.color}; "
          f"epochs={args.epochs}")

    a = train_supervised(task_x, task_y, target=False,
                         evals={"source": source_test,
                                "target": final_target},
                         args=dirty, seed=args.seed, device=device)
    b = train_supervised(target_train_x, target_train_y, target=True,
                         evals={"target": final_target},
                         args=dirty, seed=args.seed, device=device)
    a0 = train_supervised(task_x, task_y, target=False,
                          evals={"source": clean_source_test},
                          args=clean, seed=args.seed, device=device)
    b0 = train_supervised(target_train_x, target_train_y, target=True,
                          evals={"target": clean_target_test},
                          args=clean, seed=args.seed, device=device)

    a_source = a["source"]["last2"]
    a_target = a["target"]["last2"]
    b_target = b["target"]["last2"]
    a0_source = a0["source"]["last2"]
    b0_target = b0["target"]["last2"]
    clean_bias = b0_target - a0_source
    check1 = a_target < a_source
    raw2 = b_target - a_source
    adjusted2 = raw2 - clean_bias
    check2 = adjusted2 >= 0.0

    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "A": a,
        "B": b,
        "A0": a0,
        "B0": b0,
        "summary": {
            "A_source": a_source,
            "A_target": a_target,
            "B_target": b_target,
            "A0_source_clean": a0_source,
            "B0_target_clean": b0_target,
            "clean_bias": clean_bias,
            "check1_target_worse": check1,
            "raw2_B_minus_A": raw2,
            "adjusted2": adjusted2,
            "check2_target_learnable": check2,
        },
        "elapsed_s": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    print(f"A  source={a_source:.4f} target={a_target:.4f} "
          f"(gap {100 * (a_target - a_source):+.2f} pp)")
    print(f"B  target={b_target:.4f}")
    print(f"A0 clean source={a0_source:.4f}; "
          f"B0 clean target={b0_target:.4f}; "
          f"clean bias {100 * clean_bias:+.2f} pp")
    print(f"check 1 (target harder for A): {check1}")
    print(f"check 2 raw B-A={100 * raw2:+.2f} pp; "
          f"adjusted for clean bias={100 * adjusted2:+.2f} pp "
          f"-> {check2}")
    print(f"result -> {args.out}  ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
