"""Grid search over the adversarial balance of the toy surrogate-DA critic.

Data split (variant a): task/source/source-test come from MNIST train (MC);
target-train/target-test from MNIST test (exp).  The DANN encoder sees the
unlabelled source-train and target-train; source-test and target-test are held
out.  See toy_da.py for the full table.

The grid targets the knobs that decide whether the in-game critic stays alive
under a moving encoder:

    --critic-steps K     critic updates per encoder batch
    --critic-lr          critic learning rate (separate optimiser)
    --critic-hidden      0 = linear (logistic) critic, otherwise an MLP
    --da-warmup-epochs   critic-only warm-up before the domain loss
    --da-ramp-epochs     linear ramp to da_lambda after the warm-up

Stage 1 crosses K, critic lr, capacity (0 = linear), warm-up (0/1 epoch) and
ramp (0/1 epoch) at critic batch 512 and `da_lambda = 0.1`.  One seed per
config; the surviving configs are re-run with three seeds by toy_da.py.

Usage:
    python sweep.py --workers 8 --epochs 4 --out runs/sweep_stage1.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

import toy_da as toy

ROOT = Path(__file__).resolve().parent
DATA: dict | None = None


def build_data(args: argparse.Namespace) -> dict:
    train_x, train_y, test_x, test_y = toy.load_mnist(ROOT / "data")
    if (args.n_task + args.n_source + args.n_source_test > len(train_x)
            or args.n_target_train + args.n_target_test > len(test_x)):
        raise ValueError("split exceeds the MNIST pools")
    device = train_x.device

    task_x = train_x[:args.n_task]
    task_y = train_y[:args.n_task]
    source_x = train_x[args.n_task:args.n_task + args.n_source]
    s0 = args.n_task + args.n_source
    source_test_raw = train_x[s0:s0 + args.n_source_test]
    source_test_y = train_y[s0:s0 + args.n_source_test]
    target_train_x = test_x[:args.n_target_train]
    t0 = args.n_target_train
    target_test_raw = test_x[t0:t0 + args.n_target_test]
    target_test_y = test_y[t0:t0 + args.n_target_test]

    eval_color_gen = torch.Generator(device=device).manual_seed(
        args.probe_seed + 2)
    source_gen = torch.Generator(device=device).manual_seed(20261027)
    source_test = toy.corrupt(source_test_raw, args.source_strength,
                              source_gen)
    target_gen = torch.Generator(device=device).manual_seed(20261024)
    target_test = toy.corrupt(target_test_raw, args.strength, target_gen)
    if args.target_color:
        source_test = toy.to_gray_rgb(source_test)
        target_test = toy.to_subtle_color(target_test, eval_color_gen,
                                          args.color_black_frac,
                                          args.color_white_frac,
                                          args.color_tint)

    probe_gen = torch.Generator(device=device).manual_seed(20261025)
    half_s = len(source_test_raw) // 2
    half_t = len(target_test_raw) // 2
    source_probe_clean = toy.two_digit(source_test_raw[:half_s],
                                       source_test_raw[half_s:], probe_gen)
    target_probe_clean = toy.two_digit(target_test_raw[:half_t],
                                       target_test_raw[half_t:], probe_gen)
    source_probe = toy.corrupt(source_probe_clean, args.source_strength,
                               probe_gen)
    target_probe_dirty = toy.corrupt(target_probe_clean, args.strength,
                                     probe_gen)
    target_probe = target_probe_dirty
    probe_color = None
    if args.target_color:
        source_probe = toy.to_gray_rgb(source_probe)
        target_probe_gray = toy.to_gray_rgb(target_probe_dirty)
        target_probe = toy.to_subtle_color(target_probe_dirty, eval_color_gen,
                                           args.color_black_frac,
                                           args.color_white_frac,
                                           args.color_tint)
        probe_color = (target_probe_gray, target_probe)
    probe2 = (source_probe, target_probe)
    probe1 = (source_test, target_test)

    train_probe_gen = torch.Generator(device=device).manual_seed(
        args.probe_seed + 1)
    n_probe = min(2000, len(source_x) // 2, len(target_train_x) // 2)
    source_train_probe = toy.two_digit(source_x[:n_probe],
                                       source_x[n_probe:2 * n_probe],
                                       train_probe_gen)
    target_train_probe = toy.two_digit(target_train_x[:n_probe],
                                       target_train_x[n_probe:2 * n_probe],
                                       train_probe_gen)
    source_train_c = toy.corrupt(source_train_probe, args.source_strength,
                                 train_probe_gen)
    target_train_c = toy.corrupt(target_train_probe, args.strength,
                                 train_probe_gen)
    if args.target_color:
        source_train_c = toy.to_gray_rgb(source_train_c)
        target_train_c = toy.to_subtle_color(target_train_c, eval_color_gen,
                                             args.color_black_frac,
                                             args.color_white_frac,
                                             args.color_tint)
    probe2_train = (source_train_c, target_train_c)
    probe_train_test = (source_train_c, target_probe)
    probe_test_train = (source_probe, target_train_c)
    return dict(task_x=task_x, task_y=task_y, source_x=source_x,
                target_train_x=target_train_x,
                source_test=source_test, source_test_y=source_test_y,
                target_test=target_test, target_test_y=target_test_y,
                probe2=probe2, probe2_train=probe2_train,
                probe_train_test=probe_train_test,
                probe_test_train=probe_test_train, probe1=probe1,
                probe_color=probe_color)


def namespace(cfg: dict, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        epochs=args.epochs, batch_size=args.batch_size,
        strength=args.strength, source_strength=args.source_strength,
        da_lambda=cfg["da_lambda"],
        da_warmup_epochs=cfg["da_warmup_epochs"],
        da_ramp_epochs=cfg["da_ramp_epochs"],
        grl_lambda=args.grl_lambda, probe_seed=args.probe_seed,
        critic_steps=cfg["critic_steps"], critic_batch=cfg["critic_batch"],
        critic_lr=cfg["critic_lr"], critic_hidden=cfg["critic_hidden"],
        critic_dropout=cfg["critic_dropout"], critic_norm=cfg["critic_norm"],
        critic_fresh=cfg["critic_fresh"], n_classes=10,
        augment=False, da_form="two_digit", shuffle_domain=False,
        target_color=args.target_color,
        color_black_frac=args.color_black_frac,
        color_white_frac=args.color_white_frac,
        color_tint=args.color_tint,
    )


def init_worker(args: argparse.Namespace) -> None:
    """Spawned workers build their own data (fork + autograd is unsupported)."""
    global DATA
    torch.set_num_threads(args.threads)
    DATA = build_data(args)


def run_one(cfg: dict) -> dict:
    torch.set_num_threads(cfg["threads"])
    assert DATA is not None
    ns = namespace(cfg, cfg["args"])
    result = toy.train_surrogate_da(
        DATA["task_x"], DATA["task_y"], DATA["source_x"],
        DATA["target_train_x"], DATA["source_test"], DATA["source_test_y"],
        DATA["target_test"], DATA["target_test_y"], DATA["probe2"],
        DATA["probe2_train"], DATA["probe_train_test"],
        DATA["probe_test_train"], DATA["probe1"], ns, cfg["seed"],
        probe_color=DATA["probe_color"])
    return {**cfg, **result}


def grid_stage1(args: argparse.Namespace) -> list[dict]:
    """Critic strength x domain-loss schedule, at a non-destructive lambda."""
    configs = []
    for steps, lr, hidden, warmup, ramp in itertools.product(
            (5, 10), (1e-3, 3e-3), (0, 128), (0, 1), (0, 1)):
        configs.append(dict(
            critic_steps=steps, critic_lr=lr, critic_hidden=hidden,
            critic_norm=False, critic_batch=512, critic_dropout=0.0,
            critic_fresh=False, da_lambda=0.1,
            da_warmup_epochs=warmup, da_ramp_epochs=ramp, seed=0,
            threads=args.threads, args=args,
        ))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs" / "sweep_stage1.json")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--n-task", type=int, default=12000)
    parser.add_argument("--n-source", type=int, default=12000)
    parser.add_argument("--n-source-test", type=int, default=4000)
    parser.add_argument("--n-target-train", type=int, default=6000)
    parser.add_argument("--n-target-test", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--source-strength", type=float, default=0.8)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--target-color", action="store_true")
    parser.add_argument("--color-black-frac", type=float, default=0.01)
    parser.add_argument("--color-white-frac", type=float, default=0.005)
    parser.add_argument("--color-tint", type=float, default=0.15)
    parser.add_argument("--probe-seed", type=int, default=20261026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    torch.set_num_threads(1)
    data = build_data(args)
    global DATA
    DATA = data

    base_args = namespace(dict(da_lambda=0.0, critic_steps=1,
                               critic_batch=256, critic_lr=1e-3,
                               critic_hidden=128, critic_dropout=0.0,
                               critic_norm=False, critic_fresh=False,
                               da_warmup_epochs=0.0, da_ramp_epochs=0.0,
                               args=args), args)
    baseline = toy.train_baseline(
        data["task_x"], data["task_y"], data["source_test"],
        data["source_test_y"], data["target_test"], data["target_test_y"],
        data["probe2"], data["probe2_train"], data["probe_train_test"],
        data["probe_test_train"], data["probe1"], base_args, 0,
        probe_color=data["probe_color"])
    print(f"baseline (source {args.source_strength:g} vs target "
          f"{args.strength:g}): source={baseline['source']:.4f} "
          f"target={baseline['target']:.4f} probe2={baseline['probe2']:.4f} "
          f"probe1={baseline['probe1']:.4f}")

    configs = grid_stage1(args)
    print(f"{len(configs)} configs, {args.workers} workers, "
          f"{args.epochs} epochs")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=init_worker,
                  initargs=(args,)) as pool:
        results = pool.map(run_one, configs)

    alive = [r for r in results if r["critic_final"] >= 0.75]
    clean_da = [r for r in alive
                if r["target"] >= baseline["target"] - 0.02]
    results.sort(key=lambda r: (-r["critic_final"], r["probe2"]))
    print("\n  K   lr     h  warm ramp  critic  c_train  probe2  p2train  "
          "target source")
    for r in results:
        print(f"{r['critic_steps']:3d} {r['critic_lr']:.0e} "
              f"{r['critic_hidden']:3d} {r['da_warmup_epochs']:4.0f} "
              f"{r['da_ramp_epochs']:4.0f}  {r['critic_final']:.3f}  "
              f"{r['critic_train_final']:.3f}  {r['probe2']:.4f}  "
              f"{r['probe2_train']:.4f}  {r['target']:.3f}  "
              f"{r['source']:.3f}")
    print(f"\nalive (critic>=0.75): {len(alive)}/{len(results)}")
    print(f"alive and target within 2 pp of baseline: {len(clean_da)}")
    if alive:
        best = min(alive, key=lambda r: r["probe2"])
        print(f"lowest probe2 among alive: {best['probe2']:.4f} "
              f"(K={best['critic_steps']}, lr={best['critic_lr']:.0e}, "
              f"h={best['critic_hidden']}, warm={best['da_warmup_epochs']}, "
              f"ramp={best['da_ramp_epochs']}, target={best['target']:.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
               "baseline": baseline,
               "results": [{k: v for k, v in r.items() if k != "args"}
                           for r in results],
               "elapsed_s": time.time() - started}
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"\nresult -> {args.out}  ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
