#!/usr/bin/env python3
"""Summarise toy-DA runs over the last N epochs.

Reads the JSON files written by ``toy_da.py`` and averages every per-epoch
history over its last ``--last`` epochs.  Reading the final epoch alone is
what flipped the earlier conclusions, so this script is the single source of
the numbers quoted in README.md and in the chat report.

The nonlinear probes and the score-tail diagnostics are computed once, at the
end of a run; they are printed as final-epoch values and marked as such.

Usage:
    python analyze.py runs/new_*.json --last 3
    python analyze.py runs/new_target3_l02.json --last 3 --out runs/a.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def tail_mean(history: list[float], n: int) -> float:
    values = history[-n:] if n > 0 else history
    return float(np.mean(values))


def opt(d: dict, *keys, default: float = float("nan")):
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return float(d)


def per_seed(run: dict, scenario: str, n: int) -> dict:
    r = run[scenario]
    out = {
        "source": tail_mean(r["source_history"], n),
        "target": tail_mean(r["target_history"], n),
        "probe2": tail_mean(r["probe2_history"], n),
        "probe1": tail_mean(r["probe1_history"], n),
        "probe2_mlp": opt(r, "probe2_mlp"),
        "probe2_knn": opt(r, "probe2_knn"),
        "probe1_mlp": opt(r, "probe1_mlp"),
        "probe1_knn": opt(r, "probe1_knn"),
        "probe_color": opt(r, "probe_color"),
        "score_maxp_ks": opt(r, "score_tails", "maxp", "ks"),
        "score_maxp_frac": opt(r, "score_tails", "maxp",
                               "target_frac_above_source_p99"),
        "score_truep_ks": opt(r, "score_tails", "truep", "ks"),
        "score_truep_frac": opt(r, "score_tails", "truep",
                                "target_frac_above_source_p99"),
    }
    if "probe2_train_history" in r:
        out["probe2_train"] = tail_mean(r["probe2_train_history"], n)
    else:
        # baseline computes the in-sample probe only at the end
        out["probe2_train"] = float(r.get("probe2_train", float("nan")))
    if "critic_history" in r:
        out["critic"] = tail_mean(r["critic_history"], n)
    return out


def aggregate(run: dict, scenario: str, n: int) -> dict:
    rows = [per_seed(r, scenario, n) for r in run["runs"]]
    out = {}
    for key in rows[0]:
        values = np.array([row[key] for row in rows])
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return out


def fmt(stats: dict, key: str) -> str:
    s = stats[key]
    return f"{s['mean']:.4f}±{s['std']:.4f}"


def label(path: Path) -> str:
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--last", type=int, default=3,
                        help="epochs averaged at the end of a run")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs" / "summary_lastN.json")
    args = parser.parse_args()

    summary = {}
    for path in args.runs:
        run = json.loads(path.read_text())
        name = label(path)
        summary[name] = {
            "config": run["config"],
            "last": args.last,
            "baseline": aggregate(run, "baseline", args.last),
            "surrogate_da": aggregate(run, "surrogate_da", args.last),
        }

    print(f"mean of last {args.last} epochs; ± is the seed spread")
    print("\nrun                          scen   source         target         "
          "probe2         p2_train       probe1         critic")
    for name, s in summary.items():
        for scen in ("baseline", "surrogate_da"):
            a = s[scen]
            print(f"{name:28s} {scen[:4]:4s}  {fmt(a, 'source'):14s} "
                  f"{fmt(a, 'target'):14s} {fmt(a, 'probe2'):14s} "
                  f"{fmt(a, 'probe2_train'):14s} {fmt(a, 'probe1'):14s} "
                  f"{fmt(a, 'critic') if 'critic' in a else '—':14s}")

    print("\nDA - baseline (percentage points; last-epoch nonlinear probes)")
    print("run                          source   target   probe2   p2train  "
          "probe1   | MLP p2   kNN p2")
    for name, s in summary.items():
        b, d = s["baseline"], s["surrogate_da"]
        delta = {k: {"mean": 100.0 * (d[k]["mean"] - b[k]["mean"])}
                 for k in ("source", "target", "probe2", "probe2_train",
                           "probe1")}
        print(f"{name:28s} {delta['source']['mean']:+7.2f}  "
              f"{delta['target']['mean']:+7.2f}  "
              f"{delta['probe2']['mean']:+7.2f}  "
              f"{delta['probe2_train']['mean']:+7.2f}  "
              f"{delta['probe1']['mean']:+7.2f}  | "
              f"{b['probe2_mlp']['mean']:.3f}->{d['probe2_mlp']['mean']:.3f}  "
              f"{b['probe2_knn']['mean']:.3f}->{d['probe2_knn']['mean']:.3f}")

    print("\ncolour probe, paired (same target images with/without colour), "
          "final epoch")
    for name, s in summary.items():
        b, d = s["baseline"], s["surrogate_da"]
        if np.isfinite(b["probe_color"]["mean"]):
            print(f"{name:28s} {b['probe_color']['mean']:.3f} -> "
                  f"{d['probe_color']['mean']:.3f}")

    print("\nscore tails, final epoch (max-softmax): KS and target frac "
          "above source p99")
    print("run                          KS base->DA      frac base->DA")
    for name, s in summary.items():
        b, d = s["baseline"], s["surrogate_da"]
        print(f"{name:28s} {b['score_maxp_ks']['mean']:.3f}->"
              f"{d['score_maxp_ks']['mean']:.3f}        "
              f"{b['score_maxp_frac']['mean']:.4f}->"
              f"{d['score_maxp_frac']['mean']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    print(f"\nresult -> {args.out}")


if __name__ == "__main__":
    main()
