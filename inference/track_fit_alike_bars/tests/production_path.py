"""Smoke test of the path the fit will actually be used on: `exp_full`.

    python inference/track_fit_alike_bars/tests/production_path.py
    python inference/track_fit_alike_bars/tests/production_path.py --part part_s2020_c03_r0348

There is no BARS reconstruction of `exp_full` to compare with, so this is not a
validation; it answers the questions that come before one.  Does the DQM solution
file load, does the fit run, and on how much of the sample does it succeed?

Two samples are reported.  A random one, because that is what the trigger stream
is, and the brightest 0.1% by total charge, because the analysis fits the events
its network scores highly, which are the muon-like ones.  Writes `summary.md` and
a provenance sidecar under `validation/<stamp>/`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference.reader.spec import SOURCES  # noqa: E402
from inference.track_fit_alike_bars import fit_events, masks  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "validation"


def sample(h5, part: str, idx: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"source": "exp_full", "data_class": "exp_full",
                         "part_key": part, "local_idx": idx})


def report(name: str, fits: pd.DataFrame) -> str:
    counts = fits["status"].value_counts()
    ok = fits[fits["status"] == "ok"]
    line = (f"| {name} | {len(fits)} | {counts.get('ok', 0)} "
            f"({counts.get('ok', 0) / len(fits):.1%}) | "
            f"{counts.get('no_clique', 0)} | "
            f"{ok['n_hits'].median() if len(ok) else float('nan'):.0f} | "
            f"{(ok['cov_matrix_status'] == 3).mean() if len(ok) else float('nan'):.0%} |")
    return line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", default="part_s2020_c03_r0348")
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--n-bright", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    started = time.time()
    spec = SOURCES["exp_full"]
    if args.part in spec.bad_runs:
        raise SystemExit(f"{args.part} is a known-bad run of exp_full")
    rng = np.random.default_rng(args.seed)
    with h5py.File(spec.h5, "r") as f:
        g = f["exp_full"]
        starts = g["raw/ev_starts"][args.part]["data"][:].astype(np.int64)
        charge = g["raw/data"][args.part]["data"][:, 0]
    n_events = len(starts) - 1
    sum_q = np.add.reduceat(charge, starts[:-1])
    random_idx = np.sort(rng.choice(n_events, size=args.n_random, replace=False))
    bright_idx = np.sort(rng.choice(np.argsort(-sum_q)[:max(args.n_bright * 10, 1)],
                                    size=args.n_bright, replace=False))

    provider = lambda rows: masks.dqm_masks("exp_full", rows["part_key"].iloc[0])  # noqa: E731
    lines = []
    for name, idx in (("random", random_idx), ("brightest 0.1% by charge", bright_idx)):
        fits = fit_events(sample(g, args.part, idx), n_workers=args.workers,
                          channel_mask=provider)
        lines.append(report(name, fits))

    mask = masks.dqm_masks("exp_full", args.part, None)
    text = "\n".join([
        f"# The fit on {args.part} (exp_full, no BARS reference)", "",
        f"DQM solution file {masks.solution_path(*masks.part_run(args.part))} "
        f"switches off {int((~mask).sum())} of {len(mask)} channels.", "",
        "| sample | events | fitted `ok` | `no_clique` | median `n_hits` of those | `cov==3` |",
        "|---|---|---|---|---|---|", *lines, "",
    ])
    out = OUT_DIR / datetime.now().strftime("%y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(text)
    (out / "summary.md.meta.json").write_text(json.dumps({
        "script": "inference/track_fit_alike_bars/tests/production_path.py",
        "argv": sys.argv, "args": vars(args), "part": args.part,
        "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        "runtime_s": round(time.time() - started, 1),
        "written": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
