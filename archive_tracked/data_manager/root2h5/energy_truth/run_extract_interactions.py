#!/usr/bin/env python3
"""Batch driver for extract_interactions.py on cluster62.

Runs the extractor over many ROOT files with a small worker pool and one .npz per input
file, so the run is resumable: finished files are skipped on restart. Each file is handled
in its own subprocess, which keeps ROOT's global state from leaking between files and means
a single corrupt input cannot take the whole run down.

Worker count is deliberately capped well below the node's core count: cluster62 has 12
cores, no batch system, and other users on it, so nothing but this setting arbitrates load.

Usage (on cluster62, under nohup):
    nohup python3 run_extract_interactions.py \\
        --src /home3/ivkhar/Baikal/data/initial_data/MC_2020/nue2_100pev/root/all \\
        --out ~/mc_energy_truth_extract/nue2 --workers 4 --limit 10 > run.log 2>&1 &
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "extract_interactions.py"


def run_one(root_file: Path, out_dir: Path) -> tuple:
    out = out_dir / (root_file.stem + ".npz")
    if out.exists() and out.stat().st_size > 0:
        return "SKIP", root_file.name, 0.0, ""
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(SCRIPT), str(root_file), str(out)],
                          capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0 or not out.exists():
        if out.exists():          # never leave a partial file: it would be skipped later
            out.unlink()
        err = (proc.stderr or proc.stdout).strip().splitlines()
        return "FAIL", root_file.name, dt, err[-1][:300] if err else "no output"
    return "OK", root_file.name, dt, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory with .root files")
    ap.add_argument("--out", required=True, help="directory for .npz output")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all files")
    args = ap.parse_args()

    src, out_dir = Path(args.src), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.root"))
    if args.limit:
        files = files[:args.limit]
    print(f"files={len(files)}  workers={args.workers}  out={out_dir}", flush=True)

    t_start = time.time()
    counts = {"OK": 0, "SKIP": 0, "FAIL": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (status, name, dt, err) in enumerate(
                pool.map(lambda f: run_one(f, out_dir), files), start=1):
            counts[status] += 1
            msg = f"[{i}/{len(files)}] {status} {name} ({dt:.1f}s)"
            print(msg + (f" :: {err}" if err else ""), flush=True)

    total = time.time() - t_start
    size_mb = sum(p.stat().st_size for p in out_dir.glob("*.npz")) / 1e6
    print(f"\ndone in {total/60:.1f} min — OK={counts['OK']} SKIP={counts['SKIP']} "
          f"FAIL={counts['FAIL']}", flush=True)
    print(f"output: {len(list(out_dir.glob('*.npz')))} npz, {size_mb:.1f} MB total", flush=True)
    if counts["OK"]:
        print(f"mean per file: {total/max(counts['OK'],1):.1f}s", flush=True)


if __name__ == "__main__":
    main()
