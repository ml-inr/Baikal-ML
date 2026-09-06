#!/usr/bin/env python
"""Extend a nu-classifier NPY dataset with GROUND-TRUTH zenith theta.

Adds `theta.npy` (per-event, degrees, aligned with labels.npy) by reading
`prime_prty[:,0]` from the source MC h5 via the dataset's back-links
(particle_types + h5_part_keys + h5_local_event_ids). No full rebuild.

Needed for the horizon-aware (theta) loss: MC events near theta=90 deg are the
up/down-ambiguous ones the classifier confuses (see the 2026-07-06 report).

Usage:
    python -m data_manager.nu_classifier_ds_builder.add_theta \
        --npy-dir data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8 \
        --mc-h5   data_manager/data/h5datasets/baikal_mc_merged.h5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--mc-h5", default="data_manager/data/h5datasets/baikal_mc_merged.h5")
    ap.add_argument("--out", default="theta.npy")
    args = ap.parse_args()

    d = Path(args.npy_dir)
    info = json.load(open(d / "dataset_info.json"))
    # int code -> h5 group (particle type)
    code2type = {v: k for k, v in info["particle_encode"].items()}

    ptype = np.load(d / "particle_types.npy")                       # (N,) int8
    part_keys = np.asarray(np.load(d / "h5_part_keys.npy", allow_pickle=True), dtype=str)
    local_ids = np.asarray(np.load(d / "h5_local_event_ids.npy"), dtype=np.int64)
    N = len(ptype)
    assert len(part_keys) == N == len(local_ids), "back-link length mismatch"

    theta = np.full(N, np.nan, dtype=np.float32)
    with h5py.File(args.mc_h5, "r") as f:
        for code, gname in code2type.items():
            grp = f[gname]["prime_prty"]
            sel = np.where(ptype == code)[0]
            if not len(sel):
                continue
            # group this class's events by part, read theta column once per part
            pk_sel = part_keys[sel]
            order = np.argsort(pk_sel, kind="stable")
            sel_sorted = sel[order]
            pk_sorted = pk_sel[order]
            # iterate contiguous part blocks
            uniq, starts = np.unique(pk_sorted, return_index=True)
            starts = list(starts) + [len(pk_sorted)]
            for j, pk in enumerate(uniq):
                block = sel_sorted[starts[j]:starts[j + 1]]
                key = f"{pk}/data"
                if key not in grp:
                    continue
                th = grp[key][:, 0].astype(np.float32)              # prime_prty[:,0] = theta (deg)
                li = local_ids[block]
                ok = li < len(th)
                theta[block[ok]] = th[li[ok]]
            done = np.isfinite(theta[sel]).sum()
            print(f"  {gname:12s}: {done:,}/{len(sel):,} events got theta")

    out = d / args.out
    np.save(out, theta)
    n_nan = int(np.isnan(theta).sum())
    fin = theta[np.isfinite(theta)]
    print(f"\nsaved {out}  (N={N:,}, NaN={n_nan:,})")
    print(f"theta[deg]: min={fin.min():.1f} max={fin.max():.1f} "
          f"median={np.median(fin):.1f}")
    for code, gname in code2type.items():
        t = theta[ptype == code]; t = t[np.isfinite(t)]
        if len(t):
            print(f"  {gname:12s}: median={np.median(t):6.1f}  "
                  f"frac|theta-90|<15={np.mean(np.abs(t-90) < 15):.3f}  "
                  f"[{t.min():.0f},{t.max():.0f}]")


if __name__ == "__main__":
    main()
