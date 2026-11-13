#!/usr/bin/env python
"""Fingerprint the sig-noise probs HDF5 so an append run can be proven non-destructive.

Records, per particle type: the full sorted part list, and for a deterministic sample of
parts the shape and an md5 of every stored array. Run before and after appending and diff
the two JSON files: any change to a pre-existing part shows up immediately.

    python .../fingerprint_probs_h5.py --out before.json
    ... append run ...
    python .../fingerprint_probs_h5.py --out after.json
    python .../fingerprint_probs_h5.py --compare before.json after.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

DEFAULT = ("/home/albert/Baikal2025/data_manager/data/h5datasets/"
           "baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5")
CLASSES = ["muatm_2020", "nuatm_2020", "nue2_2020"]
GROUPS = ["probs", "ev_starts", "channels", "n_gt_sig_hits", "n_gt_sig_strings",
          "n_sig_hits_0.5", "n_sig_strings_0.5", "n_sig_hits_0.8", "n_sig_strings_0.8"]
N_SAMPLE = 12


def fingerprint(path: str) -> dict:
    out = {"path": path, "classes": {}}
    with h5py.File(path, "r") as f:
        for cls in CLASSES:
            if cls not in f:
                continue
            parts = sorted(f[f"{cls}/probs"].keys())
            # deterministic spread over the part list
            idx = np.linspace(0, len(parts) - 1, min(N_SAMPLE, len(parts))).astype(int)
            sampled = {}
            for i in np.unique(idx):
                pk = parts[i]
                entry = {}
                for g in GROUPS:
                    key = f"{cls}/{g}/{pk}/data"
                    if key not in f:
                        continue
                    a = f[key][:]
                    entry[g] = {"shape": list(a.shape), "dtype": str(a.dtype),
                                "md5": hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()}
                sampled[pk] = entry
            out["classes"][cls] = {"n_parts": len(parts), "parts": parts,
                                   "sampled": sampled}
            print(f"  {cls}: {len(parts):,} parts, fingerprinted {len(sampled)}", flush=True)
    return out


def compare(a_path: str, b_path: str) -> int:
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    bad = 0
    for cls in CLASSES:
        ca, cb = a["classes"].get(cls), b["classes"].get(cls)
        if ca is None:
            continue
        if cb is None:
            print(f"  {cls}: MISSING in the later file")
            bad += 1
            continue
        old, new = set(ca["parts"]), set(cb["parts"])
        lost = old - new
        added = new - old
        print(f"\n  {cls}: {ca['n_parts']:,} -> {cb['n_parts']:,} parts"
              f"  (+{len(added):,} added, -{len(lost):,} lost)")
        if lost:
            bad += 1
            print(f"    !! LOST PARTS: {sorted(lost)[:10]}")
        changed = []
        for pk, entry in ca["sampled"].items():
            other = cb["sampled"].get(pk)
            if other is None:
                changed.append((pk, "absent afterwards"))
                continue
            for g, v in entry.items():
                w = other.get(g)
                if w is None:
                    changed.append((pk, f"{g} absent"))
                elif w["md5"] != v["md5"]:
                    changed.append((pk, f"{g} md5 differs"))
        if changed:
            bad += 1
            print(f"    !! {len(changed)} pre-existing arrays CHANGED: {changed[:6]}")
        else:
            print(f"    all {len(ca['sampled'])} fingerprinted parts byte-identical")
    print("\n" + ("FAILED: pre-existing data was modified" if bad
                  else "PASS: nothing pre-existing was lost or altered"))
    return bad


def verify(before_path: str, h5_path: str) -> int:
    """Re-read from the live file exactly the parts recorded in the BEFORE fingerprint.

    This is the correct check after an append: `compare` samples parts by position, so a
    changed part count makes it sample different parts and report spurious differences.
    """
    a = json.load(open(before_path))
    bad = 0
    with h5py.File(h5_path, "r") as f:
        for cls, ca in a["classes"].items():
            now = set(f[f"{cls}/probs"].keys())
            lost = set(ca["parts"]) - now
            print(f"\n  {cls}: {ca['n_parts']:,} -> {len(now):,} parts "
                  f"(+{len(now) - ca['n_parts']:,} added, -{len(lost):,} lost)")
            if lost:
                bad += 1
                print(f"    !! LOST: {sorted(lost)[:10]}")
            n_ok = n_bad = 0
            for pk, entry in ca["sampled"].items():
                for g, v in entry.items():
                    key = f"{cls}/{g}/{pk}/data"
                    if key not in f:
                        print(f"    !! {key} disappeared")
                        n_bad += 1
                        continue
                    arr = f[key][:]
                    md5 = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
                    if md5 != v["md5"]:
                        print(f"    !! {key} CHANGED")
                        n_bad += 1
                    else:
                        n_ok += 1
            print(f"    re-read {len(ca['sampled'])} pre-existing parts: "
                  f"{n_ok} arrays identical, {n_bad} changed")
            bad += n_bad
    print("\n" + ("FAILED: pre-existing data was modified" if bad
                  else "PASS: every pre-existing part is present and byte-identical"))
    return bad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT)
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    p.add_argument("--verify", metavar="BEFORE",
                   help="re-read the BEFORE parts from --input and compare md5 (use this "
                        "after an append run)")
    a = p.parse_args()
    if a.verify:
        sys.exit(verify(a.verify, a.input))
    if a.compare:
        sys.exit(compare(*a.compare))
    fp = fingerprint(a.input)
    Path(a.out).write_text(json.dumps(fp, indent=1))
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
