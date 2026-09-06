"""Check the recomputed exp probs before anything is built on them.

Two jobs:

1. **Completeness** — every part of the source h5 is present, hit and event counts line up
   with the source, no NaNs, and the batch-size attribute says what we think it says.

2. **Quantify the change** — how far the 256 selection moves from the superseded 512 one,
   measured on the experimental data itself rather than on MC muons. The number that matters
   downstream is not how many hit probabilities shift, but how many *events* enter or leave
   the h8s3 selection, because that is what the excess is counted in.

Usage:
    python gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/validate_exp_probs.py
    ... --new <path> --old <path> --group exp_full --max-parts 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data_manager/data/h5datasets"
MODEL_TAG = "k_nsol_labelneq0_da_hs128_k0p0001"

THRESHOLD = 0.8
MIN_HITS, MIN_STRINGS = 8, 3   # the h8s3 event selection


def completeness(src_path: Path, new_path: Path, group: str) -> list[str]:
    """Part coverage, count agreement and attributes. Returns the part list."""
    with h5py.File(src_path, "r") as src, h5py.File(new_path, "r") as new:
        src_parts = sorted(k for k in src[f"{group}/raw/data"].keys() if k.startswith("part_"))
        new_parts = sorted(new[f"{group}/probs"].keys())

        bs = new.attrs.get("sig_noise_batch_size")
        print(f"  batch size attribute: {bs}" + ("" if bs is not None else "  <-- MISSING"))
        print(f"  parts: source {len(src_parts)}, probs {len(new_parts)}")

        missing = [p for p in src_parts if p not in new_parts]
        extra = [p for p in new_parts if p not in src_parts]
        if missing:
            print(f"  MISSING {len(missing)}: {missing[:5]}")
        if extra:
            print(f"  UNEXPECTED {len(extra)}: {extra[:5]}")

        bad = 0
        for pk in new_parts:
            n_src_hits = src[f"{group}/raw/data/{pk}/data"].shape[0]
            n_new_hits = new[f"{group}/probs/{pk}/data"].shape[0]
            n_src_ev = src[f"{group}/raw/ev_starts/{pk}/data"].shape[0] - 1
            n_new_ev = new[f"{group}/n_sig_hits_{THRESHOLD}/{pk}/data"].shape[0]
            if (n_src_hits, n_src_ev) != (n_new_hits, n_new_ev):
                bad += 1
                print(f"  COUNT MISMATCH {pk}: hits {n_src_hits} vs {n_new_hits}, "
                      f"events {n_src_ev} vs {n_new_ev}")
        print(f"  count agreement: {len(new_parts) - bad}/{len(new_parts)} parts")
    return new_parts


def compare(new_path: Path, old_path: Path, group: str, parts: list[str]) -> None:
    """How much did the hit selection and the h8s3 event selection actually move?"""
    hits_tot = hits_flip = ev_tot = ev_in_new = ev_in_old = ev_flip = 0
    max_delta = 0.0
    n_nan = 0

    with h5py.File(new_path, "r") as new, h5py.File(old_path, "r") as old:
        for i, pk in enumerate(parts, 1):
            pn = new[f"{group}/probs/{pk}/data"][:]
            po = old[f"{group}/probs/{pk}/data"][:]
            n_nan += int(np.isnan(pn).sum())
            max_delta = max(max_delta, float(np.abs(pn - po).max()))
            hits_tot += pn.size
            hits_flip += int(((pn > THRESHOLD) != (po > THRESHOLD)).sum())

            hn = new[f"{group}/n_sig_hits_{THRESHOLD}/{pk}/data"][:]
            sn = new[f"{group}/n_sig_strings_{THRESHOLD}/{pk}/data"][:]
            ho = old[f"{group}/n_sig_hits_{THRESHOLD}/{pk}/data"][:]
            so = old[f"{group}/n_sig_strings_{THRESHOLD}/{pk}/data"][:]
            in_new = (hn >= MIN_HITS) & (sn >= MIN_STRINGS)
            in_old = (ho >= MIN_HITS) & (so >= MIN_STRINGS)
            ev_tot += in_new.size
            ev_in_new += int(in_new.sum())
            ev_in_old += int(in_old.sum())
            ev_flip += int((in_new != in_old).sum())
            print(f"    [{i}/{len(parts)}] {pk}: h8s3 {int(in_old.sum()):,} -> "
                  f"{int(in_new.sum()):,}")

    print(f"\n  hits compared:        {hits_tot:,}")
    print(f"  max |delta prob|:     {max_delta:.4f}")
    print(f"  NaNs in new probs:    {n_nan:,}")
    print(f"  hits crossing {THRESHOLD}:   {hits_flip:,}  ({100 * hits_flip / hits_tot:.4f}%)")
    print(f"\n  events:               {ev_tot:,}")
    print(f"  passing h8s3 at 512:  {ev_in_old:,}  ({100 * ev_in_old / ev_tot:.4f}%)")
    print(f"  passing h8s3 at 256:  {ev_in_new:,}  ({100 * ev_in_new / ev_tot:.4f}%)")
    print(f"  changed membership:   {ev_flip:,}  ({100 * ev_flip / ev_tot:.4f}% of all events,"
          f" {100 * ev_flip / max(ev_in_old, 1):.2f}% of the old selection)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(D / "exp_full.h5"))
    ap.add_argument("--new", default=str(D / f"exp_full_probs_{MODEL_TAG}.h5"))
    ap.add_argument("--old", default=str(D / f"exp_full_probs_{MODEL_TAG}.DEPRECATED_bs512.h5"))
    ap.add_argument("--group", default="exp_full")
    ap.add_argument("--max-parts", type=int, default=0,
                    help="Compare only the first N parts (0 = all). Completeness always "
                         "covers everything.")
    args = ap.parse_args()

    print("== completeness ==")
    parts = completeness(Path(args.source), Path(args.new), args.group)

    if not Path(args.old).exists():
        print("\n  no superseded file to compare against — skipping the comparison")
        return

    sel = parts[: args.max_parts] if args.max_parts else parts
    print(f"\n== 256 vs 512, on {len(sel)} part(s) ==")
    compare(Path(args.new), Path(args.old), args.group, sel)


if __name__ == "__main__":
    main()
