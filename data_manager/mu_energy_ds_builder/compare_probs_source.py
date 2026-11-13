"""Compare a model-built dataset with the same dataset built from the probs HDF5.

The model path and the stored-probability path should select the same hits.  They need not
select the same *events*: a hit whose probability sits on the 0.8 threshold can flip when
the model is re-run on another GPU or in another batch, which changes ``n_sig_hits`` by one
and the h8s3 decision with it.  This script quantifies exactly that difference instead of
assuming it away.

It reports, for the two dataset directories:
    * event-key set difference, by particle class;
    * scalar fields per common event (target, status, n_tracks, sign, theta, sig counts);
    * per-event selected-hit counts and, where they agree, the feature rows and probabilities.

Usage:
    python -m data_manager.mu_energy_ds_builder.compare_probs_source \
        --reference data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_pilot \
        --candidate data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_pilot_probscheck
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCALARS = (
    "energy",
    "energy_status",
    "n_tracks",
    "sign",
    "event_weights",
    "theta",
    "n_sig_hits",
    "n_sig_strings",
    "event_fk",
)


def _load_keys(d: Path) -> list[tuple[str, str, int]]:
    """(particle class, part, local event id) — part numbers repeat across classes."""
    encode = json.load(open(d / "dataset_info.json"))["particle_encode"]
    code2name = {int(c): name for name, c in encode.items()}
    ptype = np.load(d / "particle_types.npy")
    parts = np.load(d / "h5_part_keys.npy", allow_pickle=True)
    events = np.load(d / "h5_local_event_ids.npy")
    return [
        (code2name[int(t)], str(p), int(e))
        for t, p, e in zip(ptype, parts, events)
    ]


def _index(keys: list[tuple[str, str, int]]) -> dict[tuple[str, str, int], int]:
    out: dict[tuple[str, str, int], int] = {}
    for i, k in enumerate(keys):
        if k in out:
            raise ValueError(f"duplicate event key {k}")
        out[k] = i
    return out




def compare(reference: str, candidate: str) -> dict:
    ref, cand = Path(reference), Path(candidate)
    ref_keys, cand_keys = _load_keys(ref), _load_keys(cand)
    ref_idx, cand_idx = _index(ref_keys), _index(cand_keys)
    ref_set, cand_set = set(ref_idx), set(cand_idx)

    only_ref = sorted(ref_set - cand_set)
    only_cand = sorted(cand_set - ref_set)
    common = sorted(ref_set & cand_set)

    def counts(keys):
        out: dict[str, int] = {}
        for k in keys:
            out[k[0]] = out.get(k[0], 0) + 1
        return out

    report: dict = {
        "reference": str(ref),
        "candidate": str(cand),
        "n_reference": len(ref_keys),
        "n_candidate": len(cand_keys),
        "n_common": len(common),
        "n_only_reference": len(only_ref),
        "n_only_candidate": len(only_cand),
        "only_reference": [list(k) for k in only_ref],
        "only_candidate": [list(k) for k in only_cand],
        "class_counts": {
            "reference": counts(ref_keys),
            "candidate": counts(cand_keys),
        },
    }

    # Scalars over common events.
    common_ref_rows = np.array([ref_idx[k] for k in common], dtype=np.int64)
    common_cand_rows = np.array([cand_idx[k] for k in common], dtype=np.int64)
    scalar_report = {}
    for name in SCALARS:
        a = np.load(ref / f"{name}.npy")[common_ref_rows]
        b = np.load(cand / f"{name}.npy")[common_cand_rows]
        if a.dtype.kind in "OUS":
            n_diff = int(np.sum(a != b))
            scalar_report[name] = {"n_different": n_diff}
        else:
            diff = a.astype(np.float64) - b.astype(np.float64)
            n_diff = int(np.sum(diff != 0))
            scalar_report[name] = {
                "n_different": n_diff,
                "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
            }
    report["scalars_common"] = scalar_report

    # Per-event hit slices.
    ref_off = np.load(ref / "offsets.npy")
    cand_off = np.load(cand / "offsets.npy")
    ref_feat = np.load(ref / "features.npy", mmap_mode="r")
    cand_feat = np.load(cand / "features.npy", mmap_mode="r")
    ref_prob = np.load(ref / "probs.npy", mmap_mode="r")
    cand_prob = np.load(cand / "probs.npy", mmap_mode="r")

    n_count_diff = 0
    n_features_diff = 0
    n_probs_diff = 0
    max_prob_abs = 0.0
    largest_count_diff = 0
    total_hits_ref = int(ref_off[-1])
    total_hits_cand = int(cand_off[-1])

    for k in common:
        i, j = ref_idx[k], cand_idx[k]
        s_a, e_a = int(ref_off[i]), int(ref_off[i + 1])
        s_b, e_b = int(cand_off[j]), int(cand_off[j + 1])
        n_a, n_b = e_a - s_a, e_b - s_b
        if n_a != n_b:
            n_count_diff += 1
            largest_count_diff = max(largest_count_diff, abs(n_a - n_b))
            continue
        if n_a == 0:
            continue
        fa = ref_feat[s_a:e_a]
        fb = cand_feat[s_b:e_b]
        if not np.array_equal(fa, fb):
            n_features_diff += 1
        pa = ref_prob[s_a:e_a].astype(np.float64)
        pb = cand_prob[s_b:e_b].astype(np.float64)
        d = float(np.max(np.abs(pa - pb))) if pa.size else 0.0
        if not np.array_equal(pa, pb):
            n_probs_diff += 1
            max_prob_abs = max(max_prob_abs, d)

    report["hits_common"] = {
        "total_hits_reference": total_hits_ref,
        "total_hits_candidate": total_hits_cand,
        "n_events_with_different_hit_count": n_count_diff,
        "largest_hit_count_difference": largest_count_diff,
        "n_events_with_different_features": n_features_diff,
        "n_events_with_different_probs": n_probs_diff,
        "max_abs_prob_difference": max_prob_abs,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = compare(args.reference, args.candidate)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
