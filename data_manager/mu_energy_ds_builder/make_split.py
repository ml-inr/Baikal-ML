"""Make the train / val / test split for muon-energy training.

The flat selection (``select_flat.py --target-total 1000000``) defines train+val; its
complement is the test set, as requested.  Train and val are separated **by MC part**: all
events of a part go to the same side, so the same file never appears in both.  The test
complement is not part-disjoint from train/val by construction; ``split_info.json`` records
how many test events come from a part seen in training so the caveat is explicit rather
than hidden.

Outputs, all indices into the full dataset arrays:
    split_train_indices.npy   int64
    split_train_weights.npy   float32, aligned with the train indices
    split_val_indices.npy     int64
    split_val_weights.npy     float32
    split_test_indices.npy    int64
    split_info.json

Usage:
    python -m data_manager.mu_energy_ds_builder.make_split \
        --npy-dir data_manager/datasets/mu_energy_dataset_h8s3_thr0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_split(
    npy_dir: str | Path,
    classes: tuple[str, ...] = ("nuatm_2020", "nue2_2020"),
    val_fraction: float = 0.10,
    seed: int = 42,
) -> dict:
    d = Path(npy_dir)
    info = json.load(open(d / "dataset_info.json"))
    name2code = {k: int(v) for k, v in info["particle_encode"].items()}
    code2name = {v: k for k, v in name2code.items()}

    ptype = np.load(d / "particle_types.npy")
    parts = np.load(d / "h5_part_keys.npy", allow_pickle=True)
    energy = np.load(d / "energy.npy").astype(np.float64)

    flat = np.load(d / "flat_indices.npy")
    flat_w = np.load(d / "flat_weights.npy")
    if len(flat) != len(flat_w):
        raise ValueError("flat_indices and flat_weights differ in length")

    class_mask = np.isin(ptype, [name2code[c] for c in classes])
    candidates = np.where(class_mask)[0]
    flat_set = set(int(i) for i in flat)
    if not flat_set.issubset(set(int(i) for i in candidates)):
        raise ValueError("flat selection contains non-candidate rows")
    test = np.array(
        sorted(set(int(i) for i in candidates) - flat_set), dtype=np.int64
    )

    # Group train+val by (class, part); order groups by first flat position for determinism.
    group_key = [
        (code2name[int(ptype[i])], str(parts[i])) for i in flat
    ]
    unique_groups = sorted(set(group_key))
    group_positions: dict[tuple[str, str], list[int]] = {
        g: [] for g in unique_groups
    }
    for pos, g in enumerate(group_key):
        group_positions[g].append(pos)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_groups))
    target_val = int(round(val_fraction * len(flat)))
    val_groups: list[tuple[str, str]] = []
    n_val = 0
    for gi in perm:
        g = unique_groups[gi]
        n_g = len(group_positions[g])
        if n_val + n_g > target_val and val_groups:
            continue
        val_groups.append(g)
        n_val += n_g
        if n_val >= target_val:
            break
    val_group_set = set(val_groups)

    val_pos = [p for p, g in enumerate(group_key) if g in val_group_set]
    train_pos = [p for p, g in enumerate(group_key) if g not in val_group_set]
    # Ascending row order, not random order: the loader shuffles anyway.
    val_pos = sorted(val_pos, key=lambda p: flat[p])
    train_pos = sorted(train_pos, key=lambda p: flat[p])
    train = flat[np.array(train_pos, dtype=np.int64)]
    val = flat[np.array(val_pos, dtype=np.int64)]
    train_w = flat_w[np.array(train_pos, dtype=np.int64)]
    val_w = flat_w[np.array(val_pos, dtype=np.int64)]

    train_pos_set = set(train_pos)
    val_pos_set = set(val_pos)
    train_groups = {g for p, g in enumerate(group_key) if p in train_pos_set}
    val_groups_used = {g for p, g in enumerate(group_key) if p in val_pos_set}
    seen_groups = train_groups | val_groups_used
    test_groups = {(code2name[int(ptype[i])], str(parts[i])) for i in test}
    n_test_seen = int(sum(1 for i in test
                          if (code2name[int(ptype[i])], str(parts[i])) in seen_groups))

    def class_counts(idx: np.ndarray) -> dict[str, int]:
        return {
            c: int((ptype[idx] == name2code[c]).sum()) for c in classes
        }

    def energy_summary(idx: np.ndarray) -> dict[str, float]:
        e = energy[idx]
        return {
            "min": float(e.min()) if len(e) else float("nan"),
            "median": float(np.median(e)) if len(e) else float("nan"),
            "max": float(e.max()) if len(e) else float("nan"),
        }

    info_out = {
        "npy_dir": str(d),
        "classes": list(classes),
        "val_fraction_requested": float(val_fraction),
        "val_actual_fraction": float(len(val) / len(flat)),
        "seed": int(seed),
        "n_candidates": int(len(candidates)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_train_val": int(len(train) + len(val)),
        "n_parts_total": int(len(unique_groups)),
        "n_parts_train": int(len(train_groups)),
        "n_parts_val": int(len(val_groups_used)),
        "class_counts": {
            "train": class_counts(train),
            "val": class_counts(val),
            "test": class_counts(test),
        },
        "energy_gev": {
            "train": energy_summary(train),
            "val": energy_summary(val),
            "test": energy_summary(test),
        },
        "weight_sum": {
            "train": float(train_w.sum()),
            "val": float(val_w.sum()),
        },
        "test_part_overlap": {
            "n_test_from_seen_parts": n_test_seen,
            "fraction": float(n_test_seen / len(test)) if len(test) else 0.0,
            "n_test_parts": int(len(test_groups)),
            "note": (
                "Test is the complement of the flat selection, as requested; it is not "
                "part-disjoint from train/val.  Use these counts when judging leakage."
            ),
        },
    }

    np.save(d / "split_train_indices.npy", train.astype(np.int64))
    np.save(d / "split_train_weights.npy", train_w.astype(np.float32))
    np.save(d / "split_val_indices.npy", val.astype(np.int64))
    np.save(d / "split_val_weights.npy", val_w.astype(np.float32))
    np.save(d / "split_test_indices.npy", test.astype(np.int64))
    with open(d / "split_info.json", "w") as f:
        json.dump(info_out, f, indent=2)

    print(
        f"train {len(train):,}  val {len(val):,}  test {len(test):,}  "
        f"(train+val {len(train)+len(val):,}; val parts {len(val_groups_used)}/"
        f"{len(unique_groups)}; test from seen parts {100*n_test_seen/len(test):.1f}%)"
    )
    print(f"saved split_*.npy and split_info.json in {d}")
    return info_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--classes", default="nuatm_2020,nue2_2020")
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    make_split(
        args.npy_dir,
        classes=tuple(c.strip() for c in args.classes.split(",") if c.strip()),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
