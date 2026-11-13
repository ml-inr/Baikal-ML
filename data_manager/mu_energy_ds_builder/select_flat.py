"""Flat target selection, used both by the builder and as a standalone tool.

The training target should be uniform in ``log10(E)`` **over the whole training
population**.  Equally spaced ``log10(E)`` bins are populated with a common per-bin cap:
abundant bins are randomly subsampled with weight 1; sparse bins keep all events and get
``min(cap/n, max_ratio)``.  ``target_total`` solves for the cap and fills the remainder,
so the selection is exactly N events.

``flat_select_indices`` is the reusable core.  It takes the **candidate** arrays
(log10 energy and particle type) and returns positions into those arrays, the weights and
a selection report.  The builder calls it on the h8s3/single-track candidates and writes
only the selected train+val population; the test complement is not stored in the
training dataset.

``select_flat`` is the old standalone wrapper for an already-built full NPY dataset.
It is kept for existing datasets; new builds use the integrated selection in ``build.py``.

Outputs of the standalone wrapper (in the dataset directory):
    flat_indices.npy  int64, indices into the full dataset arrays
    flat_weights.npy  float32, aligned with flat_indices
    flat_info.json    per-bin counts, weights, ESS, cap, class composition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _solve_per_bin(bin_counts: np.ndarray, target_total: int) -> tuple[int, int]:
    """Largest integer cap p with sum(min(n, p)) <= target; return (p, remainder)."""
    total_all = int(bin_counts.sum())
    if target_total > total_all:
        raise ValueError(
            f"target_total {target_total:,} exceeds the candidate count {total_all:,}"
        )
    lo, hi = 0, int(bin_counts.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if int(np.minimum(bin_counts, mid).sum()) <= target_total:
            lo = mid
        else:
            hi = mid - 1
    p = lo
    remainder = int(target_total - np.minimum(bin_counts, p).sum())
    n_slack = int((bin_counts > p).sum())
    if remainder > n_slack:
        raise RuntimeError(
            f"remainder {remainder} exceeds {n_slack} bins above the cap"
        )
    return p, remainder


def flat_select_indices(
    log_e: np.ndarray,
    particle_types: np.ndarray,
    classes: tuple[str, ...],
    code_by_name: dict[str, int],
    bins: int = 25,
    per_bin: int | None = None,
    target_total: int | None = None,
    max_ratio: float = 20.0,
    seed: int = 42,
) -> dict:
    """Select a target-flat subset of the supplied candidate arrays.

    Returns a dict with:
        selected  positions into the input arrays
        weights   float64, aligned with ``selected``
        info      the per-bin report and summary used by the dataset provenance
    """
    if len(log_e) != len(particle_types):
        raise ValueError("log_e and particle_types differ in length")
    missing = [c for c in classes if c not in code_by_name]
    if missing:
        raise ValueError(f"unknown classes {missing}")
    codes = [code_by_name[c] for c in classes]
    class_mask = np.isin(particle_types, codes)
    candidates = np.where(class_mask)[0]
    if len(candidates) == 0:
        raise ValueError("no candidates in the requested classes")

    lo_e = float(log_e[candidates].min())
    hi_e = float(log_e[candidates].max())
    edges = np.linspace(lo_e, hi_e, bins + 1)
    rng = np.random.RandomState(seed)

    bin_masks: list[np.ndarray] = []
    bin_counts = np.empty(bins, dtype=np.int64)
    for k in range(bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (log_e[candidates] >= lo) & (
            (log_e[candidates] <= hi) if k == bins - 1
            else (log_e[candidates] < hi)
        )
        bin_masks.append(mask)
        bin_counts[k] = int(mask.sum())

    if target_total is not None:
        if per_bin is not None and int(per_bin) != 2000:
            raise ValueError("pass either per_bin or target_total, not both")
        cap, remainder = _solve_per_bin(bin_counts, int(target_total))
        slack_bins = np.where(bin_counts > cap)[0]
        extra_bins = set(
            rng.choice(slack_bins, size=remainder, replace=False).tolist()
        ) if remainder else set()
    else:
        if per_bin is None:
            raise ValueError("one of per_bin / target_total is required")
        cap = int(per_bin)
        extra_bins = set()
        target_total = None

    selected_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    per_bin_report = []
    for k in range(bins):
        idx = candidates[bin_masks[k]]
        n = len(idx)
        if n == 0:
            per_bin_report.append({
                "bin": k, "lo": float(edges[k]), "hi": float(edges[k + 1]),
                "n": 0, "selected": 0, "weight": None, "capped": False,
                "extra": False,
            })
            continue
        sel_count = min(n, cap + (1 if k in extra_bins else 0))
        if n > sel_count:
            sel = rng.choice(idx, size=sel_count, replace=False)
            w = np.ones(sel_count, dtype=np.float64)
            capped = False
        else:
            sel = idx
            w_value = min(cap / n, max_ratio) if n > 0 else 1.0
            w = np.full(n, w_value, dtype=np.float64)
            capped = (cap / n) > max_ratio
        selected_parts.append(sel)
        weight_parts.append(w)
        per_bin_report.append({
            "bin": k, "lo": float(edges[k]), "hi": float(edges[k + 1]),
            "n": int(n), "selected": int(len(sel)),
            "weight": float(w[0]), "capped": bool(capped),
            "extra": bool(k in extra_bins),
        })

    selected = np.concatenate(selected_parts)
    weights = np.concatenate(weight_parts).astype(np.float64)
    ess = float(weights.sum() ** 2 / (len(weights) * (weights ** 2).sum()))
    info = {
        "classes": list(classes),
        "n_candidates": int(len(candidates)),
        "n_selected": int(len(selected)),
        "target_total": int(target_total) if target_total is not None else None,
        "log10_energy_range": [lo_e, hi_e],
        "n_bins": int(bins),
        "per_bin": int(cap),
        "max_ratio": float(max_ratio),
        "seed": int(seed),
        "weight_range": [float(weights.min()), float(weights.max())],
        "weight_median": float(np.median(weights)),
        "weight_mean": float(weights.mean()),
        "fraction_weight_one": float(np.mean(np.isclose(weights, 1.0))),
        "ess_fraction": ess,
        "class_counts_selected": {
            c: int((particle_types[selected] == code_by_name[c]).sum())
            for c in classes
        },
        "per_bin_report": per_bin_report,
        "note": (
            "The bulk is subsampled to the per-bin cap with weight 1; bins below the cap "
            "keep all events and get min(cap/n, max_ratio). The cap leaves the extreme "
            "tails underfilled on purpose."
        ),
    }
    return {"selected": selected, "weights": weights, "info": info}


def select_flat(
    npy_dir: str | Path,
    classes: tuple[str, ...] = ("nuatm_2020", "nue2_2020"),
    bins: int = 25,
    per_bin: int | None = 2000,
    target_total: int | None = None,
    max_ratio: float = 20.0,
    seed: int = 42,
    prefix: str = "flat",
) -> dict:
    """Standalone wrapper: select from an already-built full NPY dataset."""
    d = Path(npy_dir)
    info = json.load(open(d / "dataset_info.json"))
    name2code = {k: int(v) for k, v in info["particle_encode"].items()}

    energy = np.load(d / "energy.npy").astype(np.float64)
    ptype = np.load(d / "particle_types.npy")
    log_e = np.log10(energy)
    result = flat_select_indices(
        log_e=log_e,
        particle_types=ptype,
        classes=classes,
        code_by_name=name2code,
        bins=bins,
        per_bin=per_bin,
        target_total=target_total,
        max_ratio=max_ratio,
        seed=seed,
    )
    selected = result["selected"]
    weights = result["weights"].astype(np.float32)
    order = np.argsort(selected, kind="stable")
    selected = selected[order]
    weights = weights[order]
    info_out = dict(result["info"])
    info_out["npy_dir"] = str(d)
    np.save(d / f"{prefix}_indices.npy", selected.astype(np.int64))
    np.save(d / f"{prefix}_weights.npy", weights)
    with open(d / f"{prefix}_info.json", "w") as f:
        json.dump(info_out, f, indent=2)
    print(
        f"candidates {result['info']['n_candidates']:,} "
        f"-> selected {len(selected):,}  per_bin={result['info']['per_bin']:,}  "
        f"ESS {100*result['info']['ess_fraction']:.1f}%  "
        f"w=[{weights.min():.3g}, {weights.max():.3g}]  "
        f"w=1: {100*result['info']['fraction_weight_one']:.1f}%"
    )
    print(f"saved {d / f'{prefix}_indices.npy'}, "
          f"{d / f'{prefix}_weights.npy'}, {d / f'{prefix}_info.json'}")
    return info_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--classes", default="nuatm_2020,nue2_2020",
                    help="comma-separated class names")
    ap.add_argument("--bins", type=int, default=25)
    ap.add_argument("--per-bin", type=int, default=None)
    ap.add_argument("--target-total", type=int, default=None)
    ap.add_argument("--max-ratio", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="flat")
    args = ap.parse_args()
    select_flat(
        args.npy_dir,
        classes=tuple(c.strip() for c in args.classes.split(",") if c.strip()),
        bins=args.bins,
        per_bin=args.per_bin,
        target_total=args.target_total,
        max_ratio=args.max_ratio,
        seed=args.seed,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
