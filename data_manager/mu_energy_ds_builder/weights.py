"""Build per-event training weights for the muon-energy dataset.

The target is flattened to uniform in ``log10(E)`` **within each particle class**, and
the classes are optionally given equal total mass.  Both operations are importance
weights, not subsampling: every selected event stays in the dataset, and the weight file
is named explicitly by the training config (no automatic pickup).

The flattening routine is a deliberate copy of
``data_manager/nu_classifier_ds_builder/add_weights.py`` (copied 2026-10-21).  It uses
equal-count quantile bins over the 0.05–99.95 percentile range and clipped importance
weights, so the sparse tails do not produce an unbounded weight.

Outputs (in the dataset directory):
    weights.npy        float32, aligned with energy.npy / particle_types.npy
    weights_info.json  per-class counts, ESS, flatness, weight range

Usage:
    python -m data_manager.mu_energy_ds_builder.weights \
        --npy-dir data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def flatten_weights(values: np.ndarray, bins: int = 50,
                    lo_q: float = 0.0005, hi_q: float = 0.9995,
                    max_ratio: float = 20.0) -> tuple[np.ndarray, dict]:
    """Flatten a 1-D distribution to uniform with equal-count quantile bins.

    A raw histogram inversion is unusable over a decades-wide range: some bins hold one
    or two events, their inverse density explodes, and clipping the explosion away leaves
    no flattening.  Equal-count bins fix the density estimate (every bin is equally
    populated by construction), so a bin's weight is its width relative to the mean and
    is bounded by construction.  The remaining tail is clipped at ``max_ratio`` times the
    mean weight.  The realized ESS and the weighted density spread are reported.
    """
    finite = np.isfinite(values)
    v = values[finite]
    if v.size == 0:
        return np.ones_like(values, dtype=np.float32), {"n": 0}
    lo, hi = np.quantile(v, [lo_q, hi_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones_like(values, dtype=np.float32), {
            "n": int(v.size), "flat": True,
        }

    clipped = np.clip(v, lo, hi)
    edges = np.unique(np.quantile(clipped, np.linspace(0.0, 1.0, bins + 1)))
    idx = np.clip(np.searchsorted(edges, clipped, side="right") - 1,
                  0, len(edges) - 2)
    widths = np.diff(edges)
    w_bin = widths / widths.mean()

    w = np.ones(values.shape, dtype=np.float64)
    w[finite] = w_bin[idx]
    cap = float(max_ratio * w[finite].mean())
    n_capped = int((w[finite] > cap).sum())
    w = np.minimum(w, cap)
    w = w / w.mean()

    ww = w[finite]
    ess = float(ww.sum() ** 2 / (len(ww) * (ww ** 2).sum()))
    mass = np.array([ww[idx == b].sum() for b in range(len(edges) - 1)])
    density = mass / widths
    flatness = float(density.max() / max(density.min(), 1e-12))
    return w.astype(np.float32), {
        "n": int(v.size), "range": [float(lo), float(hi)],
        "bins": int(len(edges) - 1), "cap": cap, "n_capped": n_capped,
        "ess_fraction": ess, "width_ratio": float(widths.max() / widths.min()),
        "weighted_density_spread": flatness,
    }


def add_weights(
    npy_dir: str | Path,
    out_name: str = "weights.npy",
    info_name: str = "weights_info.json",
    max_ratio: float = 20.0,
    equal_class_mass: bool = True,
) -> None:
    d = Path(npy_dir)
    info = json.load(open(d / "dataset_info.json"))
    code2name = {int(v): k for k, v in info["particle_encode"].items()}

    energy = np.load(d / "energy.npy").astype(np.float64)
    ptype = np.load(d / "particle_types.npy")
    n = len(energy)
    if len(ptype) != n:
        raise ValueError(f"energy {n} != particle_types {len(ptype)}")

    weights = np.ones(n, dtype=np.float64)
    report = {
        "n_events": int(n),
        "target": "uniform in log10(E) within each class",
        "equal_class_mass": bool(equal_class_mass),
        "max_ratio": float(max_ratio),
        "per_class": {},
    }
    n_classes = len(code2name)

    for code, name in code2name.items():
        sel = np.where(ptype == code)[0]
        if len(sel) == 0:
            report["per_class"][name] = {"n": 0}
            continue
        w_e, rep_e = flatten_weights(
            np.log10(energy[sel]), max_ratio=max_ratio
        )
        class_weight = 1.0
        if equal_class_mass:
            class_weight = (n / n_classes) / len(sel)
        w = w_e.astype(np.float64) * class_weight
        weights[sel] = w
        ess = float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))
        report["per_class"][name] = {
            "n": int(len(sel)),
            "class_weight": float(class_weight),
            "weight_range": [float(w.min()), float(w.max())],
            "weight_median": float(np.median(w)),
            "ess_fraction": ess,
            "energy_flattening": rep_e,
        }

    weights = weights / weights.mean()
    ess_total = float(
        weights.sum() ** 2 / (len(weights) * (weights ** 2).sum())
    )
    report["final_range"] = [float(weights.min()), float(weights.max())]
    report["final_median"] = float(np.median(weights))
    report["ess_fraction_total"] = ess_total

    np.save(d / out_name, weights.astype(np.float32))
    with open(d / info_name, "w") as f:
        json.dump(report, f, indent=2)

    print(f"n={n:,}  ESS={100 * ess_total:.1f}%  "
          f"w=[{weights.min():.3g}, {weights.max():.3g}]  "
          f"median={np.median(weights):.3g}")
    for name, rep in report["per_class"].items():
        if "ess_fraction" in rep:
            print(f"  {name:12s} n={rep['n']:>9,}  "
                  f"ESS={100 * rep['ess_fraction']:5.1f}%  "
                  f"w=[{rep['weight_range'][0]:.3g}, "
                  f"{rep['weight_range'][1]:.3g}]  "
                  f"class_w={rep['class_weight']:.3g}")
    print(f"saved {d / out_name} and {d / info_name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--out", default="weights.npy")
    ap.add_argument("--info", default="weights_info.json")
    ap.add_argument("--max-ratio", type=float, default=20.0,
                    help="cap on weight / mean within a class")
    ap.add_argument("--no-equal-class-mass", action="store_true",
                    help="keep the natural class mixture instead of equal total mass")
    args = ap.parse_args()
    add_weights(
        args.npy_dir,
        out_name=args.out,
        info_name=args.info,
        max_ratio=args.max_ratio,
        equal_class_mass=not args.no_equal_class_mass,
    )


if __name__ == "__main__":
    main()
