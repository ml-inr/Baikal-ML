#!/usr/bin/env python
"""Extend a nu-classifier NPY dataset with per-event training weights.

Flattens two nuisance axes **within each group** (the dataset's class balance is
untouched).  The groups are the three particle types by default; with
``--group-by label`` they are muatm and the combined nuatm+nue2 mixture.

  * **energy** — uniform in ln E. E is the row's muon energy at the reference
    point, summed over the row's muons (the bundle energy is what drives the
    light); rows with no positive `e_ref` fall back to the largest valid
    `e_at_entry[:, 0]`, and rows with neither get weight 1 (counted in the
    report). Removes the "bright ⇒ neutrino" prior that comes from the ν samples
    being spectrally harder.
  * **zenith** — uniform in cos θ, θ taken from the dataset's own `theta.npy`
    (per-row ground truth, added by `add_theta.py`). Removes the density
    gradients near the horizon (μ are ~10x sparser just past the generation cut
    than ν just below it).

Both are flattened per group with equal-count quantile bins over the 0.05–99.95
percentile range (a raw histogram inversion explodes on the sparse tail bins and
its clipping removes the flattening entirely). Weights are normalised to mean 1
per class and capped at `--max-ratio` times the mean — clipped importance
weighting. The realized ESS and the weighted spread across bins are reported so
the survivors of the cap are visible.

Output: `weights.npy` (float32, aligned with `labels.npy`) and `weights_info.json`.
The dataset class does NOT pick this file up automatically — a config must name it
(`data.source_domain.weights_file`), so no other run changes behaviour by accident.

Usage:
    python -m data_manager.nu_classifier_ds_builder.add_weights \
        --npy-dir data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def row_energy(companion: h5py.File, gname: str, part: str,
               local_ids: np.ndarray) -> np.ndarray:
    """Per-row bundle energy: sum of positive e_ref, else best valid e_at_entry."""
    grp = companion[gname]
    mu_starts = grp["mu_starts"][part]["data"][:].astype(np.int64)
    e_ref = grp["e_ref"][part]["data"][:].astype(np.float32)
    e_entry = grp["e_at_entry"][part]["data"][:]          # (n_muons, 3)
    status = grp["e_at_entry_status"][part]["data"][:]

    out = np.full(len(local_ids), np.nan, dtype=np.float32)
    for i, row in enumerate(local_ids):
        ms, me = mu_starts[row], mu_starts[row + 1]
        e = e_ref[ms:me]
        alive = e[e > 0.0]
        if alive.size:
            out[i] = float(alive.sum())
            continue
        col = e_entry[ms:me, 0]
        ok = np.isin(status[ms:me, 0], (0, 1, 4, 5)) & np.isfinite(col)
        if ok.any():
            out[i] = float(col[ok].max())
    return out


def flatten_weights(values: np.ndarray, bins: int = 50,
                    lo_q: float = 0.0005, hi_q: float = 0.9995,
                    max_ratio: float = 20.0) -> tuple[np.ndarray, dict]:
    """Flatten a 1-D distribution to uniform with equal-count quantile bins.

    A raw histogram inversion is unusable here: over a decades-wide range some
    bins hold one or two events, their inverse density explodes, and clipping the
    explosion away leaves no flattening at all. Equal-count bins fix the density
    estimate (every bin is equally populated by construction), so the weight of a
    bin is just its width relative to the average -- bounded by construction --
    and the remaining tail is clipped at `max_ratio` times the mean weight, the
    standard clipped importance weighting. Both the realized ESS and the spread
    of the weighted distribution across bins are reported.
    """
    finite = np.isfinite(values)
    v = values[finite]
    if v.size == 0:
        return np.ones_like(values, dtype=np.float32), {"n": 0}
    lo, hi = np.quantile(v, [lo_q, hi_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones_like(values, dtype=np.float32), {"n": int(v.size), "flat": True}

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
    # Flatness = spread of the weighted DENSITY across bins (mass / width). The
    # mass alone is proportional to the bin width by construction with equal-count
    # bins, so it says nothing; the density is 1 when the target is reached.
    density = mass / widths
    flatness = float(density.max() / max(density.min(), 1e-12))
    return w.astype(np.float32), {
        "n": int(v.size), "range": [float(lo), float(hi)],
        "bins": int(len(edges) - 1), "cap": cap, "n_capped": n_capped,
        "ess_fraction": ess, "width_ratio": float(widths.max() / widths.min()),
        "weighted_density_spread": flatness,
    }


def apply_ratio_cap(w: np.ndarray, max_ratio: float) -> tuple[np.ndarray, dict]:
    """Cap weights at `max_ratio` times their mean (clipped importance weighting)."""
    w = w / w.mean()
    cap = float(max_ratio)
    n_capped = int((w > cap).sum())
    w = np.minimum(w, cap)
    w = w / w.mean()
    ess = float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))
    return w, {"cap_factor": cap, "n_capped": n_capped, "ess_fraction": ess}


def raking_weights(cos_v: np.ndarray, lg_v: np.ndarray,
                   n_cos: int = 40, n_lg: int = 60,
                   max_ratio: float = 20.0, lo_q: float = 0.0005,
                   hi_q: float = 0.9995, iters: int = 300,
                   tol: float = 1e-12) -> tuple[np.ndarray, dict]:
    """Iterative proportional fitting on the 2d (cos θ, lg E) table.

    Flattens BOTH marginals exactly on the 2d grid (target: equal mass per
    uniform bin = uniform density), which fixes the defects of the independent
    product-of-marginals method: the coarse widest quantile bin and the drift of
    the energy marginal caused by the cosθ-energy correlation. The joint
    dependence (copula) remains by construction; weights are still clipped at
    `max_ratio` times their mean.
    """
    fin = np.isfinite(cos_v) & np.isfinite(lg_v)
    out = np.ones(len(cos_v), dtype=np.float64)
    if fin.sum() == 0:
        return out, {"n": 0}
    c = np.clip(cos_v[fin], *np.quantile(cos_v[fin], [lo_q, hi_q]))
    e = np.clip(lg_v[fin], *np.quantile(lg_v[fin], [lo_q, hi_q]))
    edges_c = np.linspace(c.min(), c.max(), n_cos + 1)
    edges_e = np.linspace(e.min(), e.max(), n_lg + 1)
    N, _, _ = np.histogram2d(c, e, bins=[edges_c, edges_e])
    M = float(N.sum())

    tgt_r = np.full(n_cos, 1.0 / n_cos)   # равные доли массы на равные бины
    tgt_c = np.full(n_lg, 1.0 / n_lg)
    r, s = np.ones(n_cos), np.ones(n_lg)
    conv = False
    it = 0
    for it in range(1, iters + 1):
        W = N * np.outer(r, s)
        rm = W.sum(1)
        nz = rm > 0
        r[nz] *= (tgt_r[nz] * M) / rm[nz]
        W = N * np.outer(r, s)
        cm = W.sum(0)
        nz = cm > 0
        s[nz] *= (tgt_c[nz] * M) / cm[nz]
        W = N * np.outer(r, s)
        err = max(np.abs(W.sum(1) / M - tgt_r).max(),
                  np.abs(W.sum(0) / M - tgt_c).max())
        if err < tol:
            conv = True
            break

    cell_w = np.outer(r, s)
    ic = np.clip(np.searchsorted(edges_c, c, side="right") - 1, 0, n_cos - 1)
    ie = np.clip(np.searchsorted(edges_e, e, side="right") - 1, 0, n_lg - 1)
    w = cell_w[ic, ie]
    w = w / w.mean()
    n_capped = int((w > max_ratio).sum())
    w = np.minimum(w, max_ratio)
    w = w / w.mean()
    out[fin] = w

    def _spread_fine(vals, ww, bins=100):
        lo, hi = np.quantile(vals, [lo_q, hi_q])
        h, _ = np.histogram(np.clip(vals, lo, hi), bins=bins, range=(lo, hi),
                            weights=ww)
        ok = h > 0
        return float(h[ok].max() / h[ok].min())

    ess = float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))
    rep = {"n": int(fin.sum()), "grid": [n_cos, n_lg], "iters": int(it),
           "converged": bool(conv), "n_capped": n_capped,
           "ess_fraction": ess, "cap_factor": max_ratio,
           "energy_spread_fine": _spread_fine(e, w),
           "cos_spread_fine": _spread_fine(c, w),
           "final_range": [float(w.min()), float(w.max())]}
    return out.astype(np.float32), rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--companion", default="data_manager/data/h5datasets/baikal_mc_merged_energy_truth.h5")
    ap.add_argument("--theta", default="theta.npy")
    ap.add_argument("--out", default="weights.npy")
    ap.add_argument("--info", default="weights_info.json")
    ap.add_argument("--max-ratio", type=float, default=20.0,
                    help="cap on weight / mean weight (clipped importance weighting)")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="recompute the per-row energies instead of reading energy_cache.npy")
    ap.add_argument("--method", choices=["marginal", "raking"], default="marginal",
                    help="marginal = прежний метод (произведение двух 1d-выравниваний); "
                         "raking = IPF по 2d-таблице (cos θ, lg E)")
    ap.add_argument("--group-by", choices=["ptype", "label"], default="ptype",
                    help="ptype = выравнивать muatm/nuatm/nue2 по отдельности "
                         "(историческое поведение); label = отдельно muatm и "
                         "совмещённая смесь nuatm+nue2")
    ap.add_argument("--raking-cos-bins", type=int, default=40)
    ap.add_argument("--raking-lg-bins", type=int, default=60)
    args = ap.parse_args()

    d = Path(args.npy_dir)
    info = json.load(open(d / "dataset_info.json"))
    code2type = {v: k for k, v in info["particle_encode"].items()}

    ptype = np.load(d / "particle_types.npy")
    part_keys = np.asarray(np.load(d / "h5_part_keys.npy", allow_pickle=True), dtype=str)
    local_ids = np.asarray(np.load(d / "h5_local_event_ids.npy"), dtype=np.int64)
    theta = np.load(d / args.theta).astype(np.float32)
    n = len(ptype)
    assert len(part_keys) == n == len(local_ids) == len(theta), "back-link length mismatch"

    # Per-row energies are slow to recompute (a Python loop over ~5M rows of the
    # 163 GB companion); cache them next to the output so tuning the cap is cheap.
    cache = d / "energy_cache.npy"
    if args.refresh_cache or not cache.exists():
        energy = np.full(n, np.nan, dtype=np.float32)
        _fill_energy(energy, ptype, part_keys, local_ids, code2type, args.companion)
        np.save(cache, energy)
    else:
        energy = np.load(cache)
        if len(energy) != n:
            raise ValueError(f"stale energy cache: {len(energy)} != {n} rows "
                             f"({cache}); rerun with --refresh-cache")
        print(f"energy from cache {cache}")

    report = {"n_events": int(n), "per_class": {}, "max_ratio": args.max_ratio,
              "method": args.method, "group_by": args.group_by}
    weights = np.ones(n, dtype=np.float32)
    if args.group_by == "ptype":
        groups = [(gname, np.where(ptype == code)[0])
                  for code, gname in code2type.items()]
    else:
        mu_code = next(c for c, g in code2type.items() if g == "muatm_2020")
        nu_codes = [c for c, g in code2type.items() if g != "muatm_2020"]
        groups = [
            ("muatm",    np.where(ptype == mu_code)[0]),
            ("neutrino", np.where(np.isin(ptype, nu_codes))[0]),
        ]
    for gname, sel in groups:
        nan_e = int(np.isnan(energy[sel]).sum())
        if args.method == "raking":
            finite = np.isfinite(energy[sel]) & (energy[sel] > 0)
            lg = np.full(len(sel), np.nan, dtype=np.float64)
            lg[finite] = np.log10(energy[sel][finite].astype(np.float64))
            w, rep = raking_weights(np.cos(np.deg2rad(theta[sel])), lg,
                                    n_cos=args.raking_cos_bins,
                                    n_lg=args.raking_lg_bins,
                                    max_ratio=args.max_ratio)
            weights[sel] = w.astype(np.float32)
            rep["energy_nan"] = nan_e
            report["per_class"][gname] = rep
            print(f"{gname}: raking n={int(finite.sum()):,} grid={rep['grid']} "
                  f"iters={rep['iters']} conv={rep['converged']} "
                  f"w=[{w.min():.3g},{w.max():.3g}] ESS={100*rep['ess_fraction']:.0f}% "
                  f"spread(fine) E=x{rep['energy_spread_fine']:.1f} "
                  f"cos=x{rep['cos_spread_fine']:.1f}")
            continue
        w_e, rep_e = flatten_weights(energy[sel], max_ratio=args.max_ratio)
        w_t, rep_t = flatten_weights(np.cos(np.deg2rad(theta[sel])),
                                     max_ratio=args.max_ratio)
        w, rep_joint = apply_ratio_cap((w_e * w_t).astype(np.float64), args.max_ratio)
        weights[sel] = w.astype(np.float32)
        report["per_class"][gname] = {
            "n": int(len(sel)), "energy_nan": nan_e,
            "energy": rep_e, "cos_theta": rep_t, "joint": rep_joint,
            "final_range": [float(w.min()), float(w.max())],
            "ess_fraction": float(w.sum() ** 2 / (w.size * (w ** 2).sum())),
        }
        print(f"{gname}: n={len(sel):,} energy NaN={nan_e:,} "
              f"w=[{w.min():.3g},{w.max():.3g}] ESS={100*report['per_class'][gname]['ess_fraction']:.0f}%")

    np.save(d / args.out, weights)
    report["targets"] = {"energy": "uniform in ln E within group",
                         "zenith": "uniform in cos(theta) within group"}
    report["output"] = args.out
    json.dump(report, open(d / args.info, "w"), indent=2)
    print(f"saved {d / args.out} and {d / args.info}")


def _fill_energy(energy: np.ndarray, ptype: np.ndarray, part_keys: np.ndarray,
                 local_ids: np.ndarray, code2type: dict, companion_path: str) -> None:
    """Fill `energy` in place: per-row energy for every event of the dataset."""
    with h5py.File(companion_path, "r") as comp:
        for code, gname in code2type.items():
            sel = np.where(ptype == code)[0]
            if not len(sel):
                continue
            pk_sel = part_keys[sel]
            order = np.argsort(pk_sel, kind="stable")
            sel_sorted, pk_sorted = sel[order], pk_sel[order]
            uniq, starts = np.unique(pk_sorted, return_index=True)
            starts = list(starts) + [len(pk_sorted)]
            for j, pk in enumerate(uniq):
                block = sel_sorted[starts[j]:starts[j + 1]]
                if pk not in comp[gname]["e_ref"]:
                    raise KeyError(f"{pk} missing from {gname} in the companion")
                energy[block] = row_energy(comp, gname, pk, local_ids[block])
            print(f"{gname}: energy filled for {len(sel):,} rows")


if __name__ == "__main__":
    main()
