#!/usr/bin/env python3
"""Check that targets_vec reproduces targets.py exactly, on real data.

The vectorised path replaces a readable per-muon loop with index arithmetic, which is
precisely the kind of rewrite that silently shifts a boundary by one. It is therefore not
trusted on inspection: every muon of a real part is computed both ways and compared, and the
death point — the only place where a sequential walk was replaced by a prefix count — is
compared exactly.

Usage:
    python3 test_targets_vec.py [--part 3070] [--n-muons 20000]
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets import (R_LIMITS_M, cylinder_segment, death_point, deposited,  # noqa: E402
                     direction_from_angles, energy_at, target_noise_floor,
                     CLUSTER_RADIUS_M, CLUSTER_Z_HALF_M)
from targets_vec import compute_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MCH5 = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"
NPZ = ROOT / "data_manager/data/mc_energy_truth_interactions/nue2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="3070")
    ap.add_argument("--n-muons", type=int, default=20000)
    args = ap.parse_args()

    d = np.load(NPZ / f"{args.part}.npz")
    with h5py.File(MCH5, "r") as f:
        g = f["nue2_2020"]
        part = f"part_{args.part}"
        ev_ids = g[f"ev_ids/{part}/data"][:]
        root_idx = np.array([int(x.decode().rsplit("_", 1)[1]) for x in ev_ids])
        ev_starts = g[f"raw/ev_starts/{part}/data"][:].astype(np.int64)
        channels = g[f"raw/channels/{part}/data"][:]
        centres = g["clusters_centers/data"][:]

    n_rows = min(args.n_muons, len(root_idx))
    trk = root_idx[:n_rows]                       # nue2: exactly one muon per row
    cluster = np.array([np.bincount(channels[int(ev_starts[i]):int(ev_starts[i+1])] // 288).argmax()
                        for i in range(n_rows)])
    centres_of_muon = centres[cluster].astype(np.float64)

    n_inter = d["trk_n_inter"][trk]
    off_all = np.concatenate([[0], np.cumsum(d["trk_n_inter"])]).astype(np.int64)
    sel = np.concatenate([np.arange(off_all[t], off_all[t + 1]) for t in trk]) if len(trk) else np.array([], int)
    # Both paths get the same float64 inputs. The stored chain is float32, and the two
    # implementations round differently on it (the vectorised one accumulates in double, the
    # reference sums float32 slices), which shows up as a ~1e-6 haze that hides nothing but
    # also proves nothing. Promoting first makes the comparison a test of the index
    # arithmetic, which is what the rewrite actually risks getting wrong.
    int_xyz = np.stack([d["int_x"][sel], d["int_y"][sel], d["int_z"][sel]], axis=1).astype(np.float64)
    int_energy = d["int_energy"][sel].astype(np.float64)
    ref_xyz = np.stack([d["trk_x"][trk], d["trk_y"][trk], d["trk_z"][trk]], axis=1).astype(np.float64)

    theta = d["trk_theta"][trk].astype(np.float64)
    phi = d["trk_phi"][trk].astype(np.float64)

    vec = compute_targets(theta, phi, ref_xyz,
                          d["trk_energy"][trk].astype(np.float64), n_inter,
                          int_xyz, int_energy, centres_of_muon)

    # reference: the per-muon implementation
    dirs = direction_from_angles(theta, phi)
    off = np.concatenate([[0], np.cumsum(n_inter)]).astype(np.int64)
    ref = {k: np.zeros(n_rows) for k in ["E_ca", "s_ca", "b_impact", "s_first_int",
                                         "s_last_int", "s_death", "sigma_target_min"]}
    ref_dep = np.full((n_rows, len(R_LIMITS_M)), np.nan)
    ref_len = np.zeros((n_rows, len(R_LIMITS_M)))
    ref_cnt = np.zeros((n_rows, len(R_LIMITS_M)), dtype=np.int32)
    for i in range(n_rows):
        a, b = off[i], off[i + 1]
        s_int = (int_xyz[a:b] - ref_xyz[i]) @ dirs[i] if b > a else np.zeros(0)
        e_int = int_energy[a:b]
        e_ref = float(d["trk_energy"][trk[i]])
        delta = centres_of_muon[i] - ref_xyz[i]
        s_ca = float(delta @ dirs[i])
        perp = delta - s_ca * dirs[i]
        s_first = float(s_int.min()) if len(s_int) else 0.0
        s_last = float(s_int.max()) if len(s_int) else 0.0
        s_death = death_point(e_ref, s_int, e_int)
        ref["E_ca"][i] = energy_at(e_ref, s_int, e_int, s_ca)
        ref["s_ca"][i] = s_ca
        ref["b_impact"][i] = float(np.sqrt(perp @ perp))
        ref["s_first_int"][i] = s_first
        ref["s_last_int"][i] = s_last
        ref["s_death"][i] = s_death
        ref["sigma_target_min"][i] = target_noise_floor(s_ca, s_first, s_death)
        for k, r_lim in enumerate(R_LIMITS_M):
            seg = cylinder_segment(ref_xyz[i], dirs[i], centres_of_muon[i],
                                   CLUSTER_RADIUS_M + r_lim, CLUSTER_Z_HALF_M + r_lim)
            if seg is not None:
                ed, lp, ni = deposited(s_int, e_int, seg[0], seg[1])
                ref_dep[i, k], ref_len[i, k], ref_cnt[i, k] = ed, lp, ni

    print(f"part_{args.part}: {n_rows:,} muons compared\n")
    worst = 0.0
    for key in ref:
        a, b = ref[key], vec[key]
        scale = np.maximum(1.0, np.abs(a))
        rel = np.abs(a - b) / scale
        worst = max(worst, rel.max())
        print(f"  {key:18}: max rel diff {rel.max():.3e}  ({int((rel > 1e-6).sum())} muons > 1e-6)")
    for k, r_lim in enumerate(R_LIMITS_M):
        for name, a, b in [(f"E_dep_{int(r_lim)}", ref_dep[:, k], vec["E_dep"][:, k]),
                           (f"L_path_{int(r_lim)}", ref_len[:, k], vec["L_path"][:, k]),
                           (f"n_int_in_{int(r_lim)}", ref_cnt[:, k], vec["n_int_in"][:, k])]:
            same_nan = np.isnan(a) == np.isnan(b)
            fin = ~np.isnan(a) & ~np.isnan(b)
            rel = np.abs(a[fin] - b[fin]) / np.maximum(1.0, np.abs(a[fin]))
            worst = max(worst, rel.max() if len(rel) else 0.0)
            flag = "OK" if same_nan.all() and (len(rel) == 0 or rel.max() < 1e-6) else "MISMATCH"
            print(f"  {name:18}: max rel diff {rel.max() if len(rel) else 0:.3e}  nan-pattern {flag}")
    print(f"\n{'IDENTICAL (within float tolerance)' if worst < 1e-6 else f'DIFFERS: worst {worst:.2e}'}")
    sys.exit(0 if worst < 1e-6 else 1)


if __name__ == "__main__":
    main()
