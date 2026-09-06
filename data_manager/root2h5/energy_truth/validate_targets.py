"""SUPERSEDED (2026-08-21) by data_manager/energy_truth/.

This script targets the old companion schema (targets/, interactions/, muon/), which no
longer exists: baikal_mc_merged_energy_truth.h5 was rebuilt with the schema documented in
doc/hdf5_energy_truth.md. Running it against the current file will not work.
"""
#!/usr/bin/env python3
"""Physics validation of targets.py on real nue2 data, before anything is written to HDF5.

Four checks, each able to fail for a different reason:
  1. Join integrity -- npz truth must reproduce muons_prty row by row (positions and energy
     exactly, angles to float32 round-trip), since the join is positional and therefore has
     to be verified rather than assumed.
  2. Dead-at-reference tracks -- ~20% of nue2 tracks have a NEGATIVE stored energy because
     the reference point lies beyond the muon's range. Propagating backwards to the cluster
     must turn those into physical positive energies; if it does not, the sign convention or
     the chain is wrong.
  3. Energy ordering -- the muon energy at its production point cannot exceed the primary
     neutrino energy (prime_prty[:, 2]). This is an independent physical bound that no part
     of the reconstruction can see.
  4. Segment geometry -- L_path must grow with the sensitive radius, and tracks that miss
     the volume must be reported as missing rather than silently valued.

Usage:
    python3 validate_targets.py [--part part_3070] [--n-rows 4000]
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets import (R_LIMITS_M, closest_approach, direction_from_angles,  # noqa: E402
                     energy_at, targets_for_track)

ROOT = Path(__file__).resolve().parents[3]
MCH5 = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"
NPZ_DIR = ROOT / "data_manager/data/mc_energy_truth_interactions/nue2"
PTYPE = "nue2_2020"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="part_3070")
    ap.add_argument("--n-rows", type=int, default=4000)
    args = ap.parse_args()

    file_id = args.part.split("_")[-1]
    d = np.load(NPZ_DIR / f"{file_id}.npz")
    f = h5py.File(MCH5, "r")
    g = f[PTYPE]

    ev_ids = g[f"ev_ids/{args.part}/data"][:]
    root_idx = np.array([int(x.decode().rsplit("_", 1)[1]) for x in ev_ids])
    mu_starts = g[f"muons_prty/mu_starts/{args.part}/data"][:]
    individ = g[f"muons_prty/individ/{args.part}/data"][:]
    ev_starts = g[f"raw/ev_starts/{args.part}/data"][:]
    channels = g[f"raw/channels/{args.part}/data"]
    centres = g["clusters_centers/data"][:]
    prime = g[f"prime_prty/{args.part}/data"][:]

    n = min(args.n_rows, len(root_idx))
    print(f"{args.part}: {len(root_idx):,} rows, checking {n:,}")

    # ---- 1. join integrity ----
    k = mu_starts[:n]
    ok = True
    for name, h5v, npv in [
        ("x", individ[k, 2], d["trk_x"][root_idx[:n]]),
        ("y", individ[k, 3], d["trk_y"][root_idx[:n]]),
        ("z", individ[k, 4], d["trk_z"][root_idx[:n]]),
        ("E", individ[k, 6], d["trk_energy"][root_idx[:n]]),
    ]:
        bad = int((np.abs(h5v - npv) > 0).sum())
        ok &= bad == 0
        print(f"  join {name}: {bad} mismatches")
    ang = np.abs(np.degrees(individ[k, 0]) - d["trk_theta"][root_idx[:n]]).max()
    print(f"  join theta: max |diff| = {ang:.2e} deg")
    print(f"  -> join {'OK' if ok and ang < 1e-3 else 'FAILED'}")

    # ---- targets ----
    int_off = np.concatenate([[0], np.cumsum(d["trk_n_inter"])]).astype(np.int64)
    dirs = direction_from_angles(d["trk_theta"], d["trk_phi"])
    refs = np.stack([d["trk_x"], d["trk_y"], d["trk_z"]], axis=1)
    pts = np.stack([d["int_x"], d["int_y"], d["int_z"]], axis=1)

    res = []
    for row in range(n):
        r = root_idx[row]
        s0, s1 = int(ev_starts[row]), int(ev_starts[row + 1])
        cluster = int(np.bincount(channels[s0:s1] // 288).argmax())
        s_int = (pts[int_off[r]:int_off[r + 1]] - refs[r]) @ dirs[r]
        t = targets_for_track(refs[r], dirs[r], float(d["trk_energy"][r]),
                              s_int, d["int_energy"][int_off[r]:int_off[r + 1]],
                              centres[cluster])
        t["E_ref"] = float(d["trk_energy"][r])
        t["E_nu"] = float(prime[row, 2])
        res.append(t)

    E_ca = np.array([t["E_ca"] for t in res])
    E_ref = np.array([t["E_ref"] for t in res])
    E_nu = np.array([t["E_nu"] for t in res])
    s_ca = np.array([t["s_ca"] for t in res])
    b_imp = np.array([t["b_impact"] for t in res])
    L = np.stack([t["L_path"] for t in res])

    # ---- 2. tracks dead at the reference point ----
    dead = E_ref < 0
    print(f"\n  tracks with E_ref < 0: {dead.sum():,} ({100*dead.mean():.1f}%)")
    if dead.any():
        rec = E_ca[dead]
        print(f"    after propagation to the cluster: {100*(rec > 0).mean():.1f}% positive, "
              f"median {np.median(rec):.1f} GeV")

    # ---- 3. physical bound against the primary neutrino ----
    finite = np.isfinite(E_ca) & (E_nu > 0)
    viol = (E_ca > E_nu) & finite
    print(f"\n  E_ca > E_nu (must not happen): {viol.sum():,} of {finite.sum():,} "
          f"({100*viol.mean():.2f}%)")
    if viol.any():
        worst = (E_ca[viol] / E_nu[viol]).max()
        print(f"    worst ratio E_ca/E_nu = {worst:.2f}")

    # ---- 4. segment geometry ----
    print(f"\n  impact parameter: median {np.median(b_imp):.1f} m, "
          f"p90 {np.percentile(b_imp, 90):.1f} m")
    print(f"  |s_ca| (propagation distance): median {np.median(np.abs(s_ca)):.1f} m, "
          f"p90 {np.percentile(np.abs(s_ca), 90):.1f} m")
    print("  L_path by sensitive radius:")
    for i, r_lim in enumerate(R_LIMITS_M):
        miss = int((L[:, i] == 0).sum())
        print(f"    R_lim={r_lim:5.0f} m: median {np.median(L[:, i]):6.1f} m, "
              f"{miss:,} tracks miss the volume ({100*miss/n:.1f}%)")
    grows = bool(np.all(np.diff(np.median(L, axis=0)) > 0))
    print(f"  L_path grows with radius: {grows}")
    f.close()


if __name__ == "__main__":
    main()
