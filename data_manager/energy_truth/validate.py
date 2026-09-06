#!/usr/bin/env python3
"""Validate a finished energy-truth companion against the main HDF5.

Stage 5 (doc/energy_twin_plan.md). The builder checks each part as it writes it; this checks
the finished file as a whole -- totals, the alignment on parts chosen at random rather than
the first one, and the distributions that would reveal a systematic mistake the per-part
assertions cannot see (a status code that never fires, a sentinel population of the wrong size).

Usage:
    python3 validate.py [--production nuatm_2020] [--sample 20]
"""
from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np

import truth

MAIN_H5 = "/home/albert/Baikal2025/data_manager/data/h5datasets/baikal_mc_merged.h5"
TWIN_H5 = "/home/albert/Baikal2025/data_manager/data/h5datasets/baikal_mc_merged_energy_truth.h5"

STATUS_NAMES = ("propagated", "born inside", "died before", "misses volume",
                "sentinel", "sentinel inside")
VOLUME_LABELS = ("r=60", "r=90", "r=120")

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def validate(production: str, main_path: str, twin_path: str, n_sample: int) -> None:
    main = h5py.File(main_path, "r")
    twin = h5py.File(twin_path, "r")
    if production not in twin:
        print(f"{production}: not present in the companion — skipped")
        return

    main_parts = sorted(main[f"{production}/ev_ids"].keys())
    twin_parts = sorted(twin[f"{production}/ev_ids"].keys())
    print(f"\n=== {production}")
    check("every part of the main file is present",
          main_parts == twin_parts,
          f"{len(twin_parts)} of {len(main_parts)}")
    if not twin_parts:
        return

    # Totals over the whole production: cheap, since only the index arrays are read.
    rows = muons = showers = 0
    for p in twin_parts:
        ms = twin[f"{production}/mu_starts/{p}/data"]
        rows += ms.shape[0] - 1
        muons += int(ms[-1])
        showers += int(twin[f"{production}/shower_starts/{p}/data"][-1])
    rows_main = sum(main[f"{production}/ev_ids/{p}/data"].shape[0] for p in main_parts)
    check("row count matches the main file", rows == rows_main,
          f"{rows:,} rows, {muons:,} muons, {showers:,} showers")

    rng = np.random.default_rng(0)
    sample = rng.choice(len(twin_parts), size=min(n_sample, len(twin_parts)), replace=False)
    status_hist = np.zeros((len(VOLUME_LABELS), len(STATUS_NAMES)), dtype=np.int64)
    n_mu_seen = 0
    worst_perp = 0.0
    bad_upstream = bad_time = bad_ids = bad_starts = 0
    n_sentinel = 0
    inside_frac = []

    for i in sample:
        p = twin_parts[i]
        ev_t = twin[f"{production}/ev_ids/{p}/data"][:]
        ev_m = main[f"{production}/ev_ids/{p}/data"][:]
        bad_ids += int(not np.array_equal(ev_t, ev_m))
        ms_t = twin[f"{production}/mu_starts/{p}/data"][:]
        ms_m = main[f"{production}/muons_prty/mu_starts/{p}/data"][:]
        bad_starts += int(not np.array_equal(ms_t, ms_m))

        ind = main[f"{production}/muons_prty/individ/{p}/data"][:]
        ref = twin[f"{production}/ref_xyz/{p}/data"][:].astype(np.float64)
        ang = twin[f"{production}/direction/{p}/data"][:]
        d = truth.direction_from_angles(ang[:, 0], ang[:, 1])
        worst_perp = max(worst_perp, float(np.abs(np.einsum("ij,ij->i", ref, d)).max()))

        e_ref = twin[f"{production}/e_ref/{p}/data"][:]
        s_start = twin[f"{production}/s_track_start/{p}/data"][:]
        sent = e_ref == np.float32(truth.SENTINEL_E_REF)
        n_sentinel += int(sent.sum())
        if sent.any() and not (s_start[sent] > 0).all():
            _failures.append(f"{p}: a sentinel muon starts upstream")
        # Zero belongs to the non-sentinel side: a muon whose trajectory begins exactly at
        # the reference point is alive there, which is what "not a sentinel" means. It occurs
        # (t_first_muon = 0.0 gives s_track_start = -0.0, and -0.0 < 0 is false in IEEE).
        if not (s_start[~sent] <= 0).all():
            _failures.append(f"{p}: a non-sentinel muon starts downstream")

        sh = twin[f"{production}/showers/{p}/data"][:]
        st = twin[f"{production}/shower_starts/{p}/data"][:]
        bad_starts += int(st[-1] != len(sh))
        owner = np.repeat(np.arange(len(s_start)), np.diff(st))
        s_sh = np.einsum("ij,ij->i", sh[:, 2:5].astype(np.float64) - ref[owner], d[owner])
        bad_upstream += int(np.count_nonzero(s_sh < s_start[owner] - 1e-3))
        bad_time += int(np.count_nonzero(sh[:, 1] < -1e-3))

        status = twin[f"{production}/e_at_entry_status/{p}/data"][:]
        for k in range(len(VOLUME_LABELS)):
            status_hist[k] += np.bincount(status[:, k], minlength=len(STATUS_NAMES))
        n_mu_seen += len(status)

        centres = main[f"{production}/clusters_centers/data"][:]
        cl = main[f"{production}/raw/cluster_ids/{p}/data"][:]
        row_of_mu = np.repeat(np.arange(len(ms_t) - 1), np.diff(ms_t))
        c = centres[cl][row_of_mu][owner]
        inside = (np.hypot(sh[:, 2] - c[:, 0], sh[:, 3] - c[:, 1]) <= 60.0) & \
                 (np.abs(sh[:, 4] - c[:, 2]) <= 265.0)
        inside_frac.append(float(inside.mean()) if len(inside) else 0.0)

    print(f"  sampled {len(sample)} parts, {n_mu_seen:,} muons")
    check("ev_ids identical to the main file", bad_ids == 0)
    check("mu_starts identical and shower_starts closed", bad_starts == 0)
    check("reference points are closest approaches", worst_perp < 0.01,
          f"worst {worst_perp * 1000:.2f} mm")
    check("showers upstream of their track start are the known rare exception",
          bad_upstream <= 1e-4 * max(showers, 1),
          f"{bad_upstream} in the sample")
    check("negative shower times do not exceed the upstream count", bad_time <= bad_upstream)

    print(f"  sentinels: {100 * n_sentinel / max(n_mu_seen, 1):.2f} % of sampled muons")
    print(f"  showers inside the cluster volume: {100 * np.mean(inside_frac):.2f} %")
    print(f"  {'status':16s}" + "".join(f"{v:>12s}" for v in VOLUME_LABELS))
    for c in range(len(STATUS_NAMES)):
        print(f"  {STATUS_NAMES[c]:16s}" +
              "".join(f"{100 * status_hist[k, c] / max(n_mu_seen, 1):11.2f}%"
                      for k in range(len(VOLUME_LABELS))))
    main.close()
    twin.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", action="append",
                    help="repeatable; default is all three 2020 productions")
    ap.add_argument("--main", default=MAIN_H5)
    ap.add_argument("--twin", default=TWIN_H5)
    ap.add_argument("--sample", type=int, default=20)
    args = ap.parse_args()

    for p in args.production or ["nue2_2020", "nuatm_2020", "muatm_2020"]:
        validate(p, args.main, args.twin, args.sample)

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: " + "; ".join(_failures[:10]))
        sys.exit(1)
    print("validation passed")


if __name__ == "__main__":
    main()
