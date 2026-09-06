#!/usr/bin/env python3
"""Assemble the energy-truth companion from the extracted npz and the main HDF5.

Stage 3 of three (doc/energy_twin_plan.md). Reads `baikal_mc_merged.h5` and the npz written by
read_root.py, and writes `baikal_mc_merged_energy_truth.h5` with rows corresponding one to one
with the main file. The format is documented in doc/hdf5_energy_truth.md.

The work that matters here is not the arithmetic -- `truth.py` does that -- but the mapping
from ROOT entries to HDF5 rows. `root2h5.py` drops events, duplicates multi-cluster ones across
their clusters, and writes single-cluster rows before split ones, so rows are not in ROOT entry
order. The mapping is reconstructed from `ev_ids`, whose numeric tail is the ROOT entry index,
and then proved: the track reference points and energies shipped in the npz must reproduce
`muons_prty/individ` value by value. A reference point is a continuous 3-vector, so a wrong
mapping cannot match by accident.

Every check is fatal. A part is written whole or the run stops.

Usage:
    python3 build_h5.py --production nuatm_2020 --npz-dir DIR [--limit-parts N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np

import truth

MAIN_H5 = "/home/albert/Baikal2025/data_manager/data/h5datasets/baikal_mc_merged.h5"
OUT_H5 = "/home/albert/Baikal2025/data_manager/data/h5datasets/baikal_mc_merged_energy_truth.h5"

# Cylinders around the centre of the row's own cluster, as (radius, half-height) in metres.
# The cluster itself is r = 60, |z| <= 265; the two larger ones ask the same question with a
# margin, because the instrumented volume has no sharp edge.
VOLUMES = ((60.0, 265.0), (90.0, 295.0), (120.0, 325.0))

# Label codes in the main file's raw/labels: noise is 0 here (it is 1 in the upstream .dat),
# muon track light is -(10^6 - j), shower light is k*10^6 + j.
LABEL_BASE = 1_000_000
NOISE_LABEL = 0

# The generator writes every pulse word as a float32 (see doc/mc_binary_formats.md), and
# float32 represents integers exactly only up to 2^24. A shower code k*10^6 + j therefore
# loses its low digits -- the muon index -- once k >= 17, and comes back as an even number:
# j decodes as 0, 2, 12, 64 ... which is rounding noise, not a muon. Such hits are light
# whose muon is unknowable; attributing them would invent a muon. Track-light codes are
# around -10^6 and are always exact.
FLOAT32_EXACT_INT = 2 ** 24

ATTRS = {
    "frame": "global array frame, NOT cluster-centred",
    "angles": "radians, copied verbatim from muons_prty/individ",
    "shower_columns": "energy_gev,time_ns,x,y,z",
    "time_origin": "muon birth: t = t_first_muon + s/c, c = 0.299792458 m/ns",
    "shower_order": "as in ROOT, unsorted",
    "volumes": "dr=0: r=60,|z|<=265 ; dr=30: r=90,|z|<=295 ; dr=60: r=120,|z|<=325",
    "energy_model": "E(s) = e_ref - 0.24*(s - s_ref) - sum(showers); no b*E term",
    "s_track_start": "-c*t_first_muon; birth for sentinels, volume entry otherwise",
    "status_codes": ("0 propagated, 1 born inside, 2 died before entry, 3 misses volume, "
                     "4 sentinel from e_bundle_reg, 5 sentinel born inside"),
    "seasons": "2020 only; the 2019 groups of the main file are deliberately not built",
    "source_file": "baikal_mc_merged.h5",
    "builder": "data_manager/energy_truth/build_h5.py",
}


def flat_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenation of range(starts[i], starts[i] + counts[i]) without a Python loop."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    idx = np.arange(offsets[-1], dtype=np.int64)
    block = np.repeat(np.arange(len(counts)), counts)
    return idx - offsets[block] + np.asarray(starts, dtype=np.int64)[block]


def entries_of_rows(ev_ids: np.ndarray) -> np.ndarray:
    """ROOT entry index behind each row, from the numeric tail of the identifier.

    `root2h5.py` builds the identifier as f"{particle}_{file}_{entry}" where entry counts from
    zero after the start-index offset -- the same offset read_root.py reproduces.
    """
    return np.array([int(s.rsplit(b"_", 1)[1]) for s in ev_ids], dtype=np.int64)


def check_mapping(npz, track_idx: np.ndarray, individ: np.ndarray, part: str) -> dict:
    """Prove the entry-to-row mapping against the track kinematics shipped in the npz."""
    ref_root = npz["trk_xyz"][track_idx]
    ref_h5 = individ[:, 2:5]
    if not np.array_equal(ref_root, ref_h5):
        bad = int(np.flatnonzero(np.any(ref_root != ref_h5, axis=1))[0])
        raise RuntimeError(
            f"{part}: reference points disagree at muon {bad} -- ROOT {ref_root[bad]} vs "
            f"HDF5 {ref_h5[bad]}. The mapping from ROOT entries to rows is wrong.")

    e_root = npz["trk_energy_gev"][track_idx]
    if not np.array_equal(e_root, individ[:, 6]):
        raise RuntimeError(f"{part}: fMuonEnergy disagrees between ROOT and the HDF5")

    # Angles pass through a degrees-to-radians conversion in root2h5.py, so exact equality is
    # not owed; float32 precision is.
    worst = float(np.abs(np.radians(npz["trk_theta_deg"][track_idx].astype(np.float64))
                         - individ[:, 0]).max(initial=0.0))
    if worst > 1e-5:
        raise RuntimeError(f"{part}: theta disagrees by up to {worst:.3g} rad")
    return {"worst_theta_rad": worst}


def count_lighting_muons(labels: np.ndarray, ev_starts: np.ndarray,
                         n_mu_per_row: np.ndarray, part: str) -> tuple[np.ndarray, int]:
    """Distinct muons of the row whose light reached this cluster, from raw/labels.

    Both track light and shower light count: a muon seen only through one of its cascades did
    light the cluster. Hits whose muon index was destroyed by the float32 encoding are not
    attributed to any muon, but they are not thrown away either -- a row lit only by such hits
    was lit by at least one muon, and when the row holds a single muon that is not a lower
    bound but the exact answer.

    Returns the counts, the number of unattributable hits, and the number of rows whose count
    rests on them.
    """
    n_rows = len(ev_starts) - 1
    lab = labels.astype(np.int64)
    row = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(ev_starts))

    signal = lab != NOISE_LABEL
    lost = signal & (lab >= FLOAT32_EXACT_INT)
    j = np.where(lab < 0, lab + LABEL_BASE, lab % LABEL_BASE)
    keep = signal & ~lost & (j > 0)
    row_k, j_k = row[keep], j[keep]

    over = j_k > n_mu_per_row[row_k]
    if over.any():
        i = int(np.flatnonzero(over)[0])
        raise RuntimeError(
            f"{part}: hit label points at muon {j_k[i]} of a row that has only "
            f"{n_mu_per_row[row_k[i]]} -- j is not indexed within the row as assumed")

    pairs = np.unique(row_k * LABEL_BASE + j_k)
    counts = np.bincount(pairs // LABEL_BASE, minlength=n_rows)

    # Rows lit only by hits whose index is unrecoverable: at least one muon lit them.
    only_lost = (counts == 0) & (np.bincount(row[lost], minlength=n_rows) > 0)
    counts[only_lost] = 1
    return counts.astype(np.int16), int(np.count_nonzero(lost)), int(only_lost.sum())


def build_part(main: h5py.File, npz, ptype: str, part: str,
               centres: np.ndarray) -> tuple[dict, dict]:
    g = f"{ptype}"
    ev_ids = main[f"{g}/ev_ids/{part}/data"][:]
    mu_starts = main[f"{g}/muons_prty/mu_starts/{part}/data"][:].astype(np.int64)
    individ = main[f"{g}/muons_prty/individ/{part}/data"][:]
    aggregate = main[f"{g}/muons_prty/aggregate/{part}/data"][:]
    cluster_ids = main[f"{g}/raw/cluster_ids/{part}/data"][:]
    ev_starts = main[f"{g}/raw/ev_starts/{part}/data"][:].astype(np.int64)
    labels = main[f"{g}/raw/labels/{part}/data"][:]

    n_rows = len(ev_ids)
    entries = entries_of_rows(ev_ids)

    n_tracks_per_entry = npz["n_tracks_per_entry"].astype(np.int64)
    n_showers_per_track = npz["n_showers_per_track"].astype(np.int64)
    trk_off = np.concatenate([[0], np.cumsum(n_tracks_per_entry)])
    shw_off = np.concatenate([[0], np.cumsum(n_showers_per_track)])

    if entries.max(initial=-1) >= len(n_tracks_per_entry):
        raise RuntimeError(f"{part}: row refers to ROOT entry {entries.max()} but the npz has "
                           f"only {len(n_tracks_per_entry)}")

    n_mu_per_row = np.diff(mu_starts)
    if not np.array_equal(n_mu_per_row, n_tracks_per_entry[entries]):
        raise RuntimeError(f"{part}: muons per row disagree with tracks per ROOT entry")

    track_idx = flat_ranges(trk_off[entries], n_mu_per_row)
    map_info = check_mapping(npz, track_idx, individ, part)

    # Showers of each muon, in ROOT order, gathered into row order.
    n_showers_per_muon = n_showers_per_track[track_idx]
    shower_idx = flat_ranges(shw_off[track_idx], n_showers_per_muon)
    shower_starts = np.concatenate([[0], np.cumsum(n_showers_per_muon)]).astype(np.int64)

    ref = individ[:, 2:5].astype(np.float64)
    d = truth.direction_from_angles(individ[:, 0], individ[:, 1])
    e_ref = individ[:, 6].astype(np.float64)
    t_first = individ[:, 5].astype(np.float64)
    s_start = truth.track_start(t_first)

    owner = np.repeat(np.arange(len(track_idx)), n_showers_per_muon)
    shower_xyz = npz["xyz"][shower_idx].astype(np.float64)
    shower_e = npz["energy_gev"][shower_idx].astype(np.float64)
    shower_s = truth.s_along(shower_xyz, ref[owner], d[owner])
    shower_t = truth.shower_times(shower_s, t_first[owner])

    chain = truth.build_chain(shower_s, shower_e, shower_starts)

    # Per-muon quantities that live per row in the main file.
    row_of_muon = np.repeat(np.arange(n_rows), n_mu_per_row)
    e_reg = aggregate[:, 1].astype(np.float64)[row_of_muon]
    centre = centres[cluster_ids][row_of_muon]

    e_entry = np.empty((len(track_idx), len(VOLUMES)), dtype=np.float64)
    status = np.empty((len(track_idx), len(VOLUMES)), dtype=np.int8)
    for k, (radius, z_half) in enumerate(VOLUMES):
        s_in, s_out = truth.cylinder_crossing(ref, d, centre, radius, z_half)
        e_entry[:, k], status[:, k] = truth.entry_energy(chain, e_ref, s_start, e_reg,
                                                         s_in, s_out)

    n_lit, n_lost_hits, n_rows_from_lost = count_lighting_muons(
        labels, ev_starts, n_mu_per_row, part)

    data = {
        "ev_ids": ev_ids,
        "mu_starts": mu_starts,
        "shower_starts": shower_starts,
        "showers": np.column_stack([shower_e, shower_t, shower_xyz]).astype(np.float32),
        "n_muons_lighting_cluster": n_lit,
        "ref_xyz": individ[:, 2:5].astype(np.float32),
        "direction": individ[:, 0:2].astype(np.float32),
        "e_ref": individ[:, 6].astype(np.float32),
        "s_track_start": s_start.astype(np.float32),
        "e_at_entry": e_entry.astype(np.float32),
        "e_at_entry_status": status,
    }
    stats = {
        "rows": n_rows, "muons": len(track_idx), "showers": len(shower_idx),
        "lost_index_hits": n_lost_hits, "rows_from_lost": n_rows_from_lost,
        "sentinels": int(np.count_nonzero(
            individ[:, 6] == np.float32(truth.SENTINEL_E_REF))),
        **map_info,
    }
    stats.update(verify_part(data, individ, shower_s, s_start, part))
    return data, stats


# A shower cannot precede the muon that made it, but the rule has a measured exception rate of
# about one in seven million: a single 6 PeV cascade in nue2 sits 6 m upstream of its muon's
# track start. Aborting the build on that would be wrong, and silently dropping it would be
# worse, so the check is on the *share*: a real misalignment puts a large fraction of showers
# upstream, never one in a million.
MAX_UPSTREAM_SHARE = 1e-4


def verify_part(data: dict, individ: np.ndarray, shower_s: np.ndarray,
                s_start: np.ndarray, part: str) -> dict:
    """Checks on the assembled arrays, before anything is written."""
    if not np.array_equal(data["ref_xyz"], individ[:, 2:5]):
        raise RuntimeError(f"{part}: ref_xyz is not a faithful copy")
    if not np.array_equal(data["e_ref"], individ[:, 6]):
        raise RuntimeError(f"{part}: e_ref is not a faithful copy")
    if data["shower_starts"][-1] != len(data["showers"]):
        raise RuntimeError(f"{part}: shower_starts does not close on the shower count")

    owner = np.repeat(np.arange(len(s_start)), np.diff(data["shower_starts"]))
    offset = shower_s - s_start[owner]
    upstream = offset < -1e-3
    n_up = int(np.count_nonzero(upstream))
    if n_up and n_up > MAX_UPSTREAM_SHARE * max(len(offset), 1):
        raise RuntimeError(
            f"{part}: {n_up} of {len(offset):,} showers lie upstream of their muon's track "
            f"start (worst {offset.min():.2f} m) -- that share means a misalignment, not the "
            f"known rare exception")

    # Times are t_first + s/c, so a negative time is the same statement as an upstream shower;
    # this catches an arithmetic slip in the conversion rather than a new physical case.
    n_neg = int(np.count_nonzero(data["showers"][:, 1] < -1e-3))
    if n_neg > n_up:
        raise RuntimeError(f"{part}: {n_neg} negative shower times against {n_up} upstream "
                           f"showers -- the time conversion disagrees with the geometry")
    return {"upstream": n_up, "worst_upstream_m": float(offset.min(initial=0.0))}


def write_part(out: h5py.File, ptype: str, part: str, data: dict) -> None:
    big = {"showers", "ev_ids"}
    for name, arr in data.items():
        kw = dict(compression="gzip", compression_opts=4) if name in big else {}
        out.create_dataset(f"{ptype}/{name}/{part}/data", data=arr, **kw)


def build(production: str, npz_dir: str, main_path: str, out_path: str,
          limit_parts: int = 0) -> None:
    main = h5py.File(main_path, "r")
    centres = main[f"{production}/clusters_centers/data"][:]
    parts = sorted(main[f"{production}/ev_ids"].keys())
    if limit_parts:
        parts = parts[:limit_parts]

    out = h5py.File(out_path, "a")
    for key, value in ATTRS.items():
        out.attrs[key] = value

    done = set(out[production].keys()) if production in out else set()
    already = {p for p in parts if "showers" in done
               and p in out.get(f"{production}/showers", {})}
    todo = [p for p in parts if p not in already]
    print(f"{production}: {len(parts)} parts, {len(already)} already built, "
          f"{len(todo)} to do", flush=True)

    totals = {"rows": 0, "muons": 0, "showers": 0, "lost_index_hits": 0,
              "rows_from_lost": 0, "sentinels": 0, "upstream": 0}
    worst_upstream = 0.0
    t0 = time.time()
    for i, part in enumerate(todo, 1):
        npz_path = os.path.join(npz_dir, part.replace("part_", "") + ".npz")
        if not os.path.exists(npz_path):
            raise RuntimeError(f"{part}: no npz at {npz_path}")
        with np.load(npz_path) as npz:
            data, stats = build_part(main, npz, production, part, centres)
        write_part(out, production, part, data)
        for k in totals:
            totals[k] += stats[k]
        worst_upstream = min(worst_upstream, stats["worst_upstream_m"])
        if i % 25 == 0 or i == len(todo):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  [{i}/{len(todo)}] {part}  rows={totals['rows']:,} "
                  f"muons={totals['muons']:,} showers={totals['showers']:,}  "
                  f"{rate:.1f} parts/s", flush=True)
    out.close()
    main.close()

    print(f"{production} done: rows={totals['rows']:,} muons={totals['muons']:,} "
          f"showers={totals['showers']:,} sentinels={totals['sentinels']:,}")
    print(f"  hits whose muon index was lost to float32: {totals['lost_index_hits']:,}; "
          f"rows counted from them alone: {totals['rows_from_lost']:,} "
          f"({100 * totals['rows_from_lost'] / max(totals['rows'], 1):.2f} %)")
    print(f"  showers upstream of their track start: {totals['upstream']:,} "
          f"({100 * totals['upstream'] / max(totals['showers'], 1):.6f} %), "
          f"worst {worst_upstream:.2f} m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--production", required=True)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--main", default=MAIN_H5)
    ap.add_argument("--out", default=OUT_H5)
    ap.add_argument("--limit-parts", type=int, default=0)
    args = ap.parse_args()
    try:
        build(args.production, args.npz_dir, args.main, args.out, args.limit_parts)
    except RuntimeError as exc:
        sys.exit(f"BUILD ABORTED — {exc}")


if __name__ == "__main__":
    main()
