#!/usr/bin/env python3
"""Extract the muon energy-loss chain from BARS ROOT files. Runs on cluster62.

This is stage 1 of three (see doc/energy_twin_plan.md). It reads the one thing our HDF5 does
not carry -- the stochastic energy losses along each muon track, `BMCEvent.fTracks
.fInteractions` -- and writes it per ROOT file as an npz. Everything else about the muon
(direction, reference point, energy there, time) is already in `baikal_mc_merged.h5` and is
deliberately not duplicated here.

Why TTree::Draw and not uproot: `fInteractions` is a TClonesArray of the custom class
BMCInteraction. uproot cannot deserialise it, and the BARS build on cluster62 ships
dictionaries without compiled shared libraries, so the class API is unavailable. ROOT still
reads the data members through emulated classes, which `Draw(..., "goff")` exposes as flat
arrays.

Flat is the operative word: Draw returns one long array per member with the per-track nesting
erased. The nesting is reconstructed from `fInteractionN` and then *checked*, not trusted --
see `check_counts` and `check_collinearity`. A silently mis-sliced array would put a muon's
losses on another muon's track, which no downstream test would notice.

The track kinematics are written out as well, even though the main HDF5 already carries them.
They are not part of the companion file: they exist so the builder can compare them, value by
value, against `muons_prty/individ` and prove that its mapping from ROOT entries to HDF5 rows
is right. That mapping is where a silent misalignment would live -- `root2h5.py` filters events
and then writes single-cluster rows first and split multi-cluster rows after them, so rows are
not in ROOT entry order -- and a reference point is a continuous 3-vector, which no wrong
mapping reproduces by accident.

Showers are written in ROOT's own order. They do not arrive sorted along the track, and that
disorder is informative: see doc/hdf5_energy_truth.md section 6.

Usage:
    python3 read_root.py FILE.root OUT.npz
    python3 read_root.py --dir /path/to/root --out /path/to/npz [--jobs 6]

The directory mode skips inputs whose output already exists, so a killed run resumes by
being restarted.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# BARS stores interaction energies in TeV while the muon energy in the same record is in GeV.
# The reference macro energy.C multiplies by 1000; we convert on write so that everything
# downstream of this script is GeV. See doc/mc_provenance.md section 6 on why unit comments
# in BARS cannot be trusted.
TEV_TO_GEV = 1000.0

# Track members: used by the collinearity check here, and written out so the builder can
# verify its row mapping against the main HDF5. Angles are degrees in ROOT and radians in the
# HDF5 -- the conversion happens in root2h5.py, not here.
TRACK_VARS = {
    "theta": "BMCEvent.fTracks.fTheta",
    "phi": "BMCEvent.fTracks.fPhi",
    "x": "BMCEvent.fTracks.fX",
    "y": "BMCEvent.fTracks.fY",
    "z": "BMCEvent.fTracks.fZ",
    "energy": "BMCEvent.fTracks.fMuonEnergy",
}
SHOWER_VARS = {
    "x": "BMCEvent.fTracks.fInteractions.fX",
    "y": "BMCEvent.fTracks.fInteractions.fY",
    "z": "BMCEvent.fTracks.fInteractions.fZ",
    "energy": "BMCEvent.fTracks.fInteractions.fEnergy",
}
N_SHOWERS_VAR = "BMCEvent.fTracks.fInteractionN"
N_TRACKS_VAR = "BMCEvent.fResponseMuonsN"

COLLINEARITY_TOL_M = 0.1


def get_start_index(tree) -> int:
    """Reproduce root2h5.get_start_index: the converter skips a leading empty event.

    `root2h5.py` reads every branch as `[st:]` and then numbers events from zero, so HDF5 row
    `k` is ROOT entry `k + st`. Extracting from entry 0 regardless would shift a whole file by
    one event whenever st = 1. It was 0 in all 120 MC files sampled, but the converter's logic
    allows 1, so the rule is reproduced rather than assumed away.
    """
    n = tree.Draw("BEvent.fPulseN", "", "goff", 1, 0)
    return 1 if (n > 0 and tree.GetV1()[0] == 0) else 0


def draw(tree, expr: str, n_entries: int, first: int) -> np.ndarray:
    """Read one flattened member array through TTree::Draw."""
    n = tree.Draw(expr, "", "goff", n_entries, first)
    if n < 0:
        raise RuntimeError(f"Draw failed for {expr}")
    buf = tree.GetV1()
    return np.frombuffer(buf, dtype=np.float64, count=n).copy()


def check_counts(n_showers_per_track: np.ndarray, n_showers: int,
                 n_tracks_per_entry: np.ndarray, n_tracks: int) -> None:
    """Both nesting levels must be explained by their declared counts.

    This catches a truncated Draw buffer (SetEstimate too small) and TClonesArray slots that
    are allocated but never filled -- the failure that once produced a bogus "44% of pulses
    have fMagic = 0". It says nothing about order: a permutation still sums correctly.
    """
    declared = int(n_showers_per_track.sum())
    if declared != n_showers:
        raise RuntimeError(
            f"count mismatch: fInteractionN sums to {declared:,} but {n_showers:,} showers "
            f"were read -- the buffer was truncated or slots are unfilled")
    declared = int(n_tracks_per_entry.sum())
    if declared != n_tracks:
        raise RuntimeError(
            f"count mismatch: fResponseMuonsN sums to {declared:,} but {n_tracks:,} tracks "
            f"were read")


def check_collinearity(trk: dict, xyz: np.ndarray, n_showers_per_track: np.ndarray) -> float:
    """Every shower must lie on the line of the track it was sliced into.

    This is what actually establishes the *ordering* of the flat array, which the count check
    cannot see: distinct tracks are distinct lines in 3D, so a shower landing in the wrong
    slice misses its line by tens to hundreds of metres. Returns the worst deviation in metres.

    The check weakens for muon bundles (muatm), where tracks are nearly parallel and metres
    apart rather than hundreds; bundles were covered separately by re-reading single events
    through an independent Draw call (764/764 showers correctly assigned).
    """
    if len(xyz) == 0:
        return 0.0
    th, ph = np.radians(trk["theta"]), np.radians(trk["phi"])
    d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    ref = np.stack([trk["x"], trk["y"], trk["z"]], axis=1)

    owner = np.repeat(np.arange(len(n_showers_per_track)), n_showers_per_track)
    v = xyz - ref[owner]
    along = np.einsum("ij,ij->i", v, d[owner])
    perp = np.linalg.norm(v - along[:, None] * d[owner], axis=1)
    worst = float(perp.max())
    if worst > COLLINEARITY_TOL_M:
        raise RuntimeError(
            f"misaligned: showers deviate up to {worst:.3f} m from their track line "
            f"(tolerance {COLLINEARITY_TOL_M} m) -- the flat array is not ordered track by "
            f"track, do not use this output")
    return worst


def extract(in_root: str, out_npz: str, max_events: int = 0) -> dict:
    """Read one ROOT file and write one npz. Returns a summary for logging."""
    import ROOT  # imported here so the module can be imported without ROOT present

    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    f = ROOT.TFile.Open(in_root)
    if not f or f.IsZombie():
        raise RuntimeError(f"cannot open {in_root}")
    tree = f.Get("Events")
    st = get_start_index(tree)
    n_entries = int(tree.GetEntries()) - st
    if max_events:
        n_entries = min(n_entries, max_events)

    # Draw buffers everything it reads before returning it; size for the worst case seen in
    # these files (bundles of many muons, each with tens of showers).
    tree.SetEstimate(max(n_entries * 200, 1_000_000))

    n_tracks_per_entry = draw(tree, N_TRACKS_VAR, n_entries, st).astype(np.int32)
    n_showers_per_track = draw(tree, N_SHOWERS_VAR, n_entries, st).astype(np.int32)
    trk = {k: draw(tree, e, n_entries, st) for k, e in TRACK_VARS.items()}
    trk_xyz = np.stack([trk["x"], trk["y"], trk["z"]], axis=1)
    xyz = np.stack([draw(tree, SHOWER_VARS[k], n_entries, st) for k in ("x", "y", "z")], axis=1)
    energy = draw(tree, SHOWER_VARS["energy"], n_entries, st) * TEV_TO_GEV

    check_counts(n_showers_per_track, len(xyz), n_tracks_per_entry, len(trk["theta"]))
    worst_perp = check_collinearity(trk, xyz, n_showers_per_track)

    np.savez_compressed(
        out_npz,
        n_tracks_per_entry=n_tracks_per_entry,
        n_showers_per_track=n_showers_per_track,
        trk_theta_deg=trk["theta"].astype(np.float32),
        trk_phi_deg=trk["phi"].astype(np.float32),
        trk_xyz=trk_xyz.astype(np.float32),
        trk_energy_gev=trk["energy"].astype(np.float32),
        xyz=xyz.astype(np.float32),
        energy_gev=energy.astype(np.float32),
        n_entries=np.array([n_entries], dtype=np.int64),
        start_index=np.array([st], dtype=np.int8),
    )
    return dict(file=in_root, st=st, entries=n_entries, tracks=len(trk["theta"]),
                showers=len(xyz), worst_perp_mm=worst_perp * 1000)


def _one(job: tuple[str, str]) -> str:
    in_root, out_npz = job
    try:
        s = extract(in_root, out_npz)
    except Exception as exc:                       # one bad file must not kill the batch
        return f"FAIL {Path(in_root).name}: {exc}"
    return (f"ok   {Path(in_root).name}  entries={s['entries']:,} tracks={s['tracks']:,} "
            f"showers={s['showers']:,} perp={s['worst_perp_mm']:.2f}mm")


def run_dir(root_dir: str, out_dir: str, jobs: int) -> int:
    """Extract every .root of a directory, skipping those already done."""
    os.makedirs(out_dir, exist_ok=True)
    todo = []
    for name in sorted(os.listdir(root_dir)):
        if not name.endswith(".root"):
            continue
        out = os.path.join(out_dir, name[:-5] + ".npz")
        if not os.path.exists(out):
            todo.append((os.path.join(root_dir, name), out))
    print(f"{len(todo)} files to extract into {out_dir} ({jobs} workers)", flush=True)

    n_fail = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, line in enumerate(pool.map(_one, todo), 1):
            if line.startswith("FAIL"):
                n_fail += 1
            print(f"[{i}/{len(todo)}] {line}", flush=True)
    print(f"done: {len(todo) - n_fail} ok, {n_fail} failed")
    return n_fail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_root", nargs="?")
    ap.add_argument("out_npz", nargs="?")
    ap.add_argument("--dir", help="extract a whole directory of .root files")
    ap.add_argument("--out", help="output directory, used with --dir")
    # cluster62 has 12 cores and no batch system; more than 6 workers starves everyone else.
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--max-events", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    if args.dir:
        if not args.out:
            sys.exit("--dir requires --out")
        sys.exit(1 if run_dir(args.dir, args.out, args.jobs) else 0)

    if not args.in_root or not args.out_npz:
        sys.exit("give FILE.root OUT.npz, or --dir/--out")
    s = extract(args.in_root, args.out_npz, args.max_events)
    print(f"{s['file']}\n  st={s['st']} entries={s['entries']:,} tracks={s['tracks']:,} "
          f"showers={s['showers']:,} worst perpendicular deviation {s['worst_perp_mm']:.2f} mm"
          f"\n  -> {args.out_npz}")


if __name__ == "__main__":
    main()
