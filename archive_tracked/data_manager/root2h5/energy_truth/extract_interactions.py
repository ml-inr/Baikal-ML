#!/usr/bin/env python3
"""Extract the MC muon interaction chain from BARS ROOT files (runs on cluster62).

Why this exists: the muon energy stored in our HDF5 (`muons_prty/individ[:, 6]`) is the
energy at the MC reference point -- the closest approach to the centre of the WHOLE
detector -- which for a single-cluster event can be hundreds of metres away. To define an
energy target at the cluster (E at closest approach, or <dE/dx> along the traversed
segment) the muon must be propagated, and doing that accurately requires the individual
stochastic interactions, not an average dE/dx: over 100-200 m the stochastic term is tens
of percent with a heavy tail, comparable to the resolution we want to measure.

That chain lives in `BMCEvent.fTracks.fInteractions`, a TClonesArray of the custom class
BMCInteraction. uproot cannot deserialise it, and the local BARS build has dictionaries
but no compiled shared libraries, so the class API is unavailable. ROOT can still read the
data members through emulated classes, which is what this script does via TTree::Draw with
"goff" -- verified to work for fInteractions.fX / .fEnergy.

Output (one .npz per ROOT file), all arrays flat and aligned by cumulative counts:
    n_resp_mu    (n_events,)  int32    response muons per event (from fResponseMuonsN)
    n_tracks     (n_events,)  int32    tracks per event (length of fTracks)
    trk_*        (n_tracks,)  float32  theta, phi (deg), x, y, z (m, global), energy (GeV)
                                       at the reference point -- the closest approach to the
                                       centre of the WHOLE detector; it is an extrapolation
                                       and is negative for ~20% of tracks (muon already dead
                                       there), which is precisely why the chain is needed.
    trk_n_inter  (n_tracks,)  int32    interactions per track
    int_*        (n_inter,)   float32  x, y, z (m, global), energy (GeV) per interaction,
                                       SORTED along the track direction within each track
    int_block    (n_inter,)   int16    block index in the original file order (see
                                       sort_interactions_along_track): the only surviving
                                       trace of the interaction-category grouping
Track i of event e spans trk_offsets[e]:trk_offsets[e+1]; interactions of track i span
int_offsets[i]:int_offsets[i+1], both recoverable by cumsum.

Two independent checks guard the reconstructed nesting, because Draw returns flat arrays
and the nesting is inferred, not read:
  1. sum(trk_n_inter) == n_inter. `fInteractionN` (the declared count, read separately from
     the interaction members) must match the number of interaction rows actually returned.
     This catches truncation (SetEstimate too small) and TClonesArray slots that are
     allocated but unfilled. It does NOT constrain the ORDER: a permutation would still sum
     correctly.
  2. Collinearity on a random sample of tracks: every interaction assigned to a track must
     lie on that track's line. Different tracks have different lines in 3D, so interactions
     landing in the wrong slice would be off by tens to hundreds of metres; measured
     deviation is ~1.5 mm (numerical precision) over 28k tracks. This is what actually
     establishes the ordering.
Check 2 loses power for muon bundles (muatm), where tracks are nearly parallel and only
metres apart. Bundles are covered separately by check_bundle_alignment.py, which re-reads
single events through an independent Draw call and compares element by element, and also
verifies that each interaction is closest to the track it was assigned to. On muatm both
passed: 300/300 events identical on re-read, 764/764 interactions correctly assigned at a
median track separation of 23 m.

Interactions are NOT stored in propagation order in the file; this script sorts them along
the track before writing, so consumers can rely on the output order (see
sort_interactions_along_track for why, and for what `int_block` preserves).

Usage:
    python3 extract_interactions.py <in.root> <out.npz> [--max-events N]
"""
import argparse
import sys

import numpy as np
import ROOT

# Interaction energies are stored in TeV in the BARS files; the reference macro
# (energy.C) multiplies by 1000 to get GeV, and we follow it.
INTERACTION_ENERGY_TO_GEV = 1000.0

EVENT_VARS = {
    "n_resp_mu": "BMCEvent.fResponseMuonsN",
    # Intrinsic identifiers, kept for traceability only. They cannot serve as a join key:
    # the HDF5 stores just a generated positional counter (root2h5 builds ev_ids as
    # np.arange), and these fields are filled only in muatm -- nue2 and nuatm carry -1.
    # The join is therefore positional, and must be verified against the muon truth the
    # HDF5 already stores rather than assumed (see create_energy_h5.py).
    "run_n": "BMCEvent.fRunN",
    "event_n": "BMCEvent.fEventN",
}
TRACK_VARS = {
    "trk_theta":  "BMCEvent.fTracks.fTheta",
    "trk_phi":    "BMCEvent.fTracks.fPhi",
    "trk_x":      "BMCEvent.fTracks.fX",
    "trk_y":      "BMCEvent.fTracks.fY",
    "trk_z":      "BMCEvent.fTracks.fZ",
    "trk_energy": "BMCEvent.fTracks.fMuonEnergy",
    "trk_n_inter": "BMCEvent.fTracks.fInteractionN",
}
INTER_VARS = {
    "int_x":      "BMCEvent.fTracks.fInteractions.fX",
    "int_y":      "BMCEvent.fTracks.fInteractions.fY",
    "int_z":      "BMCEvent.fTracks.fInteractions.fZ",
    "int_energy": "BMCEvent.fTracks.fInteractions.fEnergy",
}


def sort_interactions_along_track(out: dict) -> dict:
    """Order each track's interactions by their projection on the track direction.

    Interactions do not arrive in propagation order. Backward steps are rare (2.4% on nue2)
    but large -- median 142 m against 8.5 m for forward steps -- and every affected track
    has exactly one of them, so each track is stored as two concatenated blocks, each
    ordered along the track. The second block is small (2.1% of interactions), consistent
    with a rare interaction category being written separately. Sorting here makes the output
    canonical so that no consumer can integrate along the track in the wrong order.

    BMCInteraction carries no type field, so that block structure is the only trace of the
    grouping, and sorting would destroy it. The block index (assigned in the original file
    order, incremented at each backward jump) is therefore stored as `int_block` before the
    sort: categories matter physically, since electromagnetic and hadronic cascades of equal
    energy do not yield the same light.
    """
    offsets = np.concatenate([[0], np.cumsum(out["trk_n_inter"])]).astype(np.int64)
    th, ph = np.radians(out["trk_theta"]), np.radians(out["trk_phi"])
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    refs = np.stack([out["trk_x"], out["trk_y"], out["trk_z"]], axis=1)
    pts = np.stack([out["int_x"], out["int_y"], out["int_z"]], axis=1)

    block = np.zeros(len(pts), dtype=np.int16)
    order = np.arange(len(pts), dtype=np.int64)
    for i in range(len(out["trk_n_inter"])):
        s, e = offsets[i], offsets[i + 1]
        if e - s < 2:
            continue
        along = (pts[s:e] - refs[i]) @ dirs[i]
        block[s:e] = np.concatenate([[0], np.cumsum(np.diff(along) < 0)]).astype(np.int16)
        order[s:e] = s + np.argsort(along, kind="stable")

    for key in ("int_x", "int_y", "int_z", "int_energy"):
        out[key] = out[key][order]
    out["int_block"] = block[order]
    return out


def check_collinearity(out: dict, n_sample: int = 2000, tol_m: float = 0.1) -> float:
    """Verify the reconstructed slicing: each track's interactions must lie on its own line.

    Returns the worst perpendicular deviation found. Raises if any track exceeds tol_m,
    which would mean the flat interaction array is not ordered track by track.
    """
    offsets = np.concatenate([[0], np.cumsum(out["trk_n_inter"])]).astype(np.int64)
    n_tracks = len(out["trk_n_inter"])
    if n_tracks == 0:
        return 0.0
    th, ph = np.radians(out["trk_theta"]), np.radians(out["trk_phi"])
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    refs = np.stack([out["trk_x"], out["trk_y"], out["trk_z"]], axis=1)
    pts = np.stack([out["int_x"], out["int_y"], out["int_z"]], axis=1)

    rng = np.random.default_rng(0)
    idx = rng.choice(n_tracks, size=min(n_sample, n_tracks), replace=False)
    worst = 0.0
    for i in idx:
        s, e = offsets[i], offsets[i + 1]
        if e <= s:
            continue
        v = pts[s:e] - refs[i]
        perp = np.linalg.norm(v - np.outer(v @ dirs[i], dirs[i]), axis=1)
        worst = max(worst, float(perp.max()))
    if worst > tol_m:
        raise RuntimeError(
            f"MISALIGNED: interactions deviate up to {worst:.3f} m from their track line "
            f"(tolerance {tol_m} m) — the flat array is not ordered track by track")
    return worst


def get_start_index(tree) -> int:
    """Replicate root2h5.get_start_index: the converter skips a leading empty event.

    root2h5.py reads every array as `[st:]` and then numbers events from zero, so the
    identifier stored in the HDF5 (`ev_ids`) refers to ROOT entry `k + st`. Extracting from
    entry 0 regardless would silently shift the whole file by one event whenever st=1. The
    offset is 0 in all 120 MC files sampled, but the converter's logic allows 1, so it is
    reproduced here rather than assumed away.
    """
    n = tree.Draw("BEvent.fPulseN", "", "goff", 1, 0)
    return 1 if (n > 0 and tree.GetV1()[0] == 0) else 0


def draw(tree, expr, n_entries, first=0):
    """Read one flattened member array through TTree::Draw (emulated classes)."""
    n = tree.Draw(expr, "", "goff", n_entries, first)
    if n < 0:
        raise RuntimeError(f"Draw failed for {expr}")
    v = tree.GetV1()
    return np.array([v[i] for i in range(n)], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_root")
    ap.add_argument("out_npz")
    ap.add_argument("--max-events", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    f = ROOT.TFile.Open(args.in_root)
    if not f or f.IsZombie():
        sys.exit(f"cannot open {args.in_root}")
    tree = f.Get("Events")
    st = get_start_index(tree)          # entry offset the converter applies (see above)
    n_events = int(tree.GetEntries()) - st
    if args.max_events:
        n_events = min(n_events, args.max_events)

    # Draw buffers everything it reads; size it for the worst case seen in these files
    # (bundles with many muons, each with tens of interactions).
    tree.SetEstimate(max(n_events * 200, 1_000_000))

    out = {}
    for name, expr in EVENT_VARS.items():
        out[name] = draw(tree, expr, n_events, st).astype(np.int32)
    for name, expr in TRACK_VARS.items():
        arr = draw(tree, expr, n_events, st)
        out[name] = arr.astype(np.int32 if name == "trk_n_inter" else np.float32)
    for name, expr in INTER_VARS.items():
        arr = draw(tree, expr, n_events, st)
        if name == "int_energy":
            arr = arr * INTERACTION_ENERGY_TO_GEV
        out[name] = arr.astype(np.float32)

    n_tracks = len(out["trk_theta"])
    n_inter = len(out["int_x"])

    # Check 1 (counts): catches truncation and unfilled TClonesArray slots. Says nothing
    # about ordering — see check 2.
    total_from_counts = int(out["trk_n_inter"].sum())
    if total_from_counts != n_inter:
        sys.exit(f"COUNT MISMATCH: sum(trk_n_inter)={total_from_counts} but read {n_inter} "
                 f"interactions -- do not use this output")

    # Check 2 (geometry): this is what establishes that the flat array is ordered track by
    # track, which the count check cannot see. Run before sorting, so it tests the nesting
    # as read rather than a reordering of our own making.
    worst_perp = check_collinearity(out)

    out = sort_interactions_along_track(out)

    # tracks per event: fResponseMuonsN counts response muons; verify it explains n_tracks
    out["n_tracks"] = out["n_resp_mu"].copy()
    if int(out["n_resp_mu"].sum()) != n_tracks:
        print(f"WARNING: sum(n_resp_mu)={int(out['n_resp_mu'].sum())} != n_tracks={n_tracks}; "
              f"storing n_tracks from fResponseMuonsN may be wrong", file=sys.stderr)

    out["n_events"] = np.array([n_events], dtype=np.int64)
    out["start_index"] = np.array([st], dtype=np.int8)   # npz row k == ROOT entry k+st == h5 ev_id k
    np.savez_compressed(args.out_npz, **out)

    print(f"{args.in_root}")
    print(f"  st={st}  events={n_events:,}  tracks={n_tracks:,}  interactions={n_inter:,}  "
          f"(counts ok; max perp deviation {worst_perp*1000:.2f} mm)")
    print(f"  n_resp_mu: min={out['n_resp_mu'].min()} max={out['n_resp_mu'].max()} "
          f"mean={out['n_resp_mu'].mean():.2f}")
    print(f"  muon E [GeV]: median={np.median(out['trk_energy']):.1f} "
          f"max={out['trk_energy'].max():.1f}")
    print(f"  interactions/track: mean={out['trk_n_inter'].mean():.1f} "
          f"max={out['trk_n_inter'].max()}")
    print(f"  -> {args.out_npz}")


if __name__ == "__main__":
    main()
