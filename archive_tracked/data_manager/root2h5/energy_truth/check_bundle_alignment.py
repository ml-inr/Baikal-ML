#!/usr/bin/env python3
"""Strong alignment test for muon bundles (muatm), where the collinearity check is weak.

extract_interactions.py reconstructs the event/track nesting from flat arrays returned by
TTree::Draw. For single-muon samples the collinearity check settles this: an interaction
placed in the wrong track's slice would sit tens to hundreds of metres off that track's
line. In a bundle the tracks are nearly parallel and metres apart, so a misassignment
barely moves the point off the line and the check loses its power.

Two independent tests are used instead:

  A. Per-entry re-read (exact, no physics). Read one event's interactions directly with
     Draw(..., nentries=1, firstentry=e) and compare element by element with the slice the
     bulk read produced for that event. This proves the EVENT boundaries exactly, since the
     two reads share no code path beyond ROOT itself.

  B. Nearest-track assignment (track level inside an event). For every interaction, find
     which of the event's tracks it is closest to and check that it is the track it was
     assigned to. The separation between the event's tracks is reported alongside, because
     the test only has power when the tracks are further apart than the measurement noise:
     a result is only meaningful for events whose tracks are well separated.

Usage (on cluster62):
    python3 check_bundle_alignment.py <in.root> <extracted.npz> [--n-events 200]
"""
import argparse
import sys

import numpy as np
import ROOT

INTER_EXPR = {
    "x": "BMCEvent.fTracks.fInteractions.fX",
    "y": "BMCEvent.fTracks.fInteractions.fY",
    "z": "BMCEvent.fTracks.fInteractions.fZ",
    "e": "BMCEvent.fTracks.fInteractions.fEnergy",
}


def read_entry(tree, expr, entry):
    n = tree.Draw(expr, "", "goff", 1, entry)
    v = tree.GetV1()
    return np.array([v[i] for i in range(n)], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_root")
    ap.add_argument("npz")
    ap.add_argument("--n-events", type=int, default=200)
    args = ap.parse_args()

    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    d = np.load(args.npz)
    # Keep the TFile referenced: if it is garbage-collected the tree becomes a dangling
    # pointer and the next Draw segfaults.
    root_file = ROOT.TFile.Open(args.in_root)
    if not root_file or root_file.IsZombie():
        sys.exit(f"cannot open {args.in_root}")
    tree = root_file.Get("Events")
    tree.SetEstimate(1_000_000)

    trk_off = np.concatenate([[0], np.cumsum(d["n_tracks"])]).astype(np.int64)
    int_off = np.concatenate([[0], np.cumsum(d["trk_n_inter"])]).astype(np.int64)
    n_events = min(args.n_events, int(d["n_events"][0]))

    # ---- A. per-entry re-read ----
    bad_a = 0
    checked_a = 0
    for e in range(n_events):
        t0, t1 = trk_off[e], trk_off[e + 1]
        if t1 <= t0:
            continue
        s, ee = int_off[t0], int_off[t1]          # all interactions of this event
        ref = read_entry(tree, INTER_EXPR["e"], e) * 1000.0   # TeV -> GeV, as in extraction
        got = d["int_energy"][s:ee]
        checked_a += 1
        if len(ref) != len(got) or not np.allclose(ref, got, rtol=1e-4, atol=1e-3):
            bad_a += 1
            if bad_a <= 3:
                print(f"  event {e}: re-read {len(ref)} vs sliced {len(got)} interactions"
                      + ("" if len(ref) != len(got) else " (values differ)"))

    print(f"A. per-entry re-read: {checked_a} events checked, {bad_a} mismatched")

    # ---- B. nearest-track assignment ----
    th, ph = np.radians(d["trk_theta"]), np.radians(d["trk_phi"])
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    refs = np.stack([d["trk_x"], d["trk_y"], d["trk_z"]], axis=1)
    pts = np.stack([d["int_x"], d["int_y"], d["int_z"]], axis=1)

    n_int = n_ok = 0
    seps = []
    multi_events = 0
    for e in range(n_events):
        t0, t1 = trk_off[e], trk_off[e + 1]
        if t1 - t0 < 2:                            # single-track events carry no information
            continue
        multi_events += 1
        idx = np.arange(t0, t1)
        # separation scale: pairwise distance between track lines at the reference points
        sep = np.linalg.norm(refs[idx][:, None, :] - refs[idx][None, :, :], axis=-1)
        seps.append(np.median(sep[sep > 0]) if (sep > 0).any() else 0.0)
        for ti in idx:
            s, ee = int_off[ti], int_off[ti + 1]
            if ee <= s:
                continue
            v = pts[s:ee][:, None, :] - refs[idx][None, :, :]
            along = np.einsum("ijk,jk->ij", v, dirs[idx])
            perp = np.linalg.norm(v - along[:, :, None] * dirs[idx][None, :, :], axis=-1)
            n_int += ee - s
            n_ok += int((perp.argmin(axis=1) == (ti - t0)).sum())

    if multi_events == 0:
        print("B. no multi-track events in this sample — bundle test not applicable")
    else:
        print(f"B. nearest-track assignment: {n_ok:,}/{n_int:,} interactions "
              f"({100*n_ok/max(n_int,1):.2f}%) closest to their assigned track")
        print(f"   over {multi_events} multi-track events; median track separation "
              f"{np.median(seps):.1f} m (test has power only well above ~0 m)")

    sys.exit(1 if bad_a else 0)


if __name__ == "__main__":
    main()
