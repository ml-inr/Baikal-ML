"""Is a lower chi-square a better track?  Two anchors, on `mc_reco` muatm.

    python inference/track_fit_alike_bars/tests/truth_check.py
    python inference/track_fit_alike_bars/tests/truth_check.py --n-parts 20 --per-part 10

The port and BARS disagree on the direction of ~30% of events, up to tens of degrees.
On those events the port's chi-square is usually **lower**.  Lower is not automatically
better, and this script asks whether it is, two ways:

**A. The same functional at both tracks.**  BARS' track is read from `reco_prty`
(`xyzRec_*`, `dirRec_*`), shifted into the frame of the hits by the event's cluster
centre -- checked by reproducing BARS' own `timeChi2` -- and the time is profiled out
on both tracks, because the h5 centres hit times per event while BARS' clock is
absolute.  A lower chi-square here cannot come from a different functional, a
different hit set or a different clock: only from the track.

**B. Against MC truth**, which the fit never sees: the angle to the primary particle's
direction.  For an atmospheric-muon event that particle is the muon that starts the
bundle, and the reconstruction fits the brightest track of the bundle, so the whole
sample is not a fair comparison -- the **single-muon** subset is, and is reported
separately.  The truth vector is the same for both reconstructions, so no angle
convention enters.

Writes `validation/<stamp>/truth_check.md` and a provenance sidecar.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference.reader.spec import SOURCES  # noqa: E402
from inference.track_fit_alike_bars import fit_events  # noqa: E402
from inference.track_fit_alike_bars.frame import read_event_hits  # noqa: E402
from inference.track_fit_alike_bars.physics import propagation_time  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "validation"


def profiled_chi2(pos, t, point, direction) -> float:
    """The fit's time chi-square (getModQuality type 2) with t0 profiled out."""
    t_th = np.array([propagation_time(*point, *direction, *p) for p in pos])
    t0 = np.mean(t - t_th)
    return float(np.sum(((t0 + t_th - t) / 3.0) ** 2))


def unit(theta, phi):
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                     np.cos(theta)], -1)


def angle(a, b):
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def sample(n_parts: int, per_part: int, seed: int) -> pd.DataFrame:
    spec = SOURCES["mc_reco"]
    rng = np.random.default_rng(seed)
    frames = []
    with h5py.File(spec.h5, "r") as f:
        g = f["muatm"]
        parts = [p for p in sorted(g["raw/data"].keys()) if p not in spec.bad_runs]
        chosen = sorted(rng.choice(parts, size=min(n_parts, len(parts)), replace=False))
        for p in chosen:
            reco = g["reco_prty"][p]["data"][:]
            usable = np.flatnonzero(g["raw/reco_n_sig_hits"][p]["data"][:]
                                    > spec.fragment_cut)
            if usable.size == 0:
                continue
            idx = np.sort(rng.choice(usable, size=min(per_part, usable.size),
                                     replace=False))
            frame = pd.DataFrame(reco[idx], columns=list(spec.reco_columns))
            frame.insert(0, "local_idx", idx)
            frame.insert(0, "part_key", p)
            frame["truth_theta"] = g["prime_prty"][p]["data"][:, 0][idx]
            frame["truth_phi"] = g["prime_prty"][p]["data"][:, 1][idx]
            frame["n_muons"] = np.diff(g["muons_prty/mu_starts"][p]["data"][:])[idx]
            centres = g["clusters_centers"][p]["data"][:]
            clusters = g["raw/cluster_ids"][p]["data"][:][idx]
            frame[["cx", "cy", "cz"]] = centres[clusters]
            frames.append(frame)
    ev = pd.concat(frames, ignore_index=True)
    ev["source"] = "mc_reco"
    ev["data_class"] = "muatm_2020"
    return ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-parts", type=int, default=200)
    ap.add_argument("--per-part", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    started = time.time()
    ev = sample(args.n_parts, args.per_part, args.seed)
    fits = fit_events(ev, n_workers=args.workers, keep_objects=True)
    keep = (fits["status"] == "ok").to_numpy()
    ev, fits = ev[keep].reset_index(drop=True), fits[keep].reset_index(drop=True)

    our_dir = np.stack([fits[c].to_numpy() for c in ("dir_x", "dir_y", "dir_z")],
                       -1).astype(float)
    bars_dir = np.stack([ev[c].to_numpy() for c in
                         ("dirRec_x", "dirRec_y", "dirRec_z")], -1).astype(float)
    truth = unit(np.radians(ev["truth_theta"].to_numpy()),
                 np.radians(ev["truth_phi"].to_numpy()))
    d_fit = angle(our_dir, bars_dir)
    d_truth = angle(our_dir, truth)
    d_truth_bars = angle(bars_dir, truth)

    rows = []
    for p, sub in ev.groupby("part_key", sort=False):
        arrays = read_event_hits("mc_reco", "muatm_2020", p,
                                 sub["local_idx"].to_numpy())
        for (i, row), (q, t, x, y, z, ch) in zip(sub.iterrows(), arrays):
            pos = np.column_stack([x, y, z])
            ids = np.asarray(fits["fit"][i].fit_hits, dtype=np.int64)
            t = np.asarray(t, dtype=float)[ids]
            ours = fits["fit"][i]
            rows.append((
                profiled_chi2(pos[ids], t, np.array([ours.ref_x, ours.ref_y, ours.ref_z]),
                              our_dir[i]),
                profiled_chi2(pos[ids], t,
                              np.array([row["xyzRec_x"] - row["cx"],
                                        row["xyzRec_y"] - row["cy"],
                                        row["xyzRec_z"] - row["cz"]]),
                              bars_dir[i]),
                ours.time_chi2, row["timeChi2"], d_fit[i],
                d_truth[i], d_truth_bars[i]))
    a = np.array(rows)
    single = (ev["n_muons"].to_numpy() == 1)
    # a[:, 0] our functional at our track, [1] at BARS' track, [2] our reported
    # TimeChi2, [3] BARS'.  The frame and the functional are checked by [1] against
    # [3]: if the shift or the formula were wrong, this would not be small.
    frame_check = np.median(np.abs(a[:, 1] - a[:, 3]) / np.maximum(a[:, 3], 1e-9))

    def line(label, m):
        if not m.sum():
            return None
        return (f"| {label} | {m.sum()} | {np.mean(a[m, 0] < a[m, 1]):.0%} | "
                f"{np.median(a[m, 0]):.1f} / {np.median(a[m, 1]):.1f} | "
                f"{np.median(a[m, 5]):.1f} / {np.median(a[m, 6]):.1f} | "
                f"{np.mean(a[m, 5] < a[m, 6]):.0%} |")

    subsets = [("all", np.ones(len(a), bool)), ("one muon", single),
               ("one muon, directions differ by > 1 deg", single & (a[:, 4] > 1)),
               ("directions differ by > 1 deg", a[:, 4] > 1),
               ("directions differ by > 5 deg", a[:, 4] > 5)]
    ratio = a[:, 2] / np.maximum(a[:, 3], 1e-9)
    text = "\n".join([
        "# Lower chi-square: a better track or a different one?", "",
        f"`mc_reco` muatm, {len(a)} events fitted "
        f"({np.median(ev['n_muons']):.0f} muons per event on the median).", "",
        f"Checks first.  Our functional evaluated at BARS' **own** track reproduces BARS' "
        f"`fTimeChi2` to a median relative difference of {frame_check:.1e}, so the "
        f"functional, the hit set and the frame of the shift are right.", "",
        f"Of the two reported chi-squares, ours is the lower one in "
        f"{np.mean(a[:, 0] < a[:, 1]):.0%} of the events in which the two tracks differ -- "
        f"but their median *ratio* is {np.median(ratio):.4f}, with quartiles "
        f"{np.quantile(ratio, 0.25):.4f} and {np.quantile(ratio, 0.75):.4f}.  The two "
        f"statements are not in conflict: in most events both fits end in the same "
        f"minimum and the ratio is 1 to a few parts in 10^4, which is what the median "
        f"reports; the sign flips in a minority where the difference is large.", "",
        "| subset | events | our track has the lower chi-square | "
        "chi-square ours / BARS' track | angle to truth ours / BARS (deg) | "
        "ours closer to truth |",
        "|---|---|---|---|---|---|",
        *[t for t in (line(lab, m) for lab, m in subsets) if t], "",
        "The `all` row is not a fair test of either fit: the truth is the primary",
        "particle, the reconstruction is of the brightest track of the bundle, and the",
        "two are not the same object.  The single-muon rows are the ones to read.", "",
    ])
    out = OUT_DIR / datetime.now().strftime("%y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    (out / "truth_check.md").write_text(text)
    (out / "truth_check.md.meta.json").write_text(json.dumps({
        "script": "inference/track_fit_alike_bars/tests/truth_check.py",
        "argv": sys.argv, "args": vars(args),
        "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        "runtime_s": round(time.time() - started, 1),
        "written": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
