"""Compare the port with BARS' own reconstruction of the same pulses.

    python -m inference.track_fit_alike_bars.validate_vs_bars \\
        --source exp_reco --n-parts 200 --per-part 10 --workers 8
    python -m inference.track_fit_alike_bars.validate_vs_bars \\
        --source mc_reco --data-class muatm_2020 --n-parts 100 --per-part 10

The reco HDF5 files store, per event, the BEvent pulses BARS reconstructed, the
BRecoMuon it wrote (`reco_prty`) and the ScanfitMask (`raw/labels` for `exp_reco`,
`raw/reco_mask` for `mc_reco`).  So the reference is external to this code: BARS
itself, on identical input.  The BARS version that produced the files is not
recorded in them.

For `exp_reco` the port is run twice, without and with the DQM channel mask that
BARS applied (`Mask.UseDQMChannelMask 1` for data), read per event from the run's
ROOT file.  `mc_reco` has no mask to apply -- BARS reconstructs MC with
`UseDQMChannelMask 0` -- which makes it the control: if the agreement is as good
there as on masked data, the data/mask story is closed.

Writes `validation/<stamp>/` with the per-event table, a provenance sidecar and
`summary.md`.  Cluster 1 and the bad runs of `exp_reco` are excluded
(`inference.reader.spec`).
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
import awkward as ak
import uproot

from inference.reader.spec import SOURCES, h5_group

from . import geometry, isolation, masks
from .config import FitConfig
from .frame import fit_events, read_event_hits

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).with_name("validation")

#: The analysis tree's preliminary cuts that both sides can evaluate.
CUTS = {
    "cov_status==3": ("cov_matrix_status", "covMatrixStatus", lambda s: s == 3),
    "scf_max_theta<1.7": ("scf_max_theta", "scfMaxTheta", lambda s: s < 1.7),
    "z_dist>70": ("z_dist", "zDist", lambda s: s > 70),
    "theta_err<2.5": ("theta_err", "thetaErr", lambda s: s < 2.5),
    "n_calls<350": ("n_calls", "nCalls", lambda s: s < 350),
    "n_hits>=8": ("n_hits", "nHits", lambda s: s >= 8),
    "theta_rec<80": ("theta_rec", "thetaRec", lambda s: s < 80),
}

#: Where the ScanfitMask is stored, which the converter decided per source.
SCANFIT_MASK = {"exp_reco": "raw/labels", "mc_reco": "raw/reco_mask"}

#: BARS' own isolation variable, in the reco ROOT file next to the pulses.  It is
#: not among the 25 `reco_prty` columns the HDF5 converter keeps, so the ROOT file
#: is the only source -- and it is the reference for our `pnonhit`: the same
#: perpendicular light model, the same 22 m window, "empty" = unused by the final
#: track.  Our other two differences (the light-path distance, and "empty" =
#: not-fired) are variations, and are not comparable to this number.
BARS_PNONHIT = "BRecoMuon./BRecoMuon.fPnonHit"
#: BARS' final track, the one `fPnonHit` belongs to (BMuonTrack).
BARS_TRACK = ("BRecoMuon./BRecoMuon.fMuonTrack/BRecoMuon.fMuonTrack.fXYZ")


def excluded(spec, part_key: str) -> bool:
    """A part the source spec refuses, by name or by cluster.

    Part keys differ between sources (`part_s2020_c04_r0100` against
    `part_2020_cl1_run10000_scl_nu_MC_s19-21`), so the cluster is only read when
    the key has one.
    """
    if part_key in spec.bad_runs:
        return True
    try:
        cluster = masks.part_run(part_key)[1]
    except ValueError:
        return False
    return cluster in spec.bad_clusters


def sample_events(source: str, data_class: str | None, n_parts: int, per_part: int,
                  seed: int) -> pd.DataFrame:
    """Random events of random parts, with whatever the fit needs to find them."""
    spec = SOURCES[source]
    rng = np.random.default_rng(seed)
    # `data_class` is the catalogue's name; the HDF5 group may be spelled differently
    group = h5_group(source, data_class or spec.groups[0])
    with h5py.File(spec.h5, "r") as f:
        g = f[group]
        parts = sorted(g[SCANFIT_MASK[source]].keys())
        parts = [p for p in parts if not excluded(spec, p)]
        chosen = sorted(rng.choice(parts, size=min(n_parts, len(parts)),
                                   replace=False))
        frames = []
        for p in chosen:
            reco = g["reco_prty"][p]["data"][:]
            usable = np.arange(len(reco))
            if spec.fragment_cut is not None:
                # A multi-cluster event is stored once per cluster, each row
                # carrying the reconstruction of the WHOLE event beside the hits of
                # one piece; those pieces are not events BARS reconstructed as such.
                n_sig = g["raw/reco_n_sig_hits"][p]["data"][:]
                usable = np.flatnonzero(n_sig > spec.fragment_cut)
            if usable.size == 0:
                continue
            idx = np.sort(rng.choice(usable, size=min(per_part, usable.size),
                                     replace=False))
            fr = pd.DataFrame(reco[idx], columns=list(spec.reco_columns))
            fr.insert(0, "local_idx", idx)
            fr.insert(0, "part_key", p)
            if spec.has_header:
                fr["event_id"] = g["header_prty"][p]["data"][idx, 3]
            if "clusters_centers" in g:
                fr["cluster_center_z"] = float(g["clusters_centers"][p]["data"][0, 2])
            frames.append(fr)
    ev = pd.concat(frames, ignore_index=True)
    ev.insert(0, "source", source)
    ev["data_class"] = group
    return ev


def dqm_provider(source: str):
    """Callable for `fit_events`: the DQM mask BARS used, one per event.

    None for a source that has none.  The mask of a part is read once and reused;
    an event whose id the ROOT file does not have raises rather than silently
    fitting without a mask.
    """
    if source != "exp_reco":
        return None
    cache: dict[str, dict[int, np.ndarray]] = {}

    def provider(rows: pd.DataFrame) -> np.ndarray:
        part = str(rows["part_key"].iloc[0])
        if part not in cache:
            cache[part] = masks.dqm_masks(source, part, rows["event_id"].to_numpy())
        found = cache[part]
        ids = [int(i) for i in rows["event_id"]]
        missing = [i for i in ids if i not in found]
        if missing:
            raise KeyError(f"{part}: no ChannelMaskDQM for event ids {missing[:5]}")
        return np.array([found[i] for i in ids])

    return provider


def bars_scanfit_masks(ev: pd.DataFrame, source: str) -> list[np.ndarray]:
    """ScanfitMask of each event, in HDF5 hit order (= the order fits report in)."""
    out: list = [None] * len(ev)
    with h5py.File(SOURCES[source].h5, "r") as f:
        g = f[h5_group(source, ev["data_class"].iloc[0])]
        for p, rows in ev.groupby("part_key", sort=False):
            starts = g["raw/ev_starts"][p]["data"][:].astype(np.int64)
            lo = starts[rows["local_idx"].min()]
            hi = starts[rows["local_idx"].max() + 1]
            block = g[SCANFIT_MASK[source]][p]["data"][lo:hi]
            for row, i in zip(rows.index, rows["local_idx"]):
                out[row] = block[starts[i] - lo:starts[i + 1] - lo] != 0
    return out


def bars_pnonhit(ev: pd.DataFrame, source: str):
    """`BRecoMuon.fPnonHit` per event, and the track it was computed at.

    Returns `(pnonhit, ref, dir)`: the reference point and direction of BARS' own
    `fMuonTrack` (absolute coordinates, as BARS works in), so the same variable
    can be recomputed at BARS' own track -- which is what separates a disagreement
    of this tool from a disagreement of the fit under it.
    """
    out = np.full(len(ev), np.nan)
    ref = np.full((len(ev), 3), np.nan)
    direction = np.full((len(ev), 3), np.nan)
    for part, rows in ev.groupby("part_key", sort=False):
        with uproot.open(masks.reco_root_path(part)) as f:
            tree = f["Events"]
            ids = tree[masks.EVENT_ID_BRANCH].array(library="np")
            entry_of = {int(v): i for i, v in enumerate(ids)}
            values = tree[BARS_PNONHIT].array(library="np")
            pos = np.asarray(ak.unzip(tree[BARS_TRACK + ".pos"].array(library="ak")),
                             dtype=float).T
            dirs = np.asarray(ak.unzip(tree[BARS_TRACK + ".dir"].array(library="ak")),
                              dtype=float).T
            for i, event_id in zip(rows.index, rows["event_id"]):
                entry = entry_of.get(int(event_id))
                if entry is not None:
                    out[i] = float(values[entry])
                    ref[i] = pos[entry]
                    direction[i] = dirs[entry]
    return out, ref, direction


def our_pnonhit(ev: pd.DataFrame, fits: pd.DataFrame, source: str,
                *, use_mask: bool, track=None) -> np.ndarray:
    """Our `pnonhit` in the strict BARS reading, for the same events.

    The strict reading is what makes the comparison meaningful: the perpendicular
    distance of the legacy fit path, the 22 m window, and "empty" = a channel the
    final track did not use (its used channels are `fit_hits` mapped back onto the
    event's channels).  The DQM mask is the per-event one BARS used, when the run
    under comparison was fitted with it.

    `track` replaces the track of `fits` -- `(ref, dir)` in absolute coordinates,
    as `bars_pnonhit` returns -- while the used channels stay ours.  Recomputing at
    BARS' own track is what tells the tool's disagreement from the fit's.
    """
    out = np.full(len(ev), np.nan)
    for part, rows in ev.groupby("part_key", sort=False):
        order = np.argsort(rows["local_idx"].to_numpy())
        local = rows["local_idx"].to_numpy()[order]
        geometry.prefill(source, part, [int(v) for v in local])
        pulses = read_event_hits(source, str(rows["data_class"].iloc[0]), part,
                                 np.sort(local))
        channels_of = {int(k): np.asarray(p[5], dtype=np.int64)
                       for k, p in zip(np.sort(local), pulses)}
        event_mask = (masks.dqm_masks(source, part, list(rows["event_id"]))
                      if use_mask else {})
        centre = geometry.centre_of(source, part)
        for i, local_idx, event_id in zip(rows.index, rows["local_idx"],
                                          rows["event_id"]):
            fit = fits.loc[i, "fit"]
            if fit.status != "ok":
                continue
            if track is not None and np.isfinite(track[0][i]).all():
                line = dict(zip(isolation.TRACK_FIELDS,
                                [*(track[0][i] - centre), *track[1][i]]))
            else:
                line = fit
            om = geometry.oms(source, part, local_idx=int(local_idx))
            used = channels_of[int(local_idx)][np.asarray(fit.fit_hits, np.int64)]
            iso = isolation.pnonhit(line, om, fired=used, distance="perpendicular",
                                    enabled=event_mask.get(int(event_id)))
            out[i] = iso.pnonhit
    return out


def unit(theta_deg, phi_deg):
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                     np.cos(th)], -1)


def compare(ev: pd.DataFrame, fits: pd.DataFrame, bars_masks) -> pd.DataFrame:
    out = ev.copy()
    for c in fits.columns:
        if c != "fit":
            out[f"our_{c}"] = fits[c].to_numpy()
    ours_mask = [f.scanfit_mask for f in fits["fit"]]
    out["scanfit_mask_equal"] = [
        m is not None and len(m) == len(b) and np.array_equal(m, b)
        for m, b in zip(ours_mask, bars_masks)]
    out["scanfit_jaccard"] = [
        (np.sum(m & b) / max(np.sum(m | b), 1)) if m is not None else 0.0
        for m, b in zip(ours_mask, bars_masks)]
    cos = np.einsum("ij,ij->i", unit(out["our_theta_rec"], out["our_phi_rec"]),
                    unit(out["thetaRec"], out["phiRec"]))
    out["angle_diff_deg"] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    if "cluster_center_z" in out:
        out["our_ev_center_z_abs"] = out["our_ev_center_z"] + out["cluster_center_z"]
    return out


def summarise(cmp: pd.DataFrame, label: str) -> str:
    ok = cmp["our_status"] == "ok"
    c = cmp[ok]
    lines = [f"### {label}", "",
             f"events: {len(cmp)}, fitted: {ok.sum()} "
             f"({ok.mean():.1%}); statuses: "
             + ", ".join(f"{k} {v}" for k, v in
                         cmp["our_status"].value_counts().items()), ""]

    def q(s):
        return " / ".join(f"{s.quantile(p):.3g}" for p in (0.5, 0.9, 0.99))

    lines += ["| quantity | agreement |", "|---|---|"]
    ad = c["angle_diff_deg"]
    lines.append(f"| direction, angle to BARS (deg), median/90%/99% | {q(ad)} |")
    for lim in (0.1, 1.0, 5.0):
        lines.append(f"| direction within {lim} deg | {(ad < lim).mean():.1%} |")
    lines.append(f"| scanfit mask identical | {c['scanfit_mask_equal'].mean():.1%} |")
    lines.append(f"| scanfit mask Jaccard, median | "
                 f"{c['scanfit_jaccard'].median():.3f} |")
    for ours, bars in [("n_hits", "nHits"), ("n_strings", "nStrings"),
                       ("n_oms", "nOMs"), ("n_triplets", "nTriplets"),
                       ("cov_matrix_status", "covMatrixStatus")]:
        lines.append(f"| {ours} equal | {(c[f'our_{ours}'] == c[bars]).mean():.1%} |")
    lines.append(f"| scf_max_theta equal (1e-6) | "
                 f"{(np.abs(c['our_scf_max_theta'] - c['scfMaxTheta']) < 1e-6).mean():.1%} |")
    for ours, bars in [("time_chi2", "timeChi2"), ("func_value", "funcValue"),
                       ("theta_err", "thetaErr"), ("z_dist", "zDist"),
                       ("p_hit", "pHit")]:
        rel = (np.abs(c[f"our_{ours}"] - c[bars])
               / np.maximum(np.abs(c[bars]), 1e-9))
        lines.append(f"| {ours} relative difference, median/90%/99% | {q(rel)} |")
    lines.append(f"| charge_term, ours median / BARS median | "
                 f"{c['our_charge_term'].median():.3g} / {c['chargeTerm'].median():.3g} |")
    if "our_pnonhit" in c:
        # BARS returns 1e-9 as a sentinel when nothing is silent in the window;
        # ours is NaN there, so those events are counted, not compared
        sentinel = c["bars_pnonhit"] <= 1e-8
        both = ~sentinel & np.isfinite(c["our_pnonhit"]) & np.isfinite(c["bars_pnonhit"])
        rel = (np.abs(c.loc[both, "our_pnonhit"] - c.loc[both, "bars_pnonhit"])
               / np.maximum(np.abs(c.loc[both, "bars_pnonhit"]), 1e-12))
        lines.append(f"| pnonhit: compared / BARS sentinel / ours NaN | "
                     f"{int(both.sum())} / {int(sentinel.sum())} / "
                     f"{int((~sentinel & ~np.isfinite(c['our_pnonhit'])).sum())} |")
        if both.any():
            lines.append(f"| pnonhit relative difference at our track, "
                         f"median/90%/99% | {q(rel)} |")
            if "our_pnonhit_at_bars" in c:
                atlas = np.isfinite(c.loc[both, "our_pnonhit_at_bars"])
                rel_b = (np.abs(c.loc[both, "our_pnonhit_at_bars"]
                                - c.loc[both, "bars_pnonhit"])
                         / np.maximum(np.abs(c.loc[both, "bars_pnonhit"]),
                                      1e-12))[atlas]
                lines.append(f"| pnonhit relative difference at BARS' track, "
                             f"median/90%/99% | {q(rel_b)} |")
                lines.append(f"| ... within 1e-6 relative | "
                             f"{(rel_b < 1e-6).mean():.1%} |")
    d_ncalls = np.abs(c["our_n_calls"] - c["nCalls"])
    lines.append(f"| n_calls absolute difference, median/90%/99% | {q(d_ncalls)} |")
    for column, what in (("our_ev_center_z", "in hit coordinates"),
                         ("our_ev_center_z_abs", "plus the cluster centre")):
        if column in c:
            dz = np.abs(c[column] - c["evCenterZ"])
            lines.append(f"| ev_center_z {what} vs BARS (m), median/90%/99% | "
                         f"{q(dz)} |")

    lines += ["", "Preliminary cuts, pass rate and per-event agreement:", "",
              "| cut | pass ours | pass BARS | same decision |", "|---|---|---|---|"]
    for name, (ours, bars, rule) in CUTS.items():
        a, b = rule(c[f"our_{ours}"]), rule(c[bars])
        lines.append(f"| {name} | {a.mean():.1%} | {b.mean():.1%} | "
                     f"{(a == b).mean():.1%} |")

    conf = pd.crosstab(c["covMatrixStatus"].astype(int),
                       c["our_cov_matrix_status"].astype(int),
                       rownames=["BARS"], colnames=["ours"])
    lines += ["", "covMatrixStatus, rows BARS, columns ours:", "", "```",
              conf.to_string(), "```"]
    return "\n".join(lines)


def provenance(args, n_rows: int, started: float) -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True,
                              text=True).stdout.strip()
    h5 = SOURCES[args.source].h5
    return {
        "script": "inference/track_fit_alike_bars/validate_vs_bars.py",
        "argv": sys.argv, "args": vars(args),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain",
                              "inference/track_fit_alike_bars")),
        "config": FitConfig().as_dict(),
        "input": {"path": str(h5), "size": h5.stat().st_size,
                  "mtime": h5.stat().st_mtime},
        "rows": n_rows, "runtime_s": round(time.time() - started, 1),
        "written": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="exp_reco", choices=sorted(SOURCES))
    ap.add_argument("--data-class", default=None,
                    help="HDF5 group; default is the source's first")
    ap.add_argument("--n-parts", type=int, default=40)
    ap.add_argument("--per-part", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    started = time.time()
    ev = sample_events(args.source, args.data_class, args.n_parts, args.per_part,
                       args.seed)
    bars = bars_scanfit_masks(ev, args.source)
    runs = [("without channel mask", None)]
    provider = dqm_provider(args.source)
    if provider is not None:
        runs.append(("with the DQM channel mask BARS used", provider))

    out = OUT_DIR / datetime.now().strftime("%y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    sections = []
    first = None
    for label, mask_arg in runs:
        fits = fit_events(ev, n_workers=args.workers, keep_objects=True,
                          channel_mask=mask_arg)
        cmp = compare(ev, fits, bars)
        bars_value, bars_ref, bars_dir = bars_pnonhit(ev, args.source)
        cmp["bars_pnonhit"] = bars_value
        cmp["our_pnonhit"] = our_pnonhit(ev, fits, args.source,
                                         use_mask=mask_arg is not None)
        cmp["our_pnonhit_at_bars"] = our_pnonhit(
            ev, fits, args.source, use_mask=mask_arg is not None,
            track=(bars_ref, bars_dir))
        key = "mask" if mask_arg is not None else "nomask"
        cmp.to_parquet(out / f"events_{key}.parquet", index=False)
        sections.append(summarise(cmp, f"{args.source} {label}"))
        first = cmp if first is None else first

    meta = provenance(args, len(first), started)
    (out / "events.parquet.meta.json").write_text(json.dumps(meta, indent=2))
    text = "\n\n".join(sections)
    (out / "summary.md").write_text(
        f"# Port vs BARS on {args.source}\n\nGenerated by `{meta['script']}` "
        f"({' '.join(sys.argv[1:])}), commit {meta['git_commit'][:10]}"
        f"{' (dirty)' if meta['git_dirty'] else ''}, {meta['written']}.\n\n"
        + text + "\n")
    print(text)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
