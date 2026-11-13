"""Stage-by-stage comparison with BRecoMuon, on the ROOT file BARS wrote.

    python inference/track_fit_alike_bars/tests/stage_diagnostics.py
    python inference/track_fit_alike_bars/tests/stage_diagnostics.py 200 0 --show
    python inference/track_fit_alike_bars/tests/stage_diagnostics.py --no-mask

`validate_vs_bars.py` compares the final answer through the HDF5 files; this
compares the intermediate ones, on the same pulses BARS read -- the ROOT file
already holds them in BARS' own order and with absolute times, and it holds the
DQM channel mask (`ChannelMaskDQM`), so the input can be reproduced exactly.

Stages, in the order the chain reaches them:

| field | written by |
|---|---|
| `ScfTheta`, `FuncValueScf`, `TimeChi2Scf` | the scanfit and trial 0 |
| `NHitsInit`, `ThetaInit`, `PhiInit`, `TimeChi2Init` | the best trial before the exclusive fit |
| `NHits`, `ThetaRec`, `TimeChi2`, `CovMatrixStatus`, `NCalls` | the final fit |

Exit status is 0 unless a stage that must be identical is not (the scanfit mask
and the scanfit fields); the rest are reported as rates, because the minimiser is
a different implementation and cannot be expected to agree call for call.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference.track_fit_alike_bars import fit_event  # noqa: E402
from inference.track_fit_alike_bars import masks  # noqa: E402

PULSES = "BEvent./BEvent.fPulses/BEvent.fPulses."
GEOMETRY = ("BGeomTel./BGeomTel.BGeomTel/BGeomTel.BGeomTel.fOMs/"
            "BGeomTel.BGeomTel.fOMs.fPosition")
DEFAULT_FILE = ("data_manager/data/exp_reco_root/exp_reco_files/"
                "2020_cl4_run100_scl_nu_DATA2020.root")

#: (`TrackFit` field, `BRecoMuon` field): the final fit
FINAL = [("n_hits", "fNHits"), ("theta_rec", "fThetaRec"), ("phi_rec", "fPhiRec"),
         ("time_chi2", "fTimeChi2"), ("func_value", "fFuncValue"),
         ("charge_term", "fChargeTerm"), ("cov_matrix_status", "fCovMatrixStatus"),
         ("min_status", "fMinStatus"), ("n_calls", "fNCalls"),
         ("theta_err", "fThetaErr"), ("z_dist", "fZDist")]
#: the best trial before the exclusive fit and the final fit
INIT = [("n_hits_init", "fNHitsInit"), ("theta_init", "fThetaInit"),
        ("phi_init", "fPhiInit"), ("time_chi2_init", "fTimeChi2Init"),
        ("func_value_init", "fFuncValueInit")]
#: the scanfit, before any track exists
SCANFIT = [("scf_theta", "fScfTheta"), ("scf_phi", "fScfPhi"),
           ("func_value_scf", "fFuncValueScf"), ("time_chi2_scf", "fTimeChi2Scf")]

BRANCHES = [f[1] for f in FINAL + INIT + SCANFIT]


def close(a, b, rel=1e-3, ab=1e-6) -> bool:
    """Equal to within `rel`, but never asking for more than `ab` absolute."""
    return bool(abs(a - b) <= max(ab, rel * abs(b)))


def read(path: Path, n: int, start: int, with_mask: bool):
    tree = uproot.open(path)["Events"]
    window = dict(entry_start=start, entry_stop=start + n)
    # variable-length branches are read as awkward arrays, scalars as numpy
    q = tree[PULSES + "fAmplitude"].array(**window)
    t = tree[PULSES + "fTime"].array(**window)
    ch = tree[PULSES + "fChannelID"].array(**window)
    geom = np.asarray(ak.unzip(tree[GEOMETRY].array(**window)), dtype=np.float64)
    mask = tree["ScanfitMask./ScanfitMask.fOrigins/ScanfitMask.fOrigins.fFlag"] \
        .array(library="np", **window)
    reco = {name: tree[f"BRecoMuon./BRecoMuon.{name}"].array(library="np", **window)
            for name in BRANCHES}
    #: the pulses BARS kept, as positions in the event's own pulse list
    reco["impulses"] = tree["BRecoMuon./BRecoMuon.fImpulseNumber"].array(
        library="np", **window)
    channel_mask = masks.root_channel_masks(tree, **window) if with_mask else None
    return q, t, ch, geom, mask, reco, channel_mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n", nargs="?", type=int, default=100, help="events")
    ap.add_argument("start", nargs="?", type=int, default=0)
    ap.add_argument("--file", type=Path, default=ROOT / DEFAULT_FILE)
    ap.add_argument("--no-mask", action="store_true",
                    help="ignore ChannelMaskDQM (BARS applied it)")
    ap.add_argument("--show", action="store_true", help="print every event")
    args = ap.parse_args()

    q, t, ch, geom, scanfit, reco, channel_mask = read(
        args.file, args.n, args.start, not args.no_mask)
    print(f"{args.file.name}: {args.n} events from entry {args.start}, "
          f"{'with' if channel_mask is not None else 'without'} the DQM mask\n")

    rows = []
    for e in range(len(q)):
        ch_e = np.asarray(ch[e], dtype=np.int64)
        pos_e = geom[:, e, :][:, ch_e].T                  # (n_hits, 3)
        fit = fit_event(np.asarray(q[e], dtype=float),
                        np.asarray(t[e], dtype=float),
                        pos_e[:, 0], pos_e[:, 1], pos_e[:, 2], ch_e,
                        channel_mask=None if channel_mask is None
                        else channel_mask[e])
        bars = {name: reco[name][e] for name in BRANCHES}
        row = {"status": fit.status,
               "scanfit_equal": fit.scanfit_mask is not None and np.array_equal(
                   fit.scanfit_mask, np.asarray(scanfit[e]) != 0)}
        for ours, name in FINAL + INIT + SCANFIT:
            row[f"{ours}|{name}"] = (getattr(fit, ours), bars[name])
        n_bars = int(bars["fNHits"])
        row["hits_equal"] = fit.fit_hits is not None and set(
            fit.fit_hits.tolist()) == set(np.asarray(reco["impulses"][e])[:n_bars])
        rows.append(row)

    print(f"fitted ok: {sum(r['status'] == 'ok' for r in rows)}/{len(rows)}")
    print(f"scanfit mask identical: "
          f"{np.mean([r['scanfit_equal'] for r in rows]):.2f}")
    print(f"final hit set identical: "
          f"{np.mean([r['hits_equal'] for r in rows]):.2f}\n")
    #: Fields that are counts or states, where only equality means anything.  The
    #: rest are floats: minimiser-dependent ones (everything downstream of a fit)
    #: can differ in the last digits and still be right, so their spread is
    #: reported rather than a pass rate.
    integers = {"n_hits", "n_hits_init", "cov_matrix_status", "min_status", "n_calls"}
    width = max(len(o) for o, _ in FINAL + INIT + SCANFIT)
    for group, items in (("scanfit", SCANFIT), ("best trial", INIT),
                         ("final fit", FINAL)):
        print(group)
        for ours, name in items:
            a = np.array([r[f"{ours}|{name}"][0] for r in rows], dtype=float)
            b = np.array([r[f"{ours}|{name}"][1] for r in rows], dtype=float)
            good = np.isfinite(a) & np.isfinite(b)
            if ours in integers:
                print(f"  {ours:<{width}}  equal to {name:<16} "
                      f"{np.mean(a[good] == b[good]):.2f}")
            else:
                rel = np.abs(a[good] - b[good]) / np.maximum(np.abs(b[good]), 1e-12)
                print(f"  {ours:<{width}}  vs {name:<16} "
                      f"within 1e-3: {np.mean(rel < 1e-3):.2f}   "
                      f"median difference: {np.median(rel):.2e}")
    if args.show:
        print()
        for r in rows:
            print({k: (round(v[0], 3), round(v[1], 3)) if isinstance(v, tuple) else v
                   for k, v in r.items()})


if __name__ == "__main__":
    main()
