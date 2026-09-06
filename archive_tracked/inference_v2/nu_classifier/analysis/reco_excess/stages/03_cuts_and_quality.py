"""Stage 03 -- what the recommended reco cuts actually do, and the known data faults.

Two jobs.

**Quantify the cuts.**  The recommended selection is a list of thirteen
conditions on ``BRecoMuon`` quantities.  Applying them blind is unsafe: they were
tuned for one purpose and this analysis compares two samples, so a cut that keeps
90% of one and 10% of the other reshapes the comparison rather than cleaning it.
Each condition is therefore measured alone, and then cumulatively, on both
samples.

**Check the faults the prefilter work found.**  Three are on record in
``inference/prefilter_model/report.ipynb`` and the current
``test_exp_reco_mc_reco/plots_from_preds.ipynb``:

* *cluster 1 of exp_reco is excluded* (``BAD_RECO_PARTS``).  The project memory
  records the cluster-1 exclusion as **refuted** -- but that check was made
  against ``exp_full.h5``, and the same note says the claim "probably came from
  exp_reco".  A different file; the refutation does not transfer, so cluster 1 is
  measured here rather than assumed either way.
* *charges above 10^4 p.e.* ("Bad Qmax Found!") are dropped.
* *multi-cluster artefacts in mc_reco*: an event spanning several clusters is
  stored once per cluster, and each fragment carries the reconstruction of the
  whole event while its hits are only a piece.  The prefilter work removes them
  with ``n_signal_hits_gt > 5``.

Units are not uniform: ``thetaRec``/``phiRec``/``thetaErr``/``phiErr`` are in
degrees, ``scf*`` in radians.  ``thetaRec < 80`` therefore selects up-going
events (90 deg is the horizon), while ``scfMaxTheta < 1.7`` is a radian cut.

Usage:
    python stages/03_cuts_and_quality.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import provenance                                              # noqa: E402

LOG = logging.getLogger("stage03")

# The recommended set, as given.  Two are flagged in the source as uncertain
# mappings from the slides; the code in the prefilter notebooks additionally
# applies the funcValue cut at 2.0 rather than the documented 1.1, and differs
# between notebooks on nStrings (>=2 in the report, >=3 in the current work).
CUTS: list[tuple[str, str, str]] = [
    ("thetaRec < 80",                    "up-going, degrees (90 = horizon)", ""),
    ("nHits >= 8",                       "reconstruction hit count", ""),
    ("nStrings >= 2",                    "reconstruction string count",
                                         "the current notebook uses >= 3"),
    ("covMatrixStatus == 3",             "fit covariance converged", ""),
    ("scfMaxTheta < 1.7",                "radians, not degrees", ""),
    ("log10(pHit) > -9",                 "hit-probability term",
                                         "flagged questionable in the source"),
    ("evCenterZ_rel < 220",              "event centre depth, relative to the "
                                         "cluster centre",
                                         "the raw field is in absolute coordinates "
                                         "whose origin differs between the files"),
    ("zDist > 70",                       "vertical extent", ""),
    ("pathLength > 50",                  "track length in the array", ""),
    ("log10(funcValue / (nHits - 5)) < 1.1", "reduced fit functional",
                                         "flagged questionable; code applies 2.0"),
    ("nTriplets / nHits > 0.1",          "fraction of hits in triplets", ""),
    ("thetaErr < 2.5",                   "zenith error, degrees", ""),
    ("nCalls < 350",                     "minimiser calls", ""),
]


def evaluate(frame: pd.DataFrame, expression: str) -> np.ndarray:
    """A cut as a boolean array; events with a missing input do not pass."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = frame.eval(expression, engine="python").to_numpy()
    return np.nan_to_num(mask, nan=False).astype(bool)


def cut_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pass rates per cut.  The MC side must be split by class: three of the four
    mc_reco classes are neutrinos, which really are up-going, so comparing
    experiment against the whole file makes `thetaRec` and `scfMaxTheta` look ten
    and twenty times asymmetric when against muons alone they are 1.3 and 1.5."""
    rows = []
    running = {name: np.ones(len(f), dtype=bool) for name, f in frames.items()}
    for expression, meaning, caveat in CUTS:
        row = {"cut": expression, "means": meaning, "caveat": caveat}
        for name, frame in frames.items():
            alone = evaluate(frame, expression)
            running[name] &= alone
            row[f"{name}_alone"] = float(alone.mean())
            row[f"{name}_cumulative"] = float(running[name].mean())
        rows.append(row)
        LOG.info("%-38s exp %.4f  muatm %.4f  ratio %.2f | cumulative exp %.5f "
                 "muatm %.5f", expression, row["exp_reco_alone"],
                 row["mc_muatm_alone"],
                 row["mc_muatm_alone"] / max(row["exp_reco_alone"], 1e-9),
                 row["exp_reco_cumulative"], row["mc_muatm_cumulative"])
    return pd.DataFrame(rows)


def quality_report(exp: pd.DataFrame, mc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, group in exp.groupby("cluster"):
        scored = group[group.score.notna()]
        rows.append({"check": "exp_reco by cluster", "key": str(cluster),
                     "events": len(group), "scored": len(scored),
                     "median_max_q": float(group.max_q.median()),
                     "frac_max_q_above_1e4": float((group.max_q > 1e4).mean()),
                     "median_score": float(scored.score.median()) if len(scored) else np.nan,
                     "frac_score_above_0.5": float((scored.score > 0.5).mean())
                                              if len(scored) else np.nan})
    for name, frame in (("exp_reco", exp), ("mc_reco", mc)):
        rows.append({"check": "max charge", "key": name, "events": len(frame),
                     "scored": int(frame.score.notna().sum()),
                     "median_max_q": float(frame.max_q.median()),
                     "frac_max_q_above_1e4": float((frame.max_q > 1e4).mean()),
                     "median_score": float(frame.score.median()),
                     "frac_score_above_0.5": float((frame.score > 0.5).mean())})
    if "n_gt_sig_hits" in mc:
        for threshold in (0, 5):
            keep = mc.n_gt_sig_hits > threshold
            rows.append({"check": "mc_reco multi-cluster fragments",
                         "key": f"n_gt_sig_hits > {threshold}",
                         "events": int(keep.sum()), "scored": int((keep & mc.score.notna()).sum()),
                         "median_max_q": float(mc.max_q[keep].median()),
                         "frac_max_q_above_1e4": float((mc.max_q[keep] > 1e4).mean()),
                         "median_score": float(mc.score[keep].median()),
                         "frac_score_above_0.5": float((mc.score[keep] > 0.5).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    exp = pd.read_parquet(HERE / "data" / "02_exp_reco_events.parquet")
    mc = pd.read_parquet(HERE / "data" / "02_mc_reco_events.parquet")
    LOG.info("exp_reco %d events, mc_reco %d events", len(exp), len(mc))

    cuts = cut_table({"exp_reco": exp,
                      "mc_muatm": mc[mc.group == "muatm"],
                      "mc_nuatm_conv": mc[mc.group == "nuatm_conv"],
                      "mc_nue2": mc[mc.group == "nue2"],
                      "mc_all": mc})
    quality = quality_report(exp, mc)
    LOG.info("\n%s", quality.to_string(index=False))

    for frame, name in ((cuts, "03_cut_rates"), (quality, "03_quality_checks")):
        provenance.write(frame, HERE / "data" / f"{name}.parquet",
                         stage="03_cuts_and_quality", config_path=HERE / "config.yaml",
                         inputs=[HERE / "data" / "02_exp_reco_events.parquet"],
                         started=started)
    LOG.info("stage 03 done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
