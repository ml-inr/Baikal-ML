"""Stage 04 -- score distributions on reconstructed events, and the excess.

The measurement: what fraction of experimental events falls in each band of the
classifier score, divided by the same fraction for simulated atmospheric muons.
One would mean the simulation describes the data.

Why this is worth doing separately from the `exp_full` / `mc_merged` measurement:
the reconstructed samples are selected by a *different* procedure -- BARS
reconstruction succeeded on them -- so an excess that survives the change of
selection is a property of the data rather than of one selection.

Background is ``muatm`` alone.  The other three MC classes are neutrinos and the
classifier accepts 99.4-99.9% of them; including them would put signal in the
denominator.

Three exclusions are applied, each measured in stage 03 rather than inherited:

* **cluster 1 of exp_reco.**  It accepts 6.9% of its h8s3 events where every
  other cluster accepts 0.94-1.18%, and only 30% of it survives the preselection
  at all.  The prefilter work excludes it; the project memory records a *refuted*
  cluster-1 exclusion, but that was checked against ``exp_full.h5`` and the same
  note says the claim probably came from ``exp_reco``.  Different file, and the
  measurement here is unambiguous.
* **max charge above 10^4 p.e.**, the "Bad Qmax" fault.
* **mc_reco fragments** with five or fewer true signal hits: a multi-cluster
  event is stored once per cluster and each fragment carries the reconstruction
  of the whole event beside the hits of one piece.

The recommended BARS quality cuts are reported as a separate variant rather than
folded into the headline number, because they cut the sample by a factor of a
thousand and the excess should first be quoted where the statistics are large.

An earlier version of this stage claimed they could not be applied at all, on the
grounds that they keep 17% of the simulation against 0.013% of the data.  That was
wrong twice over.  The 17% was measured over the whole of ``mc_reco``, three
quarters of which is neutrinos that really are up-going -- against ``muatm`` alone
``thetaRec < 80`` keeps 2.1% and ``scfMaxTheta < 1.7`` keeps 0.8%, against 2.7%
and 1.3% in the data.  And ``evCenterZ`` is written in absolute detector
coordinates whose origin differs between the files (361.7 m for the experimental
cluster, 0.4 m for the simulated one), so the raw threshold meant different things
in the two samples; referred to the cluster centre it keeps 93.0% and 98.4%.

With both corrections every one of the thirteen conditions falls between 0.64 and
1.14 of parity, and together they keep 0.105% of the data against 0.094% of the
muons.  The list is symmetric.

Usage:
    python stages/04_reco_excess.py
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

LOG = logging.getLogger("stage04")


def selected(frame: pd.DataFrame, cfg: dict, is_mc: bool) -> pd.DataFrame:
    """Scored, h8s3, and past the three measured exclusions."""
    settings = cfg["stage04"]
    keep = frame.score.notna() & (frame.n_sn_hits >= 8) & (frame.n_sn_strings >= 3)
    keep &= frame.max_q <= float(settings["max_charge"])
    if is_mc:
        if "n_gt_sig_hits" in frame:
            keep &= frame.n_gt_sig_hits > int(settings["min_gt_sig_hits"])
    else:
        keep &= ~frame.cluster.isin(settings["exclude_exp_clusters"])
    return frame[keep]


def bands(exp: pd.DataFrame, mc: pd.DataFrame, edges: list) -> pd.DataFrame:
    rows = []
    for lo, hi in edges:
        in_exp = ((exp.score >= lo) & (exp.score < hi))
        in_mc = ((mc.score >= lo) & (mc.score < hi))
        share_exp, share_mc = in_exp.mean(), in_mc.mean()
        rows.append({"score_lo": lo, "score_hi": min(hi, 1.0),
                     "exp_events": int(in_exp.sum()), "mc_events": int(in_mc.sum()),
                     "exp_share": float(share_exp), "mc_share": float(share_mc),
                     "excess": float(share_exp / share_mc) if share_mc > 0 else np.nan})
        LOG.info("band %.2f-%.2f: exp %8d (%.5f)  mc %7d (%.5f)  excess %.2f",
                 lo, min(hi, 1.0), rows[-1]["exp_events"], share_exp,
                 rows[-1]["mc_events"], share_mc, rows[-1]["excess"])
    return pd.DataFrame(rows)


def cumulative(exp: pd.DataFrame, mc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in (0.5, 0.8, 0.9, 0.99):
        share_exp = float((exp.score > threshold).mean())
        share_mc = float((mc.score > threshold).mean())
        rows.append({"threshold": threshold,
                     "exp_events": int((exp.score > threshold).sum()),
                     "mc_events": int((mc.score > threshold).sum()),
                     "exp_share": share_exp, "mc_share": share_mc,
                     "excess": share_exp / share_mc if share_mc > 0 else np.nan})
        LOG.info("xi > %.2f: exp %6d of %d, mc %5d of %d, excess %.2f", threshold,
                 rows[-1]["exp_events"], len(exp), rows[-1]["mc_events"], len(mc),
                 rows[-1]["excess"])
    return pd.DataFrame(rows)


def variants(exp_all: pd.DataFrame, mc_all: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The excess under each exclusion in turn, so its effect is visible."""
    from importlib import import_module
    sys.path.insert(0, str(HERE / "stages"))
    cuts = import_module("03_cuts_and_quality") if False else None
    rows = []

    def measure(label: str, exp: pd.DataFrame, mc: pd.DataFrame) -> None:
        for threshold in (0.5, 0.8):
            share_exp = float((exp.score > threshold).mean()) if len(exp) else np.nan
            share_mc = float((mc.score > threshold).mean()) if len(mc) else np.nan
            rows.append({"variant": label, "threshold": threshold,
                         "n_exp": len(exp), "n_mc": len(mc),
                         "excess": share_exp / share_mc if share_mc else np.nan})
        LOG.info("%-46s exp %9d mc %8d  excess@0.5 %.2f  @0.8 %.2f", label,
                 len(exp), len(mc), rows[-2]["excess"], rows[-1]["excess"])

    base_exp = exp_all[exp_all.score.notna() & (exp_all.n_sn_hits >= 8)
                       & (exp_all.n_sn_strings >= 3)]
    base_mc = mc_all[mc_all.score.notna() & (mc_all.n_sn_hits >= 8)
                     & (mc_all.n_sn_strings >= 3)]
    measure("h8s3 only", base_exp, base_mc)
    measure("h8s3, without cluster 1",
            base_exp[base_exp.cluster != 1], base_mc)
    measure("h8s3, without cluster 1 and bad charge",
            base_exp[(base_exp.cluster != 1) & (base_exp.max_q <= 1e4)],
            base_mc[base_mc.max_q <= 1e4])
    full_exp = selected(exp_all, cfg, is_mc=False)
    full_mc = selected(mc_all, cfg, is_mc=True)
    measure("all three exclusions (the measurement)", full_exp, full_mc)
    def bars(frame: pd.DataFrame) -> pd.DataFrame:
        """The thirteen recommended conditions, with evCenterZ in a common frame."""
        with np.errstate(divide="ignore", invalid="ignore"):
            keep = ((frame.thetaRec < 80) & (frame.nHits >= 8) & (frame.nStrings >= 2)
                    & (frame.covMatrixStatus == 3) & (frame.scfMaxTheta < 1.7)
                    & (np.log10(frame.pHit) > -9) & (frame.evCenterZ_rel < 220)
                    & (frame.zDist > 70) & (frame.pathLength > 50)
                    & (np.log10(frame.funcValue / (frame.nHits - 5)) < 1.1)
                    & (frame.nTriplets / frame.nHits > 0.1)
                    & (frame.thetaErr < 2.5) & (frame.nCalls < 350))
        return frame[keep.fillna(False)]

    measure("plus the full BARS quality selection", bars(full_exp), bars(full_mc))
    relaxed = lambda f: f[(f.thetaRec < 90)]
    measure("all exclusions, up-going only (thetaRec < 90)",
            relaxed(full_exp), relaxed(full_mc))
    return pd.DataFrame(rows)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    exp_all = pd.read_parquet(HERE / "data" / "02_exp_reco_events.parquet")
    mc_all = pd.read_parquet(HERE / "data" / "02_mc_reco_events.parquet")
    mc_all = mc_all[mc_all.group == "muatm"]
    LOG.info("exp_reco %d, mc_reco muatm %d", len(exp_all), len(mc_all))

    exp = selected(exp_all, cfg, is_mc=False)
    mc = selected(mc_all, cfg, is_mc=True)
    LOG.info("after selection: exp %d, mc %d", len(exp), len(mc))

    band_table = bands(exp, mc, cfg["stage04"]["bands"])
    cumulative_table = cumulative(exp, mc)
    variant_table = variants(exp_all, mc_all, cfg)

    edges = np.linspace(0, 1, 101)
    hist = pd.DataFrame({
        "bin_lo": edges[:-1], "bin_hi": edges[1:],
        "exp_count": np.histogram(exp.score, bins=edges)[0],
        "mc_count": np.histogram(mc.score, bins=edges)[0]})
    hist["exp_density"] = hist.exp_count / hist.exp_count.sum()
    hist["mc_density"] = hist.mc_count / hist.mc_count.sum()

    for frame, name in ((band_table, "04_bands"), (cumulative_table, "04_cumulative"),
                        (variant_table, "04_variants"), (hist, "04_histogram")):
        provenance.write(frame, HERE / "data" / f"{name}.parquet", stage="04_reco_excess",
                         config_path=HERE / "config.yaml",
                         inputs=[HERE / "data" / "02_exp_reco_events.parquet"],
                         started=started)
    LOG.info("stage 04 done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
