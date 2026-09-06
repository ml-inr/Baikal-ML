"""Stage 05 -- is the reco excess also a horizon effect?

Earlier work concluded the excess in `exp_full` is largely the classifier failing
near the horizon: accepted muons are reconstructed as up-going when they are not.
That was established with the network's own Cherenkov fit.  Here the angle comes
from BARS instead -- an independent reconstruction that the classifier never
sees -- so the same question can be asked without circularity.

Three scans, because a one-sided cut cannot answer it.  Keeping ``thetaRec < X``
removes down-going events but keeps the horizon; the horizon is a *band*, and
removing a band needs two edges.

* **up-going threshold**: keep ``thetaRec < X``.
* **horizon band removed**: drop ``|thetaRec - 90| < delta``, keeping both the
  clearly up-going and the clearly down-going.
* **up-going with the horizon removed**: keep ``thetaRec < 90 - delta``.

and then the same thing binned, which is what should actually be read: cumulative
cuts mix the regions they contain, and the answer here is a shape, not a number.

In this convention 90 degrees is the horizon and larger angles are down-going.

Usage:
    python stages/05_horizon_band.py
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
sys.path.insert(0, str(HERE / "stages"))

import provenance                                              # noqa: E402

LOG = logging.getLogger("stage05")


def selected(frame: pd.DataFrame, cfg: dict, is_mc: bool) -> pd.DataFrame:
    settings = cfg["stage04"]
    keep = frame.score.notna() & (frame.n_sn_hits >= 8) & (frame.n_sn_strings >= 3)
    keep &= frame.max_q <= float(settings["max_charge"])
    if is_mc:
        keep &= frame.n_gt_sig_hits > int(settings["min_gt_sig_hits"])
    else:
        keep &= ~frame.cluster.isin(settings["exclude_exp_clusters"])
    return frame[keep]


def excess(exp: pd.DataFrame, mc: pd.DataFrame, threshold: float) -> float:
    if not len(exp) or not len(mc):
        return np.nan
    share_mc = float((mc.score > threshold).mean())
    return float((exp.score > threshold).mean()) / share_mc if share_mc > 0 else np.nan


def scan(exp: pd.DataFrame, mc: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def record(kind: str, value: float, e: pd.DataFrame, m: pd.DataFrame) -> None:
        rows.append({"scan": kind, "value": value,
                     "n_exp": len(e), "n_mc": len(m),
                     "n_exp_accepted": int((e.score > 0.8).sum()),
                     "n_mc_accepted": int((m.score > 0.8).sum()),
                     "excess_0.5": excess(e, m, 0.5),
                     "excess_0.8": excess(e, m, 0.8)})

    record("no cut", np.nan, exp, mc)
    for x in (10, 20, 30, 40, 50, 60, 70, 80, 85, 90):
        record("up-going: thetaRec <", float(x),
               exp[exp.thetaRec < x], mc[mc.thetaRec < x])
    for delta in (0, 5, 10, 15, 20, 25, 30, 40):
        e = exp[(exp.thetaRec - 90).abs() >= delta]
        m = mc[(mc.thetaRec - 90).abs() >= delta]
        record("horizon band removed: |theta-90| >=", float(delta), e, m)
    for delta in (0, 5, 10, 15, 20, 25, 30, 40):
        e = exp[exp.thetaRec < 90 - delta]
        m = mc[mc.thetaRec < 90 - delta]
        record("up-going minus horizon: thetaRec < 90 -", float(delta), e, m)
    frame = pd.DataFrame(rows)
    for _, row in frame.iterrows():
        LOG.info("%-38s %5s  exp %8d (%5d acc)  mc %7d (%4d acc)  excess %.2f / %.2f",
                 row.scan, "" if np.isnan(row.value) else f"{row.value:.0f}",
                 row.n_exp, row.n_exp_accepted, row.n_mc, row.n_mc_accepted,
                 row["excess_0.5"], row["excess_0.8"])
    return frame


def binned(exp: pd.DataFrame, mc: pd.DataFrame,
           edges: list[float]) -> pd.DataFrame:
    """Excess in disjoint bands of the BARS zenith, with Poisson errors.

    The cumulative scans above hide the structure: removing a horizon band raises
    the excess only because what is left is dominated by the clearly down-going
    events where it is largest.  Bands show that directly.
    """
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        e = exp[(exp.thetaRec >= lo) & (exp.thetaRec < hi)]
        m = mc[(mc.thetaRec >= lo) & (mc.thetaRec < hi)]
        if len(m) < 200:
            continue
        n_e, n_m = int((e.score > 0.8).sum()), int((m.score > 0.8).sum())
        share_e, share_m = n_e / len(e), n_m / len(m)
        value = share_e / share_m if share_m > 0 else np.nan
        # relative Poisson error of the ratio, from the accepted counts
        error = value * np.sqrt(1 / max(n_e, 1) + 1 / max(n_m, 1))
        rows.append({"theta_lo": lo, "theta_hi": min(hi, 180.0),
                     "n_exp": len(e), "n_mc": len(m),
                     "accepted_exp": n_e, "accepted_mc": n_m,
                     "share_exp": share_e, "share_mc": share_m,
                     "excess": value, "excess_error": error})
        LOG.info("theta %5.0f-%5.0f: exp %8d (%5d acc)  mc %7d (%4d acc)  "
                 "excess %.2f +- %.2f", lo, min(hi, 180.0), len(e), n_e,
                 len(m), n_m, value, error)
    return pd.DataFrame(rows)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    exp = pd.read_parquet(HERE / "data" / "02_exp_reco_events.parquet")
    mc = pd.read_parquet(HERE / "data" / "02_mc_reco_events.parquet")
    mc = mc[mc.group == "muatm"]
    exp, mc = selected(exp, cfg, False), selected(mc, cfg, True)
    LOG.info("base selection: exp %d, mc %d", len(exp), len(mc))
    LOG.info("median thetaRec: exp %.1f, mc %.1f",
             exp.thetaRec.median(), mc.thetaRec.median())

    table = scan(exp, mc)
    bands = binned(exp, mc, [0, 20, 40, 60, 75, 85, 95, 105, 120, 140, 160, 180.1])
    for frame, name in ((table, "05_horizon_scan"), (bands, "05_theta_bands")):
        provenance.write(frame, HERE / "data" / f"{name}.parquet",
                         stage="05_horizon_band", config_path=HERE / "config.yaml",
                         inputs=[HERE / "data" / "02_exp_reco_events.parquet"],
                         started=started)
    LOG.info("stage 05 done in %.0f s", time.time() - started)


if __name__ == "__main__":
    main()
