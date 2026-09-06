"""Stage 10 -- PROTOCOL test 4: does the excess follow the detector noise load?

Lake Baikal's bioluminescence varies strongly between runs and seasons, and the
29 clean experimental runs span a factor of five in trigger rate (28 to 134 Hz,
stage 00).  That variation exists only in the data -- no simulation parameter was
tuned to it -- so a correlation between it and the excess cannot be an artefact
of fitting.

If a mis-modelled, noise-related detector response drives the excess, runs with a
heavier noise load should show a larger excess, and it should grow *faster in the
hard tail* than in the bulk.  If the excess is a flux deficit, it is flat in
noise load.

## Why disjoint score bins and not cumulative thresholds

Cumulative cuts are nested -- events above 0.9 are inside those above 0.8 -- so
excesses measured at several thresholds are one measurement repeated with
different weights, and the multiple tests cannot be counted honestly.  Disjoint
bins are approximately independent, they localise the effect (the cumulative
number is dominated by the lowest bin, where the excess is weakest), and they
expose the *profile*, which is what distinguishes the hypotheses: a response
error should bite the hard tail harder than the bulk.

Each run is therefore reduced to **two** numbers rather than one per threshold:

``level``
    mean log-excess across the bins -- how high the profile sits.
``slope``
    fit of log-excess against bin index -- how fast it rises with score.

Two regressions per proxy, so multiplicity stays interpretable.

## Noise proxies

Nothing in the prediction database can measure noise: every scalar there is
computed on hits that already survived the sig-noise filter (see
``inference_v2/nu_classifier/compute_scalars.py``).  ``n_channels`` counts
*modules among the kept hits*, so ``n_channels - n_sn_hits`` is minus the number
of repeated hits, not rejected noise -- an earlier version of this stage used it
as a noise proxy and was wrong.

``trigger_rate_hz``
    events per second of run, from every event in the HDF5 (stage 00), not from
    the 6% that reached the prediction database.  Trigger-level and independent
    of every network, but also sensitive to trigger settings.
``raw_hits_per_event``
    mean raw hit multiplicity before filtering, from ``raw/ev_starts``.  The most
    direct noise-load measure available, and it too is taken over all events.
``sn_reject_frac``
    fraction of raw hits the sig-noise network discards.  Direct, but inherits
    whatever that network gets wrong.

``repeat_hits_per_event`` is carried along as a diagnostic, correctly named.

Usage:
    python stages/10_run_noise.py [--smoke]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import h5io                                                    # noqa: E402
import provenance                                              # noqa: E402

LOG = logging.getLogger("stage10")
BINS = ((0.5, 0.8), (0.8, 0.9), (0.9, 1.01))
PROXIES = ["trigger_rate_hz", "raw_hits_per_event", "sn_reject_frac",
           "repeat_hits_per_event", "q_total_mean", "cluster"]


def _bin_sql(prefix: str) -> str:
    return ", ".join(
        f"sum(CASE WHEN {prefix}.score >= {lo} AND {prefix}.score < {hi} "
        f"THEN 1 ELSE 0 END) AS n_bin{i}" for i, (lo, hi) in enumerate(BINS))


def mc_reference(cfg: h5io.Config) -> pd.DataFrame:
    """Fraction of quality MC atmospheric-muon events falling in each score bin."""
    db = cfg.path("preds") / cfg["model"] / "mc_merged_thr0p8.duckdb"
    with duckdb.connect(str(db), read_only=True) as con:
        con.execute("PRAGMA threads=8")
        frame = con.execute(f"""
            SELECT count(*) AS n_quality, {_bin_sql('p')}
            FROM predictions p JOIN splits s USING (event_fk)
            WHERE s.data_class LIKE 'muatm%'
              AND p.n_sn_hits >= 8 AND p.n_sn_strings >= 3
        """).df()
    rows = []
    for i, (lo, hi) in enumerate(BINS):
        share = float(frame[f"n_bin{i}"][0]) / float(frame["n_quality"][0])
        rows.append({"bin": i, "lo": lo, "hi": hi,
                     "mc_events": int(frame[f"n_bin{i}"][0]),
                     "mc_quality": int(frame["n_quality"][0]),
                     "mc_share": share})
        LOG.info("MC muatm bin %.2f-%.2f: %d events, share %.3e",
                 lo, hi, rows[-1]["mc_events"], share)
    return pd.DataFrame(rows)


def noise_load(cfg: h5io.Config) -> pd.DataFrame:
    """Raw hit multiplicity per part, taken over every triggered event."""
    path = cfg.path("h5", "exp")
    rows = []
    for part in h5io.exp_parts(path):
        counts = h5io.raw_hits_per_event(path, "exp_full", part)
        rows.append({"part": part,
                     "raw_hits_per_event": float(counts.mean()),
                     "raw_hits_median": float(np.median(counts)),
                     "n_events_total": int(len(counts))})
        LOG.info("%s: raw hits/event mean %.2f median %.0f over %d events",
                 part, rows[-1]["raw_hits_per_event"],
                 rows[-1]["raw_hits_median"], len(counts))
    return pd.DataFrame(rows)


def run_table(cfg: h5io.Config, reference: pd.DataFrame) -> pd.DataFrame:
    """Per-run bin occupancies, excesses, profile level and slope."""
    db = cfg.path("preds") / cfg["model"] / "exp_full_thr0p8.duckdb"
    with duckdb.connect(str(db), read_only=True) as con:
        con.execute("PRAGMA threads=8")
        frame = con.execute(f"""
            SELECT s.part_key, count(*) AS n_quality, {_bin_sql('p')},
                   avg(p.n_sn_hits)                  AS sn_hits_per_event,
                   avg(p.n_sn_hits - c.n_channels)   AS repeat_hits_per_event,
                   avg(c.q_total)                    AS q_total_mean,
                   avg(c.prob_mean)                  AS sig_prob_mean
            FROM predictions p
            JOIN splits s  USING (event_fk)
            JOIN scalars c USING (event_fk)
            WHERE p.n_sn_hits >= 8 AND p.n_sn_strings >= 3
            GROUP BY s.part_key ORDER BY s.part_key
        """).df()

    index = pd.read_parquet(HERE / "data" / "00_exp_parts.parquet")
    loads = noise_load(cfg)
    meta = index.merge(loads, on="part")
    meta["key"] = meta.part.str.removeprefix("part_")
    frame["key"] = frame.part_key.str.removeprefix("part_")
    frame = frame.merge(
        meta[["key", "cluster", "run", "rate_hz", "raw_hits_per_event",
              "raw_hits_median", "n_events_total"]], on="key", how="left")
    frame = frame.rename(columns={"rate_hz": "trigger_rate_hz"})
    frame["sn_reject_frac"] = 1 - frame.sn_hits_per_event / frame.raw_hits_per_event

    logs = []
    for i in range(len(BINS)):
        share = frame[f"n_bin{i}"] / frame.n_quality
        frame[f"excess_{i}"] = share / reference.mc_share[i]
        logs.append(np.log(frame[f"excess_{i}"].replace(0, np.nan)))
    log_matrix = np.vstack([column.to_numpy() for column in logs])
    centres = np.arange(len(BINS), dtype=float)
    centres -= centres.mean()
    frame["level"] = np.nanmean(log_matrix, axis=0)
    frame["slope"] = ((log_matrix - np.nanmean(log_matrix, axis=0)).T * centres
                      ).sum(axis=1) / (centres ** 2).sum()
    return frame.drop(columns="key")


def correlate(frame: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of profile level and slope against each proxy."""
    rows = []
    for target in ("level", "slope"):
        for proxy in PROXIES:
            good = frame[np.isfinite(frame[target]) & np.isfinite(frame[proxy])]
            rho, pval = stats.spearmanr(good[proxy], good[target])
            rows.append({"target": target, "proxy": proxy,
                         "n_runs": int(len(good)),
                         "spearman_rho": float(rho), "p_value": float(pval),
                         "target_min": float(good[target].min()),
                         "target_max": float(good[target].max())})
            LOG.info("%-6s vs %-22s rho %+.3f (p=%.3f)  range %+.2f..%+.2f",
                     target, proxy, rho, pval,
                     rows[-1]["target_min"], rows[-1]["target_max"])
    return pd.DataFrame(rows)


def partial_correlate(frame: pd.DataFrame) -> pd.DataFrame:
    """Correlations of the profile level against noise, with confounds removed.

    Two confounds are unavoidable with 29 runs.  ``cluster`` is one: clusters
    differ both in noise load and in excess, so a raw correlation cannot say
    which is acting.  ``quality_frac`` -- the share of triggered events passing
    the quality cut -- is the other: it enters the denominator of the excess
    directly, so anything that changes it moves the excess mechanically.

    Both are removed by rank regression.  The result must be read with the
    collinearity in mind: ``trigger_rate_hz`` and ``quality_frac`` correlate at
    rho = -0.95, so controlling for the latter can remove the very variance the
    test needs.  A null here means "this design cannot separate them", not
    "there is no effect".
    """
    controls = {"none": [], "cluster": ["cluster"],
                "quality_frac": ["quality_frac"],
                "both": ["cluster", "quality_frac"]}
    frame = frame.assign(quality_frac=frame.n_quality / frame.n_events_total)
    rows = []
    for target in ("level", "slope"):
        for proxy in ("trigger_rate_hz", "raw_hits_per_event", "sn_reject_frac"):
            for label, control in controls.items():
                if not control:
                    rho, pval = stats.spearmanr(frame[proxy], frame[target])
                else:
                    ranks = np.column_stack(
                        [stats.rankdata(frame[c]) for c in control]
                        + [np.ones(len(frame))])
                    residual = {}
                    for name in (proxy, target):
                        values = stats.rankdata(frame[name])
                        beta = np.linalg.lstsq(ranks, values, rcond=None)[0]
                        residual[name] = values - ranks @ beta
                    rho, pval = stats.pearsonr(residual[proxy], residual[target])
                rows.append({"target": target, "proxy": proxy, "control": label,
                             "rho": float(rho), "p_value": float(pval),
                             "n_runs": int(len(frame))})
            LOG.info("%-5s vs %-20s raw %+.3f -> cluster %+.3f -> quality %+.3f"
                     " -> both %+.3f",
                     target, proxy, *[r["rho"] for r in rows[-4:]])
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()

    cfg = h5io.load_config(HERE)
    data, config_path = HERE / "data", HERE / "config.yaml"
    model_dir = cfg.path("preds") / cfg["model"]
    inputs = [model_dir / "exp_full_thr0p8.duckdb",
              model_dir / "mc_merged_thr0p8.duckdb",
              cfg.path("h5", "exp")]

    reference = mc_reference(cfg)
    provenance.write(reference, data / "10_mc_reference.parquet",
                     stage="10_run_noise", config_path=config_path,
                     inputs=inputs[1:2], started=started)

    runs = run_table(cfg, reference)
    LOG.info("runs %d, quality events %s, excess per bin %s",
             len(runs), f"{runs.n_quality.sum():,}",
             [f"{runs[f'excess_{i}'].median():.2f}" for i in range(len(BINS))])
    correlations = correlate(runs)
    partials = partial_correlate(runs)

    provenance.write(runs, data / "10_run_noise.parquet", stage="10_run_noise",
                     config_path=config_path, inputs=inputs, started=started,
                     notes={"bins": [list(b) for b in BINS], "proxies": PROXIES})
    provenance.write(correlations, data / "10_run_noise_correlations.parquet",
                     stage="10_run_noise", config_path=config_path,
                     inputs=inputs, started=started)
    provenance.write(partials, data / "10_run_noise_partial.parquet",
                     stage="10_run_noise", config_path=config_path,
                     inputs=inputs, started=started)
    LOG.info("stage 10 done in %.1f s", time.time() - started)


if __name__ == "__main__":
    main()
