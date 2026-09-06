"""Stage 20 -- PROTOCOL test 2: can a flux reweighting alone explain the excess?

Hypothesis (a) says the generator produces the wrong *population* of muons while
the detector response is right.  If so, some weight function over MC **truth**
variables must repair the disagreement, and -- this is what gives the test teeth
-- the *same* function must repair it everywhere at once.  There are two truth
axes available in the prediction database (zenith, primary energy) against a
dozen observables, so the problem is heavily over-determined: a flux error has to
be fixable by one low-dimensional function, a response error cannot be.

The fit deliberately has more freedom than the target it is fitted to (60 truth
bins against 12 score bins), so matching the score distribution proves nothing on
its own and is expected to succeed.  The test is entirely in the **held-out
observables**, which the fit never sees.

Two things are reported besides the residuals:

* the **dynamic range of the weights**.  A flux error of a factor of two near the
  horizon is a plausible statement about a hard-to-simulate region; a factor of
  fifty is not, and would say the hypothesis survives only by absurdity.
* the mismatch **inside the accepted region**, where the excess lives, separately
  from the mismatch over all quality events.

Truth binning is finer near the horizon because that is where the accepted events
sit: MC muons with score >= 0.9 have mean zenith 107.9 deg against 136.6 deg for
the rest, while their energies are indistinguishable (log10 E 2.73 against 2.74).

Usage:
    python stages/20_flux_reweight.py [--smoke]
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
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import h5io                                                    # noqa: E402
import provenance                                              # noqa: E402

LOG = logging.getLogger("stage20")

ZENITH_EDGES = [90, 95, 100, 105, 110, 115, 120, 130, 140, 155, 180.1]
LOGE_EDGES = [-1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 6.0]
SCORE_EDGES = [0, .01, .05, .1, .2, .3, .5, .7, .8, .9, .95, .99, 1.01]
HELD_OUT = ["q_total", "q_max", "q_mean", "q_frac_max", "n_channels",
            "n_sig_strings", "t_span", "t_span_core", "z_span", "xy_span",
            "z_c", "r_vert", "prob_mean"]
# The fit is swept over smoothness rather than run at one value.  An
# unconstrained fit is pathological -- it zeroes two thirds of the MC sample and
# produces a U-shaped zenith weight -- and a pathological fit cannot refute
# anything.  The question is posed instead as a trade-off: how well can the score
# distribution be matched *while the weight stays physically plausible*.
SMOOTHNESS_SWEEP = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
SAMPLE = 400_000           # deterministic sample for the all-quality comparison
QUALITY = "p.n_sn_hits >= 8 AND p.n_sn_strings >= 3"


def _case(expr: str, edges: list[float]) -> str:
    """SQL expression mapping a value onto its bin index."""
    parts = [f"WHEN {expr} < {hi} THEN {i}" for i, hi in enumerate(edges[1:])]
    return "CASE " + " ".join(parts) + f" ELSE {len(edges) - 2} END"


def score_matrix(cfg: h5io.Config) -> tuple[np.ndarray, np.ndarray]:
    """MC counts per (truth bin, score bin), and the experimental score counts."""
    model = cfg.path("preds") / cfg["model"]
    n_truth = (len(ZENITH_EDGES) - 1) * (len(LOGE_EDGES) - 1)
    n_score = len(SCORE_EDGES) - 1

    truth_bin = (f"({_case('t.zenith_deg', ZENITH_EDGES)}) * {len(LOGE_EDGES) - 1}"
                 f" + ({_case('log10(t.primary_energy_gev)', LOGE_EDGES)})")
    with duckdb.connect(str(model / "mc_merged_thr0p8.duckdb"), read_only=True) as con:
        con.execute("PRAGMA threads=8")
        frame = con.execute(f"""
            SELECT {truth_bin} AS tb, {_case('p.score', SCORE_EDGES)} AS sb,
                   count(*) AS n
            FROM predictions p JOIN splits s USING (event_fk)
                               JOIN truth t USING (event_fk)
            WHERE s.data_class LIKE 'muatm%' AND {QUALITY}
            GROUP BY 1, 2
        """).df()
    matrix = np.zeros((n_score, n_truth))
    matrix[frame.sb.to_numpy(int), frame.tb.to_numpy(int)] = frame.n.to_numpy()

    with duckdb.connect(str(model / "exp_full_thr0p8.duckdb"), read_only=True) as con:
        con.execute("PRAGMA threads=8")
        frame = con.execute(f"""
            SELECT {_case('p.score', SCORE_EDGES)} AS sb, count(*) AS n
            FROM predictions p WHERE {QUALITY} GROUP BY 1
        """).df()
    target = np.zeros(n_score)
    target[frame.sb.to_numpy(int)] = frame.n.to_numpy()
    return matrix, target


def fit_weights(matrix: np.ndarray, target: np.ndarray,
                smoothness: float) -> np.ndarray:
    """Smooth non-negative weights per truth bin matching the exp score shape.

    Weights are fitted as ``exp(theta)``, so they are positive by construction and
    cannot be driven to zero the way a non-negative least-squares solution does.
    ``smoothness`` penalises squared differences of ``theta`` between neighbouring
    truth bins, in zenith and in energy; at zero it reproduces the free fit, and
    raising it forces the weight towards a shape a flux correction could actually
    have.

    Bins are compared as shares with Poisson weighting, so the tail keeps the
    leverage it deserves instead of being drowned by the bulk near zero.
    """
    share_exp = target / target.sum()
    sigma = np.sqrt(np.maximum(share_exp, 1e-12) / target.sum())
    counts = matrix.sum(axis=0)
    occupied = counts > 0
    n_truth = matrix.shape[1]
    n_energy = len(LOGE_EDGES) - 1
    pairs = [(j, j + n_energy) for j in range(n_truth - n_energy)]
    pairs += [(j, j + 1) for j in range(n_truth - 1) if (j + 1) % n_energy]
    pairs = [(i, j) for i, j in pairs if occupied[i] and occupied[j]]
    left = np.array([i for i, _ in pairs])
    right = np.array([j for _, j in pairs])

    def objective(theta: np.ndarray) -> float:
        weights = np.exp(np.clip(theta, -20, 20))
        predicted = matrix @ weights
        total = predicted.sum()
        if total <= 0:
            return 1e12
        residual = (predicted / total - share_exp) / sigma
        rough = np.sum((theta[left] - theta[right]) ** 2) if len(left) else 0.0
        return float(residual @ residual + smoothness * rough)

    best = minimize(objective, np.zeros(n_truth), method="L-BFGS-B",
                    options={"maxiter": 4000, "maxfun": 200_000})
    weights = np.exp(np.clip(best.x, -20, 20))
    weights[~occupied] = 0.0
    return weights / np.average(weights[occupied], weights=counts[occupied])


def achievable_bound(matrix: np.ndarray, ranges: list[float]) -> pd.DataFrame:
    """Largest excess any truth-based reweighting can produce, without fitting.

    If the weight depends only on truth variables ``T``, the reweighted
    acceptance is ``E[w(T) a(T)] / E[w(T)]`` where ``a(T)`` is the MC acceptance
    at that truth.  Maximising a ratio of linear forms over a box
    ``w in [1/sqrt(R), sqrt(R)]`` has its optimum at a vertex: weight up every
    truth bin whose acceptance exceeds a threshold, down every other.  Sorting
    the bins by acceptance and scanning the cut therefore gives the exact
    maximum -- an upper bound that no fit, however clever, can beat.

    This is what makes the test conclusive rather than suggestive: if the bound
    for a physically moderate ``R`` falls below the observed excess, hypothesis
    (a) is dead in the truth variables available, whatever weight one chooses.
    """
    rows = []
    for band, (lo, hi) in enumerate(zip(SCORE_EDGES[:-1], SCORE_EDGES[1:])):
        if lo < 0.8:
            continue
        accepted = matrix[band]
        totals = matrix.sum(axis=0)
        keep = totals > 0
        a = accepted[keep] / totals[keep]
        n = totals[keep]
        baseline = accepted.sum() / totals.sum()
        order = np.argsort(-a)
        a, n = a[order], n[order]
        cum_a = np.cumsum(a * n)
        cum_n = np.cumsum(n)
        for allowed in ranges:
            high, low = np.sqrt(allowed), 1 / np.sqrt(allowed)
            best = max(
                (high * cum_a[m] + low * (cum_a[-1] - cum_a[m]))
                / (high * cum_n[m] + low * (cum_n[-1] - cum_n[m]))
                for m in range(len(a)))
            rows.append({"score_lo": lo, "score_hi": hi,
                         "weight_range": allowed,
                         "max_enhancement": float(best / baseline)})
    return pd.DataFrame(rows)


def sample_events(cfg: h5io.Config, weights: np.ndarray) -> pd.DataFrame:
    """MC and experimental events with observables, for the held-out comparison."""
    model = cfg.path("preds") / cfg["model"]
    columns = ", ".join(f"c.{name}" for name in HELD_OUT)
    truth_bin = (f"({_case('t.zenith_deg', ZENITH_EDGES)}) * {len(LOGE_EDGES) - 1}"
                 f" + ({_case('log10(t.primary_energy_gev)', LOGE_EDGES)})")
    with duckdb.connect(str(model / "mc_merged_thr0p8.duckdb"), read_only=True) as con:
        con.execute("PRAGMA threads=8")
        mc = con.execute(f"""
            SELECT p.score, {truth_bin} AS tb, t.zenith_deg, {columns}
            FROM predictions p JOIN splits s USING (event_fk)
                               JOIN truth t USING (event_fk)
                               JOIN scalars c USING (event_fk)
            WHERE s.data_class LIKE 'muatm%' AND {QUALITY}
              AND (p.score >= 0.5 OR hash(p.event_fk) % 100 < 2)
        """).df()
    with duckdb.connect(str(model / "exp_full_thr0p8.duckdb"), read_only=True) as con:
        con.execute("PRAGMA threads=8")
        exp = con.execute(f"""
            SELECT p.score, {columns}
            FROM predictions p JOIN scalars c USING (event_fk)
            WHERE {QUALITY} AND (p.score >= 0.5 OR hash(p.event_fk) % 100 < 2)
        """).df()
    mc["w"] = weights[mc.tb.to_numpy(int)]
    mc["source"], exp["source"], exp["w"] = "mc", "exp", 1.0
    LOG.info("held-out samples: MC %d, exp %d", len(mc), len(exp))
    return pd.concat([mc, exp], ignore_index=True)


def held_out_mismatch(events: pd.DataFrame) -> pd.DataFrame:
    """Distance between MC and experimental observables, before and after w."""
    rows = []
    for region, mask in (("all quality", events.score >= 0),
                         ("accepted (score>=0.8)", events.score >= 0.8)):
        sub = events[mask]
        mc, exp = sub[sub.source == "mc"], sub[sub.source == "exp"]
        if len(mc) < 100 or len(exp) < 100:
            continue
        for name in HELD_OUT:
            x, y = mc[name].to_numpy(float), exp[name].to_numpy(float)
            good = np.isfinite(x); x, w = x[good], mc.w.to_numpy()[good]
            y = y[np.isfinite(y)]
            spread = np.std(np.concatenate([x, y])) or 1.0
            before = wasserstein_distance(x, y) / spread
            after = wasserstein_distance(x, y, u_weights=w) / spread
            rows.append({"region": region, "observable": name,
                         "distance_before": before, "distance_after": after,
                         "fraction_removed": 1 - after / before if before else np.nan,
                         "n_mc": int(len(x)), "n_exp": int(len(y))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = h5io.load_config(HERE)
    data, config_path = HERE / "data", HERE / "config.yaml"
    model = cfg.path("preds") / cfg["model"]
    inputs = [model / "mc_merged_thr0p8.duckdb", model / "exp_full_thr0p8.duckdb"]

    matrix, target = score_matrix(cfg)
    share_mc = matrix.sum(axis=1) / matrix.sum()
    share_exp = target / target.sum()
    counts = matrix.sum(axis=0)
    occupied = counts > 0
    n_zenith, n_energy = len(ZENITH_EDGES) - 1, len(LOGE_EDGES) - 1

    sweep_rows, fits = [], {}
    for smoothness in SMOOTHNESS_SWEEP:
        weights = fit_weights(matrix, target, smoothness)
        fits[smoothness] = weights
        share_rw = (matrix @ weights) / (matrix @ weights).sum()
        excess = share_exp / share_rw
        low, high = weights[occupied].min(), weights[occupied].max()
        zeroed = float(counts[occupied][weights[occupied] < 0.05].sum()
                       / counts[occupied].sum())
        sweep_rows.append({
            "smoothness": smoothness,
            "max_abs_excess_dev": float(np.abs(excess - 1).max()),
            "median_abs_excess_dev": float(np.median(np.abs(excess - 1))),
            "weight_min": float(low), "weight_max": float(high),
            "weight_ratio": float(high / max(low, 1e-9)),
            "mc_frac_suppressed": zeroed})
        LOG.info("smoothness %7.1f: worst bin off by %.2f, weights %.2f-%.2f "
                 "(ratio %.0f), %.0f%% of MC suppressed", smoothness,
                 sweep_rows[-1]["max_abs_excess_dev"], low, high,
                 sweep_rows[-1]["weight_ratio"], 100 * zeroed)
    sweep = pd.DataFrame(sweep_rows)

    # the fit used downstream: the smoothest one that still matches every score
    # bin to within 25%, i.e. the most plausible weight that does the job
    good = sweep[sweep.max_abs_excess_dev <= 0.25]
    chosen = float(good.smoothness.max()) if len(good) else 0.0
    weights = fits[chosen]
    LOG.info("chosen smoothness %.1f (smoothest fit still matching within 25%%)",
             chosen)

    share_rw = (matrix @ weights) / (matrix @ weights).sum()
    fit = pd.DataFrame({
        "score_lo": SCORE_EDGES[:-1], "score_hi": SCORE_EDGES[1:],
        "mc_share": share_mc, "mc_share_reweighted": share_rw,
        "exp_share": share_exp,
        "excess_before": share_exp / share_mc,
        "excess_after": share_exp / share_rw})
    for _, row in fit.iterrows():
        LOG.info("score %.2f-%.2f: excess %.2f -> %.2f",
                 row.score_lo, row.score_hi, row.excess_before, row.excess_after)

    grid = pd.DataFrame({
        "truth_bin": np.arange(len(weights)),
        "zenith_lo": np.repeat(ZENITH_EDGES[:-1], n_energy),
        "zenith_hi": np.repeat(ZENITH_EDGES[1:], n_energy),
        "loge_lo": np.tile(LOGE_EDGES[:-1], n_zenith),
        "loge_hi": np.tile(LOGE_EDGES[1:], n_zenith),
        "weight": weights, "mc_events": counts, "smoothness": chosen})
    by_zenith = grid[grid.mc_events > 0].groupby("zenith_lo").apply(
        lambda x: np.average(x.weight, weights=x.mc_events), include_groups=False)
    LOG.info("weight by zenith: %s",
             {int(k): round(float(v), 2) for k, v in by_zenith.items()})

    bound = achievable_bound(matrix, [2.0, 4.0, 10.0, 100.0, 1e4, 1e9])
    observed = {row.score_lo: row.excess_before for _, row in fit.iterrows()}
    bound["observed_excess"] = bound.score_lo.map(observed)
    bound["reachable"] = bound.max_enhancement >= bound.observed_excess
    for _, row in bound.iterrows():
        LOG.info("band %.2f-%.2f, weights within x%-8.0f: max enhancement %5.2f "
                 "vs observed %.2f  %s", row.score_lo, row.score_hi,
                 row.weight_range, row.max_enhancement, row.observed_excess,
                 "reachable" if row.reachable else "IMPOSSIBLE")
    provenance.write(bound, data / "20_achievable_bound.parquet",
                     stage="20_flux_reweight", config_path=config_path,
                     inputs=inputs, started=started)

    events = sample_events(cfg, weights)
    mismatch = held_out_mismatch(events)
    for region in mismatch.region.unique():
        sub = mismatch[mismatch.region == region]
        LOG.info("%s: median mismatch removed %.1f%% (worst %.1f%%, best %.1f%%)",
                 region, 100 * sub.fraction_removed.median(),
                 100 * sub.fraction_removed.min(), 100 * sub.fraction_removed.max())

    for frame, name in ((fit, "20_score_fit"), (grid, "20_weights"),
                        (mismatch, "20_held_out"), (sweep, "20_smoothness_sweep")):
        provenance.write(frame, data / f"{name}.parquet", stage="20_flux_reweight",
                         config_path=config_path, inputs=inputs, started=started)
    LOG.info("stage 20 done in %.1f s", time.time() - started)


if __name__ == "__main__":
    main()
