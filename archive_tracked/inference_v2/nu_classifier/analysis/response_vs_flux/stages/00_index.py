"""Stage 00 -- event index, and whether the cross-cluster anchor is possible.

The strongest form of the response test (PROTOCOL test 1) needs one physical
muon seen by two clusters: cluster A fixes the track independently of cluster B,
so B's hit pattern can be compared against prediction without using B's own
information.  Three things have to be true for that, and none may be assumed:

1. MC must keep multi-cluster events.  The converter ran with ``split_multi``,
   so they survive as one row per cluster -- but the *fraction* decides whether
   there is any statistics.
2. Experimental runs from different clusters must overlap in wall-clock time.
   Every experimental part is a single *(cluster, run)* pair, so if no two
   clusters ever ran together there is nothing to match.
3. The cluster-controller clocks must agree well enough that true coincidences
   form a peak above the accidental background.

This stage measures all three and writes the answer.  If (2) or (3) fails, the
protocol falls back to the half-detector split, and that decision is recorded
rather than quietly taken.

Usage:
    python stages/00_index.py [--smoke]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import h5io                                                    # noqa: E402
import provenance                                              # noqa: E402

LOG = logging.getLogger("stage00")
NS_PER_S = 1_000_000_000


def exp_part_index(cfg: h5io.Config) -> pd.DataFrame:
    """One row per experimental part: cluster, run, time span, event count."""
    path = cfg.path("h5", "exp")
    rows = []
    for part in h5io.exp_parts(path):
        season, cluster, run = h5io.parse_exp_part(part)
        times = h5io.exp_event_times(path, part)
        rows.append({
            "part": part, "season": season, "cluster": cluster, "run": run,
            "n_events": int(len(times)),
            "t_start_ns": int(times.min()), "t_end_ns": int(times.max()),
            "duration_s": float((times.max() - times.min()) / NS_PER_S),
            "rate_hz": float(len(times) * NS_PER_S / (times.max() - times.min())),
        })
        LOG.info("%s  c%d r%d  %.1f h  %d events",
                 part, cluster, run, rows[-1]["duration_s"] / 3600, len(times))
    return pd.DataFrame(rows)


def overlapping_pairs(parts: pd.DataFrame) -> pd.DataFrame:
    """Cross-cluster part pairs whose wall-clock spans intersect."""
    rows = []
    for (_, a), (_, b) in combinations(parts.iterrows(), 2):
        if a.cluster == b.cluster:
            continue
        lo = max(a.t_start_ns, b.t_start_ns)
        hi = min(a.t_end_ns, b.t_end_ns)
        if hi <= lo:
            continue
        rows.append({
            "part_a": a.part, "part_b": b.part,
            "cluster_a": a.cluster, "cluster_b": b.cluster,
            "overlap_start_ns": int(lo), "overlap_end_ns": int(hi),
            "overlap_s": float((hi - lo) / NS_PER_S),
        })
    return pd.DataFrame(rows).sort_values("overlap_s", ascending=False)


def coincidence_scan(
    cfg: h5io.Config, pairs: pd.DataFrame, bin_ns: int = 100
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nearest-neighbour time-difference spectrum for every overlapping pair.

    For each event of A inside the common window, the signed distance to the
    closest event of B is histogrammed over +/- ``dt_scan_max_ns``.  Accidental
    coincidences give a flat density; real ones give a peak whose position is
    the clock offset between the two cluster controllers.  The background level
    is measured in the outer half of the window, not assumed.
    """
    path = cfg.path("h5", "exp")
    window = int(cfg["stage00"]["dt_scan_max_ns"])
    edges = np.arange(-window, window + bin_ns, bin_ns)
    centres = (edges[:-1] + edges[1:]) / 2
    cache: dict[str, np.ndarray] = {}

    def times(part: str) -> np.ndarray:
        if part not in cache:
            cache[part] = np.sort(h5io.exp_event_times(path, part))
        return cache[part]

    hist_rows, summary_rows = [], []
    for _, pair in pairs.iterrows():
        lo, hi = pair.overlap_start_ns, pair.overlap_end_ns
        t_a, t_b = times(pair.part_a), times(pair.part_b)
        t_a = t_a[(t_a >= lo) & (t_a <= hi)]
        t_b = t_b[(t_b >= lo) & (t_b <= hi)]
        if len(t_a) == 0 or len(t_b) == 0:
            continue
        delta = h5io.nearest_dt(t_a, t_b)
        counts, _ = np.histogram(delta[np.abs(delta) <= window], bins=edges)
        tag = f"c{pair.cluster_a}_{pair.part_a[-5:]}__c{pair.cluster_b}_{pair.part_b[-5:]}"
        hist_rows.append(pd.DataFrame({"pair": tag, "dt_ns": centres,
                                       "count": counts}))
        outer = np.abs(centres) > window / 2
        background = float(counts[outer].mean())
        peak_bin = int(np.argmax(counts))
        # excess concentrated within +/-5 us of the tallest bin
        near = np.abs(centres - centres[peak_bin]) <= 5_000
        excess = float(counts[near].sum() - background * near.sum())
        sigma = float(np.sqrt(max(background * near.sum(), 1.0)))
        summary_rows.append({
            "pair": tag, "part_a": pair.part_a, "part_b": pair.part_b,
            "cluster_a": int(pair.cluster_a), "cluster_b": int(pair.cluster_b),
            "overlap_s": float(pair.overlap_s),
            "n_a": int(len(t_a)), "n_b": int(len(t_b)),
            "background_per_bin": background,
            "peak_dt_ns": float(centres[peak_bin]),
            "peak_count": int(counts[peak_bin]),
            "excess_within_5us": excess,
            "significance": excess / sigma if sigma > 0 else np.nan,
        })
        LOG.info("%s: peak at %+.0f ns, excess %.0f (%.1f sigma over flat %.1f)",
                 tag, centres[peak_bin], excess,
                 summary_rows[-1]["significance"], background)
    hist = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame()
    return hist, pd.DataFrame(summary_rows)



def clock_offset_scan(
    cfg: h5io.Config, pairs: pd.DataFrame, bin_ns: int = 10_000_000
) -> pd.DataFrame:
    """Search for a coincidence peak at *any* clock offset, not just near zero.

    :func:`coincidence_scan` only sees offsets inside its +/- 100 us window.  If
    the cluster controllers disagree by more than that, real coincidences hide
    at an unknown lag, so the search has to cover the whole overlap.  Both event
    series are binned at ``bin_ns`` and cross-correlated by FFT; slow drifts of
    the trigger rate are removed with a running mean, because they dominate the
    raw correlation (19% RMS against 0.1% Poisson) and would swamp a narrow peak.

    The detection threshold is calibrated by injection, not assumed: a known
    number of fake coincidences is copied from A into B at a fixed offset and
    the search is asked to find them.  What the stage reports is therefore an
    upper limit with a demonstrated sensitivity, not a bare null.
    """
    from scipy.ndimage import uniform_filter1d

    path = cfg.path("h5", "exp")
    rng = np.random.default_rng(cfg["seed"])
    injections = cfg["stage00"]["inject_levels"]
    cache: dict[str, np.ndarray] = {}

    def times(part: str) -> np.ndarray:
        if part not in cache:
            cache[part] = np.sort(h5io.exp_event_times(path, part))
        return cache[part]

    def search(t_a: np.ndarray, t_b: np.ndarray, lo: int, n_bins: int,
               size: int) -> tuple[float, float, float]:
        """Return (robust sd, best excess in sigma, lag of best excess in s)."""
        a = np.bincount(np.clip((t_a - lo) // bin_ns, 0, n_bins - 1),
                        minlength=n_bins).astype(np.float32)
        b = np.bincount(np.clip((t_b - lo) // bin_ns, 0, n_bins - 1),
                        minlength=n_bins).astype(np.float32)
        corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)),
                            size)
        lags = np.fft.fftfreq(size, d=1 / size).astype(np.int64)
        keep = np.abs(lags) < n_bins // 2
        values = np.asarray(corr[keep], dtype=np.float64)
        lag_values = lags[keep]
        order = np.argsort(lag_values)
        values, lag_values = values[order], lag_values[order]
        resid = values - uniform_filter1d(values, 401, mode="nearest")
        sd = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        best = int(np.argmax(resid))
        return sd, resid[best] / sd, lag_values[best] * bin_ns / 1e9

    rows = []
    for index, (_, pair) in enumerate(pairs.iterrows()):
        lo, hi = int(pair.overlap_start_ns), int(pair.overlap_end_ns)
        t_a = times(pair.part_a); t_a = t_a[(t_a >= lo) & (t_a <= hi)]
        t_b = times(pair.part_b); t_b = t_b[(t_b >= lo) & (t_b <= hi)]
        if len(t_a) < 1000 or len(t_b) < 1000:
            continue
        n_bins = int((hi - lo) // bin_ns) + 1
        size = 1 << int(np.ceil(np.log2(2 * n_bins)))
        sd, best_sigma, best_lag = search(t_a, t_b, lo, n_bins, size)
        # A single pair searches ~n_bins/2 lags, so its largest fluctuation is
        # several sigma by construction.  The threshold is taken from the data
        # itself -- see config -- and validated against the injections below.
        trials = float(cfg["stage00"]["trials_sigma"])
        threshold = trials * sd
        row = {
            "pair": f"c{pair.cluster_a}_{pair.part_a[-5:]}"
                    f"__c{pair.cluster_b}_{pair.part_b[-5:]}",
            "cluster_a": int(pair.cluster_a), "cluster_b": int(pair.cluster_b),
            "overlap_s": float(pair.overlap_s),
            "n_a": int(len(t_a)), "n_b": int(len(t_b)),
            "robust_sd": float(sd),
            "best_sigma": float(best_sigma), "best_lag_s": float(best_lag),
            "detected": bool(best_sigma > trials),
            "limit_coincidence_frac": float(threshold / len(t_a)),
        }
        # calibrate the threshold on the first pair only -- it is a property of
        # the method and the rates, and each pair reports its own sd anyway
        if index == 0:
            for n_inject in injections:
                spiked = np.sort(np.concatenate(
                    [t_b, rng.choice(t_a, size=n_inject, replace=False)
                     + 137_000_000]))
                _, sigma, lag = search(t_a, spiked, lo, n_bins, size)
                row[f"inject_{n_inject}_sigma"] = float(sigma)
                row[f"inject_{n_inject}_lag_s"] = float(lag)
                LOG.info("  calibration: %d injected at +0.137 s -> found at "
                         "%+.3f s, %.1f sigma", n_inject, lag, sigma)
        rows.append(row)
        LOG.info("%s: best %.1f sigma at %+.1f s%s; limit %.3f%% of A",
                 row["pair"], best_sigma, best_lag,
                 " DETECTED" if row["detected"] else " (no signal)",
                 100 * row["limit_coincidence_frac"])
    return pd.DataFrame(rows)


def mc_multicluster_census(cfg: h5io.Config, n_parts: int) -> pd.DataFrame:
    """How often one MC event lights more than one cluster, per data class."""
    path = cfg.path("h5", "mc")
    rows = []
    for klass in cfg["mc_classes"]:
        parts = h5io.mc_parts(path, klass)[:n_parts]
        sizes: dict[int, int] = {}
        n_events = 0
        for part in parts:
            entries = h5io.mc_root_entries(path, klass, part)
            _, counts = np.unique(entries, return_counts=True)
            n_events += len(counts)
            for value, hits in zip(*np.unique(counts, return_counts=True)):
                sizes[int(value)] = sizes.get(int(value), 0) + int(hits)
        multi = sum(v for k, v in sizes.items() if k > 1)
        rows.append({
            "klass": klass, "parts_probed": len(parts), "events": n_events,
            "multi_cluster_events": multi,
            "multi_cluster_frac": multi / n_events if n_events else np.nan,
            "size_histogram": str(dict(sorted(sizes.items()))),
        })
        LOG.info("%s: %d events, %.2f%% multi-cluster, sizes %s",
                 klass, n_events, 100 * rows[-1]["multi_cluster_frac"],
                 rows[-1]["size_histogram"])
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="probe only a couple of parts, for a fast check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()

    cfg = h5io.load_config(HERE)
    data = HERE / "data"
    config_path = HERE / "config.yaml"
    inputs = [cfg.path("h5", "exp"), cfg.path("h5", "mc")]

    parts = exp_part_index(cfg)
    provenance.write(parts, data / "00_exp_parts.parquet", stage="00_index",
                     config_path=config_path, inputs=inputs[:1], started=started)

    pairs = overlapping_pairs(parts)
    LOG.info("cross-cluster overlapping pairs: %d", len(pairs))
    provenance.write(pairs, data / "00_cluster_pairs.parquet", stage="00_index",
                     config_path=config_path, inputs=inputs[:1], started=started)

    scan_pairs = pairs.head(3) if args.smoke else pairs
    hist, summary = coincidence_scan(cfg, scan_pairs)
    provenance.write(hist, data / "00_coincidence_dt.parquet", stage="00_index",
                     config_path=config_path, inputs=inputs[:1], started=started,
                     notes={"pairs_scanned": int(len(scan_pairs))})
    provenance.write(summary, data / "00_coincidence_summary.parquet",
                     stage="00_index", config_path=config_path, inputs=inputs[:1],
                     started=started)

    offsets = clock_offset_scan(cfg, scan_pairs)
    provenance.write(offsets, data / "00_clock_offsets.parquet", stage="00_index",
                     config_path=config_path, inputs=inputs[:1], started=started,
                     notes={"detected_any": bool(offsets.detected.any())
                            if len(offsets) else False})

    n_parts = (cfg["stage00"]["smoke_parts"] if args.smoke
               else cfg["stage00"]["mc_parts_probed"])
    census = mc_multicluster_census(cfg, n_parts)
    provenance.write(census, data / "00_mc_multicluster.parquet", stage="00_index",
                     config_path=config_path, inputs=inputs[1:], started=started)

    LOG.info("stage 00 done in %.1f s", time.time() - started)


if __name__ == "__main__":
    main()
