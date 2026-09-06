"""Stage 41 -- does jittered MC actually look like the data, or only score like it?

Stage 40 showed that adding the independently measured 28.5 ns of timing jitter
to MC reproduces the observed excess.  That is only meaningful if the jitter also
reproduces what was measured in the first place: the width of the leave-one-out
residual.  The jitter goes onto *raw* hits and the sig-noise filter then removes
part of what it smeared, so the surviving width is not the input sigma and has to
be measured.

Two checks, both held out from everything that was fitted:

1. **Residual width.**  Run stage 30's leave-one-out on jittered MC and compare
   with experiment.  If jittered MC lands on the experimental width, the
   perturbation is the right size and the chain closes.  If it overshoots or
   falls short, the agreement in stage 40 is a coincidence of two errors.
2. **Untouched observables.**  Charge, hit count and event extent were used
   nowhere in choosing the jitter.  If jittered MC moves *towards* experiment on
   them, the perturbation is describing the real difference; if it moves away, it
   is a knob that happens to fit one number.

Usage:
    nohup python stages/41_jitter_validate.py > /tmp/stage41.log 2>&1 &
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse                                                # noqa: E402
import logging                                                 # noqa: E402
import sys                                                     # noqa: E402
import time                                                    # noqa: E402
from concurrent.futures import ProcessPoolExecutor             # noqa: E402
from pathlib import Path                                       # noqa: E402

import h5py                                                    # noqa: E402
from zlib import crc32                                         # noqa: E402
import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parents[3]))

import h5io                                                    # noqa: E402
import provenance                                              # noqa: E402
import tracks                                                  # noqa: E402

LOG = logging.getLogger("stage41")
RDCC = {"rdcc_nbytes": 256 * 1024 * 1024, "rdcc_nslots": 100_003}
MIN_HITS, MIN_STRINGS, HELD_OUT = 8, 3, 3


def collect(cfg: h5io.Config, sigma: float, parts: list[str],
            predict_flat, sn_model, sn_device, count_hits_strings,
            rng) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Filtered hits of quality events after jittering the raw times by `sigma`."""
    events = []
    for part in parts:
        with h5py.File(cfg.path("h5", "mc"), "r", **RDCC) as src:
            starts = src[f"muatm_2020/raw/ev_starts/{part}/data"][:].astype(np.int64)
            data = src[f"muatm_2020/raw/data/{part}/data"][:].astype(np.float32)
            channels = src[f"muatm_2020/raw/channels/{part}/data"][:]
        perturbed = data.copy()
        if sigma > 0:
            perturbed[:, 1] += rng.normal(0.0, sigma, len(perturbed)).astype(np.float32)
        probs = predict_flat(sn_model, perturbed, starts,
                             batch_size=int(cfg["stage40"]["sn_batch_size"]),
                             device=sn_device, normalize=True)
        mask = probs > float(cfg["sig_noise_threshold"])
        n_hits = (starts[1:] - starts[:-1]).astype(np.int32)
        n_sn_h, n_sn_s = count_hits_strings(mask, channels, starts[:-1], n_hits,
                                            len(starts) - 1)
        for index in np.where((n_sn_h >= MIN_HITS) & (n_sn_s >= MIN_STRINGS))[0]:
            lo, hi = int(starts[index]), int(starts[index + 1])
            keep = mask[lo:hi]
            events.append((perturbed[lo:hi][keep].astype(np.float64),
                           channels[lo:hi][keep], sigma, part))
    return events


def residuals_of(event: tuple) -> list[dict]:
    """Leave-one-out residuals for one event, in a worker process."""
    hits, chan, sigma, part = event
    order = np.argsort(hits[:, 1])
    hits, chan = hits[order], chan[order]
    charge = np.clip(hits[:, 0], 0, tracks.Q_CLIP)
    step = max(len(hits) // HELD_OUT, 1)
    indices = np.arange(0, len(hits), step)[:HELD_OUT]
    out = tracks.leave_one_out(hits[:, 2:5], hits[:, 1], charge, indices=indices)
    rows = []
    for k in range(len(out["dt"])):
        rows.append({"sigma_ns": sigma, "dt": out["dt"][k], "d": out["d"][k],
                     "n_hits": len(hits),
                     "n_strings": int(np.unique(chan // tracks.STRING_DIVISOR).size),
                     "q_total": float(charge.sum()),
                     "t_span": float(hits[:, 1].max() - hits[:, 1].min())})
    return rows


def matched_calibration(jittered: pd.DataFrame,
                        experiment: pd.DataFrame) -> pd.DataFrame:
    """Input jitter that makes MC match experiment, cell by matched cell.

    The marginal comparison is misleading here: jittered MC is *wider* than
    experiment overall (IQR 63.0 against 54.1 at 30 ns) purely because the two
    samples differ in hit multiplicity -- MC's median is 13 hits, experiment's 10,
    and a residual narrows with more hits to anchor on.  Matched on geometry, the
    question becomes well posed.

    Within a cell the added variance is taken as quadratic in the input sigma,
    which is what a per-hit Gaussian smear gives, so the sigma that reproduces
    experiment follows from the two measured points.
    """
    experiment = experiment.assign(sigma_ns=-1.0)
    columns = ["dt", "d", "n_hits", "n_strings", "sigma_ns"]
    both = pd.concat([jittered[columns], experiment[columns]], ignore_index=True)
    both["hit_bin"] = pd.cut(both.n_hits, [7, 9, 11, 14, 20, 10_000],
                             labels=["8-9", "10-11", "12-14", "15-20", "21+"])
    both["dist_bin"] = pd.cut(both.d, [0, 15, 30, 50, 80, 1e9],
                              labels=["<15", "15-30", "30-50", "50-80", "80+"])
    both["string_bin"] = np.clip(both.n_strings, 3, 6)
    spread = both.groupby(["hit_bin", "string_bin", "dist_bin", "sigma_ns"],
                          observed=True).dt.agg(
        n="size", iqr=lambda x: x.quantile(0.75) - x.quantile(0.25)).unstack("sigma_ns")
    high = float(max(s for s in jittered.sigma_ns.unique() if s > 0))
    enough = ((spread[("n", 0.0)] >= 120) & (spread[("n", high)] >= 120)
              & (spread[("n", -1.0)] >= 120))
    spread = spread[enough]
    base, smeared, data = (spread[("iqr", 0.0)], spread[("iqr", high)],
                           spread[("iqr", -1.0)])
    out = pd.DataFrame({"n_exp": spread[("n", -1.0)], "iqr_mc": base,
                        "iqr_mc_jittered": smeared, "iqr_exp": data})
    out["sigma_needed_ns"] = high * np.sqrt(
        np.clip((data ** 2 - base ** 2) / (smeared ** 2 - base ** 2), 0, None))
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = h5io.load_config(HERE)
    data_dir, config_path = HERE / "data", HERE / "config.yaml"

    from inference_v2.shared.model_utils import load_sn_model
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (  # noqa: E501
        predict_flat)
    from data_manager.nu_classifier_ds_builder.io import _count_sig_hits_strings

    sn_model, _, sn_device = load_sn_model(device=cfg["stage40"]["device"])
    with h5py.File(cfg.path("h5", "mc"), "r") as src:
        all_parts = sorted(src["muatm_2020"]["raw"]["ev_starts"].keys())
    # crc32, not Python's hash(): string hashing is salted per process, so
    # hash() selected a different set of parts on every run and the stage was
    # not reproducible -- two runs disagreed by 9% on the calibrated jitter.
    order = np.argsort([crc32(p.encode()) for p in all_parts])
    n_parts = 2 if args.smoke else int(cfg["stage41"]["mc_parts"])
    parts = [all_parts[i] for i in order[:n_parts]]

    rng = np.random.default_rng(cfg["seed"])
    frames = []
    for sigma in cfg["stage41"]["sigmas_ns"]:
        events = collect(cfg, float(sigma), parts, predict_flat, sn_model,
                         sn_device, _count_sig_hits_strings, rng)
        cap = int(cfg["stage41"]["max_events"])
        if len(events) > cap:
            keep = np.random.default_rng(cfg["seed"]).choice(len(events), cap,
                                                             replace=False)
            events = [events[i] for i in sorted(keep)]
        LOG.info("sigma %.0f ns: %d quality events, running leave-one-out",
                 sigma, len(events))
        rows = []
        with ProcessPoolExecutor(max_workers=int(cfg["stage41"]["workers"])) as pool:
            for result in pool.map(residuals_of, events, chunksize=32):
                rows.extend(result)
        frames.append(pd.DataFrame(rows))
        LOG.info("  %d residuals, %.0f s", len(rows), time.time() - started)

    jittered = pd.concat(frames, ignore_index=True)
    experiment = pd.read_parquet(data_dir / "30_loo_residuals.parquet")
    experiment = experiment[experiment.source == "exp"]

    def iqr(values) -> float:
        return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))

    summary = []
    for sigma, group in jittered.groupby("sigma_ns"):
        summary.append({"sample": f"MC jitter {sigma:.0f} ns", "n": len(group),
                        "dt_iqr": iqr(group.dt),
                        "tail_50ns": float((group.dt.abs() > 50).mean()),
                        "n_hits_med": float(group.n_hits.median()),
                        "q_total_med": float(group.q_total.median()),
                        "t_span_med": float(group.t_span.median())})
    summary.append({"sample": "experiment", "n": len(experiment),
                    "dt_iqr": iqr(experiment.dt),
                    "tail_50ns": float((experiment.dt.abs() > 50).mean()),
                    "n_hits_med": float(experiment.n_hits.median()),
                    "q_total_med": np.nan, "t_span_med": np.nan})
    table = pd.DataFrame(summary)
    for _, row in table.iterrows():
        LOG.info("%-20s n=%7d  dt IQR %6.2f  tail %.3f  hits %.1f",
                 row["sample"], row["n"], row.dt_iqr, row.tail_50ns,
                 row.n_hits_med)

    provenance.write(jittered, data_dir / "41_jitter_residuals.parquet",
                     stage="41_jitter_validate", config_path=config_path,
                     inputs=[cfg.path("h5", "mc")], started=started)
    calibration = matched_calibration(jittered, experiment)
    needed = calibration.sigma_needed_ns
    LOG.info("matched cells %d, exp wider than unjittered MC in %d",
             len(calibration), int((calibration.iqr_exp > calibration.iqr_mc).sum()))
    LOG.info("input jitter needed to match experiment: median %.1f ns "
             "(IQR %.1f-%.1f)", needed.median(), needed.quantile(0.25),
             needed.quantile(0.75))
    scan = pd.read_parquet(data_dir / "40_jitter_summary.parquet")
    induced = np.interp([needed.quantile(0.25), needed.median(),
                         needed.quantile(0.75)], scan.sigma_ns, scan.induced_excess)
    LOG.info("which produces an excess of %.2f (range %.2f-%.2f); observed 2.87",
             induced[1], induced[0], induced[2])
    provenance.write(calibration, data_dir / "41_matched_calibration.parquet",
                     stage="41_jitter_validate", config_path=config_path,
                     inputs=[data_dir / "30_loo_residuals.parquet"], started=started,
                     notes={"sigma_needed_median_ns": float(needed.median()),
                            "induced_excess_at_that_sigma": float(induced[1])})

    provenance.write(table, data_dir / "41_validation.parquet",
                     stage="41_jitter_validate", config_path=config_path,
                     inputs=[cfg.path("h5", "mc"),
                             data_dir / "30_loo_residuals.parquet"], started=started)
    LOG.info("stage 41 done in %.1f s", time.time() - started)


if __name__ == "__main__":
    main()
