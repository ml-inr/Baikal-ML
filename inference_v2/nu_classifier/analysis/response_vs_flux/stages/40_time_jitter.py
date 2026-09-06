"""Stage 40 -- PROTOCOL test 3: does the measured response error reproduce the excess?

Stage 31 measured what MC lacks: matched on geometry, experimental hits scatter
around the fitted Cherenkov track **28.7 ns** more than simulated ones (median
over 66 cells, all 66 in the same direction).  The measured per-channel
calibration residual is about 10 ns, so most of that gap is not calibration.

This stage feeds the measurement back: add exactly that much Gaussian jitter to
MC hit times, re-run the sig-noise filter and the classifier from scratch, and
see how much excess it manufactures.  The perturbation is therefore *calibrated
by an independent measurement*, not tuned until the answer comes out right --
which is what separates this from fitting the excess with a free parameter.

Both networks are re-run, not just the classifier.  Jitter changes which hits
survive the sig-noise filter, and skipping that step would answer a different
question.  The sig-noise batch size is pinned at 256, the pipeline convention:
it is not a performance knob, it defines the hit selection.

What to read:

* ``acceptance`` at each sigma, and its ratio to sigma = 0.  If ~28 ns of jitter
  produces the observed factor of ~2.9, a response error of the measured size is
  sufficient to explain the excess.
* if it produces far less, the measured response error is real but not the cause,
  and the excess needs something else.

Usage:
    nohup python stages/40_time_jitter.py > /tmp/stage40.log 2>&1 &
"""
from __future__ import annotations

import os

# Must be set before torch is imported anywhere.  PyTorch's default device order
# is "fastest first", so `cuda:N` is NOT nvidia-smi's index N: on this machine the
# default order lists the three RTX 5000 cards first and the P100 last, so a run
# asking for cuda:1 landed on nvidia-smi index 2, on top of another user's 21 GB
# job.  Pinning the order to the PCI bus makes the device in config.yaml mean what
# nvidia-smi shows.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import logging
import sys
import time
from pathlib import Path

from zlib import crc32

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parents[3]))

import h5io                                                    # noqa: E402
import provenance                                              # noqa: E402

LOG = logging.getLogger("stage40")
RDCC = {"rdcc_nbytes": 256 * 1024 * 1024, "rdcc_nslots": 100_003}
BANDS = [(0.5, 0.8), (0.8, 0.9), (0.9, 1.01)]
MIN_HITS, MIN_STRINGS = 8, 3


def scan_part(part: str, h5_group: str, cfg: h5io.Config, model, norm, with_probs,
              sn_model, sn_device, predict_flat, predict_scores,
              count_hits_strings, rng) -> list[dict]:
    """Score one MC part at every jitter level, re-running both networks."""
    with h5py.File(cfg.path("h5", "mc"), "r", **RDCC) as src:
        starts = src[f"{h5_group}/raw/ev_starts/{part}/data"][:].astype(np.int64)
        data = src[f"{h5_group}/raw/data/{part}/data"][:].astype(np.float32)
        channels = src[f"{h5_group}/raw/channels/{part}/data"][:]
    n_events = len(starts) - 1
    n_hits = (starts[1:] - starts[:-1]).astype(np.int32)
    rows = []
    for sigma in cfg["stage40"]["jitter_sigma_ns"]:
        perturbed = data.copy()
        if sigma > 0:
            perturbed[:, 1] += rng.normal(0.0, sigma, len(perturbed)).astype(np.float32)
        probs = predict_flat(sn_model, perturbed, starts,
                             batch_size=int(cfg["stage40"]["sn_batch_size"]),
                             device=sn_device, normalize=True)
        mask = probs > float(cfg["sig_noise_threshold"])
        n_sn_h, n_sn_s = count_hits_strings(mask, channels, starts[:-1], n_hits,
                                            n_events)
        selected = np.where((n_sn_h >= MIN_HITS) & (n_sn_s >= MIN_STRINGS))[0]
        if len(selected) == 0:
            continue
        features = []
        for index in selected:
            lo, hi = int(starts[index]), int(starts[index + 1])
            event = perturbed[lo:hi][mask[lo:hi]]
            if with_probs:
                event = np.column_stack([event, probs[lo:hi][mask[lo:hi]]])
            features.append(event)
        scores = predict_scores(model, features, norm,
                                batch_size=int(cfg["stage40"]["batch_size"]),
                                device=cfg["stage40"]["device"],
                                feats_with_probs=with_probs, with_tqdm=False)
        row = {"part": part, "sigma_ns": float(sigma),
               "n_events": n_events, "n_quality": int(len(selected)),
               "mean_sn_hits": float(n_sn_h[selected].mean()),
               }
        for lo, hi in BANDS:
            row[f"n_{lo}"] = int(((scores >= lo) & (scores < hi)).sum())
        row["n_accepted"] = int((scores >= 0.8).sum())
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = h5io.load_config(HERE)
    data_dir, config_path = HERE / "data", HERE / "config.yaml"

    from inference_v2.shared.model_utils import (load_model, load_sn_model,
                                                 predict_scores)
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (  # noqa: E501
        predict_flat)
    from data_manager.nu_classifier_ds_builder.io import _count_sig_hits_strings

    device = cfg["stage40"]["device"]
    import torch
    if device.startswith("cuda"):
        index = int(device.split(":")[1]) if ":" in device else 0
        free, total = torch.cuda.mem_get_info(index)
        LOG.info("using %s = %s, %.1f of %.1f GB free", device,
                 torch.cuda.get_device_name(index), free / 1e9, total / 1e9)
        if free / total < 0.5:
            LOG.warning("%s is more than half occupied by other work", device)
    checkpoint = cfg.root / cfg["stage40"]["checkpoint"]
    model, norm, train_config = load_model(str(checkpoint), device=device)
    with_probs = train_config.get("model", {}).get("input_dim", 5) == 6
    sn_model, _, sn_device = load_sn_model(device=device)
    LOG.info("classifier and sig-noise model on %s (input_dim %d)",
             device, 6 if with_probs else 5)

    h5_group = "muatm_2020"
    with h5py.File(cfg.path("h5", "mc"), "r") as src:
        parts = sorted(src[h5_group]["raw"]["ev_starts"].keys())
    # crc32, not Python's hash(): string hashing is salted per process, so
    # hash() selected a different set of parts on every run and the stage was
    # not reproducible -- two runs disagreed by 9% on the calibrated jitter.
    order = np.argsort([crc32(p.encode()) for p in parts])
    n_parts = 2 if args.smoke else int(cfg["stage40"]["mc_parts"])
    parts = [parts[i] for i in order[:n_parts]]
    LOG.info("%d parts, jitter levels %s", len(parts),
             cfg["stage40"]["jitter_sigma_ns"])

    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for index, part in enumerate(parts, 1):
        rows.extend(scan_part(part, h5_group, cfg, model, norm, with_probs,
                              sn_model, sn_device, predict_flat, predict_scores,
                              _count_sig_hits_strings, rng))
        if index % 5 == 0 or index == len(parts):
            LOG.info("%d/%d parts, %.0f s", index, len(parts),
                     time.time() - started)

    frame = pd.DataFrame(rows)
    summary = frame.groupby("sigma_ns").agg(
        n_events=("n_events", "sum"), n_quality=("n_quality", "sum"),
        n_accepted=("n_accepted", "sum"),
        mean_sn_hits=("mean_sn_hits", "mean")).reset_index()
    summary["acceptance"] = summary.n_accepted / summary.n_quality
    baseline = summary.loc[summary.sigma_ns == 0, "acceptance"]
    summary["induced_excess"] = summary.acceptance / float(baseline.iloc[0])
    for _, row in summary.iterrows():
        LOG.info("sigma %5.1f ns: quality %8d, accepted %6d, acceptance %.3e, "
                 "induced excess %.2f", row.sigma_ns, row.n_quality,
                 row.n_accepted, row.acceptance, row.induced_excess)

    for table, name in ((frame, "40_jitter_parts"), (summary, "40_jitter_summary")):
        provenance.write(table, data_dir / f"{name}.parquet", stage="40_time_jitter",
                         config_path=config_path, inputs=[cfg.path("h5", "mc")],
                         started=started)
    LOG.info("stage 40 done in %.1f s", time.time() - started)


if __name__ == "__main__":
    main()
