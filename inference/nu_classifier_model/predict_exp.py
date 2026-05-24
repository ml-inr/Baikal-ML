"""Predict nu-classifier scores for experimental data from exp_reco.h5.

Runs the gplotnikov sig-noise model on-the-fly to filter hits per event.
Streaming: one part at a time.

Output: {output_dir}/exp_preds_{model_tag}.h5
Structure:
    /exp_reco/scores/{pk}/data          (n_sel,) float32
    /exp_reco/n_sig_hits/{pk}/data      (n_sel,) int32
    /exp_reco/n_sig_strings/{pk}/data   (n_sel,) int32
    /exp_reco/event_ids/{pk}/data       (n_sel,) int32

event_ids[i] is the 0-based event index within part pk in the source exp_reco.h5,
such that src["exp"]["raw"]["ev_starts"][pk]["data"][event_ids[i]] gives the
hit start for that event.

Usage (from project root):
    python inference/nu_classifier_model/predict_exp.py \\
        --checkpoint experiments/my_run/best.pt \\
        [--exp-h5 data_manager/data/h5datasets/exp_reco.h5] \\
        [--threshold 0.5] [--min-hits 5] [--min-strings 0] \\
        [--batch-size 512] [--device auto] \\
        [--output-dir inference/nu_classifier_model/results]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.nu_classifier_model.utils import load_model, predict_scores
from data_manager.nu_classifier_ds_builder.io import _count_sig_hits_strings

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _append_to_h5(
    dst: h5py.File,
    group: str,
    pk: str,
    scores: np.ndarray,
    n_sig_hits: np.ndarray,
    n_sig_strings: np.ndarray,
    event_ids: np.ndarray,
) -> None:
    grp = dst.require_group(group)
    for name, arr in [
        ("scores",        scores.astype(np.float32)),
        ("n_sig_hits",    n_sig_hits.astype(np.int32)),
        ("n_sig_strings", n_sig_strings.astype(np.int32)),
        ("event_ids",     event_ids.astype(np.int32)),
    ]:
        grp.create_dataset(f"{name}/{pk}/data", data=arr, compression="lzf")


def predict_exp(
    checkpoint: str,
    exp_h5: str,
    threshold: float = 0.5,
    min_hits: int = 5,
    min_strings: int = 0,
    batch_size: int = 512,
    device: str = "auto",
    output_dir: str = "inference/nu_classifier_model/results",
) -> str:
    dev = _resolve_device(device)
    model_tag = Path(checkpoint).parent.name
    output_path = Path(output_dir) / f"exp_preds_{model_tag}.h5"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Exp h5:     {exp_h5}")
    logger.info(f"Output:     {output_path}")

    model, norm_config, train_config = load_model(checkpoint, device=dev)
    model.eval()

    with_probs = train_config.get("model", {}).get("input_dim", 5) == 6
    logger.info(f"with_probs={with_probs}, device={dev}")

    # Load sig-noise model for on-the-fly hit filtering
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
        load_model as load_snmodel,
        predict_flat,
    )
    sn_model, _, sn_dev = load_snmodel(device=device)
    logger.info(f"Sig-noise model loaded on {sn_dev}")

    t_total = time.time()
    n_events_total = 0

    with h5py.File(exp_h5, "r") as src, h5py.File(output_path, "a") as dst:
        src_key = "exp_reco" if "exp_reco" in src else "exp"
        exp_grp = src[src_key]
        parts = sorted(exp_grp["raw"]["data"].keys())
        n_parts = len(parts)
        logger.info(f"\nexp: {n_parts} parts")

        t0 = time.time()
        for i, pk in tqdm(enumerate(parts), total=len(parts), desc='Predicting parts'):
            ev_starts = exp_grp[f"raw/ev_starts/{pk}/data"][:].astype(np.int64)
            data_raw  = exp_grp[f"raw/data/{pk}/data"][:].astype(np.float32)
            channels  = exp_grp[f"raw/channels/{pk}/data"][:].astype(np.int32)
            n_events  = len(ev_starts) - 1
            if n_events == 0:
                continue

            probs = predict_flat(
                sn_model, data_raw, ev_starts,
                batch_size=batch_size, device=sn_dev, normalize=True,
            )
            sig_mask_all = probs > threshold

            n_hits_arr = (ev_starts[1:] - ev_starts[:-1]).astype(np.int32)
            hit_starts  = ev_starts[:-1]
            n_sh, n_ss = _count_sig_hits_strings(
                sig_mask_all, channels, hit_starts, n_hits_arr, n_events,
            )

            cut_mask = (n_sh >= min_hits) & (n_ss >= min_strings)
            sel = np.where(cut_mask)[0]
            if len(sel) == 0:
                continue

            features_list: List[np.ndarray] = []
            for ev_i in sel:
                s, e = int(ev_starts[ev_i]), int(ev_starts[ev_i + 1])
                ev_sig = sig_mask_all[s:e]
                feats = data_raw[s:e][ev_sig]
                if with_probs:
                    feats = np.column_stack([feats, probs[s:e][ev_sig]])
                features_list.append(feats)

            scores = predict_scores(
                model, features_list, norm_config,
                batch_size=batch_size, device=dev,
                feats_with_probs=with_probs, with_tqdm=False
            )

            _append_to_h5(dst, src_key, pk, scores, n_sh[sel], n_ss[sel], event_ids=sel)
            n_events_total += len(sel)

            if (i + 1) % 5 == 0 or i == n_parts - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-3)
                eta = (n_parts - i - 1) / rate
                logger.info(
                    f"  [{i+1}/{n_parts}] {pk}  "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s, "
                    f"{n_events_total:,} total events written)"
                )

    logger.info(f"\nDone in {time.time() - t_total:.0f}s → {output_path}")
    logger.info(f"Total events written: {n_events_total:,}")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument(
        "--exp-h5",
        default="data_manager/data/h5datasets/exp.h5",
    )
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--min-hits",    type=int,   default=5)
    parser.add_argument("--min-strings", type=int,   default=0)
    parser.add_argument("--batch-size",  type=int,   default=512)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--output-dir",  default="inference/nu_classifier_model/results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    predict_exp(
        checkpoint=args.checkpoint,
        exp_h5=args.exp_h5,
        threshold=args.threshold,
        min_hits=args.min_hits,
        min_strings=args.min_strings,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
