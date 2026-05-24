"""Predict nu-classifier scores for MC events from baikal_mc_merged.h5.

Uses precomputed sig-noise probs from baikal_mc_merged_probs_*.h5 for hit
filtering and event cuts. Only parts from a JSON file are processed.
Streaming: one part at a time, never loading the full dataset into RAM.

Output: {output_dir}/mc_preds_{model_tag}.h5
Structure:
    /baikal_mc_merged/{ptype}/scores/{pk}/data          (n_sel,) float32
    /baikal_mc_merged/{ptype}/n_sig_hits/{pk}/data      (n_sel,) int32
    /baikal_mc_merged/{ptype}/n_sig_strings/{pk}/data   (n_sel,) int32
    /baikal_mc_merged/{ptype}/event_ids/{pk}/data       (n_sel,) int32

event_ids[i] is the 0-based event index within part pk in baikal_mc_merged.h5,
such that mc_h5[ptype]["raw"]["ev_starts"][pk]["data"][event_ids[i]] gives the
hit start for that event.

Usage (from project root):
    python inference/nu_classifier_model/predict_mc.py \\
        --checkpoint experiments/my_run/best.pt \\
        [--mc-h5 data_manager/data/h5datasets/baikal_mc_merged.h5] \\
        [--probs-h5 data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5] \\
        [--parts-json data_manager/nu_classifier_ds_builder/testds_parts.json] \\
        [--threshold 0.5] [--min-hits 5] [--min-strings 0] \\
        [--batch-size 512] [--device auto] \\
        [--output-dir inference/nu_classifier_model/results]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.nu_classifier_model.utils import load_model, predict_scores

logger = logging.getLogger(__name__)

PARTICLE_MAP: Dict[str, str] = {
    "muatm": "muatm_2020",
    "nue2":  "nue2_2020",
    "nuatm": "nuatm_2020",
}


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


def predict_mc(
    checkpoint: str,
    mc_h5: str,
    probs_h5: str,
    parts_json: str,
    threshold: float = 0.5,
    min_hits: int = 5,
    min_strings: int = 0,
    batch_size: int = 512,
    device: str = "auto",
    output_dir: str = "inference/nu_classifier_model/results",
) -> str:
    dev = _resolve_device(device)
    model_tag = Path(checkpoint).parent.name
    output_path = Path(output_dir) / f"mc_preds_{model_tag}.h5"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"MC h5:      {mc_h5}")
    logger.info(f"Probs h5:   {probs_h5}")
    logger.info(f"Output:     {output_path}")

    model, norm_config, train_config = load_model(checkpoint, device=dev)
    model.eval()

    with_probs = train_config.get("model", {}).get("input_dim", 5) == 6
    logger.info(f"with_probs={with_probs}, device={dev}")

    thr_str = str(threshold)

    with open(parts_json) as f:
        parts_dict = json.load(f)

    t_total = time.time()
    n_events_total = 0

    with (
        h5py.File(mc_h5,    "r") as mh,
        h5py.File(probs_h5, "r") as ph,
        h5py.File(output_path, "w") as dst,
    ):
        for ptype_short, part_nums in parts_dict.items():
            ptype_full = PARTICLE_MAP[ptype_short]
            if ptype_full not in mh:
                logger.warning(f"{ptype_full} not found in mc_h5 — skipping")
                continue
            if ptype_full not in ph:
                logger.warning(f"{ptype_full} not found in probs_h5 — skipping")
                continue

            sorted_parts = [f"part_{n}" for n in sorted(part_nums)]
            n_parts = len(sorted_parts)
            logger.info(f"\n{ptype_full}: {n_parts} parts")

            t0 = time.time()
            for i, pk in tqdm(enumerate(sorted_parts), total=len(sorted_parts), desc='Predicting parts'):
                n_sh_path = f"{ptype_full}/n_sig_hits_{thr_str}/{pk}/data"
                n_ss_path = f"{ptype_full}/n_sig_strings_{thr_str}/{pk}/data"
                if n_sh_path not in ph:
                    logger.warning(f"  {pk}: threshold key '{thr_str}' not in probs_h5 — skipping")
                    continue

                n_sig_hits    = ph[n_sh_path][:]
                n_sig_strings = ph[n_ss_path][:]
                ev_starts     = ph[f"{ptype_full}/ev_starts/{pk}/data"][:].astype(np.int64)
                probs         = ph[f"{ptype_full}/probs/{pk}/data"][:]
                data_raw      = mh[f"{ptype_full}/raw/data/{pk}/data"][:].astype(np.float32)

                cut_mask = (n_sig_hits >= min_hits) & (n_sig_strings >= min_strings)
                sel_events = np.where(cut_mask)[0]
                if len(sel_events) == 0:
                    continue

                features_list: List[np.ndarray] = []
                for ev_i in sel_events:
                    s, e = int(ev_starts[ev_i]), int(ev_starts[ev_i + 1])
                    sig_mask = probs[s:e] > threshold
                    feats = data_raw[s:e][sig_mask]
                    if with_probs:
                        feats = np.column_stack([feats, probs[s:e][sig_mask]])
                    features_list.append(feats)

                scores = predict_scores(
                    model, features_list, norm_config,
                    batch_size=batch_size, device=dev,
                    feats_with_probs=with_probs, with_tqdm=False
                )

                _append_to_h5(
                    dst, f"baikal_mc_merged/{ptype_full}", pk,
                    scores, n_sig_hits[sel_events], n_sig_strings[sel_events],
                    event_ids=sel_events,
                )
                n_events_total += len(sel_events)

                if (i + 1) % 20 == 0 or i == n_parts - 1:
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
        "--mc-h5",
        default="data_manager/data/h5datasets/baikal_mc_merged.h5",
    )
    parser.add_argument(
        "--probs-h5",
        default=(
            "data_manager/data/h5datasets/"
            "baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
        ),
    )
    parser.add_argument(
        "--parts-json",
        default="data_manager/nu_classifier_ds_builder/testds_parts.json",
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
    predict_mc(
        checkpoint=args.checkpoint,
        mc_h5=args.mc_h5,
        probs_h5=args.probs_h5,
        parts_json=args.parts_json,
        threshold=args.threshold,
        min_hits=args.min_hits,
        min_strings=args.min_strings,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
