"""Predict per-hit signal probabilities for all events in baikal_mc_merged.h5.

Output: a new HDF5 file next to the source, mirroring the parts structure:
    {particle_type}/probs/{part_key}/data      — (n_hits,) float32 sig_prob
    {particle_type}/ev_starts/{part_key}/data  — (n_events+1,) int64
    {particle_type}/channels/{part_key}/data   — (n_hits,) int32

The order of hits within each part is identical to the source file.

Usage (from project root):
    python gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/predict_mc_h5.py \
        [--input data_manager/data/h5datasets/baikal_mc_merged.h5] \
        [--device cuda:0] [--batch-size 256]

Output file is written next to the input:
    baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5
"""

# ==============================================================================
#  DISABLED ON PURPOSE — this script would overwrite the canonical probs file.
#  See doc/AUDIT.md §3.1 and question Q1.
#
#  This file is a byte-for-byte copy of
#      gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/predict_mc_h5.py
#  that was never adapted to the model living in this directory:
#
#    * it imports load_model from ...k0p0001.sig_noise_model_v3 and calls it
#      without checkpoint_path, so it scores with the k0p0001 checkpoint —
#      not with best_aug_p*.ckpt sitting next to this file;
#    * MODEL_TAG is still "k_nsol_labelneq0_da_hs128_k0p0001", and the output
#      name is built from it, so a run would write over
#          data_manager/data/h5datasets/
#              baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5
#      — the canonical 189 GB probs file that the whole nu-classifier chain
#      reads.
#
#  The directory is inconsistent beyond this script, so there is no one-line
#  fix: train_config_mc_2020.yaml here is a copy of the hidden_size=128 config,
#  while best_aug_p*.ckpt has first_layer.weight of shape (512, 5) with no
#  "encoder." prefix (63 tensors). The local model_simplified.py is likewise a
#  copy of the hidden_size=128 flavour. A shape-compatible module set lives in
#  gplotnikov_sig_noise_models/k_nsol_labelneq0_hs512_dff512/.
#
#  Per the owner (Q1) the aug_highq checkpoints were never used and the probs
#  file was produced by k0p0001, so no existing artifact is affected. This
#  guard exists so that a future run cannot affect one either.
#
#  To actually score with an augmented checkpoint, fix the module set and the
#  config first, and give the output a MODEL_TAG of its own.
# ==============================================================================
if __name__ == "__main__":
    raise SystemExit(
        "REFUSING TO RUN: this copy is not wired to the checkpoint in its own "
        "directory.\n"
        "It would score with k_nsol_labelneq0_da_hs128_k0p0001 and overwrite\n"
        "  data_manager/data/h5datasets/"
        "baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5 (189 GB).\n"
        "See the comment block at the top of this file, and doc/AUDIT.md §3.1."
    )



import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
    load_model,
    predict_flat,
)
from data_manager.nu_classifier_ds_builder.io import _count_sig_hits_strings

logger = logging.getLogger(__name__)

PARTICLE_SHORT = {"muatm": "muatm_2020", "nuatm": "nuatm_2020", "nue2": "nue2_2020"}
MODEL_TAG      = "k_nsol_labelneq0_da_hs128_k0p0001"
THRESHOLDS     = [0.5, 0.8]


def _load_parts_filter(parts_json: str) -> Dict[str, List[str]]:
    """Return {ptype_full: [part_key, ...]} from a parts JSON file."""
    with open(parts_json) as f:
        raw = json.load(f)
    return {
        PARTICLE_SHORT[short]: [f"part_{n}" for n in nums]
        for short, nums in raw.items()
        if short in PARTICLE_SHORT
    }


def predict_h5(
    input_path: str,
    device: str = "auto",
    batch_size: int = 256,
    parts_json: Optional[str] = None,
) -> str:
    input_path  = Path(input_path).resolve()
    output_path = input_path.parent / (input_path.stem + f"_probs_{MODEL_TAG}.h5")

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    parts_filter: Optional[Dict[str, List[str]]] = None
    if parts_json:
        parts_filter = _load_parts_filter(parts_json)
        total = sum(len(v) for v in parts_filter.values())
        logger.info(f"Parts filter: {parts_json} ({total} parts total)")

    model, _, dev = load_model(device=device)
    logger.info(f"Model loaded on {dev}")

    t_total = time.time()

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        for ptype in (parts_filter.keys() if parts_filter else PARTICLE_SHORT.values()):
            if ptype not in src:
                logger.warning(f"  {ptype} not found in source — skipping")
                continue

            grp_src = src[ptype]["raw"]
            if parts_filter:
                part_keys = sorted(parts_filter[ptype])
            else:
                part_keys = sorted(grp_src["data"].keys())
            n_parts = len(part_keys)
            logger.info(f"\n{ptype}: {n_parts} parts")

            dst_probs     = dst.require_group(f"{ptype}/probs")
            dst_ev_starts = dst.require_group(f"{ptype}/ev_starts")
            dst_channels  = dst.require_group(f"{ptype}/channels")
            dst_gt_hits   = dst.require_group(f"{ptype}/n_gt_sig_hits")
            dst_gt_str    = dst.require_group(f"{ptype}/n_gt_sig_strings")
            dst_thr = {
                thr: (
                    dst.require_group(f"{ptype}/n_sig_hits_{thr}"),
                    dst.require_group(f"{ptype}/n_sig_strings_{thr}"),
                )
                for thr in THRESHOLDS
            }

            t0 = time.time()
            for i, pk in enumerate(part_keys):
                ev_starts = grp_src[f"ev_starts/{pk}/data"][:].astype(np.int64)
                if len(ev_starts) < 2:
                    continue
                n_events   = len(ev_starts) - 1
                hit_starts = ev_starts[:-1]
                n_hits     = (ev_starts[1:] - ev_starts[:-1]).astype(np.int32)
                data_raw   = grp_src[f"data/{pk}/data"][:].astype(np.float32)
                channels   = grp_src[f"channels/{pk}/data"][:].astype(np.int32)
                gt_labels  = grp_src[f"labels/{pk}/data"][:]

                prob = predict_flat(
                    model=model,
                    data=data_raw,
                    ev_starts=ev_starts,
                    batch_size=batch_size,
                    device=dev,
                    normalize=True,
                )

                # Per-threshold signal counts
                for thr, (grp_hits, grp_str) in dst_thr.items():
                    n_sh, n_ss = _count_sig_hits_strings(
                        prob > thr, channels, hit_starts, n_hits, n_events,
                    )
                    grp_hits.create_dataset(f"{pk}/data", data=n_sh, compression="lzf")
                    grp_str.create_dataset( f"{pk}/data", data=n_ss, compression="lzf")

                # GT signal counts
                n_gt_sh, n_gt_ss = _count_sig_hits_strings(
                    gt_labels != 0, channels, hit_starts, n_hits, n_events,
                )
                dst_gt_hits.create_dataset(f"{pk}/data", data=n_gt_sh, compression="lzf")
                dst_gt_str.create_dataset( f"{pk}/data", data=n_gt_ss, compression="lzf")

                # Store in same order as source — no shuffling
                dst_probs.create_dataset(    f"{pk}/data", data=prob,       compression="lzf")
                dst_ev_starts.create_dataset(f"{pk}/data", data=ev_starts,  compression="lzf")
                dst_channels.create_dataset( f"{pk}/data", data=channels,   compression="lzf")

                if (i + 1) % 20 == 0 or i == n_parts - 1:
                    elapsed = time.time() - t0
                    rate    = (i + 1) / max(elapsed, 1e-3)
                    eta     = (n_parts - i - 1) / rate
                    logger.info(
                        f"  [{i+1}/{n_parts}] {pk} "
                        f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
                    )

    logger.info(f"\nDone in {time.time() - t_total:.0f}s → {output_path}")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", default="data_manager/data/h5datasets/baikal_mc_merged.h5",
    )
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--parts-json", default=None,
        help="Optional path to parts JSON (e.g. testds_parts.json). "
             "If given, only those parts are processed.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    predict_h5(
        input_path=args.input,
        device=args.device,
        batch_size=args.batch_size,
        parts_json=args.parts_json,
    )


if __name__ == "__main__":
    main()
