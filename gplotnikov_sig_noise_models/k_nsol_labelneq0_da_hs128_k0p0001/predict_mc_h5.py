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
    ptypes: Optional[List[str]] = None,
    parts_json: Optional[str] = None,
    append: bool = False,
) -> str:
    input_path  = Path(input_path).resolve()
    output_path = input_path.parent / (input_path.stem + f"_probs_{MODEL_TAG}.h5")

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    # "w" truncates: a plain re-run would destroy an existing 140 GB file that took hours
    # to produce. Appending is opt-in, and refuses to start if the file is missing.
    mode = "a" if append else "w"
    if append:
        if not output_path.exists():
            raise SystemExit(f"--append given but {output_path} does not exist")
        logger.info("APPEND mode: existing parts are kept and skipped")
    elif output_path.exists():
        raise SystemExit(
            f"{output_path} already exists ({output_path.stat().st_size/2**30:.1f} GB).\n"
            f"Refusing to truncate it. Pass --append to add the missing parts instead.")

    parts_filter: Optional[Dict[str, List[str]]] = None
    if parts_json:
        parts_filter = _load_parts_filter(parts_json)
        total = sum(len(v) for v in parts_filter.values())
        logger.info(f"Parts filter: {parts_json} ({total} parts total)")

    model, _, dev = load_model(device=device)
    logger.info(f"Model loaded on {dev}")

    t_total = time.time()

    with h5py.File(input_path, "r") as src, h5py.File(output_path, mode) as dst:
        # Appending with a different batch size would mix two hit selections in one file.
        # The sig-noise model's float padding mask makes batch size part of the selection;
        # see doc/sig_noise_batch_size.md.
        stored_bs = dst.attrs.get("sig_noise_batch_size")
        if stored_bs is not None and int(stored_bs) != batch_size:
            raise SystemExit(
                f"{output_path.name} was written with batch_size={int(stored_bs)}, "
                f"but this run uses {batch_size}. Resume with the stored value.")
        dst.attrs["sig_noise_batch_size"] = batch_size
        dst.attrs["sig_noise_model"] = MODEL_TAG

        src_keys = list(src.keys())
        if parts_filter:
            wanted = list(parts_filter.keys())
        elif ptypes:
            wanted = list(ptypes)
        else:
            wanted = list(PARTICLE_SHORT.values())

        processed = []
        for ptype in wanted:
            if ptype not in src:
                logger.warning(f"  {ptype} not found in source — skipping")
                continue
            processed.append(ptype)

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
            n_skipped = 0
            for i, pk in enumerate(part_keys):
                if f"{pk}/data" in dst_probs:      # already computed in an earlier run
                    n_skipped += 1
                    continue
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
                    n_done  = i + 1 - n_skipped
                    rate    = max(n_done, 1) / max(elapsed, 1e-3)
                    eta     = (n_parts - i - 1) / rate
                    logger.info(
                        f"  [{i+1}/{n_parts}] {pk} "
                        f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s, skipped {n_skipped})"
                    )
            if n_skipped:
                logger.info(f"  {ptype}: {n_skipped} parts already present, skipped")

    if not processed:
        # An empty output reported as success is worse than a crash: a run of
        # baikal_mc_reco.h5 wrote a 6 kB file and exited 0, because its groups are
        # named muatm / nuatm_conv / nuatm_prompt / nue2 while the default list
        # carries the mc_merged names.  Fail where the mistake happened.
        raise SystemExit(
            f"nothing was processed: none of {wanted} exists in {input_path.name}. "
            f"Available groups: {sorted(src_keys)}. Pass --ptypes explicitly.")
    logger.info(f"\nDone in {time.time() - t_total:.0f}s → {output_path} "
                f"({len(processed)} groups: {processed})")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", default="data_manager/data/h5datasets/baikal_mc_merged.h5",
    )
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--ptypes", default=None,
                        help="comma-separated top-level groups to process. Default "
                             "is the mc_merged 2020 set; baikal_mc_reco.h5 needs "
                             "muatm,nuatm_conv,nuatm_prompt,nue2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--parts-json", default=None,
        help="Optional path to parts JSON (e.g. testds_parts.json). "
             "If given, only those parts are processed.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Add missing parts to an existing output file instead of truncating it. "
             "Parts already present are skipped, so the run is resumable.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    predict_h5(
        ptypes=[p.strip() for p in args.ptypes.split(",")] if args.ptypes else None,
        input_path=args.input,
        device=args.device,
        batch_size=args.batch_size,
        parts_json=args.parts_json,
        append=args.append,
    )


if __name__ == "__main__":
    main()
