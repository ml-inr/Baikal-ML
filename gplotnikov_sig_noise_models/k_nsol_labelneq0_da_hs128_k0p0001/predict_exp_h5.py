"""Precompute per-hit sig-noise probabilities for an exp-family HDF5 file.

Analog of ``predict_mc_h5.py`` for experimental data (single top-level group,
no ground-truth labels). Runs the sig-noise model ONCE over every event and
stores the result so downstream steps (NPY builder, predict, analysis) can read
probabilities instead of re-running the model.

Output (next to the input), mirroring the source part structure:
    {group}/probs/{part_key}/data          — (n_hits,) float32 sig_prob
    {group}/ev_starts/{part_key}/data       — (n_events+1,) int64
    {group}/channels/{part_key}/data        — (n_hits,) int32
    {group}/n_sig_hits_{thr}/{part}/data     — (n_events,) int32
    {group}/n_sig_strings_{thr}/{part}/data  — (n_events,) int32

Hit order within each part is identical to the source.

Usage (from project root):
    python gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/predict_exp_h5.py \
        --input data_manager/data/h5datasets/exp_full.h5 \
        [--group exp_full] [--device cuda:0] [--batch-size 512] \
        [--exclude-parts part_s2020_c02_r0020 part_s2020_c02_r0249]

Output: exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

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

MODEL_TAG  = "k_nsol_labelneq0_da_hs128_k0p0001"
THRESHOLDS = [0.5, 0.8]

# The sig-noise model passes a float padding mask, so padding stays visible to attention and
# a hit's probability depends on how much padding shares its batch. Batch size is therefore
# part of the hit-selection definition, and it must match the MC probs file (256) or MC and
# data get different selections. See doc/sig_noise_batch_size.md.
DEFAULT_BATCH_SIZE = 256


def predict_exp_h5(
    input_path: str,
    group: Optional[str] = None,
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    exclude_parts: Optional[List[str]] = None,
) -> str:
    input_path  = Path(input_path).resolve()
    output_path = input_path.parent / (input_path.stem + f"_probs_{MODEL_TAG}.h5")
    exclude = set(exclude_parts or [])

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    model, _, dev = load_model(device=device)
    logger.info(f"Model loaded on {dev}")

    t_total = time.time()
    # append mode → safe to resume: skip parts already written
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "a") as dst:
        # Resuming with a different batch size would silently mix two hit selections in
        # one file, which is exactly the inconsistency this convention exists to prevent.
        stored_bs = dst.attrs.get("sig_noise_batch_size")
        if stored_bs is not None and int(stored_bs) != batch_size:
            raise SystemExit(
                f"{output_path.name} was written with batch_size={int(stored_bs)}, "
                f"but this run uses {batch_size}. Resume with the stored value or start "
                f"a new file — see doc/sig_noise_batch_size.md")
        dst.attrs["sig_noise_batch_size"] = batch_size
        dst.attrs["sig_noise_model"] = MODEL_TAG

        top = group or list(src.keys())[0]
        grp_src = src[top]["raw"]
        part_keys = sorted(k for k in grp_src["data"].keys() if k.startswith("part_"))
        part_keys = [p for p in part_keys if p not in exclude]
        n_parts = len(part_keys)
        logger.info(f"group='{top}': {n_parts} parts ({len(exclude)} excluded)")

        dst_probs     = dst.require_group(f"{top}/probs")
        dst_ev_starts = dst.require_group(f"{top}/ev_starts")
        dst_channels  = dst.require_group(f"{top}/channels")
        dst_thr = {
            thr: (dst.require_group(f"{top}/n_sig_hits_{thr}"),
                  dst.require_group(f"{top}/n_sig_strings_{thr}"))
            for thr in THRESHOLDS
        }

        t0 = time.time()
        for i, pk in enumerate(part_keys):
            if f"{pk}/data" in dst_probs:          # resume: already done
                continue
            ev_starts = grp_src[f"ev_starts/{pk}/data"][:].astype(np.int64)
            if len(ev_starts) < 2:
                continue
            n_events   = len(ev_starts) - 1
            hit_starts = ev_starts[:-1]
            n_hits     = (ev_starts[1:] - ev_starts[:-1]).astype(np.int32)
            data_raw   = grp_src[f"data/{pk}/data"][:].astype(np.float32)
            channels   = grp_src[f"channels/{pk}/data"][:].astype(np.int32)

            prob = predict_flat(
                model=model, data=data_raw, ev_starts=ev_starts,
                batch_size=batch_size, device=dev, normalize=True,
            )

            for thr, (grp_hits, grp_str) in dst_thr.items():
                n_sh, n_ss = _count_sig_hits_strings(
                    prob > thr, channels, hit_starts, n_hits, n_events,
                )
                grp_hits.create_dataset(f"{pk}/data", data=n_sh, compression="lzf")
                grp_str.create_dataset( f"{pk}/data", data=n_ss, compression="lzf")

            dst_probs.create_dataset(    f"{pk}/data", data=prob,      compression="lzf")
            dst_ev_starts.create_dataset(f"{pk}/data", data=ev_starts, compression="lzf")
            dst_channels.create_dataset( f"{pk}/data", data=channels,  compression="lzf")
            dst.flush()

            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-3)
            eta     = (n_parts - i - 1) / rate
            logger.info(f"  [{i+1}/{n_parts}] {pk}  n_ev={n_events:,}  "
                        f"({elapsed:.0f}s, ETA {eta/60:.0f}m)")

    logger.info(f"\nDone in {(time.time() - t_total)/60:.0f}m → {output_path}")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default="data_manager/data/h5datasets/exp_full.h5")
    parser.add_argument("--group", default=None, help="Top-level h5 group (auto if omitted)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Must match the MC probs file (256). Changing it changes "
                             "which hits pass the signal threshold.")
    parser.add_argument("--exclude-parts", nargs="*", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    predict_exp_h5(
        input_path=args.input, group=args.group, device=args.device,
        batch_size=args.batch_size, exclude_parts=args.exclude_parts,
    )


if __name__ == "__main__":
    main()
