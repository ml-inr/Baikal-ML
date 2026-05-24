"""
Predict signal/noise probabilities for baikal_mc_merged.h5.

Source: baikal_mc_merged.h5 (raw, unnormalized)
Output: h5_catalogs/signoise_<model>_preds/catalogs_mc_merged/<particle>.parquet

Columns:
  - sig_prob (Float32): per-hit signal probability
  - gt_signoise (Boolean): ground truth signal flag (labels != 0)
  - part_num (Int32): h5 part number (extracted from ev_ids: {particle}_{part_num}_{local_id})
  - local_event_id (Int32): local event id within part (extracted from ev_ids)

Row ordering matches the catalog build order (parts sorted lexicographically),
so hit_start_idx / hit_end_idx from existing catalogs index directly.

Usage:
    python predict_mc_merged.py --particles muatm_2020
    python predict_mc_merged.py --particles nue2_2020 nuatm_2020
    python predict_mc_merged.py  # all particle types
    python predict_mc_merged.py --particles muatm_2020 --events-limit 10000
    python predict_mc_merged.py --device cpu
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from sig_noise_model_v2 import load_model, predict_flat
#from sig_noise_model_v2 import load_model, predict_flat

H5_PATH = "/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5"
CATALOGS_DIR = Path(__file__).resolve().parent.parent / "data_manager" / "h5_catalogs"


def main():
    parser = argparse.ArgumentParser(description="Predict sig/noise for mc_merged.")
    parser.add_argument("--particles", nargs="+", default=None,
                        help="Particle types. Default: all.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--events-limit", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="encoder_nl5_nh1_dff512_hs512_bs128")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    output_dir = CATALOGS_DIR / "catalogs_mc_merged" / f"signoise_{args.model_name}_preds"
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {"device": args.device}
    if args.checkpoint:
        load_kwargs["checkpoint_path"] = args.checkpoint
    if args.config:
        load_kwargs["config_path"] = args.config
    model, _, device = load_model(**load_kwargs)
    print(f"Model loaded on {device}.")

    with h5py.File(H5_PATH, "r") as h5f:
        particles = args.particles or [k for k in h5f.keys() if "raw" in h5f[k]]

        for particle in particles:
            print(f"\n{'='*60}")
            print(f"Particle: {particle}")

            parts = sorted(h5f[f"{particle}/raw/data"].keys())
            total_events = 0
            total_hits = 0

            out_path = output_dir / f"preds_{particle}.parquet"
            schema = pa.schema([
                ("sig_prob", pa.float32()),
                ("gt_signoise", pa.bool_()),
                ("part_num", pa.int32()),
                ("local_event_id", pa.int32()),
            ])

            with pq.ParquetWriter(str(out_path), schema, compression="zstd") as writer, \
                 tqdm(total=len(parts)) as pbar:
                for p_idx, part_name in enumerate(parts):
                    data = h5f[f"{particle}/raw/data/{part_name}/data"][:]
                    ev_starts = h5f[f"{particle}/raw/ev_starts/{part_name}/data"][:]
                    n_ev = len(ev_starts) - 1
                    pbar.set_description(f"{particle} | {part_name} | {n_ev} ev", refresh=(p_idx==0))

                    if args.events_limit is not None:
                        remaining = args.events_limit - total_events
                        if remaining <= 0:
                            break
                        n_ev = min(n_ev, remaining)
                        ev_starts = ev_starts[:n_ev + 1]
                        data = data[:int(ev_starts[-1])]

                    part_probs = predict_flat(
                        model, data, ev_starts, args.batch_size, device,
                        normalize=True,
                        desc="",
                    )

                    # Read ground truth labels
                    labels = h5f[f"{particle}/raw/labels/{part_name}/data"][:int(ev_starts[-1])]

                    # Extract part_num and local_event_id from ev_ids: "{particle}_{part_num}_{local_id}"
                    ev_ids = h5f[f"{particle}/ev_ids/{part_name}/data"][:n_ev]
                    part_nums = np.empty(n_ev, dtype=np.int32)
                    local_eids = np.empty(n_ev, dtype=np.int32)
                    for i, eid in enumerate(ev_ids):
                        parts_str = eid.decode().rsplit("_", 2)
                        part_nums[i] = int(parts_str[1])
                        local_eids[i] = int(parts_str[2])

                    # Broadcast per-event values to per-hit
                    hits_per_event = np.diff(ev_starts).astype(np.int32)

                    batch = pa.record_batch([
                        pa.array(part_probs, type=pa.float32()),
                        pa.array(labels != 0, type=pa.bool_()),
                        pa.array(np.repeat(part_nums, hits_per_event), type=pa.int32()),
                        pa.array(np.repeat(local_eids, hits_per_event), type=pa.int32()),
                    ], schema=schema)
                    writer.write_batch(batch)

                    total_events += n_ev
                    total_hits += len(part_probs)
                    if p_idx>0: pbar.update(1)

            print(f"  {particle}: {total_events} events, {total_hits} hits → {out_path.name}")
            print(f"  Size: {out_path.stat().st_size / 1e6:.1f} MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
