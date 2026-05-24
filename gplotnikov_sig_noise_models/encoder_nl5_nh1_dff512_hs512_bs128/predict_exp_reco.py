"""
Predict signal/noise probabilities for exp_reco.h5.

Source: exp_reco.h5 (raw, unnormalized)
Output: h5_catalogs/signoise_<model>_preds/catalogs_exp_reco/exp_reco.parquet

Columns:
  - sig_prob (Float32): per-hit signal probability
  - gt_signoise (Boolean): ScanfitMask signal flag (labels != 0)
  - season_id (Int32): e.g. 2020
  - cluster_id (Int32): e.g. 1
  - run_id (Int32): e.g. 27
  - local_event_id (Int32): event index within the part (0-based)

Row ordering matches the catalog build order (parts sorted lexicographically),
so hit_start_idx / hit_end_idx from the existing catalog index directly.

Usage:
    python predict_exp_reco.py
    python predict_exp_reco.py --events-limit 10000
    python predict_exp_reco.py --device cpu
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from tqdm import tqdm

#from sig_noise_model import load_model, predict_flat
from sig_noise_model_v2 import load_model, predict_flat

H5_PATH = Path(__file__).resolve().parent.parent / "data_manager" / "data" / "h5datasets" / "exp_reco.h5"
CATALOGS_DIR = Path(__file__).resolve().parent.parent / "data_manager" / "h5_catalogs"


def parse_exp_reco_part_name(part_name: str) -> tuple:
    """'part_s2020_c01_r0027' → (2020, 1, 27)"""
    m = re.search(r's(\d+)_c(\d+)_r(\d+)', part_name)
    if not m:
        raise ValueError(f"Cannot parse part name: {part_name}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    parser = argparse.ArgumentParser(description="Predict sig/noise for exp_reco.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--events-limit", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="encoder_nl5_nh1_dff512_hs512_bs128")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    output_dir = CATALOGS_DIR / "catalogs_exp_reco" / f"signoise_{args.model_name}_preds"
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {"device": args.device}
    if args.checkpoint:
        load_kwargs["checkpoint_path"] = args.checkpoint
    if args.config:
        load_kwargs["config_path"] = args.config
    model, _, device = load_model(**load_kwargs)
    print(f"Model loaded on {device}.")

    with h5py.File(str(H5_PATH), "r") as h5f:
        parts = sorted(h5f["exp_reco/raw/data"].keys())
        print(f"exp_reco: {len(parts)} parts")

        all_sig_prob = []
        all_gt_signoise = []
        all_season = []
        all_cluster = []
        all_run = []
        all_local_eid = []
        total_events = 0

        with tqdm(total=len(parts), desc="exp_reco") as pbar:
            for part_name in parts:
                data = h5f[f"exp_reco/raw/data/{part_name}/data"][:]
                ev_starts = h5f[f"exp_reco/raw/ev_starts/{part_name}/data"][:]
                n_ev = len(ev_starts) - 1

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

                # Read ground truth labels (ScanfitMask)
                labels = h5f[f"exp_reco/raw/labels/{part_name}/data"][:int(ev_starts[-1])]
                gt_signoise = labels != 0

                season_id, cluster_id, run_id = parse_exp_reco_part_name(part_name)
                hits_per_event = np.diff(ev_starts).astype(np.int32)
                n_hits = len(part_probs)

                all_sig_prob.append(part_probs)
                all_gt_signoise.append(gt_signoise)
                all_season.append(np.full(n_hits, season_id, dtype=np.int32))
                all_cluster.append(np.full(n_hits, cluster_id, dtype=np.int32))
                all_run.append(np.full(n_hits, run_id, dtype=np.int32))
                all_local_eid.append(np.repeat(np.arange(n_ev, dtype=np.int32), hits_per_event))
                total_events += n_ev
                pbar.set_description(f"exp_reco | {part_name} | {n_ev} ev, {n_hits} hits")
                pbar.update(1)

    sig_prob = np.concatenate(all_sig_prob)
    gt_signoise = np.concatenate(all_gt_signoise)
    season_id = np.concatenate(all_season)
    cluster_id = np.concatenate(all_cluster)
    run_id = np.concatenate(all_run)
    local_event_id = np.concatenate(all_local_eid)

    out_path = output_dir / "preds_exp_reco.parquet"
    pl.DataFrame({
        "sig_prob": pl.Series(sig_prob, dtype=pl.Float32),
        "gt_signoise": pl.Series(gt_signoise, dtype=pl.Boolean),
        "season_id": pl.Series(season_id, dtype=pl.Int32),
        "cluster_id": pl.Series(cluster_id, dtype=pl.Int32),
        "run_id": pl.Series(run_id, dtype=pl.Int32),
        "local_event_id": pl.Series(local_event_id, dtype=pl.Int32),
    }).write_parquet(out_path)
    print(f"\n  {total_events} events, {len(sig_prob)} hits → {out_path.name}")
    print(f"  Size: {out_path.stat().st_size / 1e6:.1f} MB")
    print("\nDone.")


if __name__ == "__main__":
    main()
