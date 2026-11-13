#!/usr/bin/env python3
"""Run a bundled angle/track/uncertainty model on a prepared Baikal HDF5 split.

Input hits must already have passed the signal/noise model. The bundled model
does not perform hit classification; it expects the selected-hit HDF5 schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from trained_models.angle_track_uncertainty.release_v3.model import load_bundle  # noqa: E402

FIELDS = (
    "event_id",
    "n_hits",
    "n_strings",
    "track_point_scope",
    "theta_deg",
    "phi_deg",
    "direction_x",
    "direction_y",
    "direction_z",
    "anchor_x_m",
    "anchor_y_m",
    "anchor_z_m",
    "sigma_angle_deg",
    "angle_r68_deg",
    "angle_r95_deg",
    "sigma_anchor_m",
    "anchor_r68_m",
    "anchor_r95_m",
)


def read_batch(h5: h5py.File, split: str, begin: int, end: int, prep: dict):
    group = h5[split]
    starts = np.asarray(group["ev_starts/data"][begin : end + 1], dtype=np.int64)
    lengths = np.diff(starts)
    if "num_un_strings" in group:
        strings = np.asarray(group["num_un_strings/data"][begin:end], dtype=np.int32)
    else:
        # Some GT-signal comparison files retain channels but omit the cached
        # per-event string count.  Reconstruct the same 36-channel grouping.
        channels = np.asarray(group["channels/data"][starts[0] : starts[-1]], dtype=np.int32)
        strings = np.asarray(
            [
                len(np.unique(channels[starts[i] - starts[0] : starts[i + 1] - starts[0]] // 36))
                for i in range(len(lengths))
            ],
            dtype=np.int32,
        )
    if np.any(lengths < 8) or np.any(strings < 2):
        raise ValueError("Input has events outside the required 2-string/8-hit selection")
    features = np.asarray(group["data/data"][starts[0] : starts[-1]], dtype=np.float32)
    ids = np.asarray(group["ev_ids/data"][begin:end])
    stored_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
    stored_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    expected_mean = np.asarray(prep["input_norm_mean"], dtype=np.float32)
    expected_std = np.asarray(prep["input_norm_std"], dtype=np.float32)
    if features.shape[1] != 5:
        raise ValueError(f"Expected five input features, got {features.shape[1]}")
    max_length = int(prep["max_seq_len"])
    used = np.minimum(lengths, max_length)
    x = np.zeros((end - begin, int(used.max()), 5), dtype=np.float32)
    mask = np.zeros(x.shape[:2], dtype=bool)
    centers = np.zeros((end - begin, 3), dtype=np.float32)
    for index, length in enumerate(lengths):
        event = features[starts[index] - starts[0] : starts[index + 1] - starts[0]]
        # The point head predicts an offset from the centroid of *all* selected
        # hits, even when the network sees only the first max_seq_len hits.
        centers[index] = (event[:, 2:5] * stored_std[2:5] + stored_mean[2:5]).mean(0)
        normalized = (
            event[: used[index]] * stored_std + stored_mean - expected_mean
        ) / expected_std
        if prep["center_time"]:
            normalized[:, 1] -= normalized[:, 1].mean()
        x[index, : used[index]] = normalized
        mask[index, : used[index]] = True
    return ids, lengths, strings, x, mask, centers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    model, bundle = load_bundle(args.checkpoint, device=args.device)
    prep = bundle["preprocessing"]
    factors = bundle["calibration"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    first_row = None
    with h5py.File(args.data, "r") as h5, args.output.open("w", newline="") as stream:
        if args.split not in h5:
            raise KeyError(f"Missing split {args.split!r} in {args.data}")
        total = len(h5[f"{args.split}/ev_starts/data"]) - 1
        if args.max_events is not None:
            total = min(total, args.max_events)
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for begin in range(0, total, args.batch_size):
            end = min(begin + args.batch_size, total)
            ids, lengths, strings, x, mask, centers = read_batch(h5, args.split, begin, end, prep)
            with torch.inference_mode():
                direction, offset, log_angle, log_anchor = model(
                    torch.from_numpy(x).to(args.device),
                    torch.from_numpy(mask).to(args.device),
                )
            direction = direction.cpu().numpy()
            anchor = centers + offset.cpu().numpy()
            sigma_angle = log_angle.exp().cpu().numpy()
            sigma_anchor = log_anchor.exp().cpu().numpy()
            theta = np.rad2deg(np.arccos(np.clip(direction[:, 2], -1, 1)))
            phi = np.mod(np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0])), 360)
            for index in range(end - begin):
                event_id = ids[index].decode() if isinstance(ids[index], bytes) else str(ids[index])
                row = {
                    "event_id": event_id,
                    "n_hits": int(lengths[index]),
                    "n_strings": int(strings[index]),
                    "track_point_scope": (
                        "validated_single_track_MC"
                        if event_id.startswith(("nuatm_", "nue2_"))
                        else "not_validated_for_possible_multi_track_or_experiment"
                    ),
                    "theta_deg": float(theta[index]),
                    "phi_deg": float(phi[index]),
                    "direction_x": float(direction[index, 0]),
                    "direction_y": float(direction[index, 1]),
                    "direction_z": float(direction[index, 2]),
                    "anchor_x_m": float(anchor[index, 0]),
                    "anchor_y_m": float(anchor[index, 1]),
                    "anchor_z_m": float(anchor[index, 2]),
                    "sigma_angle_deg": float(sigma_angle[index]),
                    "angle_r68_deg": float(min(180.0, sigma_angle[index] * factors["angle"]["68"])),
                    "angle_r95_deg": float(min(180.0, sigma_angle[index] * factors["angle"]["95"])),
                    "sigma_anchor_m": float(sigma_anchor[index]),
                    "anchor_r68_m": float(sigma_anchor[index] * factors["line"]["68"]),
                    "anchor_r95_m": float(sigma_anchor[index] * factors["line"]["95"]),
                }
                writer.writerow(row)
                if first_row is None:
                    first_row = row
    print(
        json.dumps(
            {
                "output": str(args.output),
                "events": total,
                "first_prediction": first_row,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
