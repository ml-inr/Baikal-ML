#!/usr/bin/env python3
"""Audit a bundled direction/point model after rethresholding prepared hits.

The HDF5 must contain per-hit signal probabilities and channel IDs. Events
that fail the post-threshold two-string/eight-hit cut are omitted. Format-2
bundles output calibrated radii only for unchanged p>=0.70 events; low-threshold
reruns expose raw uncertainty scales but leave nominal radii blank.
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

from trained_models.angle_track_uncertainty.release_v3.model import (  # noqa: E402
    load_bundle,
)

FIELDS = (
    "event_id",
    "source_event_index",
    "n_hits",
    "n_strings",
    "theta_deg",
    "phi_deg",
    "direction_x",
    "direction_y",
    "direction_z",
    "anchor_x_m",
    "anchor_y_m",
    "anchor_z_m",
)
UNCERTAINTY_FIELDS = (
    "track_point_scope",
    "sigma_angle_deg",
    "angle_r68_deg",
    "angle_r95_deg",
    "sigma_anchor_m",
    "anchor_r68_m",
    "anchor_r95_m",
    "uncertainty_calibration_scope",
)


def selected_batch(
    group: h5py.Group,
    begin: int,
    end: int,
    threshold: float,
    prep: dict,
    stored_mean: np.ndarray,
    stored_std: np.ndarray,
    channels_per_string: int,
    sample_start_event: int,
    event_stride: int,
    score_feature: bool = False,
    center_time_alpha: float = 0.0,
):
    starts = np.asarray(group["ev_starts/data"][begin : end + 1], dtype=np.int64)
    first, last = int(starts[0]), int(starts[-1])
    features = np.asarray(group["data/data"][first:last], dtype=np.float32)
    probabilities = np.asarray(group["signal_probability/data"][first:last], dtype=np.float32)
    channels = np.asarray(group["channels/data"][first:last], dtype=np.int32)
    ids = np.asarray(group["ev_ids/data"][begin:end])
    expected_mean = np.asarray(prep["input_norm_mean"], dtype=np.float32)
    expected_std = np.asarray(prep["input_norm_std"], dtype=np.float32)
    max_len = int(prep["max_seq_len"])

    kept = []
    for i, event_id in enumerate(ids):
        if (begin + i - sample_start_event) % event_stride:
            continue
        lo, hi = int(starts[i] - first), int(starts[i + 1] - first)
        choose = probabilities[lo:hi] >= threshold
        hit_count = int(choose.sum())
        if hit_count < 8:
            continue
        event_channels = channels[lo:hi][choose]
        n_strings = len(np.unique(event_channels // channels_per_string))
        if n_strings < 2:
            continue
        selected = features[lo:hi][choose]
        center_m = (selected[:, 2:5] * stored_std[2:5] + stored_mean[2:5]).mean(0)
        normalized = (selected[:max_len] * stored_std + stored_mean - expected_mean) / expected_std
        alpha = 1.0 if prep["center_time"] else center_time_alpha
        if alpha:
            normalized[:, 1] -= alpha * normalized[:, 1].mean()
        if score_feature:
            selected_probabilities = probabilities[lo:hi][choose][:max_len, None]
            normalized = np.concatenate((normalized, selected_probabilities), axis=1)
        kept.append(
            (
                event_id.decode() if isinstance(event_id, bytes) else str(event_id),
                begin + i,
                hit_count,
                n_strings,
                center_m,
                normalized,
            )
        )
    if not kept:
        return [], None, None
    max_used = max(len(row[5]) for row in kept)
    x = np.zeros((len(kept), max_used, 5 + int(score_feature)), dtype=np.float32)
    mask = np.zeros((len(kept), max_used), dtype=bool)
    for i, row in enumerate(kept):
        used = len(row[5])
        x[i, :used] = row[5]
        mask[i, :used] = True
    return kept, x, mask


def added_hits_pass_photon_gate(
    group: h5py.Group,
    begin: int,
    end: int,
    source_indices: np.ndarray,
    directions: np.ndarray,
    anchors: np.ndarray,
    high_threshold: float,
    low_threshold: float,
    maximum_residual_ns: float,
    stored_mean: np.ndarray,
    stored_std: np.ndarray,
) -> np.ndarray:
    """Check every added hit against the first-pass track and high-hit clock."""
    starts = np.asarray(group["ev_starts/data"][begin : end + 1], dtype=np.int64)
    first, last = int(starts[0]), int(starts[-1])
    features = np.asarray(group["data/data"][first:last], dtype=np.float64)
    features = features * stored_std.astype(np.float64) + stored_mean.astype(np.float64)
    probabilities = np.asarray(group["signal_probability/data"][first:last])
    passed = np.zeros(len(source_indices), dtype=bool)
    photon_factor = np.sqrt(1.36**2 - 1.0)
    for j, source_index in enumerate(source_indices):
        event = int(source_index) - begin
        lo, hi = int(starts[event] - first), int(starts[event + 1] - first)
        probability = probabilities[lo:hi]
        high = probability >= high_threshold
        added = (probability >= low_threshold) & ~high
        if not high.any() or not added.any():
            continue
        direction = directions[j] / np.linalg.norm(directions[j])
        delta = features[lo:hi, 2:5] - anchors[j]
        along = delta @ direction
        transverse = np.linalg.norm(delta - along[:, None] * direction, axis=1)
        expected_ns = (along + photon_factor * transverse) / 0.299792458
        residual_ns = features[lo:hi, 1] - expected_ns
        clock_ns = np.median(residual_ns[high])
        passed[j] = np.max(np.abs(residual_ns[added] - clock_ns)) < maximum_residual_ns
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--conditional-low-threshold",
        type=float,
        help="Rerun upward events only when the lower threshold adds selected hits",
    )
    parser.add_argument("--gate-theta-max-deg", type=float, default=90.0)
    parser.add_argument(
        "--max-added-photon-residual-ns",
        type=float,
        help="Require every newly selected hit to agree with the first-pass photon clock",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--start-event", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--event-stride", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--channels-per-string", type=int, default=36)
    parser.add_argument(
        "--center-time-alpha",
        type=float,
        default=0.0,
        help="Diagnostic removal of the event-mean time from the model input; 0 preserves the bundle",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not 0 < args.threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if args.conditional_low_threshold is not None and not (
        0 < args.conditional_low_threshold < args.threshold
    ):
        raise ValueError("Conditional threshold must be positive and below the first threshold")
    if not 0 < args.gate_theta_max_deg < 180:
        raise ValueError("Gate theta must be between 0 and 180 degrees")
    if args.max_added_photon_residual_ns is not None and (
        args.max_added_photon_residual_ns <= 0 or args.conditional_low_threshold is None
    ):
        raise ValueError("Positive photon gate requires a conditional low threshold")
    if args.batch_size < 1 or args.threads < 1 or args.channels_per_string < 1:
        raise ValueError("Batch size, threads, and channels per string must be positive")
    if args.start_event < 0 or (args.max_events is not None and args.max_events < 1):
        raise ValueError("Start event must be nonnegative and max events positive")
    if args.event_stride < 1:
        raise ValueError("Event stride must be positive")
    if not 0.0 <= args.center_time_alpha <= 1.0:
        raise ValueError("Center-time alpha must be between zero and one")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    model, bundle = load_bundle(args.bundle, device=str(device))
    with_uncertainty = (
        bundle.get("format_version") == 2
        and bundle.get("model_type") == "photon_corrected_uncertainty"
    )
    prep = bundle["preprocessing"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # A prepared file cannot restore hits discarded at its original threshold.
    # Validate before opening the output so an invalid request leaves no CSV.
    with h5py.File(args.data, "r") as h5:
        metadata = json.loads(h5.attrs.get("metadata_json", "{}"))
        source_threshold = metadata.get("arguments", {}).get("threshold")
        if source_threshold is not None and args.threshold < float(source_threshold) - 1e-6:
            raise ValueError(
                f"Requested threshold {args.threshold} is below the prepared "
                f"source threshold {source_threshold}; the omitted hits cannot be recovered"
            )
        if (
            source_threshold is not None
            and args.conditional_low_threshold is not None
            and args.conditional_low_threshold < float(source_threshold) - 1e-6
        ):
            raise ValueError(
                f"Conditional threshold {args.conditional_low_threshold} is below the "
                f"prepared source threshold {source_threshold}"
            )

    kept_total = 0
    conditional_gate_total = 0
    conditional_total = 0
    with h5py.File(args.data, "r") as h5, args.output.open("w", newline="") as stream:
        group = h5[args.split]
        for key in ("signal_probability/data", "channels/data"):
            if key not in group:
                raise KeyError(f"Missing {args.split}/{key} in {args.data}")
        stored_mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        stored_std = np.asarray(h5["norm_param/std"], dtype=np.float32)
        total = len(group["ev_starts/data"]) - 1
        if args.start_event >= total:
            raise ValueError(f"Start event {args.start_event} is outside {total} events")
        end_event = (
            total if args.max_events is None else min(total, args.start_event + args.max_events)
        )
        fields = FIELDS + (("selected_threshold",) if args.conditional_low_threshold else ())
        if with_uncertainty:
            fields += UNCERTAINTY_FIELDS
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for begin in range(args.start_event, end_event, args.batch_size):
            end = min(begin + args.batch_size, end_event)
            kept, x, mask = selected_batch(
                group,
                begin,
                end,
                args.threshold,
                prep,
                stored_mean,
                stored_std,
                args.channels_per_string,
                args.start_event,
                args.event_stride,
                center_time_alpha=args.center_time_alpha,
            )
            if not kept:
                continue
            with torch.inference_mode():
                prediction = model(
                    torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)
                )
            direction, offset = prediction[:2]
            direction = direction.cpu().numpy()
            offset = offset.cpu().numpy()
            if with_uncertainty:
                sigma_angle = prediction[2].exp().cpu().numpy()
                sigma_anchor = prediction[3].exp().cpu().numpy()
            used_low_threshold = np.zeros(len(kept), dtype=bool)
            if args.conditional_low_threshold is not None:
                first_theta = np.rad2deg(np.arccos(np.clip(direction[:, 2], -1, 1)))
                gate_indices = np.flatnonzero(first_theta < args.gate_theta_max_deg)
                conditional_gate_total += len(gate_indices)
                if len(gate_indices):
                    low_kept, low_x, low_mask = selected_batch(
                        group,
                        begin,
                        end,
                        args.conditional_low_threshold,
                        prep,
                        stored_mean,
                        stored_std,
                        args.channels_per_string,
                        args.start_event,
                        args.event_stride,
                        center_time_alpha=args.center_time_alpha,
                    )
                    low_lookup = {row[1]: index for index, row in enumerate(low_kept)}
                    rerun_indices = np.asarray(
                        [
                            index
                            for index in gate_indices
                            if low_kept[low_lookup[kept[index][1]]][2] > kept[index][2]
                        ],
                        dtype=np.int64,
                    )
                    if len(rerun_indices) and args.max_added_photon_residual_ns is not None:
                        first_pass_anchors = np.stack(
                            [kept[index][4] + offset[index] for index in rerun_indices]
                        )
                        compatible = added_hits_pass_photon_gate(
                            group,
                            begin,
                            end,
                            np.asarray([kept[index][1] for index in rerun_indices]),
                            direction[rerun_indices],
                            first_pass_anchors,
                            args.threshold,
                            args.conditional_low_threshold,
                            args.max_added_photon_residual_ns,
                            stored_mean,
                            stored_std,
                        )
                        rerun_indices = rerun_indices[compatible]
                    if len(rerun_indices):
                        low_indices = [low_lookup[kept[index][1]] for index in rerun_indices]
                        with torch.inference_mode():
                            low_prediction = model(
                                torch.from_numpy(low_x[low_indices]).to(device),
                                torch.from_numpy(low_mask[low_indices]).to(device),
                            )
                        low_direction, low_offset = low_prediction[:2]
                        direction[rerun_indices] = low_direction.cpu().numpy()
                        offset[rerun_indices] = low_offset.cpu().numpy()
                        if with_uncertainty:
                            sigma_angle[rerun_indices] = low_prediction[2].exp().cpu().numpy()
                            sigma_anchor[rerun_indices] = low_prediction[3].exp().cpu().numpy()
                        for high_index, low_index in zip(rerun_indices, low_indices):
                            kept[high_index] = low_kept[low_index]
                        used_low_threshold[rerun_indices] = True
                        conditional_total += len(rerun_indices)
            theta = np.rad2deg(np.arccos(np.clip(direction[:, 2], -1, 1)))
            phi = np.mod(np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0])), 360)
            for i, row in enumerate(kept):
                anchor = row[4] + offset[i]
                record = {
                    "event_id": row[0],
                    "source_event_index": row[1],
                    "n_hits": row[2],
                    "n_strings": row[3],
                    "theta_deg": float(theta[i]),
                    "phi_deg": float(phi[i]),
                    "direction_x": float(direction[i, 0]),
                    "direction_y": float(direction[i, 1]),
                    "direction_z": float(direction[i, 2]),
                    "anchor_x_m": float(anchor[0]),
                    "anchor_y_m": float(anchor[1]),
                    "anchor_z_m": float(anchor[2]),
                }
                if args.conditional_low_threshold is not None:
                    record["selected_threshold"] = (
                        args.conditional_low_threshold if used_low_threshold[i] else args.threshold
                    )
                if with_uncertainty:
                    event_id = row[0]
                    scope = (
                        "validated_single_track_MC"
                        if event_id.startswith(("nuatm_", "nue2_"))
                        else "not_validated_for_possible_multi_track_or_experiment"
                    )
                    # Low-threshold reruns have not been calibrated. Emit raw
                    # scales but no nominal coverage radii for those rows.
                    calibrated = (
                        not used_low_threshold[i]
                        and np.isclose(args.threshold, 0.70)
                        and np.isclose(args.center_time_alpha, 0.0)
                        and args.channels_per_string == 36
                    )
                    factors = bundle["calibration"]
                    record.update(
                        {
                            "track_point_scope": scope,
                            "sigma_angle_deg": float(sigma_angle[i]),
                            "angle_r68_deg": (
                                float(min(180.0, sigma_angle[i] * factors["angle"]["68"]))
                                if calibrated
                                else None
                            ),
                            "angle_r95_deg": (
                                float(min(180.0, sigma_angle[i] * factors["angle"]["95"]))
                                if calibrated
                                else None
                            ),
                            "sigma_anchor_m": float(sigma_anchor[i]),
                            "anchor_r68_m": (
                                float(sigma_anchor[i] * factors["line"]["68"])
                                if calibrated
                                else None
                            ),
                            "anchor_r95_m": (
                                float(sigma_anchor[i] * factors["line"]["95"])
                                if calibrated
                                else None
                            ),
                            "uncertainty_calibration_scope": (
                                "p070_crossseason"
                                if calibrated
                                else (
                                    "p050_gate_unvalidated"
                                    if used_low_threshold[i]
                                    else "changed_threshold_or_transform_unvalidated"
                                )
                            ),
                        }
                    )
                writer.writerow(record)
            kept_total += len(kept)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "threshold": args.threshold,
                "conditional_low_threshold": args.conditional_low_threshold,
                "gate_theta_max_deg": args.gate_theta_max_deg,
                "max_added_photon_residual_ns": args.max_added_photon_residual_ns,
                "conditional_gate_events": conditional_gate_total,
                "conditional_reruns": conditional_total,
                "center_time_alpha": args.center_time_alpha,
                "source_events": end_event - args.start_event,
                "source_event_offset": args.start_event,
                "event_stride": args.event_stride,
                "kept_events": kept_total,
            }
        )
    )


if __name__ == "__main__":
    main()
