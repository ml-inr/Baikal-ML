"""Predict direction, track-line point and uncertainties for our HDF5 sources.

The released model (`trained_models/angle_track_uncertainty/release_v3/`) expects
already selected hits. This script streams a source HDF5, applies the release's
hit selection, runs the model, and writes one DuckDB keyed by ``event_fk``.

Two hit-selection inputs are supported:

* ``--sn-mode canonical`` (default): per-hit probabilities from the canonical SN
  model, ``{stem}_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5``;
* ``--sn-mode release``: the frozen ``signal_noise.ckpt`` shipped with the release.

The selection uses ``prob >= threshold``, matching the upstream release (the rest
of this project's signal-mask helper uses ``>``).

Calibrated radii are only populated for unchanged release-SN p>=0.70 MC rows on
CPU with the release selection conventions. Canonical-SN, experimental, gate and
changed-threshold rows keep raw sigmas but NULL radii.

Usage (from the repository root):
    python -m inference.angle_track_uncertainty.predict \
        --source data_manager/data/h5datasets/baikal_mc_merged.h5 \
        --source-name mc_merged \
        [--sn-mode canonical|release] [--parts part_0,...] [--limit N] \
        [--event-set-db OTHER_PREDICTIONS.duckdb] \
        [--conditional-low-threshold 0.50 --max-added-photon-residual-ns 200]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.angle_track_uncertainty.common import (  # noqa: E402
    CHANNELS_PER_STRING,
    SOURCE_PTYPES,
    added_hits_pass_photon_gate,
    build_model_batch,
    catalog_event_fks,
    data_class_for,
    event_set_indices,
    filter_event_set,
    infer_source_name,
    is_exp_source,
    load_norm,
    part_has_probs,
    part_keys,
    probs_window,
    read_event_block,
    release_sn_probs,
    run_model,
    select_events,
    track_point_scope,
    validate_probs_channels,
    validate_probs_ev_starts,
)
from inference.angle_track_uncertainty.store import (  # noqa: E402
    append_predictions_angle_track,
    open_angle_track_db,
)
from inference.shared.history import append_history_row  # noqa: E402
from inference.shared.run_info import probs_batch_size, write_run_info  # noqa: E402
from inference.shared.sig_counts import probs_path_for  # noqa: E402
from trained_models.angle_track_uncertainty.release_v3 import provenance  # noqa: E402
from trained_models.angle_track_uncertainty.release_v3.model import (  # noqa: E402
    load_bundle,
)
from trained_models.angle_track_uncertainty.release_v3.sn_model import (  # noqa: E402
    load_release_sn,
)

logger = logging.getLogger(__name__)

BUNDLE_DIR = PROJECT_ROOT / "trained_models/angle_track_uncertainty/release_v3"
DEFAULT_BUNDLE = BUNDLE_DIR / "angle_track_uncertainty.pt"
DEFAULT_RELEASE_SN = BUNDLE_DIR / "signal_noise.ckpt"
DEFAULT_CLASSIFIER_NORM = BUNDLE_DIR / "classifier_norm.h5"
DEFAULT_OUTPUT = PROJECT_ROOT / "inference/angle_track_uncertainty/preds"
DEFAULT_CATALOG = PROJECT_ROOT / "data_manager/catalog_v2.duckdb"
SELECTION_OP = "ge"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def run_records(
    model: torch.nn.Module,
    records: list[dict],
    mean: np.ndarray,
    std: np.ndarray,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over records in sub-batches, returning centers as well."""
    directions, offsets, log_angles, log_lines, centers = [], [], [], [], []
    for begin in range(0, len(records), batch_size):
        chunk = records[begin : begin + batch_size]
        x, mask, center = build_model_batch(chunk, mean, std, max_seq_len)
        direction, offset, log_angle, log_line = run_model(model, x, mask, device)
        directions.append(direction)
        offsets.append(offset)
        log_angles.append(log_angle)
        log_lines.append(log_line)
        centers.append(center)
    return (
        np.concatenate(directions),
        np.concatenate(offsets),
        np.concatenate(log_angles),
        np.concatenate(log_lines),
        np.concatenate(centers),
    )


def db_tag(
    sn_mode: str, threshold: float, gate_ns: float | None, gate_enabled: bool
) -> str:
    tag = f"{sn_mode}_p{int(round(threshold * 100)):03d}"
    if gate_enabled:
        tag += f"_gate{int(round(gate_ns))}" if gate_ns is not None else "_gate"
    return tag


def uncertainty_scope(
    *,
    source_name: str,
    sn_mode: str,
    calibrated: bool,
    gate_replaced: bool,
) -> str:
    if gate_replaced:
        return "p050_gate_unvalidated"
    if calibrated:
        return "p070_crossseason"
    if is_exp_source(source_name):
        return "experimental_unvalidated"
    if sn_mode == "canonical":
        return "canonical_sn_selection_unvalidated"
    return "changed_threshold_or_transform_unvalidated"


def predict(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    source_name = (
        infer_source_name(source) if args.source_name == "auto" else args.source_name
    )
    if source_name not in SOURCE_PTYPES:
        raise ValueError(f"Unknown source name: {source_name!r}")

    device = resolve_device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    bundle_model, bundle = load_bundle(checkpoint, device=str(device))
    device = next(bundle_model.parameters()).device
    prep = bundle["preprocessing"]
    calibration = bundle["calibration"]
    mean = np.asarray(prep["input_norm_mean"], dtype=np.float32)
    std = np.asarray(prep["input_norm_std"], dtype=np.float32)
    max_seq_len = int(prep["max_seq_len"])

    gate_enabled = args.conditional_low_threshold is not None
    gate_ns = args.max_added_photon_residual_ns if gate_enabled else None
    calibrated_condition = (
        args.sn_mode == "release"
        and source_name == "mc_merged"
        and np.isclose(args.threshold, 0.70)
        and args.min_hits == 8
        and args.min_strings == 2
        and args.channels_per_string == CHANNELS_PER_STRING
        and device.type == "cpu"
        and args.sn_batch_size == 256
        and max_seq_len == 128
        and not bool(prep.get("center_time", True))
    )

    probs_h5 = None
    sn_model = None
    classifier_mean = classifier_std = None
    if args.sn_mode == "canonical":
        probs_path = (
            Path(args.probs_h5).resolve()
            if args.probs_h5
            else probs_path_for(source)
        )
        if not probs_path.exists():
            raise SystemExit(
                f"SN probs file not found: {probs_path}. Pass --probs-h5 or use "
                f"--sn-mode release."
            )
        probs_h5 = h5py.File(probs_path, "r")
        probs_source = str(probs_path)
        sn_batch = probs_batch_size(str(probs_path))
    else:
        probs_source = str(Path(args.release_sn_checkpoint).resolve())
        sn_batch = args.sn_batch_size
        sn_model = load_release_sn(args.release_sn_checkpoint, device=str(device))
        classifier_mean, classifier_std = load_norm(args.classifier_norm)

    run_name = f"{checkpoint.parent.name}@{checkpoint.stem}"
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path = checkpoint_dir / (
        f"{source_name}_{db_tag(args.sn_mode, args.threshold, gate_ns, gate_enabled)}"
        "_track.duckdb"
    )
    db_source = db_path.stem
    hist_path = output_dir / "prediction_history.csv"

    logger.info("Source:           %s (%s)", source, source_name)
    logger.info("Checkpoint:       %s", checkpoint)
    logger.info("SN mode:          %s", args.sn_mode)
    logger.info("SN probs:         %s", probs_source)
    logger.info("Output DB:        %s", db_path)
    logger.info(
        "Selection:        prob >= %.2f, >=%d hits, >=%d strings",
        args.threshold, args.min_hits, args.min_strings,
    )
    if gate_enabled:
        logger.info(
            "Gate:             prob >= %.2f, theta < %.1f deg, max residual %s ns",
            args.conditional_low_threshold,
            args.gate_theta_max_deg,
            "any" if gate_ns is None else f"{gate_ns:g}",
        )

    t_start = datetime.now()
    n_inserted = 0
    n_skipped = 0
    n_selected = 0
    n_missing_fk = 0
    n_parts_skipped = 0
    n_parts_done = 0
    n_parts_not_in_event_set = 0
    n_event_set_not_selected = 0
    event_set = None
    event_set_stats: dict[str, int] = {}
    error_msg = ""
    conn = None
    total_in_db = 0

    try:
        conn = open_angle_track_db(str(db_path))
        if args.event_set_db:
            event_set, event_set_stats = event_set_indices(
                args.event_set_db,
                args.catalog,
                source_name,
                query=args.event_set_query,
            )
            existing_event_fks = conn.execute(
                "SELECT event_fk FROM predictions"
            ).df()["event_fk"].to_numpy(dtype=np.int64)
            if len(existing_event_fks):
                existing_event_fks = np.sort(np.unique(existing_event_fks))
                n_already = 0
                for key, (local_idx, event_fk) in list(event_set.items()):
                    keep = ~np.isin(
                        event_fk, existing_event_fks, assume_unique=True
                    )
                    n_already += int((~keep).sum())
                    if keep.all():
                        continue
                    if keep.any():
                        event_set[key] = (local_idx[keep], event_fk[keep])
                    else:
                        del event_set[key]
                event_set_stats["n_event_fk_already_done"] = n_already
            else:
                event_set_stats["n_event_fk_already_done"] = 0
            event_set_stats["n_event_fk_to_process"] = int(
                sum(len(fk) for _, fk in event_set.values())
            )
            logger.info(
                "Event set:        %s (%s)",
                args.event_set_db,
                event_set_stats,
            )
            if args.skip_done_parts:
                logger.warning(
                    "--skip-done-parts ignored with --event-set-db; "
                    "already-written event_fk rows are skipped individually"
                )
        done_parts: set[tuple[str, str]] = set()
        if args.skip_done_parts and not args.event_set_db:
            done_parts = {
                (str(data_class), str(part))
                for data_class, part in conn.execute(
                    "SELECT DISTINCT data_class, h5_part_key FROM predictions"
                ).fetchall()
            }
            logger.info("skip-done-parts: %d parts already present", len(done_parts))
        wanted_parts = set(args.parts.split(",")) if args.parts else None

        with h5py.File(source, "r") as h5:
            for ptype in SOURCE_PTYPES[source_name]:
                if ptype not in h5:
                    logger.warning("%s: group not found in %s", ptype, source)
                    continue
                group = h5[ptype]
                data_class = data_class_for(ptype, source_name)
                if probs_h5 is not None and ptype not in probs_h5:
                    logger.warning("%s: no probs group; skipping", ptype)
                    continue
                parts = part_keys(group)
                if wanted_parts is not None:
                    parts = [part for part in parts if part in wanted_parts]
                if event_set is not None:
                    in_set = {
                        part for (event_ptype, part) in event_set
                        if event_ptype == ptype
                    }
                    n_parts_not_in_event_set += len(parts) - len(
                        [part for part in parts if part in in_set]
                    )
                    parts = [part for part in parts if part in in_set]
                logger.info("%s: %d parts", ptype, len(parts))

                n_read_ptype = 0
                ptype_selected = 0
                ptype_t0 = time.perf_counter()
                stop = False

                for part in parts:
                    if stop:
                        break
                    if (data_class, part) in done_parts:
                        n_parts_done += 1
                        continue
                    if probs_h5 is not None and not part_has_probs(
                        probs_h5, ptype, part
                    ):
                        n_parts_skipped += 1
                        continue

                    starts_full = np.asarray(
                        group["raw"]["ev_starts"][part]["data"][:],
                        dtype=np.int64,
                    )
                    n_events_part = len(starts_full) - 1
                    if probs_h5 is not None:
                        n_hits_part = len(group["raw"]["data"][part]["data"])
                        if len(probs_h5[ptype]["probs"][part]["data"]) != n_hits_part:
                            raise ValueError(
                                f"{ptype}/{part}: probs have "
                                f"{len(probs_h5[ptype]['probs'][part]['data'])} hits, "
                                f"source has {n_hits_part}"
                            )
                        validate_probs_ev_starts(
                            probs_h5, ptype, part, starts_full
                        )

                    for begin in range(0, n_events_part, args.events_per_chunk):
                        if args.max_events_per_ptype is not None:
                            remaining = args.max_events_per_ptype - n_read_ptype
                            if remaining <= 0:
                                stop = True
                                break
                        end = min(begin + args.events_per_chunk, n_events_part)
                        if args.max_events_per_ptype is not None:
                            end = min(end, begin + remaining)
                        # Cap the window by hits, not only by events: one exp part
                        # can hold hundreds of millions of hits.
                        while (
                            end > begin + 1
                            and starts_full[end] - starts_full[begin]
                            > args.max_window_hits
                        ):
                            end = begin + max(1, (end - begin) // 2)
                        if end <= begin:
                            stop = True
                            break
                        n_read_ptype += end - begin

                        (
                            ev_starts,
                            raw,
                            channels,
                            ev_ids,
                            header,
                            hit_lo,
                        ) = read_event_block(group, part, begin, end)
                        if len(raw) == 0:
                            continue

                        if probs_h5 is not None:
                            validate_probs_channels(
                                probs_h5, ptype, part, channels, hit_lo
                            )
                            probs = probs_window(
                                probs_h5, ptype, part, hit_lo, len(raw)
                            )
                            if probs is None:
                                raise ValueError(
                                    f"{ptype}/{part}: missing probs window"
                                )
                        else:
                            probs = release_sn_probs(
                                sn_model,
                                raw,
                                ev_starts,
                                classifier_mean,
                                classifier_std,
                                args.sn_batch_size,
                                device,
                            )

                        high = select_events(
                            raw, probs, channels, ev_starts, ev_ids,
                            threshold=args.threshold,
                            min_hits=args.min_hits,
                            min_strings=args.min_strings,
                            channels_per_string=args.channels_per_string,
                        )
                        if event_set is not None:
                            high, n_not_selected = filter_event_set(
                                event_set.get((ptype, part)),
                                begin,
                                end,
                                high,
                            )
                            n_event_set_not_selected += n_not_selected
                        if not high:
                            continue

                        direction, offset, log_angle, log_line, centers = run_records(
                            bundle_model, high, mean, std, max_seq_len,
                            args.batch_size, device,
                        )
                        n_raw_hits = np.asarray(
                            [r["n_raw_hits"] for r in high], np.int64
                        )
                        n_hits = np.asarray([r["n_hits"] for r in high], np.int64)
                        n_strings = np.asarray(
                            [r["n_strings"] for r in high], np.int64
                        )
                        selected_threshold = np.full(
                            len(high), float(args.threshold), dtype=np.float32
                        )
                        gate_replaced = np.zeros(len(high), dtype=bool)

                        if gate_enabled:
                            low = select_events(
                                raw, probs, channels, ev_starts, ev_ids,
                                threshold=args.conditional_low_threshold,
                                min_hits=args.min_hits,
                                min_strings=args.min_strings,
                                channels_per_string=args.channels_per_string,
                            )
                            low_by_index = {r["index"]: j for j, r in enumerate(low)}
                            first_theta = np.rad2deg(
                                np.arccos(np.clip(direction[:, 2], -1.0, 1.0))
                            )
                            first_anchor = centers + offset
                            candidates = [
                                i
                                for i, record in enumerate(high)
                                if first_theta[i] < args.gate_theta_max_deg
                                and record["index"] in low_by_index
                                and low[low_by_index[record["index"]]]["n_hits"]
                                > record["n_hits"]
                            ]
                            if gate_ns is not None:
                                candidates = [
                                    i
                                    for i in candidates
                                    if added_hits_pass_photon_gate(
                                        raw,
                                        probs,
                                        ev_starts,
                                        high[i]["index"],
                                        direction[i],
                                        first_anchor[i],
                                        high_threshold=args.threshold,
                                        low_threshold=args.conditional_low_threshold,
                                        max_residual_ns=gate_ns,
                                        refractive_index=1.36,
                                    )
                                ]
                            if candidates:
                                low_records = [
                                    low[low_by_index[high[i]["index"]]]
                                    for i in candidates
                                ]
                                (
                                    low_direction,
                                    low_offset,
                                    low_log_angle,
                                    low_log_line,
                                    low_centers,
                                ) = run_records(
                                    bundle_model, low_records, mean, std,
                                    max_seq_len, args.batch_size, device,
                                )
                                for row, high_index in enumerate(candidates):
                                    low_record = low_records[row]
                                    direction[high_index] = low_direction[row]
                                    offset[high_index] = low_offset[row]
                                    log_angle[high_index] = low_log_angle[row]
                                    log_line[high_index] = low_log_line[row]
                                    centers[high_index] = low_centers[row]
                                    n_hits[high_index] = low_record["n_hits"]
                                    n_strings[high_index] = low_record["n_strings"]
                                    selected_threshold[high_index] = float(
                                        args.conditional_low_threshold
                                    )
                                    gate_replaced[high_index] = True

                        theta = np.rad2deg(
                            np.arccos(np.clip(direction[:, 2], -1.0, 1.0))
                        )
                        phi = np.mod(
                            np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0])),
                            360.0,
                        )
                        anchor = centers + offset
                        sigma_angle = np.exp(log_angle)
                        sigma_anchor = np.exp(log_line)

                        calibrated_rows = np.full(
                            len(high), calibrated_condition, dtype=bool
                        ) & ~gate_replaced
                        angle_r68 = np.where(
                            calibrated_rows,
                            np.minimum(180.0, sigma_angle * calibration["angle"]["68"]),
                            np.nan,
                        )
                        angle_r95 = np.where(
                            calibrated_rows,
                            np.minimum(180.0, sigma_angle * calibration["angle"]["95"]),
                            np.nan,
                        )
                        anchor_r68 = np.where(
                            calibrated_rows,
                            sigma_anchor * calibration["line"]["68"],
                            np.nan,
                        )
                        anchor_r95 = np.where(
                            calibrated_rows,
                            sigma_anchor * calibration["line"]["95"],
                            np.nan,
                        )
                        scope = track_point_scope(ptype, source_name)
                        scopes = np.asarray(
                            [
                                uncertainty_scope(
                                    source_name=source_name,
                                    sn_mode=args.sn_mode,
                                    calibrated=bool(calibrated_rows[i]),
                                    gate_replaced=bool(gate_replaced[i]),
                                )
                                for i in range(len(high))
                            ]
                        )

                        local_indices = np.asarray(
                            [r["index"] for r in high], dtype=np.int64
                        )
                        absolute_indices = begin + local_indices
                        fk_map = catalog_event_fks(
                            str(args.catalog),
                            source_name,
                            ptype,
                            part,
                            header,
                            local_indices,
                            absolute_indices,
                        )
                        positions = np.asarray(
                            [
                                i
                                for i, record in enumerate(high)
                                if int(record["index"]) in fk_map
                            ],
                            dtype=np.int64,
                        )
                        n_missing_fk += len(high) - len(positions)
                        if len(positions) == 0:
                            continue

                        df = pd.DataFrame(
                            {
                                "event_fk": [
                                    fk_map[int(high[i]["index"])] for i in positions
                                ],
                                "event_id": [
                                    high[i]["event_id"]
                                    or f"{part}:{int(absolute_indices[i])}"
                                    for i in positions
                                ],
                                "data_class": data_class,
                                "h5_part_key": part,
                                "h5_local_event_id": absolute_indices[positions],
                                "sn_mode": args.sn_mode,
                                "selection_op": SELECTION_OP,
                                "gate_applied": gate_replaced[positions],
                                "selected_threshold": selected_threshold[positions],
                                "n_raw_hits": n_raw_hits[positions],
                                "n_hits": n_hits[positions],
                                "n_strings": n_strings[positions],
                                "theta_deg": theta[positions],
                                "phi_deg": phi[positions],
                                "direction_x": direction[positions, 0],
                                "direction_y": direction[positions, 1],
                                "direction_z": direction[positions, 2],
                                "anchor_x_m": anchor[positions, 0],
                                "anchor_y_m": anchor[positions, 1],
                                "anchor_z_m": anchor[positions, 2],
                                "sigma_angle_deg": sigma_angle[positions],
                                "angle_r68_deg": angle_r68[positions],
                                "angle_r95_deg": angle_r95[positions],
                                "sigma_anchor_m": sigma_anchor[positions],
                                "anchor_r68_m": anchor_r68[positions],
                                "anchor_r95_m": anchor_r95[positions],
                                "track_point_scope": scope,
                                "uncertainty_calibration_scope": scopes[positions],
                            }
                        )
                        if args.limit is not None:
                            remaining = args.limit - n_selected
                            if remaining <= 0:
                                stop = True
                                break
                            df = df.iloc[:remaining]
                        inserted, skipped = append_predictions_angle_track(
                            conn, df, check_existing=args.check_existing
                        )
                        n_inserted += inserted
                        n_skipped += skipped
                        n_selected += len(df)
                        ptype_selected += len(df)
                        if args.limit is not None and n_selected >= args.limit:
                            stop = True
                            break

                    elapsed = time.perf_counter() - ptype_t0
                    logger.info(
                        "%s: %d parts, selected=%d, elapsed=%.0fs",
                        ptype, len(parts), ptype_selected, elapsed,
                    )

        total_in_db = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        logger.info(
            "Done: inserted=%d skipped=%d selected=%d missing_fk=%d "
            "parts_without_probs=%d parts_done=%d total_in_db=%d",
            n_inserted, n_skipped, n_selected, n_missing_fk,
            n_parts_skipped, n_parts_done, total_in_db,
        )
        write_run_info(
            checkpoint_dir / "run_info.json",
            source=db_source,
            checkpoint=str(checkpoint),
            h5_path=str(source),
            threshold=args.threshold,
            min_hits=args.min_hits,
            min_strings=args.min_strings,
            n_events_total=total_in_db,
            sn_probs_source=probs_source,
            sn_batch_size=sn_batch,
            sn_mode=args.sn_mode,
            selection_op=SELECTION_OP,
            conditional_low_threshold=args.conditional_low_threshold,
            gate_theta_max_deg=args.gate_theta_max_deg if gate_enabled else None,
            max_added_photon_residual_ns=gate_ns,
            max_seq_len=max_seq_len,
            channels_per_string=args.channels_per_string,
            device=str(device),
            bundle_sha256=provenance.sha256_file(checkpoint),
            sn_sha256=(
                provenance.sha256_file(args.release_sn_checkpoint)
                if args.sn_mode == "release"
                else None
            ),
            n_missing_event_fk=n_missing_fk,
            n_parts_without_probs=n_parts_skipped,
            n_parts_skipped_done=n_parts_done,
            event_set_db=(
                str(Path(args.event_set_db).resolve())
                if args.event_set_db else None
            ),
            event_set_query=(
                args.event_set_query if args.event_set_db else None
            ),
            n_event_fk_requested=event_set_stats.get("n_event_fk_requested"),
            n_event_fk_resolved=event_set_stats.get("n_event_fk_resolved"),
            n_event_fk_missing=event_set_stats.get("n_event_fk_missing"),
            n_event_fk_already_done=event_set_stats.get(
                "n_event_fk_already_done"
            ),
            n_event_fk_to_process=event_set_stats.get("n_event_fk_to_process"),
            n_event_set_not_selected=n_event_set_not_selected,
            n_parts_not_in_event_set=n_parts_not_in_event_set,
        )
    except Exception as exc:
        error_msg = str(exc)
        logger.exception("predict failed")
        raise
    finally:
        if conn is not None:
            conn.close()
        if probs_h5 is not None:
            probs_h5.close()
        append_history_row(
            str(hist_path),
            checkpoint=run_name,
            script="predict.py",
            source=db_source,
            db_file=str(db_path),
            threshold=args.threshold,
            min_hits=args.min_hits,
            min_strings=args.min_strings,
            n_events_new=n_inserted,
            n_events_skipped=n_skipped,
            is_successful=error_msg == "",
            error_msg=error_msg,
            timestamp=t_start.isoformat(timespec="seconds"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Source HDF5 path")
    parser.add_argument(
        "--source-name",
        choices=["auto", "mc_merged", "mc_reco", "exp_full", "exp_reco", "exp"],
        default="auto",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_BUNDLE))
    parser.add_argument(
        "--probs-h5",
        default=None,
        help="Per-hit SN probs for --sn-mode canonical (default: sibling of --source)",
    )
    parser.add_argument(
        "--sn-mode", choices=["canonical", "release"], default="canonical",
        help="canonical = existing SN probs; release = frozen release SN checkpoint",
    )
    parser.add_argument("--release-sn-checkpoint", default=str(DEFAULT_RELEASE_SN))
    parser.add_argument("--classifier-norm", default=str(DEFAULT_CLASSIFIER_NORM))
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--min-hits", type=int, default=8)
    parser.add_argument("--min-strings", type=int, default=2)
    parser.add_argument(
        "--channels-per-string", type=int, default=CHANNELS_PER_STRING
    )
    parser.add_argument(
        "--conditional-low-threshold", type=float, default=None,
        help="Second pass at this threshold for upward first-pass events",
    )
    parser.add_argument("--gate-theta-max-deg", type=float, default=90.0)
    parser.add_argument(
        "--max-added-photon-residual-ns", type=float, default=None,
        help="Require every added hit to pass this photon-time residual gate",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sn-batch-size", type=int, default=256)
    parser.add_argument("--events-per-chunk", type=int, default=4096)
    parser.add_argument("--max-window-hits", type=int, default=8_000_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parts", default=None, help="Comma-separated part keys")
    parser.add_argument(
        "--event-set-db",
        default=None,
        help=(
            "Optional DuckDB whose event_fk set restricts the inferred events. "
            "The default query reads the whole predictions table."
        ),
    )
    parser.add_argument(
        "--event-set-query",
        default="SELECT event_fk FROM predictions",
        help="SQL returning an event_fk column from --event-set-db",
    )
    parser.add_argument("--max-events-per-ptype", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max selected rows")
    parser.add_argument("--skip-done-parts", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--check-existing", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    if args.conditional_low_threshold is not None and not (
        0.0 < args.conditional_low_threshold < args.threshold
    ):
        parser.error("--conditional-low-threshold must be below --threshold")
    if (
        args.max_added_photon_residual_ns is not None
        and args.conditional_low_threshold is None
    ):
        parser.error(
            "--max-added-photon-residual-ns requires --conditional-low-threshold"
        )
    if (
        args.batch_size < 1
        or args.sn_batch_size < 1
        or args.events_per_chunk < 1
        or args.max_window_hits < 1
    ):
        parser.error("batch sizes, chunk and hit limits must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    torch.set_num_threads(args.threads)
    predict(args)


if __name__ == "__main__":
    main()
