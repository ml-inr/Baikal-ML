"""Convert our source HDF5 into the prepared schema of the released model.

The upstream `infer_unified_model.py` and `infer_rethresholded_for_audit.py`
expect a file with a ``test`` split containing already selected hits::

    test/data/data               (n_selected_hits, 5)  normalized features
    test/ev_starts/data          (n_events + 1,)
    test/channels/data           (n_selected_hits,)
    test/ev_ids/data             (n_events,)
    test/signal_probability/data (n_selected_hits,)
    test/num_un_strings/data     (n_events,)
    norm_param/mean, std

This script writes that file from our HDF5 and either the canonical SN
probabilities or the release SN checkpoint. It is the reference path for
comparing our streaming adapter (`predict.py`) with upstream; for project-scale
runs prefer `predict.py`, which writes DuckDB without an intermediate file.

Usage (from the repository root):
    python -m inference.angle_track_uncertainty.prepare_selected_h5 \
        --source data_manager/data/h5datasets/baikal_mc_merged.h5 \
        --source-name mc_merged --output /tmp/selected_p050.h5 \
        [--probs-h5 ...] [--threshold 0.50] [--parts ...] [--max-events-per-ptype N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.angle_track_uncertainty.common import (  # noqa: E402
    CHANNELS_PER_STRING,
    SOURCE_PTYPES,
    infer_source_name,
    load_norm,
    part_has_probs,
    part_keys,
    probs_window,
    read_event_block,
    release_sn_probs,
    select_events,
    validate_probs_channels,
    validate_probs_ev_starts,
)
from inference.shared.sig_counts import probs_path_for  # noqa: E402
from trained_models.angle_track_uncertainty.release_v3.sn_model import (  # noqa: E402
    load_release_sn,
)

logger = logging.getLogger(__name__)

BUNDLE_DIR = PROJECT_ROOT / "trained_models/angle_track_uncertainty/release_v3"
DEFAULT_RELEASE_SN = BUNDLE_DIR / "signal_noise.ckpt"
DEFAULT_CLASSIFIER_NORM = BUNDLE_DIR / "classifier_norm.h5"
DEFAULT_ANGLE_NORM = BUNDLE_DIR / "angle_norm.h5"


def appendable(
    parent: h5py.Group,
    name: str,
    tail: tuple[int, ...],
    dtype,
    compression: str | None = "lzf",
) -> h5py.Dataset:
    options = {"chunks": True}
    if compression:
        options["compression"] = compression
    group = parent.create_group(name)
    return group.create_dataset(
        "data", shape=(0, *tail), maxshape=(None, *tail), dtype=dtype, **options
    )


def append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    old = len(dataset)
    dataset.resize(old + len(values), axis=0)
    dataset[old:] = values


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def prepare(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_name = (
        infer_source_name(source) if args.source_name == "auto" else args.source_name
    )

    device = resolve_device(args.device)
    norm_mean, norm_std = load_norm(args.norm_source, args.norm_group)

    probs_h5 = None
    sn_model = None
    classifier_mean = classifier_std = None
    if args.sn_mode == "project":
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
        logger.info("Using project SN probs: %s", probs_path)
    else:
        sn_model = load_release_sn(args.release_sn_checkpoint, device=str(device))
        classifier_mean, classifier_std = load_norm(args.classifier_norm)
        logger.info("Using release SN checkpoint: %s", args.release_sn_checkpoint)

    wanted_parts = set(args.parts.split(",")) if args.parts else None
    summaries: dict[str, dict] = {}
    total_events = 0
    total_hits = 0
    t0 = time.time()

    with h5py.File(source, "r") as h5, h5py.File(output, "w") as out:
        split_group = out.create_group(args.split)
        data_ds = appendable(split_group, "data", (5,), np.float32, args.compression)
        channels_ds = appendable(
            split_group, "channels", (), np.int32, args.compression
        )
        probs_ds = appendable(
            split_group, "signal_probability", (), np.float32, args.compression
        )
        ev_ids_ds = appendable(
            split_group, "ev_ids", (), h5py.string_dtype("utf-8"), args.compression
        )
        strings_ds = appendable(
            split_group, "num_un_strings", (), np.int32, args.compression
        )
        starts_ds = split_group.create_dataset(
            "ev_starts/data", shape=(1,), maxshape=(None,), dtype=np.int64, chunks=True
        )
        starts_ds[0] = 0
        written_hits = 0
        written_events = 0

        for ptype in SOURCE_PTYPES[source_name]:
            if ptype not in h5:
                logger.warning("%s: group not found", ptype)
                continue
            group = h5[ptype]
            if probs_h5 is not None and ptype not in probs_h5:
                logger.warning("%s: no probs group; skipping", ptype)
                continue
            parts = part_keys(group)
            if wanted_parts is not None:
                parts = [part for part in parts if part in wanted_parts]
            ptype_events = 0
            ptype_hits = 0
            n_read = 0
            stop = False

            for part in parts:
                if stop:
                    break
                starts_full = np.asarray(
                    group["raw"]["ev_starts"][part]["data"][:], dtype=np.int64
                )
                n_events_part = len(starts_full) - 1
                if probs_h5 is not None:
                    if not part_has_probs(probs_h5, ptype, part):
                        continue
                    n_hits_part = len(group["raw"]["data"][part]["data"])
                    if len(probs_h5[ptype]["probs"][part]["data"]) != n_hits_part:
                        raise ValueError(
                            f"{ptype}/{part}: probs have "
                            f"{len(probs_h5[ptype]['probs'][part]['data'])} hits, "
                            f"source has {n_hits_part}"
                        )
                    validate_probs_ev_starts(probs_h5, ptype, part, starts_full)

                for begin in range(0, n_events_part, args.events_per_chunk):
                    if args.max_events_per_ptype is not None:
                        remaining = args.max_events_per_ptype - n_read
                        if remaining <= 0:
                            stop = True
                            break
                        end = min(begin + args.events_per_chunk, n_events_part,
                                  begin + remaining)
                    else:
                        end = min(begin + args.events_per_chunk, n_events_part)
                    while (
                        end > begin + 1
                        and starts_full[end] - starts_full[begin]
                        > args.max_window_hits
                    ):
                        end = begin + max(1, (end - begin) // 2)
                    if end <= begin:
                        stop = True
                        break
                    n_read += end - begin

                    (
                        ev_starts,
                        raw,
                        channels,
                        ev_ids,
                        _header,
                        hit_lo,
                    ) = read_event_block(group, part, begin, end)
                    if len(raw) == 0:
                        continue
                    if probs_h5 is not None:
                        validate_probs_channels(
                            probs_h5, ptype, part, channels, hit_lo
                        )
                        probs = probs_window(probs_h5, ptype, part, hit_lo, len(raw))
                        if probs is None:
                            raise ValueError(f"{ptype}/{part}: missing probs window")
                    else:
                        probs = release_sn_probs(
                            sn_model, raw, ev_starts, classifier_mean,
                            classifier_std, args.sn_batch_size, device,
                        )
                    selected = select_events(
                        raw, probs, channels, ev_starts, ev_ids,
                        threshold=args.threshold,
                        min_hits=args.min_hits,
                        min_strings=args.min_strings,
                        channels_per_string=args.channels_per_string,
                    )
                    if not selected:
                        continue
                    if args.limit is not None:
                        remaining = args.limit - written_events
                        if remaining <= 0:
                            stop = True
                            break
                        selected = selected[:remaining]

                    selected_raw = np.concatenate([row["raw"] for row in selected])
                    selected_channels = np.concatenate(
                        [row["channels"] for row in selected]
                    )
                    selected_probs = np.concatenate(
                        [row["probs"] for row in selected]
                    )
                    append(data_ds, ((selected_raw - norm_mean) / norm_std))
                    append(channels_ds, selected_channels)
                    append(probs_ds, selected_probs)
                    append(
                        ev_ids_ds,
                        np.asarray(
                            [
                                row["event_id"] or f"{part}:{begin + row['index']}"
                                for row in selected
                            ],
                            dtype=object,
                        ),
                    )
                    append(
                        strings_ds,
                        np.asarray([row["n_strings"] for row in selected]),
                    )
                    new_starts = written_hits + np.cumsum(
                        [row["n_hits"] for row in selected], dtype=np.int64
                    )
                    append(starts_ds, new_starts)
                    n_new = len(selected)
                    written_events += n_new
                    written_hits += int(new_starts[-1] - written_hits)
                    ptype_events += n_new
                    ptype_hits += int(sum(row["n_hits"] for row in selected))
                    if args.limit is not None and written_events >= args.limit:
                        stop = True
                        break

            summaries[ptype] = {
                "events": ptype_events,
                "hits": ptype_hits,
                "source_events_read": n_read,
            }
            logger.info("%s: %d events, %d hits", ptype, ptype_events, ptype_hits)
            if stop and args.limit is not None and written_events >= args.limit:
                break

        norm_group = out.create_group("norm_param")
        norm_group.create_dataset("mean", data=norm_mean)
        norm_group.create_dataset("std", data=norm_std)
        total_events = written_events
        total_hits = written_hits
        metadata = {
            "arguments": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "source_name": source_name,
            "summaries": summaries,
            "n_events": total_events,
            "n_hits": total_hits,
        }
        out.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)

    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Wrote %s (%.3f GB, %d events, %d hits) in %.1fs",
        output, output.stat().st_size / 1e9, total_events, total_hits,
        time.time() - t0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--source-name",
        choices=["auto", "mc_merged", "mc_reco", "exp_full", "exp_reco", "exp"],
        default="auto",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-hits", type=int, default=8)
    parser.add_argument("--min-strings", type=int, default=2)
    parser.add_argument(
        "--channels-per-string", type=int, default=CHANNELS_PER_STRING
    )
    parser.add_argument("--probs-h5", default=None)
    parser.add_argument(
        "--sn-mode", choices=["project", "release"], default="project"
    )
    parser.add_argument("--release-sn-checkpoint", default=str(DEFAULT_RELEASE_SN))
    parser.add_argument("--classifier-norm", default=str(DEFAULT_CLASSIFIER_NORM))
    parser.add_argument("--norm-source", default=str(DEFAULT_ANGLE_NORM))
    parser.add_argument("--norm-group", default="norm_param")
    parser.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    parser.add_argument("--parts", default=None)
    parser.add_argument("--max-events-per-ptype", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--events-per-chunk", type=int, default=4096)
    parser.add_argument("--max-window-hits", type=int, default=8_000_000)
    parser.add_argument("--sn-batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    if (
        args.events_per_chunk < 1
        or args.max_window_hits < 1
        or args.sn_batch_size < 1
    ):
        parser.error(
            "--events-per-chunk, --max-window-hits and --sn-batch-size "
            "must be positive"
        )
    if args.compression == "none":
        args.compression = None
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    torch.set_num_threads(args.threads)
    prepare(args)


if __name__ == "__main__":
    main()
