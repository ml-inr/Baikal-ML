"""Build event-level handcrafted features for the non-neural energy baseline.

The source is a hit-level energy dataset; the output is one row per event,
written next to the source arrays:

    non_neural_features.npy
    non_neural_features.yaml

The builder does not modify any existing file.  The YAML records the source
files, the charge clip, the feature groups and the row-order contract so a
stale derived file can be detected after the source dataset is rebuilt.

Run from the repository root:
    python -m data_manager.mu_energy_ds_builder.build_non_neural_features \
        --dataset-dir data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M_partsplit
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data.mu_energy_dataset.non_neural_features import (
    AMP_CLIP_DEFAULT,
    compute_non_neural_features,
    feature_groups,
    feature_names,
)

logger = logging.getLogger(__name__)

FEATURES_FILE = "non_neural_features.npy"
CONFIG_FILE = "non_neural_features.yaml"


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def build(
    dataset_dir: str | Path,
    *,
    amp_clip: float = AMP_CLIP_DEFAULT,
    overwrite: bool = False,
) -> Path:
    """Compute and write the event-level feature matrix for one dataset."""
    d = Path(dataset_dir)
    required = (
        "features.npy",
        "offsets.npy",
        "n_sig_hits.npy",
        "n_sig_strings.npy",
        "energy.npy",
        "event_fk.npy",
        "h5_part_keys.npy",
        "h5_local_event_ids.npy",
    )
    missing = [name for name in required if not (d / name).exists()]
    if missing:
        raise FileNotFoundError(f"{d}: missing {missing}")

    out_features = d / FEATURES_FILE
    out_config = d / CONFIG_FILE
    if not overwrite and (out_features.exists() or out_config.exists()):
        raise FileExistsError(
            f"{out_features} or {out_config} already exists; pass --overwrite"
        )

    features = np.load(d / "features.npy", mmap_mode="r")
    offsets = np.load(d / "offsets.npy")
    n_sig_hits = np.load(d / "n_sig_hits.npy")
    n_sig_strings = np.load(d / "n_sig_strings.npy")
    energy = np.load(d / "energy.npy")
    event_fk = np.load(d / "event_fk.npy")
    part_keys = np.load(d / "h5_part_keys.npy", allow_pickle=True)
    local_ids = np.load(d / "h5_local_event_ids.npy")

    n_events = len(offsets) - 1
    lengths = np.diff(offsets)
    if len(energy) != n_events or len(event_fk) != n_events:
        raise ValueError(
            f"{d}: offsets say {n_events} events, "
            f"energy has {len(energy)}, event_fk has {len(event_fk)}"
        )
    if not np.array_equal(lengths.astype(np.int32), n_sig_hits):
        raise ValueError(f"{d}: n_sig_hits does not match offsets")
    if len(part_keys) != n_events or len(local_ids) != n_events:
        raise ValueError(f"{d}: part/local id arrays do not match event count")

    logger.info(
        "Computing %s event-level features for %s events from %s hits",
        len(feature_names()), f"{n_events:,}", f"{len(features):,}",
    )
    matrix = compute_non_neural_features(
        features,
        offsets,
        n_sig_strings,
        amp_clip=amp_clip,
        n_sig_hits=n_sig_hits,
    )
    if matrix.shape != (n_events, len(feature_names())):
        raise RuntimeError(
            f"feature matrix shape {matrix.shape}, expected "
            f"({n_events}, {len(feature_names())})"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite values in the feature matrix")
    if len(np.unique(event_fk)) != n_events:
        raise ValueError("event_fk is not unique; row alignment is unsafe")

    np.save(out_features, matrix.astype(np.float32, copy=False))
    logger.info("Wrote %s (%s)", out_features, matrix.shape)

    dataset_info_path = d / "dataset_info.json"
    split_info_path = d / "part_split_info.json"
    metadata: dict[str, Any] = {}
    if dataset_info_path.exists():
        info = json.load(open(dataset_info_path))
        metadata["dataset_info"] = {
            "n_events": info.get("n_events"),
            "n_sig_hits_total": info.get("n_sig_hits_total"),
            "sig_noise_threshold": info.get("sig_noise_threshold"),
            "event_cuts": info.get("event_cuts"),
            "target": info.get("target"),
            "parts": info.get("parts"),
        }
    if split_info_path.exists():
        split = json.load(open(split_info_path))
        metadata["part_split"] = {
            "description": split.get("description"),
            "n_train_parts": split.get("n_train_parts"),
            "n_test_parts": split.get("n_test_parts"),
            "test_fraction": split.get("test_fraction"),
        }

    config = {
        "source_dataset": str(d.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "builder": (
            "data_manager/mu_energy_ds_builder/build_non_neural_features.py"
        ),
        "n_events": int(n_events),
        "n_features": int(matrix.shape[1]),
        "amp_clip": float(amp_clip),
        "row_order": "same_as_event_fk",
        "source_files": {
            "features.npy": _file_info(d / "features.npy"),
            "offsets.npy": _file_info(d / "offsets.npy"),
            "n_sig_hits.npy": _file_info(d / "n_sig_hits.npy"),
            "n_sig_strings.npy": _file_info(d / "n_sig_strings.npy"),
            "energy.npy": _file_info(d / "energy.npy"),
            "event_fk.npy": _file_info(d / "event_fk.npy"),
        },
        "feature_groups": feature_groups(),
        "feature_names": feature_names(),
        "source_metadata": metadata,
    }
    with open(out_config, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info("Wrote %s", out_config)
    return out_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=(
            "data_manager/datasets/"
            "mu_energy_dataset_h8s3_thr0.8_train1M_partsplit"
        ),
        help="hit-level energy dataset directory",
    )
    parser.add_argument(
        "--amp-clip",
        type=float,
        default=AMP_CLIP_DEFAULT,
        help="raw charge clip in p.e. (same as the neural model)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing non_neural_features.* files",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    build(args.dataset_dir, amp_clip=args.amp_clip, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
