"""Apply a saved non-neural energy baseline to new events.

The input is a hit-level NPY directory in the same layout as the energy dataset
(or the exp layout with ``--prefix exp``).  Features are computed on the fly with
the same ``amp_clip`` and feature definition used at training time; no tuning or
selection happens here.

Output is a DuckDB table with one row per event.  For MC the table carries the
truth columns; for exp they are NaN.  The new test parts are scored separately
from the training dataset and are never used for model selection.

Usage:
    python -m inference.track_energy_reconstruction.predict_non_neural_energy \
        --run-dir trained_models/track_energy_reconstruction/<run> \
        --model hgb \
        --npy-dir <new_parts_dir> \
        --split-name test_new
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.data.mu_energy_dataset.non_neural_features import (
    compute_non_neural_features,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_optional(
    directory: Path,
    prefix: str,
    name: str,
    *,
    allow_pickle: bool = False,
) -> np.ndarray | None:
    path = directory / f"{prefix}{name}"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=allow_pickle)


def _column(values: np.ndarray | None, n: int, fill: float) -> np.ndarray:
    if values is None:
        return np.full(n, fill)
    arr = np.asarray(values)
    if len(arr) != n:
        raise ValueError(f"array has {len(arr)} rows, expected {n}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", required=True, help="training run directory"
    )
    parser.add_argument(
        "--model", required=True, help="model subdirectory under run_dir/models"
    )
    parser.add_argument("--npy-dir", required=True, help="hit-level NPY directory")
    parser.add_argument(
        "--prefix",
        default="",
        help="file prefix, e.g. 'exp' for the exp NPY layout",
    )
    parser.add_argument(
        "--output-dir",
        default="inference/track_energy_reconstruction/preds",
    )
    parser.add_argument("--split-name", default="test_new")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_dir = _resolve(args.run_dir)
    model_dir = run_dir / "models" / args.model
    model_path = model_dir / "model.joblib"
    info_path = model_dir / "model_info.json"
    if not model_path.exists() or not info_path.exists():
        raise FileNotFoundError(f"{model_dir}: model.joblib/model_info.json")
    with open(info_path) as f:
        model_info = json.load(f)
    model = joblib.load(model_path)
    scaler_path = model_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    root_info_path = run_dir / "feature_names.json"
    with open(root_info_path) as f:
        root_info = json.load(f)
    all_names = list(root_info["feature_names"])
    selected_names = list(model_info["feature_names"])
    missing = [name for name in selected_names if name not in all_names]
    if missing:
        raise ValueError(f"model asks for unknown features: {missing}")
    selected_idx = [all_names.index(name) for name in selected_names]
    amp_clip = float(model_info.get("amp_clip", root_info.get("amp_clip")))

    d = _resolve(args.npy_dir)
    prefix = args.prefix
    features = np.load(d / f"{prefix}features.npy", mmap_mode="r")
    offsets = np.load(d / f"{prefix}offsets.npy")
    n_sig_strings = np.load(d / f"{prefix}n_sig_strings.npy")
    n_sig_hits = _load_optional(d, prefix, "n_sig_hits.npy")
    energy = _load_optional(d, prefix, "energy.npy")
    event_fk = _load_optional(d, prefix, "event_fk.npy")
    part_keys = _load_optional(
        d, prefix, "h5_part_keys.npy", allow_pickle=True
    )
    local_ids = _load_optional(d, prefix, "h5_local_event_ids.npy")
    particle_types = _load_optional(d, prefix, "particle_types.npy")
    theta = _load_optional(d, prefix, "theta.npy")
    flat_weight = _load_optional(d, prefix, "flat_weights.npy")

    n_events = len(offsets) - 1
    logger.info(
        "Computing features for %s events from %s hits (%s)",
        f"{n_events:,}", f"{len(features):,}", d,
    )
    matrix = compute_non_neural_features(
        features,
        offsets,
        n_sig_strings,
        amp_clip=amp_clip,
        n_sig_hits=n_sig_hits,
    )
    if matrix.shape[1] != len(all_names):
        raise ValueError(
            f"feature matrix has {matrix.shape[1]} columns, "
            f"training config has {len(all_names)}"
        )
    X = np.asarray(matrix[:, selected_idx], dtype=np.float64)
    if scaler is not None:
        X = scaler.transform(X)
    pred = np.asarray(model.predict(X), dtype=np.float64)

    true_log10e = (
        np.log10(np.asarray(energy, dtype=np.float64))
        if energy is not None else np.full(n_events, np.nan)
    )
    out = pd.DataFrame({
        "event_fk": _column(event_fk, n_events, -1).astype(np.int64),
        "part_key": np.asarray(
            _column(part_keys, n_events, ""), dtype=object
        ),
        "local_event_id": _column(local_ids, n_events, -1).astype(np.int64),
        "split": args.split_name,
        "true_log10e": true_log10e.astype(np.float64),
        "true_energy_gev": (
            np.asarray(energy, dtype=np.float64)
            if energy is not None else np.full(n_events, np.nan)
        ),
        "pred_log10e": pred,
        "pred_energy_gev": np.power(10.0, pred),
        "n_hits": np.diff(offsets).astype(np.int64),
        "n_strings": np.asarray(n_sig_strings, dtype=np.int64),
        "particle_type": _column(particle_types, n_events, -1).astype(
            np.int64
        ),
        "theta": _column(theta, n_events, np.nan).astype(np.float64),
        "flat_weight": _column(flat_weight, n_events, 1.0).astype(
            np.float64
        ),
        "model": args.model,
        "feature_set": model_info.get("feature_set", ""),
        "amp_clip": amp_clip,
        "source_npy_dir": str(d),
    })

    out_dir = _resolve(args.output_dir) / f"{run_dir.name}@{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"{args.split_name}.duckdb"
    if db_path.exists() and not args.overwrite:
        raise FileExistsError(f"{db_path} exists; pass --overwrite")
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.register("pred_df", out)
    con.execute("CREATE TABLE predictions AS SELECT * FROM pred_df")
    con.close()
    logger.info("Wrote %s (%s rows)", db_path, f"{len(out):,}")


if __name__ == "__main__":
    main()
