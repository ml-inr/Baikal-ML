"""Train and evaluate the non-neural muon-energy baseline.

This script fits three tabular models on event-level handcrafted features:

* ``loglinear`` — LinearRegression on a small core set of log features;
* ``ridge`` — Ridge on all features, alpha selected on val-select;
* ``hgb`` — HistGradientBoostingRegressor on all features, grid selected on
  val-select.

It uses the same event-level train/val/calibration split as the neural trainer.
The held-out test parts are **not** in this dataset and are scored separately by
``predict_non_neural_energy.py``; they are never used for model selection.

Usage:
    python -m src.training.non_neural_energy_baseline \
        --config src/training/configs/track_energy_reconstruction/\
non_neural_energy_baseline.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import duckdb
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
LOG10_2 = float(np.log10(2.0))
LOG10_5 = float(np.log10(5.0))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _split_indices(
    n_events: int,
    train_split: float,
    split_seed: int,
    calibration_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce MuEnergyTrainer's event-level train/val/cal split."""
    if not 0.0 < train_split < 1.0:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if not 0.0 <= calibration_fraction < 0.5:
        raise ValueError(
            "calibration_fraction must be in [0, 0.5), "
            f"got {calibration_fraction}"
        )
    perm = np.random.RandomState(split_seed).permutation(n_events)
    n_train = int(n_events * train_split)
    train = perm[:n_train]
    val_all = perm[n_train:]
    rng = np.random.RandomState(split_seed + 1)
    perm_val = rng.permutation(len(val_all))
    n_cal = int(round(len(val_all) * calibration_fraction))
    cal = val_all[perm_val[:n_cal]]
    val_select = val_all[perm_val[n_cal:]]
    return train, val_select, cal


def _metrics(
    y: np.ndarray,
    pred: np.ndarray,
    weight: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Regression metrics in dex, unweighted and flat-weight weighted."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(y) != len(pred):
        raise ValueError(f"y/pred length mismatch: {len(y)} vs {len(pred)}")
    if len(y) == 0:
        return {"n": 0.0}
    residual = y - pred
    abs_err = np.abs(residual)
    out: dict[str, float] = {
        "n": float(len(y)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "bias": float(residual.mean()),
        "median_ae": float(np.median(abs_err)),
        "residual_p16": float(np.percentile(residual, 16)),
        "residual_p84": float(np.percentile(residual, 84)),
        "within_factor_2": float(np.mean(abs_err <= LOG10_2)),
        "within_factor_5": float(np.mean(abs_err <= LOG10_5)),
        "within_factor_10": float(np.mean(abs_err <= 1.0)),
        "frac_abs_err_gt_1dex": float(np.mean(abs_err > 1.0)),
    }
    if len(y) >= 3 and np.std(y) > 0.0 and np.std(pred) > 0.0:
        out["pearson"] = float(pearsonr(y, pred)[0])
        out["spearman"] = float(spearmanr(y, pred).statistic)
        ss_res = float(np.sum(residual * residual))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    else:
        out.update({"pearson": float("nan"), "spearman": float("nan"),
                    "r2": float("nan")})

    if weight is not None:
        w = np.asarray(weight, dtype=np.float64)
        if len(w) != len(y):
            raise ValueError(f"weight length {len(w)} != {len(y)}")
        wsum = float(w.sum())
        if wsum > 0.0:
            out.update({
                "w_mae": float(np.average(abs_err, weights=w)),
                "w_rmse": float(
                    np.sqrt(np.average(residual * residual, weights=w))
                ),
                "w_bias": float(np.average(residual, weights=w)),
                "w_median_ae": float(
                    _weighted_median(abs_err, w)
                ),
            })
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0.0:
        return float("nan")
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def _add_bins(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    pred: np.ndarray,
    split: str,
    model: str,
    feature_set: str,
    bin_col: str,
    bins: np.ndarray,
    labels: list[str],
    weight: np.ndarray,
) -> None:
    for label, (lo, hi) in zip(labels, zip(bins[:-1], bins[1:])):
        mask = (frame[bin_col].to_numpy() >= lo) & (
            frame[bin_col].to_numpy() < hi
        )
        if not mask.any():
            continue
        row = {
            "model": model,
            "feature_set": feature_set,
            "split": split,
            "bin": label,
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            **_metrics(frame["true_log10e"].to_numpy()[mask],
                       pred[mask], weight[mask]),
        }
        rows.append(row)


def _bin_labels(bins: np.ndarray) -> list[str]:
    return [f"{lo:.1f}..{hi:.1f}" for lo, hi in zip(bins[:-1], bins[1:])]


def _predict_frame(
    meta: pd.DataFrame,
    split: str,
    pred: np.ndarray,
    model: str,
    feature_set: str,
    amp_clip: float,
) -> pd.DataFrame:
    out = meta.copy()
    out["split"] = split
    out["pred_log10e"] = pred.astype(np.float64)
    out["pred_energy_gev"] = np.power(10.0, pred.astype(np.float64))
    out["model"] = model
    out["feature_set"] = feature_set
    out["amp_clip"] = float(amp_clip)
    return out


def _fit_linear(
    X: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    *,
    weighted: bool,
    kind: str,
    alphas: Iterable[float] = (),
) -> tuple[Any, StandardScaler, dict[str, Any]]:
    scaler = StandardScaler().fit(X[train])
    Xtr = scaler.transform(X[train])
    Xva = scaler.transform(X[val])
    sw = weight[train] if weighted else None
    if kind == "loglinear":
        model = LinearRegression().fit(Xtr, y[train], sample_weight=sw)
        return model, scaler, {"kind": "loglinear"}
    if kind != "ridge":
        raise ValueError(f"unknown linear kind {kind}")

    best: Optional[dict[str, Any]] = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(
            Xtr, y[train], sample_weight=sw
        )
        val_pred = model.predict(Xva)
        mae = float(np.mean(np.abs(y[val] - val_pred)))
        if best is None or mae < best["val_mae"]:
            best = {
                "kind": "ridge",
                "alpha": float(alpha),
                "val_mae": mae,
                "model": model,
            }
    if best is None:
        raise ValueError("ridge_alphas is empty")
    model = best.pop("model")
    return model, scaler, best


def _fit_hgb(
    X: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    *,
    weighted: bool,
    grid: dict[str, list[Any]],
) -> tuple[HistGradientBoostingRegressor, dict[str, Any], pd.DataFrame]:
    sw = weight[train] if weighted else None
    keys = ["max_iter", "learning_rate", "max_leaf_nodes",
            "l2_regularization", "loss"]
    combos = list(itertools.product(*(grid[k] for k in keys)))
    rows: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    for values in combos:
        params = dict(zip(keys, values))
        model = HistGradientBoostingRegressor(
            **params,
            early_stopping=False,
            random_state=42,
        )
        model.fit(X[train], y[train], sample_weight=sw)
        val_pred = model.predict(X[val])
        mae = float(np.mean(np.abs(y[val] - val_pred)))
        row = {**params, "val_mae": mae}
        rows.append(row)
        if best is None or mae < best["val_mae"]:
            best = {"params": params, "val_mae": mae, "model": model}
    if best is None:
        raise ValueError("hgb_grid is empty")
    model = best.pop("model")
    return model, best, pd.DataFrame(rows)


def _subset_features(
    names: list[str],
    groups: dict[str, list[str]],
    selected_groups: Iterable[str],
) -> list[int]:
    selected: set[str] = set()
    for group in selected_groups:
        if group not in groups:
            raise ValueError(f"unknown feature group {group!r}")
        selected.update(groups[group])
    indices = [i for i, name in enumerate(names) if name in selected]
    if not indices:
        raise ValueError(f"empty feature subset for groups {selected_groups}")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "src/training/configs/track_energy_reconstruction/"
            "non_neural_energy_baseline.yaml"
        ),
    )
    parser.add_argument("--name", default=None, help="override experiment name")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    config_path = _resolve(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if args.name:
        config["experiment"]["name"] = args.name

    seed = int(config["experiment"].get("seed", 42))
    np.random.seed(seed)

    data_cfg = config["data"]
    dataset_dir = _resolve(data_cfg["dataset_dir"])
    features_path = dataset_dir / data_cfg["features_file"]
    features_config_path = dataset_dir / data_cfg["features_config"]
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} does not exist; run "
            "data_manager.mu_energy_ds_builder.build_non_neural_features"
        )
    with open(features_config_path) as f:
        feature_config = yaml.safe_load(f)
    feature_names = list(feature_config["feature_names"])
    feature_groups = {
        key: list(value)
        for key, value in feature_config["feature_groups"].items()
    }
    amp_clip = float(feature_config["amp_clip"])

    X = np.load(features_path, mmap_mode="r")
    if X.shape != (feature_config["n_events"], len(feature_names)):
        raise ValueError(
            f"{features_path} has shape {X.shape}, expected "
            f"({feature_config['n_events']}, {len(feature_names)})"
        )
    energy = np.load(dataset_dir / "energy.npy").astype(np.float64)
    flat_weight = np.load(dataset_dir / "flat_weights.npy").astype(
        np.float64
    )
    event_fk = np.load(dataset_dir / "event_fk.npy")
    part_keys = np.load(dataset_dir / "h5_part_keys.npy", allow_pickle=True)
    local_ids = np.load(dataset_dir / "h5_local_event_ids.npy")
    n_hits = np.load(dataset_dir / "n_sig_hits.npy")
    n_strings = np.load(dataset_dir / "n_sig_strings.npy")
    particle_types = np.load(dataset_dir / "particle_types.npy")
    theta = np.load(dataset_dir / "theta.npy").astype(np.float64)

    n_events = len(energy)
    if not (
        len(X) == len(flat_weight) == len(event_fk) == len(part_keys)
        == len(local_ids) == len(n_hits) == len(n_strings)
        == len(particle_types) == len(theta) == n_events
    ):
        raise ValueError("source array lengths do not match")
    y = np.log10(energy)
    if not np.isfinite(y).all():
        raise ValueError("non-finite log10 energy")
    if not np.isfinite(X).all():
        raise ValueError("non-finite feature matrix")

    train, val, cal = _split_indices(
        n_events,
        float(data_cfg.get("train_split", 0.9)),
        int(data_cfg.get("split_seed", seed)),
        float(data_cfg.get("calibration_fraction", 0.0)),
    )
    logger.info(
        "Split: train %s, val-select %s, calibration %s",
        f"{len(train):,}", f"{len(val):,}", f"{len(cal):,}",
    )

    run_dir = _resolve(config["output"]["root"]) / config["experiment"]["name"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} already exists and is not empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir()

    meta = pd.DataFrame({
        "event_fk": event_fk,
        "part_key": np.asarray(part_keys, dtype=object),
        "local_event_id": local_ids,
        "true_log10e": y,
        "true_energy_gev": energy,
        "n_hits": n_hits,
        "n_strings": n_strings,
        "particle_type": particle_types,
        "theta": theta,
        "flat_weight": flat_weight,
    })
    weighted = bool(data_cfg.get("weighted", True))
    weight = flat_weight if weighted else np.ones_like(flat_weight)

    prediction_frames: list[pd.DataFrame] = []
    global_rows: dict[str, dict[str, dict[str, float]]] = {
        "train": {}, "val_select": {}, "cal": {},
    }
    by_energy: list[dict[str, Any]] = []
    by_class: list[dict[str, Any]] = []
    by_n_hits: list[dict[str, Any]] = []
    by_theta: list[dict[str, Any]] = []
    part_rows: list[dict[str, Any]] = []
    linear_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    hgb_grid_rows: list[pd.DataFrame] = []
    selection: dict[str, Any] = {}

    energy_bins = np.arange(-1.0, 8.5, 0.5)
    energy_labels = _bin_labels(energy_bins)
    nh_bins = np.array([8, 12, 16, 24, 40, 80, 160, 400], dtype=float)
    nh_labels = _bin_labels(nh_bins)
    theta_bins = np.array([0.0, 30.0, 60.0, 90.01], dtype=float)
    theta_labels = _bin_labels(theta_bins)

    train_models = list(config["models"]["train"])
    core_names = list(config["models"]["loglinear_features"])
    core_idx = [feature_names.index(name) for name in core_names]
    all_idx = list(range(len(feature_names)))

    def record_model(
        name: str,
        feature_set: str,
        model: Any,
        scaler: Optional[StandardScaler],
        indices: list[int],
        extra: dict[str, Any],
    ) -> None:
        Xm = np.asarray(X[:, indices], dtype=np.float64)
        if scaler is not None:
            Xm = scaler.transform(Xm)
        train_pred = model.predict(Xm[train])
        val_pred = model.predict(Xm[val])
        cal_pred = model.predict(Xm[cal])
        for split, idx, pred in (
            ("train", train, train_pred),
            ("val_select", val, val_pred),
            ("cal", cal, cal_pred),
        ):
            global_rows[split][name] = _metrics(
                y[idx], pred, weight[idx]
            )
        model_dir = models_dir / name
        model_dir.mkdir()
        joblib.dump(model, model_dir / "model.joblib")
        if scaler is not None:
            joblib.dump(scaler, model_dir / "scaler.joblib")
        model_info = {
            "model": name,
            "feature_set": feature_set,
            "feature_names": [feature_names[i] for i in indices],
            "scaler": scaler is not None,
            "amp_clip": amp_clip,
            "selection": extra,
        }
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        selection[name] = extra

        for split, idx, pred in (
            ("val_select", val, val_pred),
            ("cal", cal, cal_pred),
        ):
            prediction_frames.append(
                _predict_frame(meta.iloc[idx].reset_index(drop=True),
                               split, pred, name, feature_set, amp_clip)
            )
            _add_bins(
                by_energy, meta.iloc[idx], pred, split, name, feature_set,
                "true_log10e", energy_bins, energy_labels, weight[idx],
            )
            for code in np.unique(particle_types[idx]):
                mask = particle_types[idx] == code
                by_class.append({
                    "model": name,
                    "feature_set": feature_set,
                    "split": split,
                    "particle_type": int(code),
                    **_metrics(y[idx][mask], pred[mask], weight[idx][mask]),
                })
            _add_bins(
                by_n_hits, meta.iloc[idx], pred, split, name, feature_set,
                "n_hits", nh_bins, nh_labels, weight[idx],
            )
            _add_bins(
                by_theta, meta.iloc[idx], pred, split, name, feature_set,
                "theta", theta_bins, theta_labels, weight[idx],
            )
            frame = meta.iloc[idx].reset_index(drop=True)
            frame["pred"] = pred
            for part_key, group in frame.groupby("part_key", sort=True):
                part_rows.append({
                    "model": name,
                    "feature_set": feature_set,
                    "split": split,
                    "part_key": str(part_key),
                    **_metrics(
                        group["true_log10e"].to_numpy(),
                        group["pred"].to_numpy(),
                        group["flat_weight"].to_numpy(),
                    ),
                })
        if feature_set == "all" and scaler is not None:
            for feature, coef in zip(
                [feature_names[i] for i in indices],
                np.ravel(model.coef_),
            ):
                linear_rows.append({
                    "model": name,
                    "feature": feature,
                    "coefficient": float(coef),
                })

    if "loglinear" in train_models:
        logger.info("Fitting loglinear on %s features", len(core_idx))
        model, scaler, extra = _fit_linear(
            np.asarray(X[:, core_idx], dtype=np.float64),
            y, weight, train, val, weighted=weighted, kind="loglinear",
        )
        record_model(
            "loglinear", "core", model, scaler, core_idx, extra
        )

    if "ridge" in train_models:
        logger.info("Fitting Ridge on %s features", len(all_idx))
        model, scaler, extra = _fit_linear(
            np.asarray(X[:, all_idx], dtype=np.float64),
            y, weight, train, val,
            weighted=weighted,
            kind="ridge",
            alphas=config["models"]["ridge_alphas"],
        )
        record_model(
            "ridge", "all", model, scaler, all_idx, extra
        )

    hgb_model: Optional[HistGradientBoostingRegressor] = None
    if "hgb" in train_models:
        logger.info("Fitting HGB grid on %s features", len(all_idx))
        hgb_model, extra, grid_df = _fit_hgb(
            np.asarray(X[:, all_idx], dtype=np.float64),
            y, weight, train, val,
            weighted=weighted,
            grid=config["models"]["hgb_grid"],
        )
        record_model("hgb", "all", hgb_model, None, all_idx, extra)
        grid_df.insert(0, "selected", False)
        grid_df.loc[grid_df["val_mae"].idxmin(), "selected"] = True
        hgb_grid_rows.append(grid_df)
        logger.info("HGB best: %s", extra)
        logger.info("Computing permutation importance on val-select")
        imp = permutation_importance(
            hgb_model,
            np.asarray(X[val][:, all_idx], dtype=np.float64),
            y[val],
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=seed,
            n_jobs=1,
        )
        for feature, mean, std in zip(
            [feature_names[i] for i in all_idx],
            imp.importances_mean,
            imp.importances_std,
        ):
            importance_rows.append({
                "model": "hgb",
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
            })

    if (
        hgb_model is not None
        and config.get("ablations", {}).get("enabled", False)
    ):
        for ablation_name, groups in (
            config["ablations"].get("hgb", {}) or {}
        ).items():
            if ablation_name == "all":
                continue
            indices = _subset_features(
                feature_names, feature_groups, groups
            )
            logger.info(
                "Fitting HGB ablation %s (%s features)", ablation_name,
                len(indices),
            )
            model = HistGradientBoostingRegressor(
                **selection["hgb"]["params"],
                early_stopping=False,
                random_state=seed,
            )
            sw = weight[train] if weighted else None
            model.fit(
                np.asarray(X[train][:, indices], dtype=np.float64),
                y[train], sample_weight=sw,
            )
            record_model(
                f"hgb_{ablation_name}", ablation_name, model, None,
                indices, {"params": selection["hgb"]["params"]},
            )

    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
    else:
        predictions = pd.DataFrame()

    if not predictions.empty:
        db_path = run_dir / "predictions.duckdb"
        con = duckdb.connect(str(db_path))
        con.register("pred_df", predictions)
        con.execute("CREATE TABLE predictions AS SELECT * FROM pred_df")
        con.close()
        logger.info("Wrote %s (%s rows)", db_path, f"{len(predictions):,}")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "run": {
                "name": config["experiment"]["name"],
                "seed": seed,
                "weighted": weighted,
                "amp_clip": amp_clip,
                "dataset_dir": str(dataset_dir),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            "split_sizes": {
                "train": int(len(train)),
                "val_select": int(len(val)),
                "cal": int(len(cal)),
            },
            "models": train_models,
            "global": global_rows,
            "model_selection": selection,
        }, f, indent=2)

    def save_frame(name: str, rows: list[dict[str, Any]]) -> None:
        if rows:
            pd.DataFrame(rows).to_parquet(run_dir / name, index=False)
            logger.info("Wrote %s (%s rows)", name, len(rows))

    save_frame("metrics_by_true_energy.parquet", by_energy)
    save_frame("metrics_by_class.parquet", by_class)
    save_frame("metrics_by_n_hits.parquet", by_n_hits)
    save_frame("metrics_by_theta.parquet", by_theta)
    save_frame("part_metrics.parquet", part_rows)
    save_frame("linear_coefficients.parquet", linear_rows)
    save_frame("feature_importance.parquet", importance_rows)
    if hgb_grid_rows:
        pd.concat(hgb_grid_rows, ignore_index=True).to_parquet(
            run_dir / "hgb_grid_results.parquet", index=False
        )

    with open(run_dir / "feature_names.json", "w") as f:
        json.dump({
            "feature_names": feature_names,
            "feature_groups": feature_groups,
            "core_features": core_names,
            "amp_clip": amp_clip,
            "source_features_config": str(features_config_path),
        }, f, indent=2)
    with open(run_dir / "run_info.json", "w") as f:
        json.dump({
            "config_path": str(config_path),
            "dataset_dir": str(dataset_dir),
            "features_config": feature_config,
            "split_sizes": {
                "train": int(len(train)),
                "val_select": int(len(val)),
                "cal": int(len(cal)),
            },
        }, f, indent=2)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info("Run directory: %s", run_dir)


if __name__ == "__main__":
    main()
