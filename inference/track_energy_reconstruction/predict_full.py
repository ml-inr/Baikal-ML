"""Predict muon energy on the full h8s3 MC source.

The training dataset written by ``data_manager.mu_energy_ds_builder`` contains only
the flat-selected train+val population.  This script streams the **full** h8s3 pool
from the MC HDF5 and the stored sig-noise probabilities, applies the same event cuts
and the same ``prob > threshold`` hit selection, and writes predictions to DuckDB.

Every source event is marked with the provenance of the checkpoint's training
dataset: ``train`` / ``val`` for events present in ``--npy-dir`` and ``test`` for
everything else.  ``--subset all`` writes all events; ``--subset test`` writes only
the held-out complement.  The DB is keyed by ``event_fk`` with ``INSERT OR IGNORE``,
so the two subsets can be accumulated in either order into the same file.

The checkpoint stores the target z-normalization, the feature normalization and
``max_hits``; the model output is converted back to ``log10(E/GeV)`` before writing.

Usage (from the repository root):
    python -m inference.track_energy_reconstruction.predict_full \
        --checkpoint trained_models/track_energy_reconstruction/<run>/best_model.pth \
        --npy-dir data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M \
        [--subset all|test] [--batch-size 512] [--device auto] \
        [--output-dir inference/track_energy_reconstruction/preds] \
        [--limit N] [--check-existing]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data_manager.mu_energy_ds_builder.build import (  # noqa: E402
    DEFAULT_CATALOG_PATH,
    DEFAULT_COMPANION_PATH,
    DEFAULT_H5_PATH,
    DEFAULT_PROBS_PATH,
    PARTICLE_ENCODE,
    PARTICLE_MAP,
    _read_event_metadata,
    _read_targets,
)
from data_manager.mu_energy_ds_builder.io import (  # noqa: E402
    read_part_metadata_from_probs,
)
from inference.shared.catalog_query import get_event_fks_mc  # noqa: E402
from inference.shared.history import append_history_row  # noqa: E402
from inference.shared.run_info import probs_batch_size, write_run_info  # noqa: E402
from inference.track_energy_reconstruction.store import (  # noqa: E402
    append_predictions_energy,
    open_energy_db,
)
from src.data.mu_energy_dataset import mu_energy_collate_fn  # noqa: E402
from src.models.mu_energy_model import create_mu_energy_model  # noqa: E402

logger = logging.getLogger(__name__)

FLUSH_EVERY = 50_000
CODE_TO_NAME = {v: k for k, v in PARTICLE_ENCODE.items()}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_energy_model(
    checkpoint: str, device: str
) -> tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load the energy regressor and its target/feature normalization."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    norm_config = ckpt.get("normalization_config")
    target_norm = (
        ckpt.get("target_normalization")
        or config.get("data", {}).get("target_normalization")
        or {"enabled": False}
    )

    model_cfg = copy.deepcopy(config["model"])
    raw_clip = model_cfg.get("amp_clip")
    if raw_clip is not None and norm_config is not None:
        model_cfg["amp_clip"] = (
            float(raw_clip) - norm_config["means"][0]
        ) / norm_config["stds"][0]

    model = create_mu_energy_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(
        "Loaded %s: epoch %s, best_metric=%s",
        checkpoint, ckpt.get("epoch"), ckpt.get("best_metric"),
    )
    return model, norm_config, target_norm, ckpt


def _load_parts_by_class(path: str) -> Dict[str, set[str]]:
    """Load class -> part-key sets from a YAML/JSON file.

    The file uses the builder's keys (``nuatm``, ``nue2``) and part numbers.  A
    class absent from the file has no parts to score; this is intentional, so a
    part number shared by two classes cannot leak a train class into a test scan.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    out: Dict[str, set[str]] = {}
    for key, values in raw.items():
        if key not in PARTICLE_MAP:
            raise ValueError(f"unknown particle key {key!r} in {path}")
        out[PARTICLE_MAP[key]] = {f"part_{int(v)}" for v in values}
    return out


def _list_parts(
    h5: h5py.File, probs: h5py.File, ptype: str, threshold: float,
    only: Optional[set[str]] = None,
) -> list[str]:
    """Part keys that have both raw hits and stored SN counts at this threshold."""
    key = f"n_sig_hits_{threshold:g}"
    if key not in probs[ptype]:
        raise KeyError(
            f"{ptype}: {key} not stored in the probs file; available: "
            f"{[k for k in probs[ptype].keys() if k.startswith('n_sig_hits_')]}"
        )
    raw = set(h5[ptype]["raw/ev_starts"].keys())
    stored = set(probs[ptype][key].keys())
    parts = raw & stored
    if only is not None:
        parts &= only
    return sorted(parts, key=lambda s: int(s.rsplit("_", 1)[-1]))


def _load_provenance(npy_dir: str, ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Load the checkpoint's training-dataset provenance and reproduce its split.

    Two layouts are supported:
      * new: the dataset contains only flat train+val rows, ``flat_weights.npy`` is
        aligned with every row, and the split comes from the checkpoint's training
        YAML (``data.train_split`` / ``data.split_seed``);
      * legacy: split files are present and ``flat_weights.npy`` is aligned with
        ``flat_indices.npy``.  These datasets predate the refactor and are kept
        readable so existing checkpoints remain analysable.
    """
    d = Path(npy_dir)
    info = json.loads((d / "dataset_info.json").read_text())
    n_total = int(info["n_events"])
    fk = np.load(d / "event_fk.npy").astype(np.int64)
    part_keys = np.load(d / "h5_part_keys.npy", allow_pickle=True)
    local_ids = np.load(d / "h5_local_event_ids.npy").astype(np.int64)
    for arr, name in (
        (fk, "event_fk"), (part_keys, "h5_part_keys"),
        (local_ids, "h5_local_event_ids"),
    ):
        if len(arr) != n_total:
            raise ValueError(
                f"{d / name}: {len(arr)} rows, dataset_info says {n_total}"
            )

    split = np.full(n_total, "test", dtype=object)
    flat_weight = np.full(n_total, np.nan, dtype=np.float32)
    legacy_train = d / "split_train_indices.npy"
    if legacy_train.exists():
        for name in ("train", "val", "test"):
            path = d / f"split_{name}_indices.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            split[np.load(path)] = name
        flat_indices = d / "flat_indices.npy"
        if flat_indices.exists():
            fi = np.load(flat_indices)
            fw = np.load(d / "flat_weights.npy")
            if len(fi) != len(fw):
                raise ValueError("flat_indices and flat_weights differ in length")
            flat_weight[fi] = fw
        split_mode = "legacy split files"
        train_frac = split_seed = None
    else:
        fw = np.load(d / "flat_weights.npy").astype(np.float32)
        if len(fw) != n_total:
            raise ValueError(
                f"{d / 'flat_weights.npy'} has {len(fw)} weights, expected {n_total}"
            )
        flat_weight[:] = fw
        data_cfg = ckpt["config"].get("data", {})
        train_frac = float(data_cfg.get("train_split", 0.9))
        split_seed = int(
            data_cfg.get(
                "split_seed",
                ckpt["config"].get("experiment", {}).get("seed", 42),
            )
        )
        perm = np.random.RandomState(split_seed).permutation(n_total)
        n_train = int(n_total * train_frac)
        split[perm[:n_train]] = "train"
        split[perm[n_train:]] = "val"
        split_mode = "checkpoint YAML train_split"

    fk_to_row = {
        int(f): i for i, f in enumerate(fk) if int(f) >= 0
    }
    if len(fk_to_row) != int((fk >= 0).sum()):
        raise ValueError(f"{d}: event_fk is not unique")
    return {
        "info": info,
        "n_total": n_total,
        "event_fk": fk,
        "split": split,
        "flat_weight": flat_weight,
        "used_for_labels": split == "train",
        "fk_to_row": fk_to_row,
        "split_mode": split_mode,
        "train_frac": train_frac,
        "split_seed": split_seed,
    }


def _selection_mask(
    ptype_full: str,
    n_tracks: np.ndarray,
    energy: np.ndarray,
    status: np.ndarray,
    n_sig_hits: np.ndarray,
    n_sig_strings: np.ndarray,
    min_hits: int,
    min_strings: int,
    valid_status: tuple[int, ...],
    born_inside_status: tuple[int, ...],
    exclude_nuatm_inside: bool,
) -> np.ndarray:
    """The builder's event cuts, replayed on the full source."""
    mask = (
        (n_sig_hits >= min_hits)
        & (n_sig_strings >= min_strings)
        & (n_tracks == 1)
        & np.isfinite(energy)
        & np.isin(status, valid_status)
    )
    if exclude_nuatm_inside and ptype_full == "nuatm_2020":
        mask &= ~np.isin(status, born_inside_status)
    return mask


def predict_full(
    checkpoint: str,
    npy_dir: str,
    output_dir: str = "inference/track_energy_reconstruction/preds",
    subset: str = "all",
    batch_size: int = 512,
    device: str = "auto",
    limit: Optional[int] = None,
    check_existing: bool = False,
    parts: Optional[list[str]] = None,
    parts_file: Optional[str] = None,
) -> None:
    if subset not in ("all", "test", "val"):
        raise ValueError(f"subset must be 'all', 'test' or 'val', got {subset!r}")
    if parts and parts_file:
        raise ValueError("pass either --parts or --parts-file, not both")

    t_start = datetime.now()
    dev = _resolve_device(device)
    ckpt_path = Path(checkpoint)
    checkpoint_name = f"{ckpt_path.parent.name}@{ckpt_path.stem}"
    npy_path = Path(npy_dir)

    model, norm_config, target_norm, ckpt = load_energy_model(checkpoint, dev)
    has_variance = bool(
        getattr(model, "heteroscedastic", False)
        and hasattr(model, "forward_with_variance")
    )
    config = ckpt["config"]
    data_cfg = config.get("data", {})
    max_hits = data_cfg.get("max_hits", 500)
    if target_norm.get("enabled", True):
        target_mean = float(target_norm["mean"])
        target_std = float(target_norm["std"])
    else:
        target_mean, target_std = 0.0, 1.0
    logger.info(
        "Target normalization: z=(log10E - %.6f)/%.6f (enabled=%s); max_hits=%s",
        target_mean, target_std, target_norm.get("enabled", True), max_hits,
    )

    prov = _load_provenance(npy_dir, ckpt)
    ds_info = prov["info"]
    threshold = float(ds_info.get("sig_noise_threshold", 0.8))
    cuts = ds_info.get("event_cuts", {})
    min_hits = int(cuts.get("min_hits", 8))
    min_strings = int(cuts.get("min_strings", 3))
    target = ds_info.get("target", {})
    target_column = int(target.get("column", 1))
    valid_status = tuple(int(x) for x in target.get("valid_status", [0, 1, 4, 5]))
    born_inside_status = tuple(
        int(x) for x in target.get("born_inside_status", [1, 5])
    )
    exclude_nuatm_inside = bool(target.get("exclude_nuatm_born_inside", True))

    flat_cfg = ds_info.get("flat_selection") or {}
    classes = tuple(flat_cfg.get("classes") or ["nuatm_2020", "nue2_2020"])
    only_parts = set(parts) if parts else None
    parts_by_class = _load_parts_by_class(parts_file) if parts_file else None
    unknown = [c for c in classes if c not in PARTICLE_MAP.values()]
    if unknown:
        raise ValueError(f"unknown classes in dataset_info: {unknown}")
    h5_path = ds_info.get("h5_source", DEFAULT_H5_PATH)
    companion_path = ds_info.get("companion", DEFAULT_COMPANION_PATH)
    catalog_path = ds_info.get("catalog", DEFAULT_CATALOG_PATH)
    probs_path = ds_info.get("sig_noise_probs_path", DEFAULT_PROBS_PATH)

    thr_tag = str(threshold).replace(".", "p")
    cut_tag = f"h{min_hits}s{min_strings}"
    source = "mc_merged"
    db_name = f"{source}_{cut_tag}_thr{thr_tag}.duckdb"

    run_dir = Path(output_dir) / checkpoint_name
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / db_name
    run_info_path = run_dir / "run_info.json"
    history_path = Path(output_dir) / "prediction_history.csv"
    log_path = run_dir / f"predict_full_{t_start.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )
    logger.info("Checkpoint: %s", checkpoint)
    logger.info("Training dataset provenance: %s (%s)", npy_dir, prov["split_mode"])
    logger.info("Full source: %s", h5_path)
    logger.info("SN probs:    %s (threshold %s)", probs_path, threshold)
    logger.info("Subset: %s | output DB: %s", subset, db_path)

    # ---- Pass 1: per-event metadata and selection on the full source --------
    logger.info("\nPass 1: full-source event selection")
    coll: Dict[str, list[np.ndarray]] = {
        "ptype": [], "part_key": [], "local_id": [], "hit_start": [],
        "hit_end": [], "energy": [], "status": [], "n_tracks": [],
        "n_sig_hits": [], "n_sig_strings": [], "theta": [],
        "event_weight": [], "sign": [],
    }
    n_seen = 0
    t0 = time.time()
    stop = False
    with h5py.File(h5_path, "r") as h5, h5py.File(companion_path, "r") as comp, \
            h5py.File(probs_path, "r") as pf:
        for ptype_full in classes:
            only = (
                parts_by_class.get(ptype_full)
                if parts_by_class is not None else only_parts
            )
            if parts_by_class is not None and only is None:
                logger.info("  %s: no parts in --parts-file; skipped", ptype_full)
                continue
            part_list = _list_parts(
                h5, pf, ptype_full, threshold, only=only,
            )
            logger.info("  %s: %s parts", ptype_full, f"{len(part_list):,}")
            for i, pk in enumerate(part_list):
                meta = read_part_metadata_from_probs(
                    h5[ptype_full], pf[ptype_full], pk, threshold,
                )
                if meta is None:
                    continue
                n_ev = len(meta["n_hits"])
                targets = _read_targets(
                    comp, ptype_full, pk, n_ev, target_column
                )
                extra = _read_event_metadata(h5, ptype_full, pk, n_ev)
                keep = np.flatnonzero(_selection_mask(
                    ptype_full,
                    targets["n_tracks"], targets["energy"], targets["status"],
                    meta["n_sig_hits"], meta["n_sig_strings"],
                    min_hits, min_strings, valid_status, born_inside_status,
                    exclude_nuatm_inside,
                ))
                if limit is not None:
                    keep = keep[: max(0, int(limit) - n_seen)]
                if len(keep):
                    coll["ptype"].append(
                        np.full(len(keep), PARTICLE_ENCODE[ptype_full], np.int8)
                    )
                    coll["part_key"].append(
                        np.full(len(keep), pk, dtype=object)
                    )
                    coll["local_id"].append(keep.astype(np.int32))
                    coll["hit_start"].append(meta["hit_starts"][keep])
                    coll["hit_end"].append(meta["hit_ends"][keep])
                    coll["energy"].append(targets["energy"][keep])
                    coll["status"].append(targets["status"][keep])
                    coll["n_tracks"].append(targets["n_tracks"][keep])
                    coll["n_sig_hits"].append(meta["n_sig_hits"][keep])
                    coll["n_sig_strings"].append(meta["n_sig_strings"][keep])
                    coll["theta"].append(extra["theta"][keep])
                    coll["event_weight"].append(extra["event_weight"][keep])
                    coll["sign"].append(extra["sign"][keep])
                    n_seen += len(keep)
                if (i + 1) % 100 == 0 or i == len(part_list) - 1:
                    logger.info(
                        "    [%d/%d] %s/%s: %s candidates (%.1fs)",
                        i + 1, len(part_list), ptype_full, pk,
                        f"{n_seen:,}", time.time() - t0,
                    )
                if limit is not None and n_seen >= int(limit):
                    stop = True
                    break
            if stop:
                break

    if n_seen == 0:
        raise RuntimeError("no events passed the full-source selection")
    meta = {
        k: np.concatenate(v) if v else np.array([])
        for k, v in coll.items()
    }
    n_full = len(meta["energy"])
    logger.info(
        "Pass 1 done: %s candidates in %.0fs", f"{n_full:,}", time.time() - t0
    )

    # ---- event_fk and training-dataset provenance ---------------------------
    logger.info("Catalog lookup for %s candidates...", f"{n_full:,}")
    t0 = time.time()
    ptype_names = np.array(
        [CODE_TO_NAME[int(c)] for c in meta["ptype"]], dtype=object
    )
    seasons = np.array(
        [int(name.rsplit("_", 1)[-1]) for name in ptype_names], dtype=np.int32
    )
    df = get_event_fks_mc(
        catalog_path=catalog_path,
        source=source,
        data_classes=ptype_names,
        seasons=seasons,
        runs=meta["part_key"],
        event_ids=meta["local_id"].astype(np.int64),
    )
    event_fk = np.full(n_full, -1, dtype=np.int64)
    idx = df["query_idx"].to_numpy(dtype=np.int64)
    event_fk[idx] = df["event_fk"].to_numpy(dtype=np.int64)
    n_missing_fk = int((event_fk < 0).sum())
    logger.info(
        "  event_fk: %s matched, %s missing (%.0fs)",
        f"{len(idx):,}", f"{n_missing_fk:,}", time.time() - t0,
    )

    split = np.full(n_full, "test", dtype=object)
    flat_weight = np.full(n_full, np.nan, dtype=np.float32)
    used_for_labels = np.zeros(n_full, dtype=bool)
    n_row_matched = 0
    fk_to_row = prov["fk_to_row"]
    for i in range(n_full):
        row = fk_to_row.get(int(event_fk[i]))
        if row is None:
            continue
        n_row_matched += 1
        split[i] = prov["split"][row]
        flat_weight[i] = prov["flat_weight"][row]
        used_for_labels[i] = bool(prov["used_for_labels"][row])
    split_counts = Counter(split.tolist())
    logger.info(
        "Provenance: dataset rows matched %s / %s; full-source split counts: %s",
        f"{n_row_matched:,}", f"{prov['n_total']:,}",
        dict(sorted(split_counts.items())),
    )
    if n_row_matched != prov["n_total"]:
        if limit is None and parts is None and parts_file is None:
            logger.warning(
                "Only %s / %s training-dataset rows were found in the full source; "
                "check that --npy-dir matches this checkpoint",
                f"{n_row_matched:,}", f"{prov['n_total']:,}",
            )
        else:
            logger.info(
                "Partial scan (--limit/--parts/--parts-file): %s / %s "
                "training-dataset rows matched; this is expected",
                f"{n_row_matched:,}", f"{prov['n_total']:,}",
            )

    # ---- Choose what to write ----------------------------------------------
    write_idx = (
        np.arange(n_full, dtype=np.int64)
        if subset == "all"
        else np.flatnonzero(split == subset)
    )
    write_idx = write_idx[event_fk[write_idx] >= 0]
    if limit is not None:
        write_idx = write_idx[: int(limit)]
    logger.info(
        "Selected %s events (subset=%s; %s test, %s train, %s val)",
        f"{len(write_idx):,}", subset,
        f"{int((split[write_idx] == 'test').sum()):,}",
        f"{int((split[write_idx] == 'train').sum()):,}",
        f"{int((split[write_idx] == 'val').sum()):,}",
    )

    conn = open_energy_db(str(db_path))
    n_existing_skipped = 0
    if not check_existing:
        existing = conn.execute(
            "SELECT event_fk FROM predictions"
        ).df()["event_fk"].to_numpy(dtype=np.int64)
        if len(existing):
            keep = ~np.isin(event_fk[write_idx], existing, assume_unique=True)
            n_existing_skipped = int((~keep).sum())
            write_idx = write_idx[keep]
            logger.info(
                "Skipping %s events already in the DB; %s to predict",
                f"{n_existing_skipped:,}", f"{len(write_idx):,}",
            )

    # Group output events by part for one sequential read per part.
    groups: Dict[tuple[str, str], list[int]] = {}
    for i in write_idx:
        key = (CODE_TO_NAME[int(meta["ptype"][i])], str(meta["part_key"][i]))
        groups.setdefault(key, []).append(int(i))
    part_order = sorted(
        groups, key=lambda k: (k[0], int(k[1].rsplit("_", 1)[-1]))
    )

    # ---- Pass 2: stream hits, run the model, write DB -----------------------
    n_new = n_skipped = 0
    n_written = 0
    error_msg = ""
    t0 = time.perf_counter()
    chunks: list[pd.DataFrame] = []
    pending_items: list[Dict[str, Any]] = []
    pending_rows: list[int] = []
    n_parts_done = 0

    def flush() -> None:
        nonlocal n_new, n_skipped, n_written
        if not pending_items:
            return
        rows = np.asarray(pending_rows, dtype=np.int64)
        batch = mu_energy_collate_fn(
            pending_items,
            normalization_config=norm_config,
            target_normalization=None,
            augmentation_config=None,
        )
        model_input = {
            "features": batch["features"].to(dev, non_blocking=True),
            "mask": batch["mask"].to(dev, non_blocking=True),
            "lengths": batch["lengths"].to(dev, non_blocking=True),
        }
        with torch.no_grad():
            if has_variance:
                pred_z, log_var_z = model.forward_with_variance(model_input)
                sigma_dex = (
                    torch.exp(0.5 * log_var_z.float().cpu()).numpy()
                    * target_std
                )
            else:
                pred_z = model(model_input)
                sigma_dex = np.full(len(pred_z), np.nan, dtype=np.float32)
        pred_log = pred_z.float().cpu().numpy() * target_std + target_mean
        df_out = pd.DataFrame({
            "event_fk":          event_fk[rows],
            "pred_log10e":       pred_log.astype(np.float32),
            "pred_energy_gev":   np.power(10.0, pred_log.astype(np.float64)),
            "pred_sigma_dex":    np.asarray(sigma_dex, dtype=np.float32),
            "true_log10e":       np.log10(meta["energy"][rows]).astype(np.float32),
            "true_energy_gev":   meta["energy"][rows].astype(np.float64),
            "energy_status":     meta["status"][rows].astype(np.int64),
            "particle_type":     meta["ptype"][rows].astype(np.int64),
            "n_sig_hits":        meta["n_sig_hits"][rows].astype(np.int64),
            "n_sig_strings":     meta["n_sig_strings"][rows].astype(np.int64),
            "theta":             meta["theta"][rows].astype(np.float32),
            "split":             split[rows],
            "flat_weight":       flat_weight[rows],
            "used_for_labels":   used_for_labels[rows],
            "h5_part_key":       meta["part_key"][rows],
            "h5_local_event_id": meta["local_id"][rows].astype(np.int64),
        })
        chunks.append(df_out)
        n_written += len(rows)
        pending_items.clear()
        pending_rows.clear()
        if sum(len(c) for c in chunks) >= FLUSH_EVERY:
            new, skipped = append_predictions_energy(
                conn, pd.concat(chunks, ignore_index=True),
                check_existing=check_existing,
            )
            n_new += new
            n_skipped += skipped
            chunks.clear()

    try:
        with h5py.File(h5_path, "r") as h5, h5py.File(probs_path, "r") as pf:
            for ptype_full, pk in part_order:
                data_raw = h5[ptype_full][f"raw/data/{pk}/data"][:].astype(
                    np.float32
                )
                prob = pf[ptype_full][f"probs/{pk}/data"][:].astype(np.float32)
                if len(prob) != len(data_raw):
                    raise ValueError(
                        f"{ptype_full}/{pk}: probs have {len(prob)} entries, "
                        f"hits {len(data_raw)}"
                    )
                for i in groups[(ptype_full, pk)]:
                    s, e = int(meta["hit_start"][i]), int(meta["hit_end"][i])
                    ev_mask = prob[s:e] > threshold
                    n_sel = int(ev_mask.sum())
                    if n_sel != int(meta["n_sig_hits"][i]):
                        raise ValueError(
                            f"{ptype_full}/{pk}: event {int(meta['local_id'][i])} "
                            f"has {n_sel} selected hits, stored count says "
                            f"{int(meta['n_sig_hits'][i])}"
                        )
                    hits = data_raw[s:e][ev_mask]
                    if max_hits is not None:
                        hits = hits[: int(max_hits)]
                    length = len(hits)
                    pending_items.append({
                        "features": torch.from_numpy(hits),
                        "target": torch.tensor(
                            np.log10(float(meta["energy"][i])), dtype=torch.float32
                        ),
                        "energy": torch.tensor(
                            float(meta["energy"][i]), dtype=torch.float32
                        ),
                        "weight": torch.tensor(1.0, dtype=torch.float32),
                        "lengths": torch.tensor(length, dtype=torch.long),
                        "original_lengths": torch.tensor(length, dtype=torch.long),
                        "signal_hit_count": int(meta["n_sig_hits"][i]),
                        "signal_string_count": int(meta["n_sig_strings"][i]),
                        "particle_type": int(meta["ptype"][i]),
                        "theta": torch.tensor(
                            float(meta["theta"][i]), dtype=torch.float32
                        ),
                        "event_fk": int(event_fk[i]),
                        "part_key": pk,
                        "local_event_id": int(meta["local_id"][i]),
                        "energy_status": int(meta["status"][i]),
                    })
                    pending_rows.append(int(i))
                    if len(pending_items) >= batch_size:
                        flush()
                n_parts_done += 1
                if n_parts_done % 20 == 0 or n_parts_done == len(part_order):
                    logger.info(
                        "  [%d/%d] parts, %s / %s events written",
                        n_parts_done, len(part_order),
                        f"{n_written + len(pending_rows):,}", f"{len(write_idx):,}",
                    )
        flush()
        if chunks:
            new, skipped = append_predictions_energy(
                conn, pd.concat(chunks, ignore_index=True),
                check_existing=check_existing,
            )
            n_new += new
            n_skipped += skipped
        n_total_db = conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]
        conn.close()

        logger.info(
            "Written: %s new | skipped: %s existing | total in DB: %s",
            f"{n_new:,}", f"{n_skipped:,}", f"{n_total_db:,}",
        )
        write_run_info(
            run_info_path, source=source,
            checkpoint=checkpoint,
            npy_dir=str(npy_path),
            subset=subset,
            threshold=threshold,
            min_hits=min_hits,
            min_strings=min_strings,
            classes=list(classes),
            sn_probs_source=probs_path,
            sn_batch_size=probs_batch_size(probs_path),
            n_full_candidates=n_full,
            split_counts=dict(sorted(split_counts.items())),
            n_dataset_rows=prov["n_total"],
            n_dataset_rows_matched=n_row_matched,
            n_event_fk_missing=n_missing_fk,
            n_events_predicted=n_written,
            n_written_this_run=n_written,
            n_existing_skipped=n_existing_skipped,
            n_events_total=n_total_db,
            split_mode=prov["split_mode"],
            train_split=prov["train_frac"],
            split_seed=prov["split_seed"],
            target_normalization=target_norm,
            checkpoint_epoch=ckpt.get("epoch"),
            best_metric=ckpt.get("best_metric"),
            db_file=str(db_path),
            script="predict_full.py",
            limit=limit,
            parts_file=parts_file,
            parts_by_class=(
                {k: len(v) for k, v in parts_by_class.items()}
                if parts_by_class is not None else None
            ),
            elapsed_s=round(time.perf_counter() - t0, 1),
        )
    except Exception as exc:
        error_msg = str(exc)
        logger.exception("predict_full failed")
        raise
    finally:
        elapsed = (datetime.now() - t_start).total_seconds()
        logger.info("Total elapsed: %.0fs", elapsed)
        append_history_row(
            str(history_path),
            checkpoint=checkpoint_name,
            script="predict_full.py",
            source=source,
            db_file=str(db_path),
            threshold=threshold,
            min_hits=min_hits,
            min_strings=min_strings,
            n_events_new=n_new,
            n_events_skipped=n_skipped + n_existing_skipped,
            is_successful=(error_msg == ""),
            error_msg=error_msg,
            timestamp=t_start.isoformat(timespec="seconds"),
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to a track_energy_reconstruction checkpoint")
    ap.add_argument("--npy-dir", required=True,
                    help="Training dataset that supplied this checkpoint's rows")
    ap.add_argument("--output-dir",
                    default="inference/track_energy_reconstruction/preds")
    ap.add_argument("--subset", choices=("all", "test", "val"), default="all",
                    help="Write all full-source events, the test complement, or val rows")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N full-source candidates (smoke test)")
    ap.add_argument("--parts", default=None,
                    help="Comma-separated part keys to process (e.g. part_1006)")
    ap.add_argument("--parts-file", default=None,
                    help="YAML/JSON class -> part-number lists (nuatm, nue2); "
                         "filters each class separately")
    ap.add_argument("--check-existing", action="store_true",
                    help="Warn if rerun predictions differ from stored ones")
    args = ap.parse_args()
    predict_full(
        checkpoint=args.checkpoint,
        npy_dir=args.npy_dir,
        output_dir=args.output_dir,
        subset=args.subset,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        check_existing=args.check_existing,
        parts=[p.strip() for p in args.parts.split(",") if p.strip()]
        if args.parts else None,
        parts_file=args.parts_file,
    )


if __name__ == "__main__":
    main()
