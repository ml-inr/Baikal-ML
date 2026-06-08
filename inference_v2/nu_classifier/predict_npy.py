"""Predict nu-classifier scores for events from the pre-built NPY dataset.

Reads the memory-mapped NPY dataset (data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8/).
Features are already sig-noise-filtered — no on-the-fly sig-noise model needed.

Outputs to: preds/{checkpoint_name}/mc_merged_thr{threshold}.duckdb
Schema: predictions(event_fk BIGINT PK, score FLOAT, n_sn_hits INTEGER, n_sn_strings INTEGER)

Incremental: INSERT OR IGNORE — safe to re-run on overlapping subsets.

Usage (from project root):
    python inference_v2/nu_classifier/predict_npy.py \\
        --checkpoint experiments/numu/260508_1724_.../best.pt \\
        --npy-dir data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8 \\
        [--min-hits 8] [--min-strings 2] \\
        [--batch-size 512] [--device auto] \\
        [--output-dir inference_v2/nu_classifier/preds] \\
        [--catalog data_manager/catalog_v2.duckdb] \\
        [--check-existing]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inference_v2.shared.model_utils import load_model, predict_scores, predict_scores_and_embeddings
from inference_v2.shared.catalog_query import (
    get_event_fks_mc,
    open_predictions_db,
    NU_CLASSIFIER_SCHEMA,
    EMBEDDINGS_SCHEMA,
    append_predictions_nu_classifier,
    append_embeddings,
)
from inference_v2.shared.history import append_history_row

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _update_run_info(run_info_path: Path, checkpoint: str, source: str, threshold: float,
                     min_hits: int, min_strings: int, n_total: int) -> None:
    info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
    info.update({
        "checkpoint":     checkpoint,
        "source":         source,
        "threshold":      threshold,
        "min_hits":       min_hits,
        "min_strings":    min_strings,
        "n_events_total": n_total,
    })
    run_info_path.write_text(json.dumps(info, indent=2))


def predict_npy(
    checkpoint: str,
    npy_dir: str,
    min_hits: int = 8,
    min_strings: int = 2,
    batch_size: int = 512,
    device: str = "auto",
    output_dir: str = "inference_v2/nu_classifier/preds",
    catalog: str = "data_manager/catalog_v2.duckdb",
    check_existing: bool = False,
    save_embeddings: bool = False,
) -> None:
    t_start = datetime.now()
    dev = _resolve_device(device)
    checkpoint_name = f"{Path(checkpoint).parent.name}@{Path(checkpoint).stem}"
    npy_path = Path(npy_dir)

    thr_tag = "unknown"
    info_file = npy_path / "dataset_info.json"
    if info_file.exists():
        with open(info_file) as f:
            ds_info = json.load(f)
        thr_tag = str(ds_info.get("sig_noise_threshold", ds_info.get("threshold", "unknown"))).replace(".", "p")
    else:
        # infer from dir name
        for part in npy_path.name.split("_"):
            if part.startswith("thr"):
                thr_tag = part[3:]

    checkpoint_dir = Path(output_dir) / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path = checkpoint_dir / f"mc_merged_thr{thr_tag}.duckdb"
    log_path = checkpoint_dir / f"predict_npy_{t_start.strftime('%Y%m%d_%H%M%S')}.log"
    history_path = Path(output_dir) / "prediction_history.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )

    logger.info(f"Checkpoint:  {checkpoint}")
    logger.info(f"NPY dir:     {npy_dir}")
    logger.info(f"Output DB:   {db_path}")
    logger.info(f"Catalog:     {catalog}")
    logger.info(f"min_hits={min_hits}  min_strings={min_strings}  device={dev}  save_embeddings={save_embeddings}")

    error_msg = ""
    n_inserted = 0
    n_skipped  = 0

    try:
        # ── Load NPY arrays ───────────────────────────────────────────────
        t0 = time.perf_counter()
        with open(info_file) as f:
            ds_meta = json.load(f)
        particle_decode = {v: k for k, v in ds_meta["particle_encode"].items()}

        particle_types  = np.load(npy_path / "particle_types.npy")
        h5_part_keys    = np.load(npy_path / "h5_part_keys.npy", allow_pickle=True)
        h5_local_ids    = np.load(npy_path / "h5_local_event_ids.npy")
        n_sig_hits_arr  = np.load(npy_path / "n_sig_hits.npy")
        n_sig_str_arr   = np.load(npy_path / "n_sig_strings.npy")
        features_mmap   = np.load(npy_path / "features.npy", mmap_mode="r")
        offsets         = np.load(npy_path / "offsets.npy")
        n_total = len(particle_types)
        logger.info(f"NPY loaded: {n_total:,} events  ({time.perf_counter()-t0:.2f}s)")

        # ── Quality cut ───────────────────────────────────────────────────
        sel = np.where(
            (n_sig_hits_arr >= min_hits) & (n_sig_str_arr >= min_strings)
        )[0]
        logger.info(f"Quality cut: {len(sel):,} / {n_total:,} events pass")
        if len(sel) == 0:
            logger.warning("No events pass quality cut. Nothing to write.")
            return

        # ── Catalog lookup ────────────────────────────────────────────────
        logger.info("Querying catalog for event_fk values...")
        t0 = time.perf_counter()
        data_classes = np.array([particle_decode[int(particle_types[i])] for i in sel])
        seasons      = np.array([int(dc.rsplit("_", 1)[-1]) for dc in data_classes], dtype=np.int32)
        runs         = h5_part_keys[sel]
        event_ids    = h5_local_ids[sel].astype(np.int64)

        cat_result = get_event_fks_mc(
            catalog, "mc_merged", data_classes, seasons, runs, event_ids
        )
        t_cat = time.perf_counter() - t0
        n_found = len(cat_result)
        if n_found != len(sel):
            logger.warning(f"  {len(sel) - n_found} events not found in catalog — skipped")
        logger.info(f"  {n_found:,} events resolved  ({t_cat:.2f}s, {n_found/max(t_cat,1e-3):.0f}/s)")

        if n_found == 0:
            logger.warning("No events found in catalog. Nothing to write.")
            return

        # Align features to catalog-found events
        found_query_idx = cat_result["query_idx"].values
        event_fks       = cat_result["event_fk"].values.astype(np.int64)
        found_sel       = sel[found_query_idx]

        # ── Load model ────────────────────────────────────────────────────
        with_probs = ds_meta.get("with_probs", False)
        model, norm_config, train_config = load_model(checkpoint, device=dev)
        model.eval()
        logger.info(f"Model loaded  with_probs={with_probs}")

        # ── Build features list ───────────────────────────────────────────
        features_list = []
        if with_probs:
            probs_mmap = np.load(npy_path / "probs.npy", mmap_mode="r")
            for i in found_sel:
                s, e = int(offsets[i]), int(offsets[i + 1])
                feats = np.array(features_mmap[s:e])
                p     = np.array(probs_mmap[s:e])
                features_list.append(np.column_stack([feats, p]))
        else:
            for i in found_sel:
                s, e = int(offsets[i]), int(offsets[i + 1])
                features_list.append(np.array(features_mmap[s:e]))

        # ── Inference ─────────────────────────────────────────────────────
        logger.info(f"Running inference on {len(features_list):,} events...")
        t0 = time.perf_counter()
        if save_embeddings:
            scores, emb_arr = predict_scores_and_embeddings(
                model, features_list, norm_config,
                batch_size=batch_size, device=dev,
                feats_with_probs=with_probs,
            )
        else:
            scores = predict_scores(
                model, features_list, norm_config,
                batch_size=batch_size, device=dev,
                feats_with_probs=with_probs, with_tqdm=True,
            )
        logger.info(f"  Inference done  ({time.perf_counter()-t0:.1f}s)")

        # ── Write to DuckDB ───────────────────────────────────────────────
        conn = open_predictions_db(str(db_path), NU_CLASSIFIER_SCHEMA)
        if save_embeddings:
            conn.execute(EMBEDDINGS_SCHEMA)
        n_inserted, n_skipped = append_predictions_nu_classifier(
            conn,
            event_fks=event_fks,
            scores=scores,
            n_sn_hits=n_sig_hits_arr[found_sel],
            n_sn_strings=n_sig_str_arr[found_sel],
            check_existing=check_existing,
        )
        if save_embeddings:
            append_embeddings(conn, event_fks, emb_arr)
        total_in_db = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()

        logger.info(
            f"Written: {n_inserted:,} new  |  skipped: {n_skipped:,} existing  "
            f"|  total in DB: {total_in_db:,}"
        )

        _update_run_info(
            checkpoint_dir / "run_info.json",
            checkpoint=checkpoint,
            source="mc_merged",
            threshold=ds_meta.get("sig_noise_threshold", ds_meta.get("threshold", float("nan"))),
            min_hits=min_hits,
            min_strings=min_strings,
            n_total=total_in_db,
        )

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("predict_npy failed")
        raise

    finally:
        elapsed = (datetime.now() - t_start).total_seconds()
        logger.info(f"Total elapsed: {elapsed:.0f}s")
        append_history_row(
            str(history_path),
            checkpoint=checkpoint_name,
            script="predict_npy.py",
            source="mc_merged",
            db_file=str(db_path),
            threshold=float(ds_meta.get("sig_noise_threshold", ds_meta.get("threshold", float("nan")))) if "ds_meta" in dir() else float("nan"),
            min_hits=min_hits,
            min_strings=min_strings,
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
    parser.add_argument("--checkpoint",   required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--npy-dir",      required=True,  help="Path to NPY dataset dir")
    parser.add_argument("--min-hits",     type=int,   default=8)
    parser.add_argument("--min-strings",  type=int,   default=2)
    parser.add_argument("--batch-size",   type=int,   default=512)
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--output-dir",   default="inference_v2/nu_classifier/preds")
    parser.add_argument("--catalog",      default="data_manager/catalog_v2.duckdb")
    parser.add_argument("--check-existing",  action="store_true",
                        help="Warn if existing predictions differ from new ones")
    parser.add_argument("--save-embeddings", action="store_true", default=False,
                        help="Also store encoder-level 128-dim feature vectors in an embeddings table")
    args = parser.parse_args()

    predict_npy(
        checkpoint=args.checkpoint,
        npy_dir=args.npy_dir,
        min_hits=args.min_hits,
        min_strings=args.min_strings,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        catalog=args.catalog,
        check_existing=args.check_existing,
        save_embeddings=args.save_embeddings,
    )


if __name__ == "__main__":
    main()
