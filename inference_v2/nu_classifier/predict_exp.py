"""Predict nu-classifier scores for experimental data from HDF5.

Runs the sig-noise model on-the-fly. Auto-detects source type (exp vs exp_reco)
from the h5 top-level key.

Output: preds/{checkpoint_name}/exp_reco_thr{thr}.duckdb  (or exp_thr{thr}.duckdb)
Schema: predictions(event_fk BIGINT PK, score FLOAT, n_sn_hits INTEGER, n_sn_strings INTEGER)

For exp_reco: event_fk looked up by (season, cluster, run, event_id_in_run) from header_prty.
For exp (no header_prty): event_fk looked up by (run=part_key, event_id=local_idx) — catalog
must have been built with source='exp'.

Incremental: safe to re-run on different part subsets — INSERT OR IGNORE.

Usage (from project root):
    python inference_v2/nu_classifier/predict_exp.py \\
        --checkpoint experiments/numu/260508_1724_.../best.pt \\
        --exp-h5 data_manager/data/h5datasets/exp_reco.h5 \\
        [--threshold 0.8] [--min-hits 8] [--min-strings 2] \\
        [--batch-size 512] [--device auto] \\
        [--output-dir inference_v2/nu_classifier/preds] \\
        [--catalog data_manager/catalog_v2.duckdb]
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

_PART_KEY_RE = re.compile(r"part_s(\d+)_c(\d+)_r(\d+)")

def _parse_exp_part_key(pk: str) -> tuple[int, int, int]:
    """Parse 'part_s2020_c01_r0027' → (season=2020, cluster=1, run=27)."""
    m = _PART_KEY_RE.match(pk)
    if not m:
        raise ValueError(f"Cannot parse exp part key: {pk!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inference_v2.shared.model_utils import load_model, load_sn_model, predict_scores, predict_scores_and_embeddings
from inference_v2.shared.catalog_query import (
    get_event_fks_exp,
    open_predictions_db,
    NU_CLASSIFIER_SCHEMA,
    EMBEDDINGS_SCHEMA,
    append_predictions_nu_classifier,
    append_embeddings,
)
from inference_v2.shared.history import append_history_row
from data_manager.nu_classifier_ds_builder.io import _count_sig_hits_strings

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    import torch
    return "cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu"


def _update_run_info(path: Path, **kwargs) -> None:
    info = json.loads(path.read_text()) if path.exists() else {}
    info.update(kwargs)
    path.write_text(json.dumps(info, indent=2))


def _get_done_parts(conn, catalog_path: str, source: str) -> set:
    """Return set of h5 part keys already present in the predictions DB for this source."""
    try:
        conn.execute(f"ATTACH '{catalog_path}' AS _cat (READ_ONLY)")
        rows = conn.execute("""
            SELECT DISTINCT l.part_key
            FROM predictions p
            JOIN _cat.events e ON p.event_fk = e.id
            JOIN _cat.h5_locations l ON l.event_fk = e.id
            WHERE e.source = ?
        """, [source]).fetchall()
        conn.execute("DETACH _cat")
        return {r[0] for r in rows}
    except Exception as exc:
        logger.warning(f"Could not query done parts: {exc}")
        try:
            conn.execute("DETACH _cat")
        except Exception:
            pass
        return set()


def predict_exp(
    checkpoint: str,
    exp_h5: str,
    threshold: float = 0.8,
    min_hits: int = 8,
    min_strings: int = 2,
    batch_size: int = 512,
    device: str = "auto",
    output_dir: str = "inference_v2/nu_classifier/preds",
    catalog: str = "data_manager/catalog_v2.duckdb",
    check_existing: bool = False,
    skip_done_parts: bool = False,
    save_embeddings: bool = False,
    checkpoint_name: Optional[str] = None,
) -> None:
    t_start = datetime.now()
    dev = _resolve_device(device)
    checkpoint_name = checkpoint_name or f"{Path(checkpoint).parent.name}@{Path(checkpoint).stem}"
    thr_tag = str(threshold).replace(".", "p")

    checkpoint_dir = Path(output_dir) / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # detect source key before we know db name
    with h5py.File(exp_h5, "r") as h5:
        src_key = "exp_reco" if "exp_reco" in h5 else "exp"

    db_path   = checkpoint_dir / f"{src_key}_thr{thr_tag}.duckdb"
    log_path  = checkpoint_dir / f"predict_exp_{t_start.strftime('%Y%m%d_%H%M%S')}.log"
    hist_path = Path(output_dir) / "prediction_history.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )

    logger.info(f"Checkpoint:       {checkpoint}")
    logger.info(f"Exp h5:           {exp_h5}  (src_key={src_key})")
    logger.info(f"Output DB:        {db_path}")
    logger.info(f"threshold={threshold}  min_hits={min_hits}  min_strings={min_strings}")
    logger.info(f"skip_done_parts:  {skip_done_parts}")
    logger.info(f"save_embeddings:  {save_embeddings}")

    error_msg  = ""
    n_inserted = 0
    n_skipped  = 0

    try:
        model, norm_config, train_config = load_model(checkpoint, device=dev)
        with_probs = train_config.get("model", {}).get("input_dim", 5) == 6
        sn_model, _, sn_dev = load_sn_model(device=device)

        from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
            predict_flat,
        )

        conn = open_predictions_db(str(db_path), NU_CLASSIFIER_SCHEMA)
        if save_embeddings:
            conn.execute(EMBEDDINGS_SCHEMA)
        has_header = (src_key == "exp_reco")

        done_part_keys: set = set()
        if skip_done_parts:
            logger.info("Querying done parts from predictions DB...")
            t0 = time.perf_counter()
            done_part_keys = _get_done_parts(conn, catalog, src_key)
            logger.info(f"  {len(done_part_keys)} already-done parts  ({time.perf_counter()-t0:.1f}s)")

        with h5py.File(exp_h5, "r") as h5:
            grp       = h5[src_key]
            all_parts = sorted(grp["raw"]["data"].keys())
            sel_parts = [p for p in all_parts if p not in done_part_keys]

            if done_part_keys:
                logger.info(f"{src_key}: {len(sel_parts)} parts to process  "
                            f"(excluded {len(all_parts) - len(sel_parts)} done)")
            else:
                logger.info(f"{src_key}: {len(sel_parts)} parts")

            part_ins = 0
            part_skp = 0
            loop_t0  = time.perf_counter()
            log_every = 10

            for part_idx, pk in enumerate(sel_parts):
                ev_starts = grp[f"raw/ev_starts/{pk}/data"][:].astype(np.int64)
                n_events  = len(ev_starts) - 1
                if n_events == 0:
                    continue

                data_raw = grp[f"raw/data/{pk}/data"][:].astype(np.float32)
                channels = grp[f"raw/channels/{pk}/data"][:].astype(np.int32)
                header   = grp[f"header_prty/{pk}/data"][:] if has_header else None

                probs    = predict_flat(sn_model, data_raw, ev_starts,
                                        batch_size=batch_size, device=sn_dev, normalize=True)
                sig_mask = probs > threshold

                n_hits_arr = (ev_starts[1:] - ev_starts[:-1]).astype(np.int32)
                n_sn_h, n_sn_s = _count_sig_hits_strings(
                    sig_mask, channels, ev_starts[:-1], n_hits_arr, n_events,
                )

                cut_mask = (n_sn_h >= min_hits) & (n_sn_s >= min_strings)
                sel_ev   = np.where(cut_mask)[0]

                if len(sel_ev) > 0:
                    features_list = []
                    for ev_i in sel_ev:
                        s, e  = int(ev_starts[ev_i]), int(ev_starts[ev_i + 1])
                        ev_sm = sig_mask[s:e]
                        feats = data_raw[s:e][ev_sm]
                        if with_probs:
                            feats = np.column_stack([feats, probs[s:e][ev_sm]])
                        features_list.append(feats)

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
                            feats_with_probs=with_probs, with_tqdm=False,
                        )

                    n_sel = len(sel_ev)
                    if has_header and header is not None:
                        seasons  = header[sel_ev, 0].astype(np.int32)
                        clusters = header[sel_ev, 1].astype(np.int32)
                        runs     = np.array([str(int(header[i, 2])) for i in sel_ev])
                        ev_ids   = header[sel_ev, 3].astype(np.int64)
                        cat_res  = get_event_fks_exp(
                            catalog, src_key, seasons, clusters, runs, ev_ids
                        )
                    else:
                        pk_season, pk_cluster, pk_run = _parse_exp_part_key(pk)
                        cat_res = get_event_fks_exp(
                            catalog, src_key,
                            seasons=np.full(n_sel, pk_season,  dtype=np.int32),
                            clusters=np.full(n_sel, pk_cluster, dtype=np.int32),
                            runs=np.array([str(pk_run)] * n_sel),
                            event_ids=sel_ev.astype(np.int64),
                        )

                    if len(cat_res) == 0:
                        logger.warning(f"  {pk}: no events found in catalog, skipping")
                    else:
                        found_qi  = cat_res["query_idx"].values
                        event_fks = cat_res["event_fk"].values.astype(np.int64)
                        ins, skp  = append_predictions_nu_classifier(
                            conn,
                            event_fks=event_fks,
                            scores=scores[found_qi],
                            n_sn_hits=n_sn_h[sel_ev[found_qi]],
                            n_sn_strings=n_sn_s[sel_ev[found_qi]],
                            check_existing=check_existing,
                        )
                        n_inserted += ins
                        n_skipped  += skp
                        part_ins   += ins
                        part_skp   += skp

                        if save_embeddings:
                            append_embeddings(conn, event_fks, emb_arr[found_qi])

                parts_done = part_idx + 1
                if parts_done % log_every == 0 or parts_done == len(sel_parts):
                    elapsed  = time.perf_counter() - loop_t0
                    rate     = parts_done / elapsed if elapsed > 0 else 0
                    eta_s    = (len(sel_parts) - parts_done) / rate if rate > 0 else float("inf")
                    eta_str  = (f"{eta_s/3600:.1f}h" if eta_s > 3600
                                else f"{eta_s/60:.0f}m" if eta_s > 60
                                else f"{eta_s:.0f}s")
                    logger.info(
                        f"  {src_key}  [{parts_done:>4}/{len(sel_parts)}]  "
                        f"ins={part_ins:,}  skip={part_skp:,}  "
                        f"rate={rate:.2f} parts/s  eta ~{eta_str}"
                    )

        total_in_db = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()

        logger.info(f"\nDone: {n_inserted:,} inserted  {n_skipped:,} skipped  "
                    f"total in DB: {total_in_db:,}")

        _update_run_info(
            checkpoint_dir / "run_info.json",
            checkpoint=checkpoint,
            source=src_key,
            h5_path=exp_h5,
            threshold=threshold,
            min_hits=min_hits,
            min_strings=min_strings,
            n_events_total=total_in_db,
        )

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("predict_exp failed")
        raise

    finally:
        elapsed = (datetime.now() - t_start).total_seconds()
        logger.info(f"Total elapsed: {elapsed:.0f}s")
        append_history_row(
            str(hist_path),
            checkpoint=checkpoint_name,
            script="predict_exp.py",
            source=src_key if "src_key" in dir() else "unknown",
            db_file=str(db_path) if "db_path" in dir() else "",
            threshold=threshold,
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
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--exp-h5",         required=True)
    parser.add_argument("--threshold",      type=float, default=0.8)
    parser.add_argument("--min-hits",       type=int,   default=8)
    parser.add_argument("--min-strings",    type=int,   default=2)
    parser.add_argument("--batch-size",     type=int,   default=512)
    parser.add_argument("--device",         default="auto")
    parser.add_argument("--output-dir",       default="inference_v2/nu_classifier/preds")
    parser.add_argument("--checkpoint-name", default=None,
                        help="Override the auto-derived output subdir name "
                             "(default: {parent_dir}@{stem})")
    parser.add_argument("--catalog",        default="data_manager/catalog_v2.duckdb")
    parser.add_argument("--check-existing",  action="store_true")
    parser.add_argument("--skip-done-parts", action="store_true", default=False,
                        help="Skip parts already present in the predictions DB (safe restart)")
    parser.add_argument("--save-embeddings", action="store_true", default=False,
                        help="Also store encoder-level 128-dim feature vectors in an embeddings table")
    args = parser.parse_args()

    predict_exp(
        checkpoint=args.checkpoint,
        exp_h5=args.exp_h5,
        threshold=args.threshold,
        min_hits=args.min_hits,
        min_strings=args.min_strings,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        catalog=args.catalog,
        check_existing=args.check_existing,
        skip_done_parts=args.skip_done_parts,
        save_embeddings=args.save_embeddings,
        checkpoint_name=args.checkpoint_name,
    )


if __name__ == "__main__":
    main()
