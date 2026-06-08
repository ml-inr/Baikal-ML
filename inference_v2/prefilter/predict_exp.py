"""Predict prefilter scores for experimental data from HDF5.

No sig-noise filtering — all raw hits are fed to the prefilter model.
Auto-detects source type (exp vs exp_reco) from h5 top-level key.

Output: preds/{checkpoint_name}/exp_reco_allhits.duckdb  (or exp_allhits.duckdb)
Schema: predictions(event_fk BIGINT PK, score FLOAT, n_hits INTEGER)

For exp_reco: event_fk looked up by (season, cluster, run, event_id_in_run) from header_prty.
For exp (no header_prty): event_fk looked up by parsed part key (season, cluster, run int).

Incremental: safe to re-run on different part subsets — INSERT OR IGNORE.

Usage (from project root):
    python inference_v2/prefilter/predict_exp.py \\
        --checkpoint experiments/numu/da_prefilter_numu_260429_.../best_da_model.pth \\
        --exp-h5 data_manager/data/h5datasets/exp_reco.h5 \\
        [--batch-size 512] [--device auto] \\
        [--output-dir inference_v2/prefilter/preds] \\
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

from inference_v2.shared.model_utils import load_model, predict_scores, predict_scores_and_embeddings
from inference_v2.shared.catalog_query import (
    get_event_fks_exp,
    open_predictions_db,
    PREFILTER_SCHEMA,
    EMBEDDINGS_SCHEMA,
    append_predictions_prefilter,
    append_embeddings,
)
from inference_v2.shared.history import append_history_row

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
    batch_size: int = 512,
    device: str = "auto",
    output_dir: str = "inference_v2/prefilter/preds",
    catalog: str = "data_manager/catalog_v2.duckdb",
    check_existing: bool = False,
    skip_done_parts: bool = False,
    save_embeddings: bool = False,
) -> None:
    t_start = datetime.now()
    dev = _resolve_device(device)
    checkpoint_name = f"{Path(checkpoint).parent.name}@{Path(checkpoint).stem}"

    with h5py.File(exp_h5, "r") as h5:
        src_key = "exp_reco" if "exp_reco" in h5 else "exp"

    checkpoint_dir = Path(output_dir) / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path   = checkpoint_dir / f"{src_key}_allhits.duckdb"
    log_path  = checkpoint_dir / f"predict_exp_{t_start.strftime('%Y%m%d_%H%M%S')}.log"
    hist_path = Path(output_dir) / "prediction_history.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )

    logger.info(f"Checkpoint:      {checkpoint}")
    logger.info(f"Exp h5:          {exp_h5}  (src_key={src_key})")
    logger.info(f"Output DB:       {db_path}")
    logger.info(f"skip_done_parts: {skip_done_parts}")
    logger.info(f"save_embeddings: {save_embeddings}")

    error_msg  = ""
    n_inserted = 0
    n_skipped  = 0

    try:
        model, norm_config, _ = load_model(checkpoint, device=dev)
        conn = open_predictions_db(str(db_path), PREFILTER_SCHEMA)
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

            part_ins  = 0
            part_skp  = 0
            emb_ins   = 0
            loop_t0   = time.perf_counter()
            log_every = 10

            for part_idx, pk in enumerate(sel_parts):
                ev_starts = grp[f"raw/ev_starts/{pk}/data"][:].astype(np.int64)
                n_events  = len(ev_starts) - 1
                if n_events == 0:
                    continue

                data_raw = grp[f"raw/data/{pk}/data"][:].astype(np.float32)
                header   = grp[f"header_prty/{pk}/data"][:] if has_header else None

                n_hits_arr    = (ev_starts[1:] - ev_starts[:-1]).astype(np.int32)
                features_list = [
                    data_raw[int(ev_starts[i]):int(ev_starts[i + 1])]
                    for i in range(n_events)
                ]

                if save_embeddings:
                    scores, emb_arr = predict_scores_and_embeddings(
                        model, features_list, norm_config,
                        batch_size=batch_size, device=dev,
                    )
                else:
                    scores = predict_scores(
                        model, features_list, norm_config,
                        batch_size=batch_size, device=dev, with_tqdm=False,
                    )

                all_ev = np.arange(n_events, dtype=np.int64)
                if has_header and header is not None:
                    seasons  = header[:, 0].astype(np.int32)
                    clusters = header[:, 1].astype(np.int32)
                    runs     = np.array([str(int(header[i, 2])) for i in range(n_events)])
                    ev_ids   = header[:, 3].astype(np.int64)
                    cat_res  = get_event_fks_exp(catalog, src_key, seasons, clusters, runs, ev_ids)
                else:
                    pk_season, pk_cluster, pk_run = _parse_exp_part_key(pk)
                    cat_res = get_event_fks_exp(
                        catalog, src_key,
                        seasons=np.full(n_events, pk_season,  dtype=np.int32),
                        clusters=np.full(n_events, pk_cluster, dtype=np.int32),
                        runs=np.array([str(pk_run)] * n_events),
                        event_ids=all_ev,
                    )

                if len(cat_res) == 0:
                    logger.warning(f"  {pk}: no events found in catalog, skipping")
                else:
                    found_qi  = cat_res["query_idx"].values
                    event_fks = cat_res["event_fk"].values.astype(np.int64)
                    ins, skp  = append_predictions_prefilter(
                        conn,
                        event_fks=event_fks,
                        scores=scores[found_qi],
                        n_hits=n_hits_arr[found_qi],
                        check_existing=check_existing,
                    )
                    n_inserted += ins
                    n_skipped  += skp
                    part_ins   += ins
                    part_skp   += skp
                    if save_embeddings:
                        ei, _ = append_embeddings(conn, event_fks, emb_arr[found_qi])
                        emb_ins += ei

                parts_done = part_idx + 1
                if parts_done % log_every == 0 or parts_done == len(sel_parts):
                    elapsed  = time.perf_counter() - loop_t0
                    rate     = parts_done / elapsed if elapsed > 0 else 0
                    eta_s    = (len(sel_parts) - parts_done) / rate if rate > 0 else float("inf")
                    eta_str  = (f"{eta_s/3600:.1f}h" if eta_s > 3600
                                else f"{eta_s/60:.0f}m" if eta_s > 60
                                else f"{eta_s:.0f}s")
                    emb_str  = f"  emb_ins={emb_ins:,}" if save_embeddings else ""
                    logger.info(
                        f"  {src_key}  [{parts_done:>4}/{len(sel_parts)}]  "
                        f"ins={part_ins:,}  skip={part_skp:,}{emb_str}  "
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
            threshold=None,
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
            threshold=float("nan"),
            min_hits=0,
            min_strings=0,
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
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--exp-h5",          required=True)
    parser.add_argument("--batch-size",      type=int, default=512)
    parser.add_argument("--device",          default="auto")
    parser.add_argument("--output-dir",      default="inference_v2/prefilter/preds")
    parser.add_argument("--catalog",         default="data_manager/catalog_v2.duckdb")
    parser.add_argument("--check-existing",  action="store_true")
    parser.add_argument("--skip-done-parts", action="store_true", default=False,
                        help="Skip parts already present in the predictions DB (safe restart)")
    parser.add_argument("--save-embeddings", action="store_true", default=False,
                        help="Also store encoder-level feature vectors in an embeddings table")
    args = parser.parse_args()

    predict_exp(
        checkpoint=args.checkpoint,
        exp_h5=args.exp_h5,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        catalog=args.catalog,
        check_existing=args.check_existing,
        skip_done_parts=args.skip_done_parts,
        save_embeddings=args.save_embeddings,
    )


if __name__ == "__main__":
    main()
