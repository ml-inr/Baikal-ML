"""Predict prefilter scores for MC events directly from HDF5.

No sig-noise filtering — all raw hits are fed to the prefilter model.
Supports mc_merged and mc_reco sources.

Output: preds/{checkpoint_name}/mc_merged_allhits.duckdb  (or mc_reco_allhits.duckdb)
Schema: predictions(event_fk BIGINT PK, score FLOAT, n_hits INTEGER)

Incremental: safe to re-run on different part subsets — INSERT OR IGNORE.

Usage (from project root):
    python inference_v2/prefilter/predict_mc.py \\
        --checkpoint experiments/numu/da_prefilter_numu_260429_.../best_da_model.pth \\
        --mc-h5 /net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5 \\
        [--source mc_merged] \\
        [--ptypes muatm_2020,nuatm_2020,nue2_2020] \\
        [--parts part_0,part_1,...] \\
        [--batch-size 512] [--device auto] \\
        [--output-dir inference_v2/prefilter/preds] \\
        [--catalog data_manager/catalog_v2.duckdb]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inference_v2.shared.model_utils import load_model, predict_scores, predict_scores_and_embeddings
from inference_v2.shared.catalog_query import (
    get_event_fks_mc,
    open_predictions_db,
    PREFILTER_SCHEMA,
    EMBEDDINGS_SCHEMA,
    append_predictions_prefilter,
    append_embeddings,
    MC_RECO_PTYPE_TO_DATA_CLASS,
)
from inference_v2.shared.history import append_history_row

logger = logging.getLogger(__name__)

MC_MERGED_PTYPES = ["muatm_2020", "nuatm_2020", "nue2_2020"]


def _resolve_device(device: str) -> str:
    import torch
    return "cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu"


def _ptype_to_data_class(ptype: str, source: str) -> str:
    if source == "mc_reco":
        return MC_RECO_PTYPE_TO_DATA_CLASS.get(ptype, ptype)
    return ptype


def _update_run_info(path: Path, **kwargs) -> None:
    info = json.loads(path.read_text()) if path.exists() else {}
    info.update(kwargs)
    path.write_text(json.dumps(info, indent=2))


def _get_done_parts(conn, catalog_path: str, source: str) -> dict:
    """Return {data_class: set_of_run_strings} for parts already present in the predictions DB."""
    try:
        conn.execute(f"ATTACH '{catalog_path}' AS _cat (READ_ONLY)")
        rows = conn.execute("""
            SELECT DISTINCT e.data_class, e.run
            FROM predictions p
            JOIN _cat.events e ON p.event_fk = e.id
            WHERE e.source = ?
        """, [source]).fetchall()
        conn.execute("DETACH _cat")
        result: dict = {}
        for data_class, run in rows:
            result.setdefault(data_class, set()).add(run)
        return result
    except Exception as exc:
        logger.warning(f"Could not query done parts: {exc}")
        try:
            conn.execute("DETACH _cat")
        except Exception:
            pass
        return {}


def predict_mc(
    checkpoint: str,
    mc_h5: str,
    source: str = "mc_merged",
    ptypes: Optional[List[str]] = None,
    parts: Optional[List[str]] = None,
    max_events_per_ptype: Optional[int] = None,
    skip_done_parts: bool = False,
    batch_size: int = 512,
    min_batch_events: int = 2048,
    device: str = "auto",
    output_dir: str = "inference_v2/prefilter/preds",
    catalog: str = "data_manager/catalog_v2.duckdb",
    check_existing: bool = False,
    save_embeddings: bool = False,
) -> None:
    t_start = datetime.now()
    dev = _resolve_device(device)
    checkpoint_name = f"{Path(checkpoint).parent.name}@{Path(checkpoint).stem}"

    checkpoint_dir = Path(output_dir) / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path   = checkpoint_dir / f"{source}_allhits.duckdb"
    log_path  = checkpoint_dir / f"predict_mc_{t_start.strftime('%Y%m%d_%H%M%S')}.log"
    hist_path = Path(output_dir) / "prediction_history.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )

    logger.info(f"Checkpoint:            {checkpoint}")
    logger.info(f"MC h5:                 {mc_h5}")
    logger.info(f"Source:                {source}")
    logger.info(f"Output DB:             {db_path}")
    logger.info(f"max_events_per_ptype:  {max_events_per_ptype if max_events_per_ptype is not None else 'unlimited'}")
    logger.info(f"skip_done_parts:       {skip_done_parts}")
    logger.info(f"save_embeddings:       {save_embeddings}")

    error_msg  = ""
    n_inserted = 0
    n_skipped  = 0

    try:
        model, norm_config, _ = load_model(checkpoint, device=dev)
        conn = open_predictions_db(str(db_path), PREFILTER_SCHEMA)
        if save_embeddings:
            conn.execute(EMBEDDINGS_SCHEMA)

        done_parts: dict = {}
        if skip_done_parts:
            logger.info("Querying done parts from predictions DB...")
            t0 = time.perf_counter()
            done_parts = _get_done_parts(conn, catalog, source)
            total_done = sum(len(v) for v in done_parts.values())
            logger.info(f"  {total_done} already-done parts found  ({time.perf_counter()-t0:.1f}s)")

        with h5py.File(mc_h5, "r") as h5:
            if ptypes is not None:
                selected_ptypes = [p for p in ptypes if p in h5]
            elif source == "mc_merged":
                selected_ptypes = [p for p in MC_MERGED_PTYPES if p in h5]
            else:
                selected_ptypes = list(h5.keys())

            for ptype in selected_ptypes:
                grp        = h5[ptype]
                data_class = _ptype_to_data_class(ptype, source)
                season     = int(data_class.rsplit("_", 1)[-1]) if "_" in data_class else 0

                all_parts = sorted(grp["raw"]["data"].keys())
                sel_parts = [p for p in all_parts if parts is None or p in parts]
                done_excl = done_parts.get(data_class, set())
                n_before_done = len(sel_parts)
                if done_excl:
                    sel_parts = [p for p in sel_parts if p not in done_excl]

                suffix = (f"  (excluded: {n_before_done - len(sel_parts)} done)"
                          if n_before_done - len(sel_parts) else "")
                logger.info(f"\n{ptype}: {len(sel_parts)} parts to process{suffix}")

                n_remaining       = max_events_per_ptype
                ptype_ins         = 0
                ptype_skp         = 0
                ptype_emb_ins     = 0
                ptype_events_read = 0
                ptype_t0          = time.perf_counter()
                log_every         = 10

                # Part buffer: accumulate small parts before running GPU inference
                # Each entry: (pk, data_raw, ev_starts, n_events)
                buf_parts:  list = []
                buf_events: int  = 0

                def _flush_buffer() -> None:
                    nonlocal n_inserted, n_skipped, ptype_ins, ptype_skp, ptype_emb_ins
                    nonlocal buf_parts, buf_events
                    if not buf_parts:
                        return

                    # Build global hit array and event→part mapping
                    all_data = np.concatenate([b[1] for b in buf_parts])
                    global_ev_starts = [0]
                    ev_part_of:  list = []
                    ev_local_of: list = []
                    for bi, (_, _, ev_starts, n_ev) in enumerate(buf_parts):
                        for j in range(n_ev):
                            global_ev_starts.append(
                                global_ev_starts[-1] + int(ev_starts[j + 1] - ev_starts[j])
                            )
                            ev_part_of.append(bi)
                            ev_local_of.append(j)
                    global_ev_starts = np.array(global_ev_starts, dtype=np.int64)
                    n_buf = len(ev_part_of)

                    features_list = [
                        all_data[int(global_ev_starts[i]):int(global_ev_starts[i + 1])]
                        for i in range(n_buf)
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

                    n_hits_arr = (global_ev_starts[1:] - global_ev_starts[:-1]).astype(np.int32)
                    sel_runs   = np.array([buf_parts[ev_part_of[i]][0] for i in range(n_buf)])
                    sel_ids    = np.array([ev_local_of[i] for i in range(n_buf)], dtype=np.int64)

                    cat_result = get_event_fks_mc(
                        catalog, source,
                        data_classes=np.array([data_class] * n_buf),
                        seasons=np.full(n_buf, season, dtype=np.int32),
                        runs=sel_runs,
                        event_ids=sel_ids,
                    )

                    if len(cat_result) == 0:
                        pks_str = ", ".join(b[0] for b in buf_parts)
                        logger.warning(f"  buffer [{pks_str}]: no events found in catalog")
                    else:
                        found_qi  = cat_result["query_idx"].values
                        event_fks = cat_result["event_fk"].values.astype(np.int64)
                        ins, skp  = append_predictions_prefilter(
                            conn,
                            event_fks=event_fks,
                            scores=scores[found_qi],
                            n_hits=n_hits_arr[found_qi],
                            check_existing=check_existing,
                        )
                        n_inserted += ins; n_skipped += skp
                        ptype_ins  += ins; ptype_skp += skp
                        if save_embeddings:
                            emb_ins, _ = append_embeddings(conn, event_fks, emb_arr[found_qi])
                            ptype_emb_ins += emb_ins

                    buf_parts.clear()
                    buf_events = 0

                for part_idx, pk in enumerate(sel_parts):
                    if n_remaining is not None and n_remaining <= 0:
                        break

                    ev_starts_full = grp[f"raw/ev_starts/{pk}/data"][:].astype(np.int64)
                    n_events_full  = len(ev_starts_full) - 1
                    if n_events_full == 0:
                        continue

                    if n_remaining is not None and n_events_full > n_remaining:
                        ev_starts = ev_starts_full[:n_remaining + 1]
                        n_events  = n_remaining
                    else:
                        ev_starts = ev_starts_full
                        n_events  = n_events_full

                    if n_remaining is not None:
                        n_remaining -= n_events
                    ptype_events_read += n_events

                    hit_end  = int(ev_starts[-1])
                    data_raw = grp[f"raw/data/{pk}/data"][:hit_end].astype(np.float32)

                    buf_parts.append((pk, data_raw, ev_starts, n_events))
                    buf_events += n_events

                    parts_done = part_idx + 1
                    is_last    = (parts_done == len(sel_parts)
                                  or (n_remaining is not None and n_remaining == 0))
                    if buf_events >= min_batch_events or is_last:
                        _flush_buffer()

                    if parts_done % log_every == 0 or is_last:
                        elapsed = time.perf_counter() - ptype_t0
                        if max_events_per_ptype is not None:
                            ev_consumed = max_events_per_ptype - (n_remaining or 0)
                            ev_rate     = ptype_events_read / elapsed if elapsed > 0 else 0
                            eta_s       = (n_remaining / ev_rate
                                          if ev_rate > 0 and n_remaining and n_remaining > 0
                                          else 0.0)
                            progress = (f"events {ev_consumed:,}/{max_events_per_ptype:,}"
                                        f"  [{parts_done} parts]")
                            rate_str = f"rate={ev_rate/1000:.1f}k ev/s"
                        else:
                            part_rate = parts_done / elapsed if elapsed > 0 else 0
                            eta_s     = ((len(sel_parts) - parts_done) / part_rate
                                         if part_rate > 0 else float("inf"))
                            progress  = f"[{parts_done:>6}/{len(sel_parts)}]"
                            rate_str  = f"rate={part_rate:.2f} parts/s"
                        eta_str     = (f"{eta_s/3600:.1f}h" if eta_s > 3600
                                       else f"{eta_s/60:.0f}m" if eta_s > 60
                                       else f"{eta_s:.0f}s")
                        emb_str     = f"  emb_ins={ptype_emb_ins:,}" if save_embeddings else ""
                        all_skipped = (ptype_ins == 0 and ptype_skp > 0
                                       and not (save_embeddings and ptype_emb_ins > 0))
                        note        = "  ⚠ all skipped — already in DB from prior run" if all_skipped else ""
                        logger.info(
                            f"  {ptype}  {progress}  "
                            f"ins={ptype_ins:,}  skip={ptype_skp:,}{emb_str}  "
                            f"{rate_str}  eta ~{eta_str}{note}"
                        )

        total_in_db = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()

        logger.info(f"\nDone: {n_inserted:,} inserted  {n_skipped:,} skipped  "
                    f"total in DB: {total_in_db:,}")

        _update_run_info(
            checkpoint_dir / "run_info.json",
            checkpoint=checkpoint,
            source=source,
            h5_path=mc_h5,
            threshold=None,
            n_events_total=total_in_db,
        )

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("predict_mc failed")
        raise

    finally:
        elapsed = (datetime.now() - t_start).total_seconds()
        logger.info(f"Total elapsed: {elapsed:.0f}s")
        append_history_row(
            str(hist_path),
            checkpoint=checkpoint_name,
            script="predict_mc.py",
            source=source,
            db_file=str(db_path),
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
    parser.add_argument("--checkpoint",           required=True)
    parser.add_argument("--mc-h5",                required=True)
    parser.add_argument("--source",               default="mc_merged",
                        choices=["mc_merged", "mc_reco"])
    parser.add_argument("--ptypes",               default=None,
                        help="Comma-separated particle types (default: all)")
    parser.add_argument("--parts",                default=None,
                        help="Comma-separated part keys to process (default: all)")
    parser.add_argument("--max-events-per-ptype", type=int, default=None,
                        help="Cap on events per particle type (first N in part order)")
    parser.add_argument("--skip-done-parts",      action="store_true", default=False,
                        help="Skip parts already present in the predictions DB (safe restart)")
    parser.add_argument("--batch-size",           type=int, default=512)
    parser.add_argument("--min-batch-events",     type=int, default=2048,
                        help="Accumulate parts until this many events before running GPU inference "
                             "(avoids tiny GPU launches for sources with small parts, e.g. mc_reco)")
    parser.add_argument("--device",               default="auto")
    parser.add_argument("--output-dir",           default="inference_v2/prefilter/preds")
    parser.add_argument("--catalog",              default="data_manager/catalog_v2.duckdb")
    parser.add_argument("--check-existing",       action="store_true")
    parser.add_argument("--save-embeddings",      action="store_true", default=False,
                        help="Also store encoder-level feature vectors in an embeddings table")
    args = parser.parse_args()

    predict_mc(
        checkpoint=args.checkpoint,
        mc_h5=args.mc_h5,
        source=args.source,
        ptypes=args.ptypes.split(",") if args.ptypes else None,
        parts=set(args.parts.split(",")) if args.parts else None,
        max_events_per_ptype=args.max_events_per_ptype,
        skip_done_parts=args.skip_done_parts,
        batch_size=args.batch_size,
        min_batch_events=args.min_batch_events,
        device=args.device,
        output_dir=args.output_dir,
        catalog=args.catalog,
        check_existing=args.check_existing,
        save_embeddings=args.save_embeddings,
    )


if __name__ == "__main__":
    main()
