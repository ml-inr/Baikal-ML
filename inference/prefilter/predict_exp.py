"""Predict prefilter scores for experimental data from HDF5.

No sig-noise filtering — all raw hits are fed to the prefilter model.
Auto-detects the source group (exp / exp_reco / exp_full) as the top-level group
holding 'raw'; event_fk lookup uses header_prty when that group has it.

Output: preds/{checkpoint_name}/{src_key}_allhits.duckdb
Schema: predictions(event_fk BIGINT PK, score FLOAT, n_hits INTEGER)

With header_prty (exp_reco, exp_full): event_fk looked up by
(season, cluster, run, event_id_in_run) read from header_prty.
Without it (exp): event_fk looked up by parsed part key (season, cluster, run int).

Incremental: safe to re-run on different part subsets — INSERT OR IGNORE.

Reproducibility: `--batch-size` and `--chunk-events` both perturb the score in its last
float32 bit. The model pads each batch to the longest event in it, so batch composition
sets the order of a floating-point reduction, and `--chunk-events` changes that
composition wherever a part is larger than the window. Measured on exp.h5 (649,976
events, one checkpoint): batch 512 against 2048 moves 37.8% of scores, chunk 5,000
against whole-part moves 5.3%, and in both cases the largest difference is 1.192e-07 —
exactly one ulp at 1.0.

This is **not** the sig-noise situation. There the float padding mask changes which hits
pass the threshold, so batch size is part of the hit selection (doc/sig_noise_batch_size.md).
Here nothing is selected and the change is arithmetic noise six orders below any bin edge
or working point used downstream. Still, two runs to be compared bit-for-bit must use the
same values for both flags.

Usage (from project root):
    python inference/prefilter/predict_exp.py \\
        --checkpoint trained_models/prefilter/da_prefilter_numu_260429_.../best_da_model.pth \\
        --exp-h5 data_manager/data/h5datasets/exp_reco.h5 \\
        [--batch-size 512] [--chunk-events 200000] [--device auto] \\
        [--output-dir inference/prefilter/preds] \\
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


# raw/data is chunked (43215, 1) + gzip: a generous chunk cache keeps a window read
# from re-decompressing the chunks that straddle its edges.
_RDCC = dict(rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=1_000_003)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.shared.model_utils import load_model, predict_scores, predict_scores_and_embeddings
from inference.shared.catalog_query import (
    get_event_fks_exp,
    open_predictions_db,
    PREFILTER_SCHEMA,
    EMBEDDINGS_SCHEMA,
    append_predictions_prefilter,
    append_embeddings,
)
from inference.shared.history import append_history_row
from inference.shared.sig_counts import (
    SN_THRESHOLD, check_lengths, probs_path_for, read_part_counts,
)

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
    output_dir: str = "inference/prefilter/preds",
    catalog: str = "data_manager/catalog_v2.duckdb",
    check_existing: bool = False,
    skip_done_parts: bool = False,
    save_embeddings: bool = False,
    chunk_events: int = 200_000,
    probs_h5: str | None = None,
    sn_threshold: float = SN_THRESHOLD,
) -> None:
    t_start = datetime.now()
    dev = _resolve_device(device)
    checkpoint_name = f"{Path(checkpoint).parent.name}@{Path(checkpoint).stem}"

    # src_key = top-level group holding the hit data ('exp', 'exp_reco', 'exp_full', ...).
    # has_header is driven by the actual presence of header_prty, not the group name,
    # so exp_full (physical CC-timestamp event_id in header_prty) is handled correctly.
    with h5py.File(exp_h5, "r") as h5:
        data_groups = [k for k in h5.keys() if "raw" in h5[k]]
        src_key = data_groups[0] if data_groups else list(h5.keys())[0]
        has_header_h5 = "header_prty" in h5[src_key]

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
    logger.info(f"chunk_events:    {chunk_events:,}")
    logger.info(f"probs h5:        {probs_h5 or 'none — hit counts will be NULL'}")

    error_msg  = ""
    n_inserted = 0
    n_skipped  = 0

    probs_file = None
    try:
        model, norm_config, _ = load_model(checkpoint, device=dev)
        if probs_h5:
            probs_file = h5py.File(probs_h5, "r", **_RDCC)
            probs_grp = probs_file[src_key] if src_key in probs_file else \
                probs_file[list(probs_file.keys())[0]]
        else:
            probs_grp = None
        conn = open_predictions_db(str(db_path), PREFILTER_SCHEMA)
        if save_embeddings:
            conn.execute(EMBEDDINGS_SCHEMA)
        has_header = has_header_h5

        done_part_keys: set = set()
        if skip_done_parts:
            logger.info("Querying done parts from predictions DB...")
            t0 = time.perf_counter()
            done_part_keys = _get_done_parts(conn, catalog, src_key)
            logger.info(f"  {len(done_part_keys)} already-done parts  ({time.perf_counter()-t0:.1f}s)")

        with h5py.File(exp_h5, "r", **_RDCC) as h5:
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

                ds_data = grp[f"raw/data/{pk}/data"]
                ds_hdr  = grp[f"header_prty/{pk}/data"] if has_header else None

                # Signal-hit counts for the whole part, sliced per window below. Tiny
                # next to the hits, so they are read once rather than per window.
                counts = read_part_counts(probs_grp, pk, sn_threshold)
                if counts.empty and probs_grp is not None:
                    logger.warning(f"  {pk}: absent from the probs file — counts NULL")
                check_lengths(counts, n_events, f"{src_key}/{pk}")

                # One exp_full part is up to 11.3M events / 1.06e9 hits = 21 GB as
                # float32, and the per-event view list on top of it would be another
                # 11.3M Python objects -- so the whole part cannot be taken at once.
                # Walk it in windows of chunk_events instead. Each window is a single
                # contiguous hit-range read and the windows advance in file order, so
                # the gzip chunks are decompressed in order; only a chunk straddling a
                # window boundary is touched twice, and the 256 MB cache above holds it
                # across the seam. Parts smaller than the window (exp, exp_reco, and
                # every mc part) take this path with one window and behave as before.
                for c0 in range(0, n_events, chunk_events):
                    c1 = min(c0 + chunk_events, n_events)
                    hit_lo, hit_hi = int(ev_starts[c0]), int(ev_starts[c1])

                    block  = ds_data[hit_lo:hit_hi].astype(np.float32)
                    loc    = ev_starts[c0:c1 + 1] - hit_lo
                    header = ds_hdr[c0:c1] if ds_hdr is not None else None

                    n_hits_arr    = (loc[1:] - loc[:-1]).astype(np.int32)
                    features_list = [
                        block[int(loc[i]):int(loc[i + 1])]
                        for i in range(c1 - c0)
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

                    n_chunk = c1 - c0
                    all_ev  = np.arange(c0, c1, dtype=np.int64)
                    if header is not None:
                        seasons  = header[:, 0].astype(np.int32)
                        clusters = header[:, 1].astype(np.int32)
                        runs     = np.array([str(int(v)) for v in header[:, 2]])
                        ev_ids   = header[:, 3].astype(np.int64)
                        cat_res  = get_event_fks_exp(
                            catalog, src_key, seasons, clusters, runs, ev_ids
                        )
                    else:
                        pk_season, pk_cluster, pk_run = _parse_exp_part_key(pk)
                        cat_res = get_event_fks_exp(
                            catalog, src_key,
                            seasons=np.full(n_chunk, pk_season,  dtype=np.int32),
                            clusters=np.full(n_chunk, pk_cluster, dtype=np.int32),
                            runs=np.array([str(pk_run)] * n_chunk),
                            event_ids=all_ev,
                        )

                    if len(cat_res) == 0:
                        logger.warning(f"  {pk}[{c0}:{c1}]: no events found in catalog, skipping")
                    else:
                        found_qi  = cat_res["query_idx"].values
                        event_fks = cat_res["event_fk"].values.astype(np.int64)
                        found     = counts.slice(c0, c1).take(found_qi)
                        ins, skp  = append_predictions_prefilter(
                            conn,
                            event_fks=event_fks,
                            scores=scores[found_qi],
                            n_hits=n_hits_arr[found_qi],
                            n_sn_hits=found.n_sn_hits,
                            n_sn_strings=found.n_sn_strings,
                            n_gt_sig_hits=found.n_gt_sig_hits,
                            n_gt_sig_strings=found.n_gt_sig_strings,
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
        with_counts = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE n_sn_hits IS NOT NULL").fetchone()[0]
        logger.info(f"rows with sig-noise counts: {with_counts:,} / {total_in_db:,}")
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
        if probs_file is not None:
            probs_file.close()
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
    parser.add_argument("--output-dir",      default="inference/prefilter/preds")
    parser.add_argument("--catalog",         default="data_manager/catalog_v2.duckdb")
    parser.add_argument("--check-existing",  action="store_true")
    parser.add_argument("--skip-done-parts", action="store_true", default=False,
                        help="Skip parts already present in the predictions DB (safe restart)")
    parser.add_argument("--save-embeddings", action="store_true", default=False,
                        help="Also store encoder-level feature vectors in an embeddings table")
    parser.add_argument("--probs-h5", default=None,
                        help="Sig-noise probabilities file supplying n_sn_hits / "
                             "n_sn_strings (and MC truth counts where it has them). "
                             "Pass 'auto' to take the one named beside --exp-h5. "
                             "Without it those columns are NULL.")
    parser.add_argument("--sn-threshold", type=float, default=SN_THRESHOLD,
                        help="Which stored sig-noise counts to copy (0.5 or 0.8). Does "
                             "not affect scoring — the prefilter reads all raw hits.")
    parser.add_argument("--chunk-events", type=int, default=200_000,
                        help="Events per read window inside a part. Bounds RAM — an "
                             "exp_full part is up to 21 GB of hits. Where a part is "
                             "larger than the window it also re-cuts the inference "
                             "batches, which moves scores by one float32 ulp; see the "
                             "note in the module docstring.")
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
        chunk_events=args.chunk_events,
        probs_h5=(str(probs_path_for(args.exp_h5)) if args.probs_h5 == "auto"
                  else args.probs_h5),
        sn_threshold=args.sn_threshold,
    )


if __name__ == "__main__":
    main()
