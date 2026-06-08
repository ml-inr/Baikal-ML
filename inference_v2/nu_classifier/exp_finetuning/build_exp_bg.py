"""Build the exp-background NPY dataset for fine-tuning.

Selects exp events with score < ξ (default 0.5) from a single checkpoint's
``exp_thr{tag}.duckdb`` predictions — these are almost certainly atmospheric
muon background and will be used with hard label 0 during fine-tuning.

Single-pass design: for each part the sig-noise model is run exactly once on
the selected events' hits.  Metadata (n_sig_hits, n_sig_strings) and the
filtered feature arrays are collected together, so offsets and features are
always consistent.

Output files (all with prefix ``exp_bg_``) in ``--output-dir``:
    exp_bg_features.npy        (total_sig_hits, 5) float32
    exp_bg_offsets.npy         (n_events+1,)       int64
    exp_bg_n_sig_hits.npy      (n_events,)         int32
    exp_bg_n_sig_strings.npy   (n_events,)         int32
    exp_bg_event_fks.npy       (n_events,)         int64
    exp_bg_scores.npy          (n_events,)         float32
    exp_bg_dataset_info.json

Usage (from project root):
    python inference_v2/nu_classifier/exp_finetuning/build_exp_bg.py \\
        --preds-dir  inference_v2/nu_classifier/preds/260508_1724_...seed32@best_da_model \\
        --exp-h5     data_manager/data/h5datasets/exp.h5 \\
        --catalog    data_manager/catalog_v2.duckdb \\
        --score-threshold 0.5 \\
        --sn-threshold    0.8
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

STRING_DIVISOR = 36


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_sig_hits_strings(
    mask: np.ndarray,
    channels: np.ndarray,
    hit_starts: np.ndarray,
    n_hits: np.ndarray,
    n_events: int,
) -> Tuple[np.ndarray, np.ndarray]:
    intp_starts = hit_starts.astype(np.intp)
    n_sig_hits  = np.add.reduceat(mask.astype(np.int32), intp_starts)

    if mask.any():
        event_idx  = np.repeat(np.arange(n_events), n_hits)
        sig_ev     = event_idx[mask]
        sig_str    = (channels[mask] // STRING_DIVISOR).astype(np.int32)
        combined   = sig_ev.astype(np.int64) * 1000 + sig_str.astype(np.int64)
        sort_idx   = combined.argsort()
        combined_s = combined[sort_idx]
        uniq       = np.empty(len(combined_s), dtype=bool)
        uniq[0]    = True
        uniq[1:]   = combined_s[1:] != combined_s[:-1]
        n_sig_strings = np.bincount(
            sig_ev[sort_idx[uniq]], minlength=n_events
        ).astype(np.int32)
    else:
        n_sig_strings = np.zeros(n_events, dtype=np.int32)

    return n_sig_hits.astype(np.int32), n_sig_strings


def _query_candidates(
    preds_dir: str,
    catalog_path: str,
    score_threshold: float,
    sn_thr: float,
) -> "pd.DataFrame":
    import pandas as pd
    thr_tag = str(sn_thr).replace(".", "p")
    db_path = Path(preds_dir) / f"exp_thr{thr_tag}.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(f"Exp predictions DB not found: {db_path}")

    conn = duckdb.connect()
    conn.execute(f"ATTACH '{db_path}' AS preds (READ_ONLY)")
    conn.execute(f"ATTACH '{catalog_path}' AS cat (READ_ONLY)")

    df = conn.execute("""
        SELECT p.event_fk, p.score, e.season, e.cluster, e.run, e.event_id
        FROM preds.predictions p
        JOIN cat.events e ON e.id = p.event_fk
        WHERE p.score < ?
        ORDER BY e.season, e.cluster, e.run, e.event_id
    """, [score_threshold]).df()
    conn.close()
    return df


def _process_part(
    exp_grp: h5py.Group,
    part_key: str,
    event_ids: List[int],
    event_fks: List[int],
    scores: List[float],
    model,
    batch_size: int,
    device: str,
    threshold: float,
) -> Optional[List[Dict]]:
    """Run sig-noise on selected events only and return metadata + filtered features.

    The model is run exactly once per part on the concatenated hits of the
    selected events.  Filtered features are captured immediately so there is
    no separate Pass 2 that could produce a different mask.
    """
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
        predict_flat,
    )

    ev_starts_full = exp_grp[f"raw/ev_starts/{part_key}/data"][:].astype(np.int64)
    if len(ev_starts_full) <= 1:
        return None

    h5_data = exp_grp[f"raw/data/{part_key}/data"]
    h5_chan  = exp_grp[f"raw/channels/{part_key}/data"]

    slices = [(int(ev_starts_full[eid]), int(ev_starts_full[eid + 1])) for eid in event_ids]

    # Load only the selected events' hits
    mini_data = np.concatenate([h5_data[s:e].astype(np.float32) for s, e in slices], axis=0)
    mini_chan  = np.concatenate([h5_chan[s:e].astype(np.int32)   for s, e in slices], axis=0)

    n_hits_arr = np.array([e - s for s, e in slices], dtype=np.int32)
    mini_ev_starts = np.zeros(len(event_ids) + 1, dtype=np.int64)
    np.cumsum(n_hits_arr.astype(np.int64), out=mini_ev_starts[1:])

    prob = predict_flat(
        model=model,
        data=mini_data,
        ev_starts=mini_ev_starts,
        batch_size=batch_size,
        device=device,
        normalize=True,
    )
    sig_mask = prob > threshold

    n_events = len(event_ids)
    n_sig_h, n_sig_str = _count_sig_hits_strings(
        sig_mask, mini_chan, mini_ev_starts[:-1], n_hits_arr, n_events,
    )

    results = []
    for i, (eid, fk, sc) in enumerate(zip(event_ids, event_fks, scores)):
        ev_s = int(mini_ev_starts[i])
        ev_e = int(mini_ev_starts[i + 1])
        ev_mask = sig_mask[ev_s:ev_e]
        results.append({
            "part_key":      part_key,
            "event_id":      eid,
            "hit_start":     slices[i][0],
            "hit_end":       slices[i][1],
            "event_fk":      fk,
            "score":         sc,
            "n_sig_hits":    int(n_sig_h[i]),
            "n_sig_strings": int(n_sig_str[i]),
            "features":      mini_data[ev_s:ev_e][ev_mask].copy(),  # filtered hits, captured once
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preds-dir",       required=True,
                        help="Checkpoint preds dir containing exp_thr*.duckdb")
    parser.add_argument("--exp-h5",          required=True)
    parser.add_argument("--catalog",         default="data_manager/catalog_v2.duckdb")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Select events with score < this (default: 0.5)")
    parser.add_argument("--sn-threshold",    type=float, default=0.8,
                        help="Sig-noise threshold for hit mask and DB filename (default: 0.8)")
    parser.add_argument("--min-hits",        type=int, default=5)
    parser.add_argument("--min-strings",     type=int, default=0)
    parser.add_argument("--batch-size",      type=int, default=512)
    parser.add_argument("--device",          default="cpu")
    parser.add_argument("--output-dir",      default=None)
    args = parser.parse_args()

    ckpt_name = Path(args.preds_dir).name
    xi_tag    = str(args.score_threshold).replace(".", "p")

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "exp_bg_datasets"
        / f"{ckpt_name}_lt{xi_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Output dir:       {out_dir}")
    logger.info(f"Preds dir:        {args.preds_dir}")
    logger.info(f"Score threshold:  score < {args.score_threshold}")
    logger.info(f"SN threshold:     {args.sn_threshold}")
    logger.info(f"Quality cuts:     n_sig_hits >= {args.min_hits}, n_sig_strings >= {args.min_strings}")

    # ── Query DB ──────────────────────────────────────────────────────────────
    logger.info("Querying exp predictions DB...")
    df = _query_candidates(
        preds_dir       = args.preds_dir,
        catalog_path    = args.catalog,
        score_threshold = args.score_threshold,
        sn_thr          = args.sn_threshold,
    )
    logger.info(f"Found {len(df):,} events with score < {args.score_threshold}")
    if df.empty:
        logger.warning("No events found — exiting")
        return

    def _part_key(row) -> str:
        return f"part_s{int(row.season)}_c{int(row.cluster):02d}_r{int(row.run):04d}"

    df["part_key"] = df.apply(_part_key, axis=1)

    from inference_v2.shared.model_utils import load_sn_model
    sn_model, _, sn_dev = load_sn_model(device=args.device)
    logger.info(f"Sig-noise model loaded (device={sn_dev})")

    # ── Single pass: run model once per part, collect metadata + features ─────
    logger.info("Processing parts (single pass)...")
    t0 = time.time()
    all_meta: List[Dict] = []

    part_keys_unique = list(df["part_key"].unique())
    n_parts = len(part_keys_unique)

    with h5py.File(args.exp_h5, "r") as h5:
        exp_grp = h5["exp"]
        for i, pk in enumerate(part_keys_unique):
            grp_df    = df[df["part_key"] == pk]
            event_ids = grp_df["event_id"].tolist()
            fks       = grp_df["event_fk"].tolist()
            scs       = grp_df["score"].tolist()

            results = _process_part(
                exp_grp   = exp_grp,
                part_key  = pk,
                event_ids = event_ids,
                event_fks = fks,
                scores    = scs,
                model     = sn_model,
                batch_size = args.batch_size,
                device    = sn_dev,
                threshold = args.sn_threshold,
            )
            if results:
                all_meta.extend(results)

            if (i + 1) % 10 == 0 or i == n_parts - 1:
                elapsed = time.time() - t0
                rate    = (i + 1) / max(elapsed, 1e-3)
                eta     = (n_parts - i - 1) / rate
                logger.info(
                    f"  [{i+1}/{n_parts}] parts done "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
                )

    logger.info(f"Pass done: {len(all_meta):,} events processed")

    # ── Quality cuts ──────────────────────────────────────────────────────────
    n_sig_hits_arr    = np.array([m["n_sig_hits"]    for m in all_meta], dtype=np.int32)
    n_sig_strings_arr = np.array([m["n_sig_strings"] for m in all_meta], dtype=np.int32)

    cut_mask = (n_sig_hits_arr >= args.min_hits) & (n_sig_strings_arr >= args.min_strings)
    selected = np.where(cut_mask)[0]
    n_selected = len(selected)
    logger.info(
        f"After cuts (n_sig_hits>={args.min_hits}, n_sig_strings>={args.min_strings}): "
        f"{n_selected:,} / {len(all_meta):,} events ({100*n_selected/max(len(all_meta),1):.1f}%)"
    )
    if n_selected == 0:
        logger.warning("No events survive quality cuts — exiting")
        return

    sel_meta = [all_meta[i] for i in selected]

    # ── Write output files ────────────────────────────────────────────────────
    logger.info(f"Writing output files to {out_dir} ...")

    # Features: concatenate already-filtered arrays (no second model run needed)
    features_arr = np.concatenate([m["features"] for m in sel_meta], axis=0)
    np.save(out_dir / "exp_bg_features.npy", features_arr)
    logger.info(
        f"  exp_bg_features.npy: {features_arr.shape} float32 "
        f"({features_arr.nbytes / 1e9:.3f} GB)"
    )
    del features_arr  # free memory

    n_sig_hits_sel    = n_sig_hits_arr[selected]
    n_sig_strings_sel = n_sig_strings_arr[selected]
    event_fks_sel     = np.array([m["event_fk"] for m in sel_meta], dtype=np.int64)
    scores_sel        = np.array([m["score"]    for m in sel_meta], dtype=np.float32)

    offsets = np.zeros(n_selected + 1, dtype=np.int64)
    np.cumsum(n_sig_hits_sel.astype(np.int64), out=offsets[1:])

    def save(name: str, arr: np.ndarray) -> None:
        np.save(out_dir / name, arr)
        logger.info(f"  {name}: {arr.shape} {arr.dtype} ({arr.nbytes / 1e6:.1f} MB)")

    save("exp_bg_offsets.npy",       offsets)
    save("exp_bg_n_sig_hits.npy",    n_sig_hits_sel)
    save("exp_bg_n_sig_strings.npy", n_sig_strings_sel)
    save("exp_bg_event_fks.npy",     event_fks_sel)
    save("exp_bg_scores.npy",        scores_sel)

    # ── Dataset info ──────────────────────────────────────────────────────────
    info = {
        "n_events":          n_selected,
        "n_sig_hits_total":  int(offsets[-1]),
        "checkpoint":        ckpt_name,
        "score_threshold":   args.score_threshold,
        "sn_threshold":      args.sn_threshold,
        "min_hits":          args.min_hits,
        "min_strings":       args.min_strings,
        "exp_h5":            args.exp_h5,
        "catalog":           args.catalog,
        "score_stats": {
            "mean":  float(scores_sel.mean()),
            "std":   float(scores_sel.std()),
            "min":   float(scores_sel.min()),
            "max":   float(scores_sel.max()),
        },
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "exp_bg_dataset_info.json").write_text(json.dumps(info, indent=2))
    logger.info(f"  exp_bg_dataset_info.json written")
    logger.info(f"=== Done: {n_selected:,} events, {int(offsets[-1]):,} sig hits → {out_dir} ===")


if __name__ == "__main__":
    main()
