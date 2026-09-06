"""Extract exp candidate events where ALL specified models predict score > threshold.

Inner-joins exp prediction DBs from N models, applies per-model score cut,
loads raw hits from exp.h5, runs the sig-noise model to produce hit masks,
writes one JSON per event.

JSON structure (matches inference/nu_classifier_model/candidates/ convention):
  {
    "eventID": <int>,   -- catalog event_id (0-based local index within run)
    "season": <int>,
    "cluster": <int>,
    "run": <int>,
    "scores": {"<checkpoint_name>": <float>, ...},
    "pulses": {
      "0": {"mask": 0|1, "amplitude": float, "charge": float,
             "time": float, "channelID": int},
      ...
    }
  }

Filename: event_part_s{season}_c{cluster:02d}_r{run:04d}_{event_id}.json

Usage (from project root):
    python inference_v2/nu_classifier/analysis/extract_moe_candidates.py \\
        --preds-dir inference_v2/nu_classifier/preds \\
        --checkpoints \\
            260507_2111_...@best_da_model \\
            260507_2121_...@best_da_model \\
        --exp-h5 data_manager/data/h5datasets/exp.h5 \\
        --score-threshold 0.95 \\
        --sn-threshold 0.8
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _part_key(season: int, cluster: int, run: int) -> str:
    return f"part_s{season}_c{cluster:02d}_r{run:04d}"


def _find_candidates(
    checkpoint_dirs: list[str],
    thr: float,
    score_thr: float,
    catalog_path: str,
) -> list[dict]:
    """Return list of dicts {event_fk, season, cluster, run, event_id, scores}."""
    n = len(checkpoint_dirs)
    thr_tag = str(thr).replace(".", "p")

    db_paths = []
    for d in checkpoint_dirs:
        p = Path(d) / f"exp_thr{thr_tag}.duckdb"
        if not p.exists():
            raise FileNotFoundError(f"Exp predictions DB not found: {p}")
        db_paths.append(str(p))

    conn = duckdb.connect()
    for i, p in enumerate(db_paths):
        conn.execute(f"ATTACH '{p}' AS p{i} (READ_ONLY)")
    conn.execute(f"ATTACH '{catalog_path}' AS cat (READ_ONLY)")

    score_cols = ", ".join(f"p{i}.predictions.score AS score_{i}" for i in range(n))
    joins = "\n        ".join(
        f"INNER JOIN p{i}.predictions USING (event_fk)" for i in range(1, n)
    )
    where = " AND ".join(f"p{i}.predictions.score > {score_thr}" for i in range(n))

    rows = conn.execute(f"""
        SELECT
            p0.predictions.event_fk,
            e.season, e.cluster, e.run, e.event_id,
            {score_cols}
        FROM p0.predictions
        {joins}
        JOIN cat.events e ON e.id = p0.predictions.event_fk
        WHERE {where}
        ORDER BY e.season, e.cluster, e.run, e.event_id
    """).df()
    conn.close()

    ckpt_names = [Path(d).name for d in checkpoint_dirs]
    candidates = []
    for _, row in rows.iterrows():
        scores = {ckpt_names[i]: float(row[f"score_{i}"]) for i in range(n)}
        candidates.append({
            "event_fk":  int(row["event_fk"]),
            "season":    int(row["season"]),
            "cluster":   int(row["cluster"]),
            "run":       int(row["run"]),
            "event_id":  int(row["event_id"]),
            "scores":    scores,
        })
    return candidates


def _extract_event(
    exp_h5: h5py.File,
    season: int,
    cluster: int,
    run: int,
    event_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (data_raw, ev_starts_2, channels) for a single event.

    data_raw: (n_hits, 5) float32 — all raw hits
    ev_starts_2: (2,) int64 — [0, n_hits] for predict_flat
    channels: (n_hits,) int32
    """
    pk = _part_key(season, cluster, run)
    grp = exp_h5["exp"]
    ev_starts = grp[f"raw/ev_starts/{pk}/data"][:]
    s, e = int(ev_starts[event_id]), int(ev_starts[event_id + 1])
    data_raw = grp[f"raw/data/{pk}/data"][s:e].astype(np.float32)
    channels = grp[f"raw/channels/{pk}/data"][s:e].astype(np.int32)
    return data_raw, np.array([0, e - s], dtype=np.int64), channels


def _build_json(
    event_id: int,
    season: int,
    cluster: int,
    run: int,
    scores: dict,
    data_raw: np.ndarray,
    channels: np.ndarray,
    sn_mask: np.ndarray,
) -> dict:
    pulses = {}
    for i in range(len(data_raw)):
        amp = float(data_raw[i, 0])
        t   = float(data_raw[i, 1])
        pulses[str(i)] = {
            "mask":      int(sn_mask[i]),
            "amplitude": amp,
            "charge":    amp,
            "time":      t,
            "channelID": int(channels[i]),
        }
    return {
        "eventID": event_id,
        "season":  season,
        "cluster": cluster,
        "run":     run,
        "scores":  scores,
        "pulses":  pulses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preds-dir",      required=True)
    parser.add_argument("--checkpoints",    nargs="+", required=True)
    parser.add_argument("--exp-h5",         required=True)
    parser.add_argument("--score-threshold", type=float, default=0.95,
                        help="All models must score above this (default: 0.95)")
    parser.add_argument("--sn-threshold",   type=float, default=0.8,
                        help="Sig-noise threshold for hit mask and DB filename (default: 0.8)")
    parser.add_argument("--batch-size",     type=int, default=512)
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--catalog",        default="data_manager/catalog_v2.duckdb")
    parser.add_argument("--output-dir",     default=None)
    args = parser.parse_args()

    preds_root = Path(args.preds_dir)
    checkpoint_dirs = [str(preds_root / c) for c in args.checkpoints]
    n_models = len(checkpoint_dirs)

    _ts = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "moe_candidates"
        / f"moe_{n_models}models_scorethr{str(args.score_threshold).replace('.','p')}_{_ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Output dir:      {out_dir}")
    logger.info(f"Models ({n_models}):  {args.checkpoints}")
    logger.info(f"Score threshold: {args.score_threshold}")
    logger.info(f"SN threshold:    {args.sn_threshold}")

    # ── Find candidates ───────────────────────────────────────────────────────
    candidates = _find_candidates(
        checkpoint_dirs = checkpoint_dirs,
        thr             = args.sn_threshold,
        score_thr       = args.score_threshold,
        catalog_path    = args.catalog,
    )
    logger.info(f"Found {len(candidates)} candidate events")
    if not candidates:
        logger.warning("No candidates — exiting")
        return

    # ── Load sig-noise model ──────────────────────────────────────────────────
    from inference_v2.shared.model_utils import load_sn_model
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
        predict_flat,
    )
    sn_model, _, sn_dev = load_sn_model(device=args.device)
    logger.info(f"Sig-noise model loaded (device={sn_dev})")

    # ── Process each candidate ────────────────────────────────────────────────
    with h5py.File(args.exp_h5, "r") as h5:
        for ev in candidates:
            season, cluster, run, event_id = ev["season"], ev["cluster"], ev["run"], ev["event_id"]
            pk = _part_key(season, cluster, run)
            logger.info(f"  {pk}  event_id={event_id}  scores={ev['scores']}")

            data_raw, ev_starts, channels = _extract_event(h5, season, cluster, run, event_id)
            n_hits = len(data_raw)

            probs = predict_flat(
                sn_model, data_raw, ev_starts,
                batch_size=args.batch_size,
                device=sn_dev,
                normalize=True,
            )
            sn_mask = (probs > args.sn_threshold).astype(np.int32)
            n_sig = int(sn_mask.sum())
            logger.info(f"    {n_hits} hits, {n_sig} sig-noise hits (mask=1)")

            payload = _build_json(
                event_id = event_id,
                season   = season,
                cluster  = cluster,
                run      = run,
                scores   = ev["scores"],
                data_raw = data_raw,
                channels = channels,
                sn_mask  = sn_mask,
            )
            fname = f"event_part_s{season}_c{cluster:02d}_r{run:04d}_{event_id}.json"
            fpath = out_dir / fname
            fpath.write_text(json.dumps(payload, indent=2))
            logger.info(f"    Saved: {fpath}")

    logger.info(f"=== Done: {len(candidates)} events written to {out_dir} ===")


if __name__ == "__main__":
    main()
