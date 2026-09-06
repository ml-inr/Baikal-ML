"""Predict SNGP (distance-aware) nu-classifier scores, persisting BOTH the mean-field
score and the GP predictive variance.

Why a separate script: the standard predict_mc.py / predict_exp.py store only `score`.
For SNGP the variance var = phi^T Sigma phi is the whole point (it is what makes the
output distance-aware), and it is computed inside RandomFeatureGPHead.forward and then
discarded. Here we recompute it explicitly and store it alongside the score.

The mean-field score reproduced here is bit-identical to what model.classifier(emb)
returns in eval mode with a valid covariance:
    phi    = head._phi(emb)
    logit  = head.beta(phi)
    var    = einsum(phi, Sigma, phi)
    score  = sigmoid(logit / sqrt(1 + mean_field_factor * var))
Pre-processing (padding, masking, normalisation, amplitude clipping) mirrors
inference_v2/shared/model_utils.predict_scores exactly, so scores are directly
comparable with the DA/fine-tuned model DBs.

Sources: mc_merged (per-ptype groups) and exp_full (single group, physical event_id
from header_prty). MC parts are large (23-39k events), so parts are processed one at a
time without buffering.

Output: preds/{checkpoint_name}/{source}_thr{thr}.duckdb
Schema: predictions(event_fk BIGINT PK, score FLOAT, gp_var FLOAT,
                    n_sn_hits INTEGER, n_sn_strings INTEGER)
Incremental: INSERT OR IGNORE, safe to re-run on further parts.

Usage (from project root):
    python inference_v2/nu_classifier/predict_sngp.py \\
        --checkpoint experiments/numu/sngp_nu_classifier_baseline/best_sngp_model.pth \\
        --source mc_merged \\
        --h5 data_manager/data/h5datasets/baikal_mc_merged.h5 \\
        --probs-h5 data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5 \\
        --npy-dir-to-exclude data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8 \\
        --max-events-per-ptype 2000000 --device cuda:0
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inference_v2.shared.model_utils import load_model, load_sn_model
from inference_v2.shared.catalog_query import get_event_fks_mc, get_event_fks_exp
from inference_v2.shared.history import append_history_row
from data_manager.nu_classifier_ds_builder.io import _count_sig_hits_strings

logger = logging.getLogger(__name__)

MC_MERGED_PTYPES = ["muatm_2020", "nuatm_2020", "nue2_2020"]
RDCC = dict(rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=1_000_003)

SNGP_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    event_fk     BIGINT  PRIMARY KEY,
    score        FLOAT,
    gp_var       FLOAT,
    n_sn_hits    INTEGER,
    n_sn_strings INTEGER
)
"""


def append_predictions_sngp(
    conn: duckdb.DuckDBPyConnection,
    event_fks: np.ndarray,
    scores: np.ndarray,
    gp_vars: np.ndarray,
    n_sn_hits: np.ndarray,
    n_sn_strings: np.ndarray,
) -> Tuple[int, int]:
    """Insert (score, gp_var) rows with INSERT OR IGNORE semantics."""
    df_new = pd.DataFrame({
        "event_fk":     event_fks.astype(np.int64),
        "score":        scores.astype(np.float32),
        "gp_var":       gp_vars.astype(np.float32),
        "n_sn_hits":    n_sn_hits.astype(np.int32),
        "n_sn_strings": n_sn_strings.astype(np.int32),
    }).drop_duplicates(subset="event_fk")

    before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.register("df_new", df_new)
    conn.execute("INSERT OR IGNORE INTO predictions SELECT * FROM df_new")
    conn.unregister("df_new")
    after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    inserted = after - before
    return inserted, len(df_new) - inserted


@torch.no_grad()
def predict_sngp_scores(
    model: torch.nn.Module,
    features_list: List[np.ndarray],
    normalization_config: Dict,
    batch_size: int = 512,
    max_hits: Optional[int] = 500,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched SNGP inference -> (mean-field scores, predictive variances)."""
    head = model.classifier
    if not hasattr(head, "covariance"):
        raise AttributeError("Checkpoint is not an SNGP model (classifier has no covariance)")
    if not bool(head.cov_valid.item()):
        raise RuntimeError("GP covariance is not valid in this checkpoint — mean-field "
                           "would be skipped; re-run the last training epoch to build it")

    means = torch.tensor(normalization_config["means"], dtype=torch.float32, device=device)
    stds = torch.tensor(normalization_config["stds"], dtype=torch.float32, device=device)

    all_scores: List[np.ndarray] = []
    all_vars: List[np.ndarray] = []

    for start in range(0, len(features_list), batch_size):
        batch_feats = features_list[start: start + batch_size]
        b = len(batch_feats)

        truncated, lengths = [], []
        for feat in batch_feats:
            nh = len(feat)
            if max_hits is not None and nh > max_hits:
                truncated.append(feat[:max_hits])
                lengths.append(max_hits)
            else:
                truncated.append(feat)
                lengths.append(nh)

        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        max_len = int(lengths_t.max().item())
        padded = torch.zeros(b, max_len, 5, dtype=torch.float32, device=device)
        for i, feat in enumerate(truncated):
            padded[i, :len(feat)] = torch.from_numpy(feat)

        mask = torch.arange(max_len, device=device)[None, :] < lengths_t[:, None]
        padded = torch.where(mask.unsqueeze(-1), (padded - means) / (stds + 1e-8), padded)

        batch_dict = {"features": padded, "lengths": lengths_t, "mask": mask}
        if hasattr(model, "_clip_amplitude"):
            batch_dict = model._clip_amplitude(batch_dict, getattr(model, "amp_clip", None))

        emb = model.feature_extractor(
            sequences=batch_dict["features"],
            lengths=batch_dict["lengths"],
            mask=batch_dict["mask"],
        )
        phi = head._phi(emb)
        logits = head.beta(phi).squeeze(-1)
        var = torch.einsum("bi,ij,bj->b", phi, head.covariance, phi).clamp_min(0.0)
        mean_field = logits / torch.sqrt(1.0 + head.mean_field_factor * var)

        all_scores.append(torch.sigmoid(mean_field).cpu().numpy())
        all_vars.append(var.cpu().numpy())

    return np.concatenate(all_scores), np.concatenate(all_vars)


def _load_npy_training_parts(npy_dir: str) -> Dict[str, set]:
    """{ptype_name: set_of_part_keys} used in training (for out-of-training scoring)."""
    npy_path = Path(npy_dir)
    with open(npy_path / "dataset_info.json") as f:
        info = json.load(f)
    decode = {v: k for k, v in info["particle_encode"].items()}
    part_keys = np.load(npy_path / "h5_part_keys.npy", allow_pickle=True)
    ptypes_arr = np.load(npy_path / "particle_types.npy")
    return {name: set(part_keys[ptypes_arr == pid].tolist()) for pid, name in decode.items()}


def _score_part(model, norm_config, sn_model, sn_dev, predict_flat,
                data_raw, channels, ev_starts, probs_raw,
                threshold, min_hits, min_strings, batch_size, dev):
    """Filter hits, apply the multiplicity cut, score. Returns (sel_ev, scores, vars, nh, ns)."""
    n_events = len(ev_starts) - 1
    probs = (probs_raw if probs_raw is not None
             else predict_flat(sn_model, data_raw, ev_starts,
                               batch_size=batch_size, device=sn_dev, normalize=True))
    sig_mask = probs > threshold
    n_hits_arr = (ev_starts[1:] - ev_starts[:-1]).astype(np.int32)
    n_sn_h, n_sn_s = _count_sig_hits_strings(
        sig_mask, channels, ev_starts[:-1], n_hits_arr, n_events)

    sel_ev = np.where((n_sn_h >= min_hits) & (n_sn_s >= min_strings))[0]
    if len(sel_ev) == 0:
        return sel_ev, None, None, n_sn_h, n_sn_s

    features_list = []
    for ev_i in sel_ev:
        s, e = int(ev_starts[ev_i]), int(ev_starts[ev_i + 1])
        features_list.append(data_raw[s:e][sig_mask[s:e]])

    scores, gp_vars = predict_sngp_scores(
        model, features_list, norm_config, batch_size=batch_size, device=dev)
    return sel_ev, scores, gp_vars, n_sn_h, n_sn_s


def run(
    checkpoint: str,
    h5_path: str,
    source: str = "mc_merged",
    probs_h5: Optional[str] = None,
    ptypes: Optional[List[str]] = None,
    max_events_per_ptype: Optional[int] = None,
    max_events_per_part: Optional[int] = None,
    npy_dir_to_exclude: Optional[str] = None,
    threshold: float = 0.8,
    min_hits: int = 8,
    min_strings: int = 2,
    batch_size: int = 512,
    device: str = "cuda:0",
    output_dir: str = "inference_v2/nu_classifier/preds",
    catalog: str = "data_manager/catalog_v2.duckdb",
    checkpoint_name: Optional[str] = None,
) -> None:
    t_start = datetime.now()
    checkpoint_name = checkpoint_name or f"{Path(checkpoint).parent.name}@{Path(checkpoint).stem}"
    thr_tag = str(threshold).replace(".", "p")

    ckpt_dir = Path(output_dir) / checkpoint_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    db_path = ckpt_dir / f"{source}_thr{thr_tag}.duckdb"
    log_path = ckpt_dir / f"predict_sngp_{t_start.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)])

    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Source:     {source}   h5: {h5_path}")
    logger.info(f"Output DB:  {db_path}")
    logger.info(f"Caps: per-ptype={max_events_per_ptype}  per-part={max_events_per_part}")

    n_inserted = n_skipped = 0
    error_msg = ""

    try:
        model, norm_config, _ = load_model(checkpoint, device=device)
        if not hasattr(model.classifier, "covariance"):
            raise SystemExit("Not an SNGP checkpoint — use predict_mc.py / predict_exp.py")
        logger.info(f"SNGP loaded: mean_field_factor={model.classifier.mean_field_factor:.4f}, "
                    f"cov_valid={bool(model.classifier.cov_valid.item())}")

        probs_ctx = h5py.File(probs_h5, "r", **RDCC) if probs_h5 else None
        if probs_ctx is not None:
            sn_model, sn_dev, predict_flat = None, None, None
            logger.info(f"Using precomputed SN probs: {probs_h5}")
        else:
            sn_model, _, sn_dev = load_sn_model(device=device)
            from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import predict_flat

        training_parts = _load_npy_training_parts(npy_dir_to_exclude) if npy_dir_to_exclude else {}

        conn = duckdb.connect(str(db_path))
        conn.execute(SNGP_SCHEMA)

        with h5py.File(h5_path, "r", **RDCC) as h5:
            if source == "mc_merged":
                groups = [p for p in (ptypes or MC_MERGED_PTYPES) if p in h5]
            else:
                groups = [k for k in h5.keys()][:1]  # exp_full: single top group
            logger.info(f"Groups: {groups}")

            for gname in groups:
                grp = h5[gname]
                has_header = "header_prty" in grp
                probs_grp = probs_ctx[gname] if (probs_ctx is not None and gname in probs_ctx) else None

                all_parts = sorted(grp["raw"]["data"].keys())
                if probs_grp is not None:
                    have = set(probs_grp["probs"].keys())
                    n_pre = len(all_parts)
                    all_parts = [p for p in all_parts if p in have]
                    logger.info(f"{gname}: {len(all_parts)}/{n_pre} parts have precomputed probs")
                excl = training_parts.get(gname, set())
                if excl:
                    n_pre = len(all_parts)
                    all_parts = [p for p in all_parts if p not in excl]
                    logger.info(f"{gname}: excluded {n_pre - len(all_parts)} training parts")

                season = int(gname.rsplit("_", 1)[-1]) if source == "mc_merged" else 0
                remaining = max_events_per_ptype
                g_ins = g_skp = 0
                t0 = time.perf_counter()

                for part_idx, pk in enumerate(all_parts):
                    if remaining is not None and remaining <= 0:
                        break

                    ev_starts_full = grp[f"raw/ev_starts/{pk}/data"][:].astype(np.int64)
                    n_events_full = len(ev_starts_full) - 1
                    if n_events_full == 0:
                        continue

                    # Per-part cap is applied BEFORE reading hits: one exp part is one
                    # physical run whose events are time-ordered but physically random,
                    # so a contiguous prefix is an unbiased subsample and keeps the read
                    # (and the scoring) proportional to the cap instead of the part size.
                    n_take = n_events_full
                    if max_events_per_part is not None:
                        n_take = min(n_take, max_events_per_part)
                    if remaining is not None:
                        n_take = min(n_take, remaining)
                    ev_starts = ev_starts_full[:n_take + 1]
                    n_events = len(ev_starts) - 1
                    if remaining is not None:
                        remaining -= n_events

                    hit_end = int(ev_starts[-1])
                    data_raw = grp[f"raw/data/{pk}/data"][:hit_end].astype(np.float32)
                    channels = grp[f"raw/channels/{pk}/data"][:hit_end].astype(np.int32)
                    probs_raw = (probs_grp[f"probs/{pk}/data"][:hit_end].astype(np.float32)
                                 if probs_grp is not None else None)

                    sel_ev, scores, gp_vars, n_sn_h, n_sn_s = _score_part(
                        model, norm_config, sn_model, sn_dev, predict_flat,
                        data_raw, channels, ev_starts, probs_raw,
                        threshold, min_hits, min_strings, batch_size, device)

                    if len(sel_ev) == 0:
                        continue

                    n_sel = len(sel_ev)
                    if source == "mc_merged":
                        cat_res = get_event_fks_mc(
                            catalog, source,
                            data_classes=np.array([gname] * n_sel),
                            seasons=np.full(n_sel, season, dtype=np.int32),
                            runs=np.array([pk] * n_sel),
                            event_ids=sel_ev.astype(np.int64))
                    else:
                        header = grp[f"header_prty/{pk}/data"][:] if has_header else None
                        if header is None:
                            logger.warning(f"  {pk}: no header_prty, skipping")
                            continue
                        cat_res = get_event_fks_exp(
                            catalog, gname,
                            seasons=header[sel_ev, 0].astype(np.int32),
                            clusters=header[sel_ev, 1].astype(np.int32),
                            runs=np.array([str(int(header[i, 2])) for i in sel_ev]),
                            event_ids=header[sel_ev, 3].astype(np.int64))

                    if len(cat_res) == 0:
                        logger.warning(f"  {pk}: no events found in catalog")
                        continue

                    qi = cat_res["query_idx"].values
                    ins, skp = append_predictions_sngp(
                        conn,
                        event_fks=cat_res["event_fk"].values.astype(np.int64),
                        scores=scores[qi], gp_vars=gp_vars[qi],
                        n_sn_hits=n_sn_h[sel_ev[qi]], n_sn_strings=n_sn_s[sel_ev[qi]])
                    n_inserted += ins; n_skipped += skp
                    g_ins += ins; g_skp += skp

                    if (part_idx + 1) % 10 == 0 or part_idx + 1 == len(all_parts):
                        el = time.perf_counter() - t0
                        rate = (part_idx + 1) / el if el > 0 else 0
                        logger.info(f"  {gname} [{part_idx+1}/{len(all_parts)}] "
                                    f"ins={g_ins:,} skip={g_skp:,} rate={rate:.2f} parts/s")

        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
        if probs_ctx is not None:
            probs_ctx.close()
        logger.info(f"\nDone: {n_inserted:,} inserted, {n_skipped:,} skipped, total in DB: {total:,}")

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("predict_sngp failed")
        raise
    finally:
        logger.info(f"Total elapsed: {(datetime.now()-t_start).total_seconds():.0f}s")
        append_history_row(
            str(Path(output_dir) / "prediction_history.csv"),
            checkpoint=checkpoint_name, script="predict_sngp.py", source=source,
            db_file=str(db_path), threshold=threshold, min_hits=min_hits,
            min_strings=min_strings, n_events_new=n_inserted, n_events_skipped=n_skipped,
            is_successful=error_msg == "", error_msg=error_msg,
            timestamp=t_start.isoformat(timespec="seconds"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5", required=True, help="Source HDF5 (mc_merged or exp_full)")
    p.add_argument("--source", default="mc_merged", choices=["mc_merged", "exp_full"])
    p.add_argument("--probs-h5", default=None, help="Precomputed SN probs (strongly recommended)")
    p.add_argument("--ptypes", default=None, help="Comma-separated ptypes (mc_merged only)")
    p.add_argument("--max-events-per-ptype", type=int, default=None)
    p.add_argument("--max-events-per-part", type=int, default=None)
    p.add_argument("--npy-dir-to-exclude", default=None,
                   help="Training NPY dir whose parts are skipped (out-of-training scoring)")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--min-hits", type=int, default=8)
    p.add_argument("--min-strings", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="inference_v2/nu_classifier/preds")
    p.add_argument("--catalog", default="data_manager/catalog_v2.duckdb")
    p.add_argument("--checkpoint-name", default=None)
    args = p.parse_args()

    run(checkpoint=args.checkpoint, h5_path=args.h5, source=args.source,
        probs_h5=args.probs_h5, ptypes=args.ptypes.split(",") if args.ptypes else None,
        max_events_per_ptype=args.max_events_per_ptype,
        max_events_per_part=args.max_events_per_part,
        npy_dir_to_exclude=args.npy_dir_to_exclude, threshold=args.threshold,
        min_hits=args.min_hits, min_strings=args.min_strings, batch_size=args.batch_size,
        device=args.device, output_dir=args.output_dir, catalog=args.catalog,
        checkpoint_name=args.checkpoint_name)


if __name__ == "__main__":
    main()
