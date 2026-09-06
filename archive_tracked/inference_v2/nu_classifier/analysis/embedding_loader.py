"""Load encoder embeddings for UMAP and other representation-space analyses.

Two loaders:
  load_mc_embeddings_from_h5()   — sample out-of-training MC events from baikal_mc_merged.h5
  load_exp_embeddings_from_preds() — sample exp events by score bucket from exp_thr0p8.duckdb
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional

import duckdb
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PART_KEY_RE = re.compile(r"part_s(\d+)_c(\d+)_r(\d+)")

MC_MERGED_PTYPES = ["muatm_2020", "nuatm_2020", "nue2_2020"]


def _parse_exp_part_key(pk: str) -> tuple[int, int, int]:
    m = _PART_KEY_RE.match(pk)
    if not m:
        raise ValueError(f"Cannot parse exp part key: {pk!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _apply_sn_and_build_features(
    data_raw: np.ndarray,
    channels: np.ndarray,
    ev_starts: np.ndarray,      # (n_events+1,) int64
    sn_model,
    sn_dev: str,
    threshold: float,
    min_hits: int,
    min_strings: int,
    batch_size: int,
    with_probs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Apply sig-noise model and quality cut.

    Returns:
        sel_ev:        (k,) indices of passing events
        n_sn_hits:     (k,) hit counts after filtering
        n_sn_strings:  (k,) string counts
        features_list: k lists of (n_sig_hits, 5 or 6) arrays
    """
    from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import predict_flat
    from data_manager.nu_classifier_ds_builder.io import _count_sig_hits_strings

    n_events   = len(ev_starts) - 1
    t0 = time.perf_counter()
    probs      = predict_flat(sn_model, data_raw, ev_starts.astype(np.int64),
                              batch_size=batch_size, device=sn_dev, normalize=True)
    logger.debug(f"    sn predict_flat: {time.perf_counter()-t0:.2f}s  ({n_events} events)")

    sig_mask   = probs > threshold
    n_hits_arr = (ev_starts[1:] - ev_starts[:-1]).astype(np.int32)

    n_sn_h, n_sn_s = _count_sig_hits_strings(
        sig_mask, channels, ev_starts[:-1].astype(np.int64), n_hits_arr, n_events,
    )
    cut_mask = (n_sn_h >= min_hits) & (n_sn_s >= min_strings)
    sel_ev   = np.where(cut_mask)[0]

    t0 = time.perf_counter()
    features_list = []
    for ev_i in sel_ev:
        s, e  = int(ev_starts[ev_i]), int(ev_starts[ev_i + 1])
        ev_sm = sig_mask[s:e]
        feats = data_raw[s:e][ev_sm]
        if with_probs:
            feats = np.column_stack([feats, probs[s:e][ev_sm]])
        features_list.append(feats)
    logger.debug(f"    feature build:   {time.perf_counter()-t0:.2f}s  ({len(sel_ev)}/{n_events} pass)")

    return sel_ev, n_sn_h[sel_ev], n_sn_s[sel_ev], features_list


def _infer_embeddings_batched(
    features_list: list[np.ndarray],
    model,
    norm_config: dict,
    batch_size: int,
    device: str,
    with_probs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, embeddings) via predict_scores_and_embeddings."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from inference_v2.shared.model_utils import predict_scores_and_embeddings

    if not features_list:
        d = model.classifier.in_features if hasattr(model.classifier, "in_features") else 128
        return np.empty(0, dtype=np.float32), np.empty((0, d), dtype=np.float32)

    t0 = time.perf_counter()
    scores, embs = predict_scores_and_embeddings(
        model, features_list, norm_config,
        batch_size=batch_size, device=device, feats_with_probs=with_probs,
    )
    logger.debug(f"    nu-classifier infer: {time.perf_counter()-t0:.2f}s  ({len(features_list)} events)")
    return scores.astype(np.float32), embs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  MC loader
# ─────────────────────────────────────────────────────────────────────────────

def load_mc_embeddings_from_h5(
    mc_h5: str,
    catalog: str,
    npy_dir: str,
    model,
    norm_config: dict,
    sn_model,
    sn_dev: str,
    threshold: float = 0.8,
    n_per_class: int = 10_000,
    min_hits: int = 5,
    min_strings: int = 0,
    batch_size: int = 512,
    device: str = "cpu",
    seed: int = 42,
    ptypes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Sample out-of-training MC events, apply sig-noise, return embeddings.

    Excludes all h5 parts that appear in the NPY training dataset.

    Returns DataFrame with columns: data_class, score, embedding (ndarray).
    """
    from inference_v2.shared.model_utils import load_model  # noqa: F401 (ensure path)

    npy_path = Path(npy_dir)
    part_keys_arr = np.load(npy_path / "h5_part_keys.npy", allow_pickle=True)
    training_parts: set[str] = set(np.unique(part_keys_arr).tolist())
    logger.info(f"Excluding {len(training_parts)} training parts from MC sampling")

    import json
    with open(npy_path / "dataset_info.json") as f:
        ds_info = json.load(f)
    with_probs = ds_info.get("with_probs", False)

    selected_ptypes = ptypes if ptypes is not None else MC_MERGED_PTYPES
    rng = np.random.default_rng(seed)

    all_rows: list[dict] = []

    cat_conn = duckdb.connect(catalog, read_only=True)

    _MAX_PARTS = min(max(n_per_class // 5, 10), 200)

    for ptype in selected_ptypes:
        logger.info(f"  Sampling {n_per_class} out-of-training events for {ptype}...")

        # Step 1: distinct parts for this ptype — fast single-table query, no h5_locations join.
        # For mc_merged: events.run == part_key, events.event_id == local_idx.
        df_parts = cat_conn.execute(
            "SELECT DISTINCT run AS part_key FROM events WHERE source='mc_merged' AND data_class=?",
            [ptype],
        ).df()
        available_parts = [p for p in df_parts["part_key"].tolist() if p not in training_parts]

        if not available_parts:
            logger.warning(f"  {ptype}: no out-of-training parts in catalog")
            continue

        # Step 2: randomly select at most _MAX_PARTS parts so each part carries enough
        # events to amortize one full-array h5 read (~0.17 s/part on local storage).
        n_parts = min(_MAX_PARTS, len(available_parts))
        sel_parts = rng.choice(available_parts, size=n_parts, replace=False).tolist()
        logger.info(f"  {ptype}: {len(available_parts)} available parts → using {n_parts}")

        # Step 3: reservoir-sample n_per_class events from selected parts only.
        # Scanning n_parts*~28k rows is ~100× faster than scanning all 576 M.
        sel_df = pd.DataFrame({"pk": sel_parts})
        cat_conn.register("_sel_parts", sel_df)
        rng_seed = int(rng.integers(0, 2**31))
        df_sample = cat_conn.execute(f"""
            SELECT event_fk, part_key, local_idx
            FROM (
                SELECT id AS event_fk, run AS part_key, event_id AS local_idx
                FROM events
                WHERE source    = 'mc_merged'
                  AND data_class = ?
                  AND run        IN (SELECT pk FROM _sel_parts)
            )
            USING SAMPLE {n_per_class} ROWS (reservoir, {rng_seed})
            ORDER BY part_key, local_idx
        """, [ptype]).df()
        cat_conn.execute("DROP VIEW IF EXISTS _sel_parts")

        if df_sample.empty:
            logger.warning(f"  {ptype}: no events sampled")
            continue

        n_unique_parts = df_sample["part_key"].nunique()
        logger.info(f"  {ptype}: sampled {len(df_sample):,} events from {n_unique_parts} parts")

        ptype_rows: list[dict] = []
        t_h5 = t_sn = t_infer = 0.0
        with h5py.File(mc_h5, "r") as h5:
            if ptype not in h5:
                logger.warning(f"  {ptype}: group not found in h5, skipping")
                continue
            grp = h5[ptype]["raw"]

            for part_key, grp_df in df_sample.groupby("part_key"):
                _t = time.perf_counter()
                ev_starts = grp[f"ev_starts/{part_key}/data"][:].astype(np.int64)
                data_raw  = grp[f"data/{part_key}/data"][:].astype(np.float32)
                channels  = grp[f"channels/{part_key}/data"][:].astype(np.int32)
                t_h5 += time.perf_counter() - _t

                local_idxs = grp_df["local_idx"].values
                event_fks  = grp_df["event_fk"].values

                mini_starts  = [0]
                mini_data    = []
                mini_chans   = []
                local_to_fk  = {}

                for pos, local_idx in enumerate(local_idxs):
                    s = int(ev_starts[local_idx])
                    e = int(ev_starts[local_idx + 1])
                    mini_data.append(data_raw[s:e])
                    mini_chans.append(channels[s:e])
                    mini_starts.append(mini_starts[-1] + (e - s))
                    local_to_fk[pos] = event_fks[pos]

                if not mini_data:
                    continue

                all_data  = np.concatenate(mini_data)
                all_chans = np.concatenate(mini_chans)
                ev_starts_mini = np.array(mini_starts, dtype=np.int64)

                _t = time.perf_counter()
                sel_ev, n_sn_h, n_sn_s, features_list = _apply_sn_and_build_features(
                    all_data, all_chans, ev_starts_mini,
                    sn_model=sn_model, sn_dev=sn_dev,
                    threshold=threshold, min_hits=min_hits, min_strings=min_strings,
                    batch_size=batch_size, with_probs=with_probs,
                )
                t_sn += time.perf_counter() - _t
                if not features_list:
                    continue

                _t = time.perf_counter()
                scores, embs = _infer_embeddings_batched(
                    features_list, model, norm_config, batch_size, device, with_probs,
                )
                t_infer += time.perf_counter() - _t
                for k, ev_pos in enumerate(sel_ev):
                    ptype_rows.append({
                        "data_class": ptype,
                        "event_fk":   int(local_to_fk[ev_pos]),
                        "score":      float(scores[k]),
                        "embedding":  embs[k],
                    })

        logger.info(
            f"  {ptype}: {len(ptype_rows):,} valid embeddings | "
            f"h5={t_h5:.1f}s  sn={t_sn:.1f}s  infer={t_infer:.1f}s"
        )
        all_rows.extend(ptype_rows)

    cat_conn.close()
    df = pd.DataFrame(all_rows)
    logger.info(f"MC loader done: {len(df):,} total events across {len(selected_ptypes)} classes")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Exp loader
# ─────────────────────────────────────────────────────────────────────────────

def load_exp_embeddings_from_preds(
    preds_db: str,
    catalog: str,
    exp_h5: str,
    model,
    norm_config: dict,
    sn_model,
    sn_dev: str,
    threshold: float = 0.8,
    n_high: int = 5_000,
    n_low: int = 5_000,
    min_hits: int = 5,
    min_strings: int = 0,
    batch_size: int = 512,
    device: str = "cpu",
    seed: int = 42,
    with_probs: bool = False,
) -> pd.DataFrame:
    """Sample exp events from predictions DB by score bucket, re-infer embeddings.

    Reads raw hits from exp_h5 via catalog h5_locations.
    Labels events as 'exp_high' (score > threshold) or 'exp_low' (score ≤ threshold).

    Returns DataFrame with columns: data_class, score, embedding (ndarray).
    """
    rng = np.random.default_rng(seed)

    preds_conn = duckdb.connect(preds_db, read_only=True)

    def _sample_fks(preds_conn: duckdb.DuckDBPyConnection, cond: str, n: int) -> np.ndarray:
        df = preds_conn.execute(
            f"SELECT event_fk FROM predictions WHERE {cond}"
        ).df()
        if df.empty:
            return np.array([], dtype=np.int64)
        idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
        return df["event_fk"].values[idx]

    fks_high = _sample_fks(preds_conn, f"score >  {threshold}", n_high)
    fks_low  = _sample_fks(preds_conn, f"score <= {threshold}", n_low)
    preds_conn.close()

    logger.info(f"Exp: sampled {len(fks_high):,} high-score + {len(fks_low):,} low-score events")

    bucket_map = {fk: "exp_high" for fk in fks_high}
    bucket_map.update({fk: "exp_low" for fk in fks_low})
    all_fks = np.concatenate([fks_high, fks_low]).astype(np.int64)

    if len(all_fks) == 0:
        return pd.DataFrame(columns=["data_class", "event_fk", "score", "embedding"])

    # Locate events in h5 via catalog
    cat_conn = duckdb.connect(catalog, read_only=True)
    fks_df = pd.DataFrame({"fk": all_fks})
    cat_conn.register("_fks", fks_df)
    df_locs = cat_conn.execute("""
        SELECT e.id AS event_fk, e.source AS src_key,
               h.part_key, h.local_idx
        FROM events e
        JOIN h5_locations h ON h.event_fk = e.id
        JOIN _fks f ON f.fk = e.id
    """).df()
    cat_conn.close()

    if df_locs.empty:
        logger.warning("No exp events found in catalog")
        return pd.DataFrame(columns=["data_class", "event_fk", "score", "embedding"])

    # Detect the h5 group name (exp or exp_reco)
    with h5py.File(exp_h5, "r") as h5:
        src_key = "exp_reco" if "exp_reco" in h5 else "exp"

    df_locs = df_locs.sort_values(["part_key", "local_idx"]).reset_index(drop=True)

    all_rows: list[dict] = []
    with h5py.File(exp_h5, "r") as h5:
        grp = h5[src_key]["raw"]
        has_header = f"header_prty" in h5[src_key]

        for part_key, grp_df in df_locs.groupby("part_key"):
            ev_starts = grp[f"ev_starts/{part_key}/data"][:].astype(np.int64)
            # Keep as h5py dataset — read per-event slices only
            data_ds  = grp[f"data/{part_key}/data"]
            chan_ds   = grp[f"channels/{part_key}/data"]

            local_idxs = grp_df["local_idx"].values
            event_fks  = grp_df["event_fk"].values

            mini_starts = [0]
            mini_data   = []
            mini_chans  = []

            for local_idx in local_idxs:
                s = int(ev_starts[local_idx])
                e = int(ev_starts[local_idx + 1])
                mini_data.append(data_ds[s:e].astype(np.float32))
                mini_chans.append(chan_ds[s:e].astype(np.int32))
                mini_starts.append(mini_starts[-1] + (e - s))

            if not mini_data:
                continue

            all_data  = np.concatenate(mini_data)
            all_chans = np.concatenate(mini_chans)
            ev_starts_mini = np.array(mini_starts, dtype=np.int64)

            sel_ev, _, _, features_list = _apply_sn_and_build_features(
                all_data, all_chans, ev_starts_mini,
                sn_model=sn_model, sn_dev=sn_dev,
                threshold=threshold, min_hits=min_hits, min_strings=min_strings,
                batch_size=batch_size, with_probs=with_probs,
            )
            if not features_list:
                continue

            scores, embs = _infer_embeddings_batched(
                features_list, model, norm_config, batch_size, device, with_probs,
            )
            for k, ev_pos in enumerate(sel_ev):
                fk = int(event_fks[ev_pos])
                all_rows.append({
                    "data_class": bucket_map.get(fk, "exp_low"),
                    "event_fk":   fk,
                    "score":      float(scores[k]),
                    "embedding":  embs[k],
                })

    df = pd.DataFrame(all_rows)
    logger.info(f"Exp loader done: {len(df):,} events "
                f"({(df['data_class']=='exp_high').sum():,} high, "
                f"{(df['data_class']=='exp_low').sum():,} low)")
    return df
