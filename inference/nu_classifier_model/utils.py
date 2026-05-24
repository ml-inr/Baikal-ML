"""Inference utilities for the prefilter model.

Functions for:
- Loading trained model from checkpoint
- Reading MC test parts (complement of training parts) from HDF5
- Reading exp / exp_reco data from HDF5
- Batched prediction with normalization
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

STRING_DIVISOR = 36

# ── Particle type encoding (matches prefilter_npy_ds_builder) ──────────
PARTICLE_ENCODE = {"muatm_2020": 0, "nue2_2020": 1, "nuatm_2020": 2}
PARTICLE_MAP = {"muatm": "muatm_2020", "nue2": "nue2_2020", "nuatm": "nuatm_2020"}

# ── reco_prty column name lists ────────────────────────────────────────
# exp_reco.h5 — 25 scalar columns (root2h5_config_exp_reco.yaml reco_ev order)
EXP_RECO_COL_NAMES: List[str] = [
    "thetaRec",        # 0  BRecoMuon.fThetaRec
    "phiRec",          # 1  BRecoMuon.fPhiRec
    "thetaErr",        # 2  BRecoMuon.fThetaErr
    "phiErr",          # 3  BRecoMuon.fPhiErr
    "funcValue",       # 4  BRecoMuon.fFuncValue
    "timeChi2",        # 5  BRecoMuon.fTimeChi2
    "chargeTerm",      # 6  BRecoMuon.fChargeTerm
    "LLFit",           # 7  BRecoMuon.fLLFit
    "nHits",           # 8  BRecoMuon.fNHits
    "nStrings",        # 9  BRecoMuon.fNStrings
    "nOMs",            # 10 BRecoMuon.fNOMs
    "pathLength",      # 11 BRecoMuon.fPathLength
    "timeXYZRec",      # 12 BRecoMuon.fTimeXYZRec
    "covMatrixStatus", # 13 BRecoMuon.fCovMatrixStatus
    "scfMaxTheta",     # 14 BRecoMuon.fScfMaxTheta
    "scfMinTheta",     # 15 BRecoMuon.fScfMinTheta
    "scfTheta",        # 16 BRecoMuon.fScfTheta
    "scfPhi",          # 17 BRecoMuon.fScfPhi
    "pHit",            # 18 BRecoMuon.fPHit
    "evCenterZ",       # 19 BRecoMuon.fEvCenterZ
    "zDist",           # 20 BRecoMuon.fZDist
    "nTriplets",       # 21 BRecoMuon.fNTriplets
    "nCalls",          # 22 BRecoMuon.fNCalls
    "classBDT",        # 23 BRecoMuon.fClassBDT
    "classBDTLowE",    # 24 BRecoMuon.fClassBDTLowE
]

# mc_reco.h5 (new format after config update) — 25 scalars (same as EXP_RECO_COL_NAMES)
# + 6 MC-specific vector components appended by reco_vectors
MC_RECO_COL_NAMES: List[str] = EXP_RECO_COL_NAMES + [
    "xyzRec_x",  # 25 BRecoMuon.fXYZRec[0]
    "xyzRec_y",  # 26 BRecoMuon.fXYZRec[1]
    "xyzRec_z",  # 27 BRecoMuon.fXYZRec[2]
    "dirRec_x",  # 28 BRecoMuon.fDirectionRec[0]
    "dirRec_y",  # 29 BRecoMuon.fDirectionRec[1]
    "dirRec_z",  # 30 BRecoMuon.fDirectionRec[2]
]

# Columns present in both exp_reco and mc_reco (first 25 of MC_RECO_COL_NAMES == EXP_RECO_COL_NAMES)
COMMON_RECO_COLS: List[str] = EXP_RECO_COL_NAMES


# ═══════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════

def load_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Load trained base model from a DA checkpoint.

    Returns:
        (model, normalization_config, train_config)
    """
    import sys, os
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.models.base_models import create_model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    norm_config = ckpt["normalization_config"]

    model = create_model(config["model"])
    model.load_state_dict(ckpt["base_model_state_dict"])

    # Normalize amp_clip from raw PE (stored in config) to model's normalized space.
    # The checkpoint stores amp_clip as raw PE (e.g. 100), but the model expects
    # a normalized value. Re-apply normalization here using the saved norm_config.
    raw_clip = config["model"].get("amp_clip")
    if raw_clip is not None and norm_config is not None:
        mean_amp = norm_config["means"][0]
        std_amp = norm_config["stds"][0]
        model.amp_clip = (raw_clip - mean_amp) / std_amp
        logger.info(
            f"amp_clip: Q={raw_clip} PE → normalized={model.amp_clip:.4f} "
            f"(mean={mean_amp}, std={std_amp})"
        )

    model.to(device)
    model.eval()

    logger.info(
        f"Loaded model from epoch {ckpt['epoch']} "
        f"(best_metric={ckpt.get('best_metric', '?'):.4f})"
    )
    return model, norm_config, config


# ═══════════════════════════════════════════════════════════════════════
#  HDF5 part-level data loading
# ═══════════════════════════════════════════════════════════════════════

def _compute_per_event_metadata(
    ev_starts: np.ndarray,
    labels_raw: np.ndarray,
    channels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute n_hits, n_signal_hits, n_signal_strings per event.

    Args:
        ev_starts: (n_events+1,) fence-post indices.
        labels_raw: (n_hits,) per-hit labels (!=0 means signal).
        channels: (n_hits,) per-hit channel ids.

    Returns:
        (n_hits, n_signal_hits, n_signal_strings) each (n_events,).
    """
    n_events = len(ev_starts) - 1
    hit_start = ev_starts[:-1].astype(np.int64)
    hit_end = ev_starts[1:].astype(np.int64)
    n_hits = (hit_end - hit_start).astype(np.int32)

    signal_mask = labels_raw != 0
    n_signal = np.add.reduceat(
        signal_mask.astype(np.int32), hit_start.astype(np.intp)
    )

    # Unique signal strings per event
    if signal_mask.any():
        string_ids = np.where(signal_mask, channels // STRING_DIVISOR, -1)
        event_idx = np.repeat(np.arange(n_events), n_hits)
        sig_events = event_idx[signal_mask]
        sig_strings = string_ids[signal_mask]
        combined = sig_events.astype(np.int64) * 1000 + sig_strings.astype(np.int64)
        sort_idx = combined.argsort()
        combined_sorted = combined[sort_idx]
        unique_mask = np.empty(len(combined_sorted), dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = combined_sorted[1:] != combined_sorted[:-1]
        unique_events = sig_events[sort_idx[unique_mask]]
        n_unique_signal_strings = np.bincount(
            unique_events, minlength=n_events
        ).astype(np.int32)
    else:
        n_unique_signal_strings = np.zeros(n_events, dtype=np.int32)

    return n_hits, n_signal.astype(np.int32), n_unique_signal_strings


def _count_signal_hits_strings(
    ev_starts: np.ndarray,
    signal_mask: np.ndarray,
    channels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized per-event n_signal_hits and n_signal_strings.

    Args:
        ev_starts: (n_events+1,) fence-post indices.
        signal_mask: (n_hits,) boolean.
        channels: (n_hits,) int per-hit channel IDs.

    Returns:
        (n_signal_hits, n_signal_strings) each (n_events,) int32.
    """
    n_events = len(ev_starts) - 1
    starts = ev_starts[:-1].astype(np.intp)
    ends   = ev_starts[1:].astype(np.int64)

    n_sig_hits = np.add.reduceat(signal_mask.astype(np.int32), starts)

    if signal_mask.any():
        string_ids = channels // STRING_DIVISOR
        event_idx  = np.repeat(np.arange(n_events), ends - ev_starts[:-1].astype(np.int64))
        sig_ev  = event_idx[signal_mask]
        sig_str = string_ids[signal_mask]
        combined = sig_ev.astype(np.int64) * 1000 + sig_str.astype(np.int64)
        sort_idx = combined.argsort()
        combined_sorted = combined[sort_idx]
        unique_mask = np.empty(len(combined_sorted), dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = combined_sorted[1:] != combined_sorted[:-1]
        n_sig_strings = np.bincount(
            sig_ev[sort_idx[unique_mask]], minlength=n_events
        ).astype(np.int32)
    else:
        n_sig_strings = np.zeros(n_events, dtype=np.int32)

    return n_sig_hits.astype(np.int32), n_sig_strings


def load_mc_test_parts(
    h5_path: str,
    train_parts_json: str,
    n_parts_per_particle: Dict[str, int],
    seed: int = 42,
) -> pd.DataFrame:
    """Load MC events from parts NOT used in training.

    Args:
        h5_path: Path to baikal_mc_merged.h5.
        train_parts_json: Path to prefilter_h5_parts.json (training parts).
        n_parts_per_particle: e.g. {"muatm": 50, "nue2": 10, "nuatm": 10}.
        seed: Random seed for part selection.

    Returns:
        DataFrame with columns:
            particle_type, n_hits, n_signal_hits, n_signal_strings,
            features (list of np.ndarray per event),
            [theta, phi, energy if prime_prty present in HDF5].
    """
    with open(train_parts_json) as f:
        train_parts = json.load(f)

    rng = np.random.RandomState(seed)
    rows: List[Dict[str, Any]] = []

    with h5py.File(h5_path, "r") as h5:
        for ptype_short, n_parts in n_parts_per_particle.items():
            ptype_full = PARTICLE_MAP[ptype_short]
            grp = h5[ptype_full]

            # All available parts
            all_parts = sorted(
                int(k.split("_")[1])
                for k in grp["raw"]["data"].keys()
                if k.startswith("part_")
            )
            # Complement: exclude training parts
            train_set = set(train_parts.get(ptype_short, []))
            complement = [p for p in all_parts if p not in train_set]

            # Randomly select n_parts
            if n_parts < len(complement):
                selected = sorted(rng.choice(complement, size=n_parts, replace=False))
            else:
                selected = complement
                logger.warning(
                    f"{ptype_short}: requested {n_parts} test parts, "
                    f"only {len(complement)} available"
                )

            logger.info(
                f"Loading {len(selected)} test parts for {ptype_full} "
                f"(complement of {len(train_set)} training parts)"
            )

            has_prime = "prime_prty" in grp

            for part_num in tqdm(selected, desc=f"MC {ptype_short}", unit="part"):
                part_name = f"part_{part_num}"
                data = grp[f"raw/data/{part_name}/data"][:]
                ev_starts = grp[f"raw/ev_starts/{part_name}/data"][:]
                labels_raw = grp[f"raw/labels/{part_name}/data"][:]
                channels = grp[f"raw/channels/{part_name}/data"][:]
                # prime_prty columns: [theta, phi, energy] per event
                prime_prty = grp[f"prime_prty/{part_name}/data"][:] if has_prime else None

                n_events = len(ev_starts) - 1
                if n_events == 0:
                    continue

                n_hits, n_sig, n_strings = _compute_per_event_metadata(
                    ev_starts, labels_raw, channels
                )

                for i in range(n_events):
                    s, e = int(ev_starts[i]), int(ev_starts[i + 1])
                    row = {
                        "local_id": i,
                        "h5_part_num": part_num,
                        "particle_type": ptype_full,
                        "n_hits": int(n_hits[i]),
                        "n_signal_hits": int(n_sig[i]),
                        "n_signal_strings": int(n_strings[i]),
                        "features": data[s:e],
                        "signal_mask": (labels_raw[s:e] != 0),
                    }
                    if prime_prty is not None:
                        row["theta"] = float(prime_prty[i, 0])
                        row["phi"] = float(prime_prty[i, 1])
                        row["energy"] = float(prime_prty[i, 2])
                    rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"MC test set: {len(df):,} events loaded")
    return df


def list_h5_parts(
    h5_path: str,
    group_name: str = "exp",
    ):
    with h5py.File(h5_path, "r") as h5:
        grp = h5[group_name]
        parts = sorted(k for k in grp["raw"]["data"].keys() if k.startswith("part_"))
    return parts

def load_exp_parts(
    h5_path: str,
    group_name: str = "exp",
    max_events: Optional[int] = None,
    max_parts: Optional[int] = None,
    seed: int = 42,
    load_reco: bool = False,
    parts_to_load: list[str] | None = None
) -> pd.DataFrame:
    """Load events from exp.h5 or exp_reco.h5.

    Args:
        h5_path: Path to HDF5 file.
        group_name: Top-level group ("exp" or "exp_reco").
        max_events: Cap total events by subsampling after loading (None = all).
        max_parts: Cap number of HDF5 parts to read — fast early exit (None = all).
        seed: Seed for subsampling if max_events is set.
        load_reco: If True, also read reco_prty (per-event scalars) and
                   reco_hit_prty (per-hit values, e.g. fTRes) columns.

    Returns:
        DataFrame with columns:
            n_hits, features,
            [signal_mask if raw/labels present],
            [reco_0..reco_N if load_reco and reco_prty present],
            [reco_hit_prty (list of (n_hits, n_fields) arrays) if load_reco
             and reco_hit_prty present].
    """
    # Collect columns as lists — much faster than list-of-dicts for pandas
    col_h5_part:            List[str]        = []
    col_local_id:           List[int]        = []
    col_n_hits:             List[int]        = []
    col_features:           List[np.ndarray] = []
    col_max_q:              List[float]      = []
    col_mean_z:             List[float]      = []
    col_signal:             List[np.ndarray] = []   # only if has_labels
    col_reco_hit:           List[np.ndarray] = []   # only if has_reco_hit
    col_n_signal_hits_reco: List[int]        = []   # only if has_reco_hit and has_channels
    col_n_signal_strs_reco: List[int]        = []
    col_season:             List[int]        = []
    col_cluster:            List[int]        = []
    col_run:                List[int]        = []
    col_event_id:           List[int]        = []
    reco_rows:              List[np.ndarray] = []

    with h5py.File(h5_path, "r") as h5:
        grp = h5[group_name]
        if parts_to_load is None:
            parts = sorted(k for k in grp["raw"]["data"].keys() if k.startswith("part_"))
        else:
            parts = parts_to_load
        if max_parts is not None:
            parts = parts[:max_parts]

        has_reco     = load_reco and "reco_prty" in grp
        has_reco_hit = load_reco and "reco_hit_prty" in grp
        has_labels   = "labels" in grp["raw"]
        has_header   = "header_prty" in grp
        has_channels = "channels" in grp["raw"]

        for part in tqdm(parts, desc=f"Exp {group_name}", unit="part"):
            data      = grp[f"raw/data/{part}/data"][:]
            ev_starts = grp[f"raw/ev_starts/{part}/data"][:]
            n_events  = len(ev_starts) - 1

            labels_raw    = grp[f"raw/labels/{part}/data"][:]    if has_labels   else None
            reco_data     = grp[f"reco_prty/{part}/data"][:]     if has_reco     else None
            reco_hit_data = grp[f"reco_hit_prty/{part}/data"][:] if has_reco_hit else None
            header_data   = grp[f"header_prty/{part}/data"][:]   if has_header   else None
            channels      = grp[f"raw/channels/{part}/data"][:]  if has_channels else None

            # Per-event n_hits via vectorised diff
            starts = ev_starts[:-1].astype(np.int64)
            ends   = ev_starts[1:].astype(np.int64)
            valid  = ends > starts

            v_idx    = np.where(valid)[0]
            v_starts = starts[v_idx]
            v_ends   = ends[v_idx]
            n_valid  = len(v_idx)

            # Scalars — fully vectorized, no Python loop
            col_h5_part.extend([part] * n_valid)
            col_local_id.extend(v_idx.tolist())
            col_n_hits.extend((v_ends - v_starts).tolist())

            # max_q and mean_z — reduceat over flat data buffer
            intp_starts = v_starts.astype(np.intp)
            col_max_q.extend(np.maximum.reduceat(data[:, 0], intp_starts).tolist())
            col_mean_z.extend(
                (np.add.reduceat(data[:, 4], intp_starts) / (v_ends - v_starts)).tolist()
            )

            # Variable-length array slices — list comprehension (faster than append loop)
            col_features.extend([data[s:e] for s, e in zip(v_starts, v_ends)])
            if has_labels and labels_raw is not None:
                col_signal.extend([labels_raw[s:e] != 0 for s, e in zip(v_starts, v_ends)])
            if has_reco_hit and reco_hit_data is not None:
                col_reco_hit.extend([reco_hit_data[s:e] for s, e in zip(v_starts, v_ends)])
                # Reco signal counts: hits where reco_hit_prty[:,0] != 0
                if channels is not None:
                    reco_sig_mask = reco_hit_data[:, 0] != 0
                    n_sig_hits, n_sig_strs = _count_signal_hits_strings(
                        ev_starts, reco_sig_mask, channels
                    )
                    col_n_signal_hits_reco.extend(n_sig_hits[v_idx].tolist())
                    col_n_signal_strs_reco.extend(n_sig_strs[v_idx].tolist())

            # Header scalars — numpy fancy indexing
            if has_header and header_data is not None:
                col_season.extend(header_data[v_idx, 0].tolist())
                col_cluster.extend(header_data[v_idx, 1].tolist())
                col_run.extend(header_data[v_idx, 2].tolist())
                col_event_id.extend(header_data[v_idx, 3].tolist())

            # Reco scalars — numpy fancy indexing (no per-row append)
            if has_reco and reco_data is not None:
                reco_rows.extend(reco_data[v_idx])

    col_dict: Dict[str, Any] = {
        "h5_part_str": col_h5_part,
        "local_id":    col_local_id,
        "n_hits":      col_n_hits,
        "max_q":       col_max_q,
        "mean_z":      col_mean_z,
        "features":    col_features,
    }
    if has_labels:
        col_dict["signal_mask"] = col_signal
    if has_reco_hit:
        col_dict["reco_hit_prty"] = col_reco_hit
        if col_n_signal_hits_reco:
            col_dict["n_signal_hits_reco"]    = col_n_signal_hits_reco
            col_dict["n_signal_strings_reco"] = col_n_signal_strs_reco
    if has_header:
        col_dict["season"]      = col_season
        col_dict["cluster"]     = col_cluster
        col_dict["run"]         = col_run
        col_dict["event_id_cc"] = col_event_id

    df = pd.DataFrame(col_dict)

    if has_reco and reco_rows:
        reco_arr = np.stack(reco_rows)
        names = EXP_RECO_COL_NAMES if len(EXP_RECO_COL_NAMES) == reco_arr.shape[1] else None
        for col_idx in range(reco_arr.shape[1]):
            col_name = names[col_idx] if names else f"reco_{col_idx}"
            df[col_name] = reco_arr[:, col_idx]

    if max_events is not None and max_events < len(df):
        rng = np.random.RandomState(seed)
        df = df.sample(n=max_events, random_state=rng).reset_index(drop=True)

    logger.info(f"Loaded {len(df):,} events from {h5_path} ({group_name})")
    return df


def load_mc_reco_parts(
    h5_path: str,
    particle_types: List[str] = ("muatm", "nuatm_conv", "nuatm_prompt"),
    max_parts_per_particle: Optional[int] = None,
    parts_to_load: Optional[List[str]] = None,
    max_events: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load events from baikal_mc_reco.h5 into a DataFrame.

    The resulting DataFrame has named reco columns matching MC_RECO_COL_NAMES:
    first 25 columns are identical to EXP_RECO_COL_NAMES (matchable with exp_reco),
    columns 25-30 are MC-specific 3D vectors (xyzRec_x/y/z, dirRec_x/y/z).

    Args:
        h5_path: Path to baikal_mc_reco.h5.
        particle_types: Which top-level groups to load (e.g. ["muatm", "nuatm_conv"]).
        max_parts_per_particle: Cap number of parts read per particle (None = all).
        max_events: Cap total events after loading by random subsampling (None = all).
        seed: Random seed for max_events subsampling.

    Returns:
        DataFrame with columns:
            particle_type, h5_part_str, local_id, n_hits,
            features (np.ndarray per event),
            signal_mask (bool array per event),
            theta_mc, phi_mc, energy_mc, nucleon_n, response_muons_n, event_weight,
            [thetaRec, phiRec, ..., classBDTLowE] — 25 common reco columns,
            [xyzRec_x, xyzRec_y, xyzRec_z, dirRec_x, dirRec_y, dirRec_z] — if present.
    """
    col_particle:         List[str]        = []
    col_h5_part:          List[str]        = []
    col_local_id:         List[int]        = []
    col_n_hits:           List[int]        = []
    col_features:         List[np.ndarray] = []
    col_max_q:            List[float]      = []
    col_mean_z:           List[float]      = []
    col_signal:           List[np.ndarray] = []
    col_n_signal_hits_gt: List[int]        = []
    col_n_signal_strs_gt: List[int]        = []
    col_theta_mc:         List[float]      = []
    col_phi_mc:           List[float]      = []
    col_energy_mc:        List[float]      = []
    col_nucleon_n:        List[float]      = []
    col_resp_mu:          List[float]      = []
    col_ev_weight:        List[float]      = []
    reco_rows:            List[np.ndarray] = []

    with h5py.File(h5_path, "r") as h5:
        for ptype in particle_types:
            if ptype not in h5:
                logger.warning(f"Particle type '{ptype}' not found in {h5_path}, skipping")
                continue

            grp = h5[ptype]
            if parts_to_load is not None:
                parts = parts_to_load
            else:
                parts = sorted(grp["raw"]["data"].keys())
                if max_parts_per_particle is not None:
                    parts = parts[:max_parts_per_particle]

            has_prime    = "prime_prty" in grp
            has_reco     = "reco_prty" in grp
            has_channels = "channels" in grp["raw"]

            logger.info(f"Loading {len(parts)} parts for '{ptype}'")

            for part in tqdm(parts, desc=f"MC-reco {ptype}", unit="part"):
                data       = grp[f"raw/data/{part}/data"][:]
                ev_starts  = grp[f"raw/ev_starts/{part}/data"][:]
                labels_raw = grp[f"raw/labels/{part}/data"][:]
                channels   = grp[f"raw/channels/{part}/data"][:] if has_channels else None
                n_events   = len(ev_starts) - 1
                if n_events == 0:
                    continue

                prime_data = grp[f"prime_prty/{part}/data"][:] if has_prime else None
                reco_data  = grp[f"reco_prty/{part}/data"][:]  if has_reco  else None

                starts = ev_starts[:-1].astype(np.int64)
                ends   = ev_starts[1:].astype(np.int64)
                valid  = ends > starts
                v_idx    = np.where(valid)[0]
                v_starts = starts[v_idx]
                v_ends   = ends[v_idx]
                n_valid  = len(v_idx)

                # GT signal counts — vectorized over full part, indexed by v_idx
                if channels is not None:
                    n_sig_hits, n_sig_strs = _count_signal_hits_strings(
                        ev_starts, labels_raw != 0, channels
                    )
                    col_n_signal_hits_gt.extend(n_sig_hits[v_idx].tolist())
                    col_n_signal_strs_gt.extend(n_sig_strs[v_idx].tolist())

                col_particle.extend([ptype] * n_valid)
                col_h5_part.extend([part] * n_valid)
                col_local_id.extend(v_idx.tolist())
                col_n_hits.extend((v_ends - v_starts).tolist())

                intp_starts = v_starts.astype(np.intp)
                col_max_q.extend(np.maximum.reduceat(data[:, 0], intp_starts).tolist())
                col_mean_z.extend(
                    (np.add.reduceat(data[:, 4], intp_starts) / (v_ends - v_starts)).tolist()
                )

                col_features.extend([data[s:e] for s, e in zip(v_starts, v_ends)])
                col_signal.extend([labels_raw[s:e] != 0 for s, e in zip(v_starts, v_ends)])

                if prime_data is not None:
                    col_theta_mc.extend(prime_data[v_idx, 0].tolist())
                    col_phi_mc.extend(prime_data[v_idx, 1].tolist())
                    col_energy_mc.extend(prime_data[v_idx, 2].tolist())
                    col_nucleon_n.extend(prime_data[v_idx, 3].tolist())
                    col_resp_mu.extend(prime_data[v_idx, 4].tolist())
                    col_ev_weight.extend(prime_data[v_idx, 5].tolist())
                if reco_data is not None:
                    reco_rows.extend(reco_data[v_idx])

    col_dict: Dict[str, Any] = {
        "particle_type": col_particle,
        "h5_part_str":   col_h5_part,
        "local_id":      col_local_id,
        "n_hits":        col_n_hits,
        "max_q":         col_max_q,
        "mean_z":        col_mean_z,
        "features":      col_features,
        "signal_mask":   col_signal,
    }
    if col_n_signal_hits_gt:
        col_dict["n_signal_hits_gt"]    = col_n_signal_hits_gt
        col_dict["n_signal_strings_gt"] = col_n_signal_strs_gt
    if col_theta_mc:
        col_dict.update({
            "theta_mc":      col_theta_mc,
            "phi_mc":        col_phi_mc,
            "energy_mc":     col_energy_mc,
            "nucleon_n":     col_nucleon_n,
            "response_muons_n": col_resp_mu,
            "event_weight":  col_ev_weight,
        })

    df = pd.DataFrame(col_dict)

    if reco_rows:
        reco_arr = np.stack(reco_rows)
        n_cols = reco_arr.shape[1]
        for col_idx in range(n_cols):
            col_name = MC_RECO_COL_NAMES[col_idx] if col_idx < len(MC_RECO_COL_NAMES) else f"reco_{col_idx}"
            df[col_name] = reco_arr[:, col_idx]

    if max_events is not None and max_events < len(df):
        rng = np.random.RandomState(seed)
        df = df.sample(n=max_events, random_state=rng).reset_index(drop=True)

    logger.info(f"MC-reco: {len(df):,} events from {h5_path} ({', '.join(particle_types)})")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Batched prediction
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_scores(
    model: nn.Module,
    features_list: List[np.ndarray],
    normalization_config: Dict[str, Any],
    batch_size: int = 512,
    max_hits: Optional[int] = 500,
    device: str = "cpu",
    feats_with_probs: bool = False,
    with_tqdm: bool = True
) -> np.ndarray:
    """Run batched inference and return sigmoid scores.

    Args:
        model: Trained base model (eval mode).
        features_list: List of (n_hits, 5) arrays per event.
        normalization_config: {"means": [...], "stds": [...]}.
        batch_size: Inference batch size.
        max_hits: Truncate events longer than this.
        device: torch device string.

    Returns:
        (n_events,) float32 array of sigmoid probabilities.
    """
    if feats_with_probs:
        normalization_config = {
            "means": normalization_config["means"] + [0.75],
            "stds":  normalization_config["stds"]  + [0.25],
        }
        features_num = 6
    else:
        features_num = 5

    means = torch.tensor(
        normalization_config["means"], dtype=torch.float32, device=device
    )
    stds = torch.tensor(
        normalization_config["stds"], dtype=torch.float32, device=device
    )

    all_scores: List[np.ndarray] = []
    n = len(features_list)

    if with_tqdm:
        starts = tqdm(range(0, n, batch_size), desc="Predicting", unit="batch", total=(n + batch_size - 1) // batch_size)
    else:
        starts = range(0, n, batch_size)
    for start in starts:
        batch_feats = features_list[start : start + batch_size]
        b = len(batch_feats)

        # Truncate + compute lengths
        lengths = []
        truncated = []
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

        # Pad
        padded = torch.zeros(b, max_len, features_num, dtype=torch.float32, device=device)
        for i, feat in enumerate(truncated):
            seq_len = len(feat)
            padded[i, :seq_len] = torch.from_numpy(feat)

        # Mask
        mask = torch.arange(max_len, device=device)[None, :] < lengths_t[:, None]

        # Normalize
        padded = torch.where(
            mask.unsqueeze(-1),
            (padded - means) / (stds + 1e-8),
            padded,
        )

        # Forward
        logits = model({"features": padded, "lengths": lengths_t, "mask": mask})
        scores = torch.sigmoid(logits).cpu().numpy().flatten()
        all_scores.append(scores)

    return np.concatenate(all_scores)


# ═══════════════════════════════════════════════════════════════════════
#  Event visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_event_3d(
    features: np.ndarray,
    signal_mask: Optional[np.ndarray] = None,
    title: str = "Event 3D view",
):
    """Plot a single event in 3D.

    Each hit is a sphere at (x, y, z) with size proportional to log10(amplitude).
    Noise hits are grey and semi-transparent.
    Signal hits are colored red→blue by time (early=red, late=blue).

    Args:
        features: (n_hits, 5) array — [amplitude, time, x, y, z].
        signal_mask: (n_hits,) bool array. None = all noise.
        title: Plot title.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    amp   = features[:, 0]
    time  = features[:, 1]
    x     = features[:, 2]
    y     = features[:, 3]
    z     = features[:, 4]

    if signal_mask is None:
        signal_mask = np.zeros(len(features), dtype=bool)

    noise_mask = ~signal_mask

    # Marker sizes: proportional to log10(amp), clipped to reasonable range
    sizes = np.clip(np.log10(np.maximum(amp, 1e-3)) + 3, 0.5, 8) ** 2 * 10

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Noise hits — grey, transparent
    if noise_mask.any():
        ax.scatter(
            x[noise_mask], y[noise_mask], z[noise_mask],
            s=sizes[noise_mask],
            c="grey", alpha=0.25, linewidths=0,
            label=f"Noise ({noise_mask.sum()})",
        )

    # Signal hits — red→blue by time
    if signal_mask.any():
        t_sig = time[signal_mask]
        t_min, t_max = t_sig.min(), t_sig.max()
        t_norm = (t_sig - t_min) / (t_max - t_min + 1e-8)

        sc = ax.scatter(
            x[signal_mask], y[signal_mask], z[signal_mask],
            s=sizes[signal_mask],
            c=t_norm, cmap="coolwarm_r", alpha=0.9, linewidths=0,
            label=f"Signal ({signal_mask.sum()})",
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label("time (red=early, blue=late)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
#  Blind reco analysis loading
# ═══════════════════════════════════════════════════════════════════════

# Rename map: ROOT branch name → DataFrame column name
_BLIND_RECO_RENAME = {
    "fSeason":  "season",
    "fCluster": "cluster",
    "fRun":     "run",
    "fEvent":   "event_id_cc",
}

def load_blind_reco(
    blind_reco_dir: str,
    prelcuts_only: bool = False,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """Load murecAnalysis blind reco ROOT files into a DataFrame.

    Each file contains a flat ``trAnalysis`` TTree with per-event analysis
    scalars (BDT scores, energy estimates, angles, cut flags, …).

    The four identifier columns are renamed to match ``load_exp_parts`` output:
    ``fSeason→season``, ``fCluster→cluster``, ``fRun→run``, ``fEvent→event_id_cc``.
    Use these to join against a DataFrame produced by ``load_exp_parts``.

    Args:
        blind_reco_dir: Directory containing ``murecAnalysis_blind_*.root`` files.
        prelcuts_only:  If True, load only the ``*_prelCuts_*`` files (small
                        subset that passed preliminary cuts).  Default is False —
                        loads the full files (all events, ``fPrelCutsPassed``
                        column indicates which ones passed).
        max_files:      Cap the number of ROOT files to load (sorted order).
                        None = load all files.

    Returns:
        DataFrame with columns season, cluster, run, event_id_cc and all
        analysis branches from the trAnalysis tree.
    """
    try:
        import uproot as ur
    except ImportError as exc:
        raise ImportError("uproot is required: pip install uproot") from exc

    blind_dir = Path(blind_reco_dir)
    all_files = sorted(blind_dir.glob("murecAnalysis_blind_*.root"))

    # Filter by variant
    if prelcuts_only:
        files = [f for f in all_files if "prelCuts" in f.name]
    else:
        files = [f for f in all_files if "prelCuts" not in f.name]

    if not files:
        raise FileNotFoundError(
            f"No blind reco ROOT files found in {blind_reco_dir} "
            f"(prelcuts_only={prelcuts_only})"
        )

    if max_files is not None:
        files = files[:max_files]

    dfs: List[pd.DataFrame] = []
    n_skipped = 0
    for fpath in tqdm(files, desc="Loading blind reco", unit="file"):
        try:
            with ur.open(str(fpath)) as rf:
                tree = rf["trAnalysis"]
                if tree.num_entries == 0:
                    continue
                chunk = tree.arrays(library="pd")
                dfs.append(chunk)
        except Exception as e:
            logger.warning(f"Skipping corrupt file {fpath.name}: {e}")
            n_skipped += 1
    if n_skipped:
        logger.warning(f"Skipped {n_skipped} corrupt/unreadable files")

    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns=_BLIND_RECO_RENAME, inplace=True)
    logger.info(
        f"Loaded {len(df):,} blind reco events from {len(files)} files "
        f"({blind_reco_dir})"
    )
    return df



# OTHER
def load_npy_exp(npy_dir: Path, max_events: int = None, seed: int = 42, feats_with_probs: bool = False) -> pd.DataFrame:
    """Load exp events from nu_classifier exp NPY dataset into a DataFrame."""
    features_mmap  = np.load(npy_dir / "exp_features.npy",      mmap_mode="r")
    probs_mmap = np.load(npy_dir / "exp_probs.npy",      mmap_mode="r")
    offsets        = np.load(npy_dir / "exp_offsets.npy")
    channels       = np.load(npy_dir / "exp_channels.npy", mmap_mode='r')
    n_sig_hits     = np.load(npy_dir / "exp_n_sig_hits.npy")
    n_sig_strings  = np.load(npy_dir / "exp_n_sig_strings.npy")
    part_names          = np.load(npy_dir / "exp_part_names.npy")

    n_total = len(n_sig_hits)
    rng = np.random.RandomState(seed)
    if max_events is not None and max_events < n_total:
        idx = np.sort(rng.choice(n_total, size=max_events, replace=False))
    else:
        idx = np.arange(n_total)

    if feats_with_probs:
        rows = {
            "part_names": part_names[idx].tolist(),
            "n_sig_hits":    n_sig_hits[idx].tolist(),
            "n_sig_strings": n_sig_strings[idx].tolist(),
            "channels":      [np.array(channels[offsets[i]:offsets[i+1]]) for i in idx],
            "features":      [np.concat([
                np.array(features_mmap[offsets[i]:offsets[i+1]]),
                np.array(probs_mmap[offsets[i]:offsets[i+1], None]),
            ], axis=1) for i in idx],
            "probs": [np.array(probs_mmap[offsets[i]:offsets[i+1]]) for i in idx],
        }
    else:
        rows = {
            "part_names": part_names[idx].tolist(),
            "n_sig_hits":    n_sig_hits[idx].tolist(),
            "n_sig_strings": n_sig_strings[idx].tolist(),
            "channels":      [np.array(channels[offsets[i]:offsets[i+1]]) for i in idx],
            "features":      [np.array(features_mmap[offsets[i]:offsets[i+1]]) for i in idx],
            "probs": [np.array(probs_mmap[offsets[i]:offsets[i+1]]) for i in idx],
        }
    return pd.DataFrame(rows)


def load_raw_hits_for_npy_exp(
    df: pd.DataFrame,
    h5_path: str,
) -> pd.DataFrame:
    """Fetch full raw hit data from exp.h5 for each event in df.

    Expects df to have a 'part_names' column with entries formatted as
    "{part_key}_{event_idx}" (e.g. "part_s2020_c01_r0027_42").

    Adds three columns in-place and returns the df:
        raw_channels  — (n_hits,) int32  — OM channel IDs, all hits
        raw_times     — (n_hits,) float32 — hit times, all hits
        raw_charges   — (n_hits,) float32 — hit amplitudes, all hits
    """
    import h5py

    # Parse part_key and event_idx from each part_name
    def _parse(name: str):
        part_key, idx_str = name.rsplit("_", 1)
        return part_key, int(idx_str)

    parsed = [_parse(n) for n in df["part_names"]]

    # Group row positions by part_key
    from collections import defaultdict
    part_to_rows: dict = defaultdict(list)   # part_key -> [(row_pos, event_idx), ...]
    for row_pos, (part_key, event_idx) in enumerate(parsed):
        part_to_rows[part_key].append((row_pos, event_idx))

    raw_channels = [None] * len(df)
    raw_times    = [None] * len(df)
    raw_charges  = [None] * len(df)

    with h5py.File(h5_path, "r") as h5:
        exp_grp = h5["exp"]
        for part_key, row_events in part_to_rows.items():
            ev_starts = exp_grp[f"raw/ev_starts/{part_key}/data"][:].astype(np.int64)
            data_raw  = exp_grp[f"raw/data/{part_key}/data"][:]       # (total_hits, 5)
            channels  = exp_grp[f"raw/channels/{part_key}/data"][:]   # (total_hits,)

            for row_pos, event_idx in row_events:
                s = int(ev_starts[event_idx])
                e = int(ev_starts[event_idx + 1])
                raw_channels[row_pos] = channels[s:e].astype(np.int32)
                raw_times[row_pos]    = data_raw[s:e, 1].astype(np.float32)
                raw_charges[row_pos]  = data_raw[s:e, 0].astype(np.float32)

    df = df.copy()
    df["raw_channels"] = raw_channels
    df["raw_times"]    = raw_times
    df["raw_charges"]  = raw_charges
    return df