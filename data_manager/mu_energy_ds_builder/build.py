"""Build the muon-energy NPY dataset from the main HDF5 and its energy-truth companion.

The task is a per-row regression: the target is the muon energy at entry into the
sensitive volume of the row's own cluster (the companion's column 1: ``r <= 90 m``,
``|z| <= 295 m``; born-inside muons use their energy at birth).  Only rows with exactly
one muon track in the cluster are kept.

This builder is deliberately self-contained (see ``io.py``).  It does not import the
nu-classifier builder: the two datasets must be free to diverge.

Pipeline
--------
1. Pass 1: obtain the per-event h8s3 counts and read the per-row target/status/track
   multiplicity from the companion plus theta, sign and the MC event weight from the main
   HDF5.  With ``sig_noise_source: probs_h5`` the counts are read from the precomputed SN
   probs file (no model, no GPU); with ``model`` they are computed by running the model.
2. Select: ``n_sig_hits >= min_hits`` AND ``n_sig_strings >= min_strings`` AND exactly one
   track in the cluster AND a finite target with a valid status.  For nuatm, rows whose
   muon was born inside the target volume are excluded (the vertex-cascade bug).
3. Flat selection (``flat_selection`` in the config): choose the train+val population by
   flattening ``log10(E)`` with the same algorithm as ``select_flat.flat_select_indices``.
   Only this population is written. The test complement stays in the full h8s3 source and
   is **not stored in the training dataset**; test predictions read the full source and
   exclude this dataset's ``event_fk`` values.
4. Pass 2: write the SN-filtered hits for the flat train+val population only.
5. Attach ``event_fk`` and write the per-event metadata and flat weights.

Usage:
    python -m data_manager.mu_energy_ds_builder --config <config.yaml>
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import yaml

from .io import (
    EventLocation,
    read_part_metadata,
    read_part_metadata_from_probs,
    write_selected_features,
    write_selected_features_from_probs,
)
from .select_flat import flat_select_indices

logger = logging.getLogger(__name__)

PARTICLE_MAP = {
    "muatm": "muatm_2020",
    "nuatm": "nuatm_2020",
    "nue2": "nue2_2020",
}
PARTICLE_ENCODE = {
    "muatm_2020": 0,
    "nuatm_2020": 1,
    "nue2_2020": 2,
}
NEUTRINO_TYPES = {"nuatm_2020", "nue2_2020"}

DEFAULT_H5_PATH = "data_manager/data/h5datasets/baikal_mc_merged.h5"
DEFAULT_COMPANION_PATH = (
    "data_manager/data/h5datasets/baikal_mc_merged_energy_truth.h5"
)
DEFAULT_CATALOG_PATH = "data_manager/catalog_v2.duckdb"
DEFAULT_PROBS_PATH = (
    "data_manager/data/h5datasets/"
    "baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
)


def _load_parts(path: str) -> Dict[str, List[int]]:
    with open(path) as f:
        parts = yaml.safe_load(f)
    missing = set(parts) - set(PARTICLE_MAP)
    if missing:
        raise ValueError(f"unknown particle keys in {path}: {sorted(missing)}")
    return parts


def _read_targets(
    companion: h5py.File,
    ptype_full: str,
    part_key: str,
    n_events: int,
    column: int,
) -> Dict[str, np.ndarray]:
    """Per-event target, status and track multiplicity for one part.

    The companion stores one row per muon.  A row of the main HDF5 is a cluster-level
    event, and ``mu_starts`` says which muons belong to it.  A row with more than one
    muon gets ``NaN``/``-1`` here; those rows are dropped by the single-track cut.
    """
    group = companion[ptype_full]
    mu_starts = group[f"mu_starts/{part_key}/data"][:].astype(np.int64)
    e_entry = group[f"e_at_entry/{part_key}/data"][:]
    status = group[f"e_at_entry_status/{part_key}/data"][:]
    if len(mu_starts) != n_events + 1:
        raise ValueError(
            f"{ptype_full}/{part_key}: mu_starts has {len(mu_starts)} entries, "
            f"expected {n_events + 1}"
        )
    first = mu_starts[:-1]
    n_tracks = np.diff(mu_starts)
    energy = np.where(n_tracks == 1, e_entry[first, column], np.nan)
    target_status = np.where(n_tracks == 1, status[first, column], -1)
    return {
        "energy": energy.astype(np.float32),
        "status": target_status.astype(np.int8),
        "n_tracks": n_tracks.astype(np.int32),
    }


def _read_event_metadata(
    h5: h5py.File,
    ptype_full: str,
    part_key: str,
    n_events: int,
) -> Dict[str, np.ndarray]:
    """Theta, primary sign (fNucleonN) and MC event weight for one part."""
    prime = h5[f"{ptype_full}/prime_prty/{part_key}/data"][:]
    if len(prime) != n_events:
        raise ValueError(
            f"{ptype_full}/{part_key}: prime_prty has {len(prime)} rows, "
            f"expected {n_events}"
        )
    return {
        "theta": prime[:, 0].astype(np.float32),
        "sign": prime[:, 3].astype(np.float32),
        "event_weight": prime[:, 5].astype(np.float32),
    }


def _attach_event_fks(
    catalog_path: str,
    locations: List[EventLocation],
    selected: np.ndarray,
) -> np.ndarray:
    """Map selected rows to ``catalog_v2.events.id``; -1 where the catalog has no row."""
    fks = np.full(len(selected), -1, dtype=np.int64)
    if not Path(catalog_path).exists():
        logger.warning("Catalog %s not found — event_fk left as -1", catalog_path)
        return fks

    from inference.shared.catalog_query import get_event_fks_mc

    data_classes = np.array(
        [locations[i].particle_type for i in selected], dtype=object
    )
    seasons = np.array(
        [int(dc.rsplit("_", 1)[-1]) for dc in data_classes], dtype=np.int32
    )
    runs = np.array([locations[i].part_key for i in selected], dtype=object)
    event_ids = np.array(
        [locations[i].event_idx_in_part for i in selected], dtype=np.int64
    )
    logger.info("  Looking up %s event_fk values...", f"{len(selected):,}")
    t0 = time.time()
    df = get_event_fks_mc(
        catalog_path=catalog_path,
        source="mc_merged",
        data_classes=data_classes,
        seasons=seasons,
        runs=runs,
        event_ids=event_ids,
    )
    idx = df["query_idx"].to_numpy(dtype=np.int64)
    fks[idx] = df["event_fk"].to_numpy(dtype=np.int64)
    logger.info(
        "  event_fk: %s matched, %s missing (%.0fs)",
        f"{len(idx):,}", f"{len(selected) - len(idx):,}", time.time() - t0,
    )
    return fks


def build_mu_energy_npy(cfg: dict) -> None:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    h5_path = cfg.get("h5_path", DEFAULT_H5_PATH)
    companion_path = cfg.get("companion_path", DEFAULT_COMPANION_PATH)
    catalog_path = cfg.get("catalog_path", DEFAULT_CATALOG_PATH)
    threshold = float(cfg.get("sig_noise_threshold", 0.8))

    cuts_cfg = cfg.get("event_cuts", {})
    min_hits = int(cuts_cfg.get("min_hits", 8))
    min_strings = int(cuts_cfg.get("min_strings", 3))

    target_cfg = cfg.get("target", {})
    target_column = int(target_cfg.get("column", 1))
    valid_status = tuple(int(x) for x in target_cfg.get("valid_status", [0, 1, 4, 5]))
    born_inside_status = tuple(
        int(x) for x in target_cfg.get("born_inside_status", [1, 5])
    )
    exclude_nuatm_inside = bool(
        target_cfg.get("exclude_nuatm_born_inside", True)
    )
    volume_radius = float(target_cfg.get("volume_radius_m", 90.0))
    volume_z_half = float(target_cfg.get("volume_z_half_m", 295.0))

    source = str(cfg.get("sig_noise_source", "model")).lower()
    if source not in ("model", "probs_h5"):
        raise ValueError(
            f"sig_noise_source must be 'model' or 'probs_h5', got {source!r}"
        )
    probs_h5_path = str(cfg.get("sig_noise_probs_path", DEFAULT_PROBS_PATH))

    model_cfg = cfg.get("sig_noise_model", {})
    model_device = model_cfg.get("device", "auto")
    batch_size = int(model_cfg.get("batch_size", 256))

    parts_dict = _load_parts(cfg["parts_file"])
    logger.info("Parts file: %s", cfg["parts_file"])
    for key, values in parts_dict.items():
        logger.info("  %s: %d parts", key, len(values))

    flat_cfg = cfg.get("flat_selection", {})
    flat_enabled = bool(flat_cfg.get("enabled", False))
    flat_classes = tuple(flat_cfg.get("classes", ["nuatm_2020", "nue2_2020"]))
    flat_target_total = int(flat_cfg.get("target_total", 1_000_000))
    flat_bins = int(flat_cfg.get("bins", 25))
    flat_max_ratio = float(flat_cfg.get("max_ratio", 20.0))
    flat_seed = int(flat_cfg.get("seed", seed))
    if flat_enabled:
        logger.info(
            "Flat selection enabled: target_total=%s, bins=%d, max_ratio=%s, "
            "classes=%s, seed=%d",
            f"{flat_target_total:,}", flat_bins, flat_max_ratio, flat_classes, flat_seed,
        )
    else:
        logger.info("Flat selection disabled: the full h8s3 pool will be written")

    if source == "model":
        logger.info("\nLoading sig-noise model...")
        from trained_models.sig_noise.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
            load_model,
        )

        model, _, device = load_model(device=model_device)
        logger.info("  Model on %s, threshold=%s", device, threshold)
    else:
        if not Path(probs_h5_path).exists():
            raise FileNotFoundError(f"SN probs file not found: {probs_h5_path}")
        model = None
        device = None
        logger.info(
            "\nUsing precomputed SN probs: %s (threshold %.2f, no model/GPU)",
            probs_h5_path, threshold,
        )

    t_total = time.time()

    # ---- Pass 1 ---------------------------------------------------------
    logger.info("\nPass 1: sig-noise metadata + target/status/track count")
    all_n_hits: List[np.ndarray] = []
    all_n_sig_hits: List[np.ndarray] = []
    all_n_sig_strings: List[np.ndarray] = []
    all_energy: List[np.ndarray] = []
    all_status: List[np.ndarray] = []
    all_n_tracks: List[np.ndarray] = []
    all_theta: List[np.ndarray] = []
    all_sign: List[np.ndarray] = []
    all_event_weight: List[np.ndarray] = []
    all_ptypes: List[np.ndarray] = []
    all_locations: List[EventLocation] = []
    candidate_rows: List[np.ndarray] = []
    row_offset = 0

    probs_handle = (
        h5py.File(probs_h5_path, "r") if source == "probs_h5" else None
    )
    with h5py.File(h5_path, "r") as h5, h5py.File(companion_path, "r") as companion:
        for ptype_short, part_nums in parts_dict.items():
            ptype_full = PARTICLE_MAP[ptype_short]
            ptype_code = PARTICLE_ENCODE[ptype_full]
            grp = h5[ptype_full]
            n_parts = len(part_nums)
            t0 = time.time()

            for i, pn in enumerate(sorted(part_nums)):
                pk = f"part_{pn}"
                if source == "probs_h5":
                    meta = read_part_metadata_from_probs(
                        grp, probs_handle[ptype_full], pk, threshold
                    )
                else:
                    meta = read_part_metadata(
                        grp, pk, model, batch_size, device, threshold
                    )
                if meta is None:
                    continue

                n_ev = len(meta["n_hits"])
                targets = _read_targets(
                    companion, ptype_full, pk, n_ev, target_column
                )
                extra = _read_event_metadata(h5, ptype_full, pk, n_ev)

                all_n_hits.append(meta["n_hits"])
                all_n_sig_hits.append(meta["n_sig_hits"])
                all_n_sig_strings.append(meta["n_sig_strings"])
                all_energy.append(targets["energy"])
                all_status.append(targets["status"])
                all_n_tracks.append(targets["n_tracks"])
                all_theta.append(extra["theta"])
                all_sign.append(extra["sign"])
                all_event_weight.append(extra["event_weight"])
                all_ptypes.append(np.full(n_ev, ptype_code, dtype=np.int8))

                part_h8s3 = (
                    (meta["n_sig_hits"] >= min_hits)
                    & (meta["n_sig_strings"] >= min_strings)
                )
                part_valid = (
                    np.isin(targets["status"], valid_status)
                    & np.isfinite(targets["energy"])
                )
                part_single = targets["n_tracks"] == 1
                part_exclude = np.zeros(n_ev, dtype=bool)
                if exclude_nuatm_inside:
                    part_exclude = (
                        (ptype_code == PARTICLE_ENCODE["nuatm_2020"])
                        & np.isin(targets["status"], born_inside_status)
                    )
                part_keep = part_h8s3 & part_valid & part_single & ~part_exclude
                keep_pos = np.flatnonzero(part_keep)
                for j in keep_pos:
                    all_locations.append(
                        EventLocation(
                            particle_type=ptype_full,
                            part_key=pk,
                            hit_start=int(meta["hit_starts"][j]),
                            hit_end=int(meta["hit_ends"][j]),
                            event_idx_in_part=int(j),
                        )
                    )
                candidate_rows.append(row_offset + keep_pos)
                row_offset += n_ev

                if (i + 1) % 20 == 0 or i == n_parts - 1:
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(elapsed, 1e-3)
                    eta = (n_parts - i - 1) / rate
                    logger.info(
                        "  [%d/%d] %s/%s (%.0fs elapsed, ETA %.0fs)",
                        i + 1, n_parts, ptype_full, pk, elapsed, eta,
                    )

            n_ev_total = sum(len(a) for a in all_n_hits)
            logger.info("  %s: done (%s events so far)", ptype_full, f"{n_ev_total:,}")

    if probs_handle is not None:
        probs_handle.close()

    n_hits = np.concatenate(all_n_hits)
    n_sig_hits = np.concatenate(all_n_sig_hits)
    n_sig_strings = np.concatenate(all_n_sig_strings)
    energy = np.concatenate(all_energy)
    target_status = np.concatenate(all_status)
    n_tracks = np.concatenate(all_n_tracks)
    theta = np.concatenate(all_theta)
    sign = np.concatenate(all_sign)
    event_weight = np.concatenate(all_event_weight)
    particle_types = np.concatenate(all_ptypes)
    n_events = len(n_hits)
    logger.info("\nPass 1 done: %s events total", f"{n_events:,}")

    # ---- Selection ------------------------------------------------------
    h8s3 = (n_sig_hits >= min_hits) & (n_sig_strings >= min_strings)
    valid = np.isin(target_status, valid_status) & np.isfinite(energy)
    single = n_tracks == 1
    exclude = np.zeros(n_events, dtype=bool)
    if exclude_nuatm_inside:
        nuatm_code = PARTICLE_ENCODE["nuatm_2020"]
        exclude = (particle_types == nuatm_code) & np.isin(
            target_status, born_inside_status
        )

    mask = h8s3 & valid & single & ~exclude
    candidates = np.where(mask)[0]
    cand_rows = (
        np.concatenate(candidate_rows)
        if candidate_rows else np.array([], dtype=np.int64)
    )
    if not np.array_equal(candidates, cand_rows):
        raise RuntimeError(
            "candidate locations are not aligned with the selection mask "
            f"({len(candidates):,} rows vs {len(cand_rows):,} locations)"
        )

    logger.info("\nSelection:")
    logger.info("  total rows:                 %s", f"{n_events:,}")
    logger.info(
        "  h8s3 (>=%d hits, >=%d strings): %s",
        min_hits, min_strings, f"{int(h8s3.sum()):,}",
    )
    logger.info("  target defined:             %s", f"{int(valid.sum()):,}")
    logger.info("  exactly one track:          %s", f"{int(single.sum()):,}")
    if exclude_nuatm_inside:
        logger.info("  excluded nuatm born inside: %s", f"{int(exclude.sum()):,}")
    logger.info("  h8s3/single/valid candidates: %s", f"{len(candidates):,}")
    for name, code in PARTICLE_ENCODE.items():
        logger.info(
            "    %-12s %s", name,
            f"{int((particle_types[candidates] == code).sum()):,}",
        )

    if flat_enabled:
        flat = flat_select_indices(
            log_e=np.log10(energy[candidates]),
            particle_types=particle_types[candidates],
            classes=flat_classes,
            code_by_name=PARTICLE_ENCODE,
            bins=flat_bins,
            per_bin=None,
            target_total=flat_target_total,
            max_ratio=flat_max_ratio,
            seed=flat_seed,
        )
        flat_pos = flat["selected"]
        flat_weights = flat["weights"].astype(np.float32)
        flat_info = flat["info"]
        selected = candidates[flat_pos]
        n_test = int(len(candidates) - len(selected))
        logger.info(
            "  flat train+val:             %s  (ESS %.1f%%, weights %.2f..%.2f)",
            f"{len(selected):,}",
            100.0 * flat_info["ess_fraction"],
            flat_info["weight_range"][0], flat_info["weight_range"][1],
        )
        logger.info(
            "  test complement:            %s  (not stored in this dataset)",
            f"{n_test:,}",
        )
        for name, code in PARTICLE_ENCODE.items():
            logger.info(
                "    %-12s train+val %s",
                name, f"{int((particle_types[selected] == code).sum()):,}",
            )
    else:
        flat_pos = np.arange(len(candidates), dtype=np.int64)
        flat_weights = np.ones(len(candidates), dtype=np.float32)
        selected = candidates
        n_test = 0
        flat_info = {
            "enabled": False,
            "n_candidates": int(len(candidates)),
            "n_selected": int(len(candidates)),
        }
        logger.info("  selected (no flat):         %s", f"{len(selected):,}")

    for st in sorted(set(target_status[selected].tolist())):
        logger.info(
            "    status %+d: %s", st,
            f"{int((target_status[selected] == st).sum()):,}",
        )

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(selected))
    selected = selected[order]
    flat_weights = flat_weights[order]
    flat_pos = flat_pos[order]

    # ---- Pass 2 ---------------------------------------------------------
    logger.info("\nPass 2: writing filtered hits to %s", output_dir)
    sig_lengths = n_sig_hits[selected].astype(np.int64)
    offsets = np.zeros(len(selected) + 1, dtype=np.int64)
    np.cumsum(sig_lengths, out=offsets[1:])

    features_path = str(output_dir / "features.npy")
    probs_path = str(output_dir / "probs.npy")
    if source == "probs_h5":
        write_selected_features_from_probs(
            h5_path=h5_path,
            probs_h5_path=probs_h5_path,
            locations=all_locations,
            selected_idx=flat_pos,
            offsets=offsets,
            features_path=features_path,
            probs_out_path=probs_path,
            threshold=threshold,
        )
    else:
        write_selected_features(
            h5_path=h5_path,
            locations=all_locations,
            selected_idx=flat_pos,
            offsets=offsets,
            features_path=features_path,
            probs_path=probs_path,
            model=model,
            batch_size=batch_size,
            device=device,
            threshold=threshold,
        )

    # ---- Per-event arrays ----------------------------------------------
    logger.info("\nWriting per-event arrays...")

    def save(name: str, arr: np.ndarray) -> None:
        path = output_dir / name
        np.save(path, arr)
        logger.info("  %s: %s %s (%.1f MB)", name, arr.shape, arr.dtype,
                    arr.nbytes / 1e6)

    save("offsets.npy", offsets)
    save("energy.npy", energy[selected].astype(np.float32))
    save("energy_status.npy", target_status[selected].astype(np.int8))
    save("n_tracks.npy", n_tracks[selected].astype(np.int32))
    save("particle_types.npy", particle_types[selected].astype(np.int8))
    save("sign.npy", sign[selected].astype(np.float32))
    save("event_weights.npy", event_weight[selected].astype(np.float32))
    save("flat_weights.npy", flat_weights.astype(np.float32))
    save("theta.npy", theta[selected].astype(np.float32))
    save("n_sig_hits.npy", n_sig_hits[selected].astype(np.int32))
    save("n_sig_strings.npy", n_sig_strings[selected].astype(np.int32))
    save("h5_part_keys.npy", np.array(
        [all_locations[i].part_key for i in flat_pos], dtype=object
    ))
    save("h5_local_event_ids.npy", np.array(
        [all_locations[i].event_idx_in_part for i in flat_pos], dtype=np.int32
    ))

    event_fk = _attach_event_fks(catalog_path, all_locations, flat_pos)
    save("event_fk.npy", event_fk)

    # ---- Dataset info ---------------------------------------------------
    energy_sel = energy[selected]
    counts = {
        name: int((particle_types[selected] == code).sum())
        for name, code in PARTICLE_ENCODE.items()
    }
    status_counts = {
        int(st): int((target_status[selected] == st).sum())
        for st in sorted(set(target_status[selected].tolist()))
    }
    npy_names = [
        "features.npy", "probs.npy", "offsets.npy", "energy.npy",
        "energy_status.npy", "n_tracks.npy", "particle_types.npy", "sign.npy",
        "event_weights.npy", "flat_weights.npy", "theta.npy", "n_sig_hits.npy",
        "n_sig_strings.npy", "h5_part_keys.npy", "h5_local_event_ids.npy",
        "event_fk.npy",
    ]
    total_bytes = sum((output_dir / name).stat().st_size for name in npy_names)

    info = {
        "n_events": int(len(selected)),
        "n_sig_hits_total": int(offsets[-1]),
        "total_bytes": int(total_bytes),
        "sig_noise_threshold": threshold,
        "sig_noise_source": source,
        "sig_noise_probs_path": probs_h5_path if source == "probs_h5" else None,
        "sig_noise_batch_size": 256 if source == "probs_h5" else batch_size,
        "event_cuts": {"min_hits": min_hits, "min_strings": min_strings},
        "target": {
            "column": target_column,
            "volume_radius_m": volume_radius,
            "volume_z_half_m": volume_z_half,
            "valid_status": list(valid_status),
            "born_inside_status": list(born_inside_status),
            "exclude_nuatm_born_inside": exclude_nuatm_inside,
            "unit": "GeV",
        },
        "h5_source": h5_path,
        "companion": companion_path,
        "catalog": catalog_path,
        "particle_encode": PARTICLE_ENCODE,
        "parts": {k: len(v) for k, v in parts_dict.items()},
        "flat_selection": {
            "enabled": flat_enabled,
            "classes": list(flat_classes),
            "target_total": flat_target_total if flat_enabled else None,
            "bins": flat_bins if flat_enabled else None,
            "max_ratio": flat_max_ratio if flat_enabled else None,
            "seed": flat_seed if flat_enabled else None,
            "n_candidates": int(len(candidates)),
            "n_train_val": int(len(selected)),
            "n_test_complement": int(n_test),
            "flat_weights_file": "flat_weights.npy",
            "test_note": (
                "Test is the complement of the flat train+val selection inside the "
                "h8s3 pool. It is not stored in this training dataset; test predictions "
                "must read the full source and exclude this dataset's event_fk values."
            ),
        },
        "counts": counts,
        "status_counts": status_counts,
        "energy_gev": {
            "min": float(np.nanmin(energy_sel)),
            "median": float(np.nanmedian(energy_sel)),
            "p95": float(np.nanpercentile(energy_sel, 95)),
            "max": float(np.nanmax(energy_sel)),
        },
        "event_fk_missing": int((event_fk < 0).sum()),
        "seed": seed,
        "source_note": (
            "Self-contained copy of the sig-noise/hit-writing code from "
            "data_manager/nu_classifier_ds_builder (2026-10-21); the two builders are "
            "kept independent on purpose."
        ),
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    selection_info = dict(flat_info)
    selection_info.update({
        "n_candidates": int(len(candidates)),
        "n_train_val": int(len(selected)),
        "n_test_complement": int(n_test),
        "flat_weights_file": "flat_weights.npy",
        "test_note": info["flat_selection"]["test_note"],
    })
    with open(output_dir / "selection_info.json", "w") as f:
        json.dump(selection_info, f, indent=2)

    logger.info(
        "\nTotal: %.2f GB, %s events", total_bytes / 1e9, f"{len(selected):,}"
    )
    logger.info("Done in %.0fs", time.time() - t_total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the muon-energy NPY dataset"
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "build_config.yaml")

    build_mu_energy_npy(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
