"""Build the nu-classifier NPY dataset for experimental data.

Two-pass sig-noise filtering pipeline applied to exp.h5:
  Pass 1 — run sig-noise model on all parts, collect per-event metadata.
  Pass 2 — write prob-filtered hit features + probs to mmap .npy files.

No labels, no balancing.  Output files are written to the same ``output_dir``
as the MC dataset, using the ``exp_`` prefix.

Output files
------------
exp_features.npy        (total_sig_hits, 5)  float32
exp_probs.npy           (total_sig_hits,)    float32
exp_offsets.npy         (n_events+1,)        int64
exp_n_sig_hits.npy      (n_events,)          int32
exp_n_sig_strings.npy   (n_events,)          int32
exp_dataset_info.json
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import h5py
import numpy as np
import torch

from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
    predict_flat,
)

logger = logging.getLogger(__name__)

STRING_DIVISOR = 36          # OM channels per string in Baikal-GVD


# ---------------------------------------------------------------------------
# Location record
# ---------------------------------------------------------------------------

class ExpEventLocation(NamedTuple):
    part_key: str
    event_idx: int   # positional index of the event within its part
    hit_start: int
    hit_end: int


# ---------------------------------------------------------------------------
# Shared model runner
# ---------------------------------------------------------------------------

def _run_model(
    data_raw: np.ndarray,
    ev_starts: np.ndarray,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    return predict_flat(
        model=model,
        data=data_raw,
        ev_starts=ev_starts,
        batch_size=batch_size,
        device=device,
        normalize=True,
        max_gpu_hits=2_000_000,
    )


# ---------------------------------------------------------------------------
# Pass 1 helpers
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
        event_idx = np.repeat(np.arange(n_events), n_hits)
        sig_ev    = event_idx[mask]
        sig_str   = (channels[mask] // STRING_DIVISOR).astype(np.int32)
        combined  = sig_ev.astype(np.int64) * 1000 + sig_str.astype(np.int64)
        sort_idx  = combined.argsort()
        combined_s = combined[sort_idx]
        uniq      = np.empty(len(combined_s), dtype=bool)
        uniq[0]   = True
        uniq[1:]  = combined_s[1:] != combined_s[:-1]
        n_sig_strings = np.bincount(
            sig_ev[sort_idx[uniq]], minlength=n_events
        ).astype(np.int32)
    else:
        n_sig_strings = np.zeros(n_events, dtype=np.int32)

    return n_sig_hits.astype(np.int32), n_sig_strings


def _read_part_metadata(
    exp_grp: h5py.Group,
    part_key: str,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    threshold: float,
    min_hits: int,
    min_strings: int,
    max_events_per_part: Optional[int],
    rng: np.random.Generator,
    probs_grp: Optional[h5py.Group] = None,
) -> Optional[Dict]:
    """Pass 1 for one exp part.

    Applies the quality cut and (optional) per-part subsample INSIDE the part,
    so the caller only accumulates the surviving/subsampled events — essential
    for exp_full where a single part holds millions of events. Returns the
    selected events' original positional indices + per-event arrays, or None.

    If ``probs_grp`` is given (precomputed sig-noise h5), per-event signal counts
    are read from it (no model, no raw-hit read) — much faster for large parts.
    """
    if probs_grp is not None:
        ev_starts = probs_grp[f"ev_starts/{part_key}/data"][:].astype(np.int64)
        n_events  = len(ev_starts) - 1
        if n_events == 0:
            return None
        hit_starts = ev_starts[:-1]
        hit_ends   = ev_starts[1:]
        n_sig_hits    = probs_grp[f"n_sig_hits_{threshold}/{part_key}/data"][:].astype(np.int32)
        n_sig_strings = probs_grp[f"n_sig_strings_{threshold}/{part_key}/data"][:].astype(np.int32)
    else:
        ev_starts = exp_grp[f"raw/ev_starts/{part_key}/data"][:].astype(np.int64)
        n_events  = len(ev_starts) - 1
        if n_events == 0:
            return None
        hit_starts = ev_starts[:-1]
        hit_ends   = ev_starts[1:]
        n_hits     = (hit_ends - hit_starts).astype(np.int32)
        data_raw = exp_grp[f"raw/data/{part_key}/data"][:].astype(np.float32)
        channels = exp_grp[f"raw/channels/{part_key}/data"][:].astype(np.int32)
        prob     = _run_model(data_raw, ev_starts, model, batch_size, device)
        n_sig_hits, n_sig_strings = _count_sig_hits_strings(
            prob > threshold, channels, hit_starts, n_hits, n_events,
        )

    # ── per-part quality cut ──────────────────────────────────────────────
    cut = (n_sig_hits >= min_hits) & (n_sig_strings >= min_strings)
    sel = np.where(cut)[0]                      # original positional indices
    if len(sel) == 0:
        return None

    # ── per-part random subsample (coverage across all parts/runs) ────────
    if max_events_per_part is not None and len(sel) > max_events_per_part:
        sel = np.sort(rng.choice(sel, size=max_events_per_part, replace=False))

    return {
        "event_idx":     sel.astype(np.int64),          # positional in part
        "n_sig_hits":    n_sig_hits[sel],
        "n_sig_strings": n_sig_strings[sel],
        "hit_starts":    hit_starts[sel],
        "hit_ends":      hit_ends[sel],
        "n_events_raw":  n_events,
    }


def _read_all_exp_metadata(
    h5_file: h5py.File,
    h5_group: str,
    exclude_parts: set,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    threshold: float,
    min_hits: int,
    min_strings: int,
    max_events_per_part: Optional[int],
    seed: int,
    probs_grp: Optional[h5py.Group] = None,
) -> Dict:
    """Pass 1: iterate all parts; cut + per-part subsample applied inside."""
    exp_grp = h5_file[h5_group]
    part_keys = sorted(
        k for k in exp_grp["raw/data"].keys() if k.startswith("part_")
    )
    part_keys = [p for p in part_keys if p not in exclude_parts]
    n_parts = len(part_keys)
    logger.info(f"  {h5_group}: {n_parts} parts ({len(exclude_parts)} excluded)")

    all_n_sig_hits: List[np.ndarray]    = []
    all_n_sig_strings: List[np.ndarray] = []
    all_locations: List[ExpEventLocation] = []

    t0 = time.time()
    for i, pk in enumerate(part_keys):
        rng  = np.random.default_rng(seed + i)   # reproducible per-part draw
        meta = _read_part_metadata(
            exp_grp, pk, model, batch_size, device, threshold,
            min_hits, min_strings, max_events_per_part, rng, probs_grp,
        )
        if meta is None:
            continue

        ev_idx = meta["event_idx"]
        all_n_sig_hits.append(meta["n_sig_hits"])
        all_n_sig_strings.append(meta["n_sig_strings"])

        for j in range(len(ev_idx)):
            all_locations.append(ExpEventLocation(
                part_key=pk,
                event_idx=int(ev_idx[j]),
                hit_start=int(meta["hit_starts"][j]),
                hit_end=int(meta["hit_ends"][j]),
            ))

        elapsed = time.time() - t0
        rate    = (i + 1) / max(elapsed, 1e-3)
        eta     = (n_parts - i - 1) / rate
        n_so_far = sum(len(a) for a in all_n_sig_hits)
        logger.info(
            f"  [{i+1}/{n_parts}] {h5_group}/{pk}  kept {len(ev_idx):,}/"
            f"{meta['n_events_raw']:,}  ({n_so_far:,} total, "
            f"{elapsed:.0f}s, ETA {eta:.0f}s)"
        )

    return {
        "n_sig_hits":    np.concatenate(all_n_sig_hits),
        "n_sig_strings": np.concatenate(all_n_sig_strings),
        "locations":     all_locations,
    }


# ---------------------------------------------------------------------------
# Pass 2: write filtered features + probs to mmap
# ---------------------------------------------------------------------------

def _write_exp_features(
    h5_path: str,
    h5_group: str,
    locations: List[ExpEventLocation],
    selected_idx: np.ndarray,
    offsets: np.ndarray,
    features_path: str,
    probs_path: str,
    channels_path: str,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    threshold: float,
    probs_h5_path: Optional[str] = None,
    probs_group: Optional[str] = None,
) -> None:
    total_sig_hits = int(offsets[-1])
    features  = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=np.float32, shape=(total_sig_hits, 5),
    )
    probs_out = np.lib.format.open_memmap(
        probs_path, mode="w+", dtype=np.float32, shape=(total_sig_hits,),
    )
    chan_out  = np.lib.format.open_memmap(
        channels_path, mode="w+", dtype=np.int32, shape=(total_sig_hits,),
    )

    # Group selected events by part_key
    part_groups: Dict[str, List[Tuple[int, ExpEventLocation]]] = {}
    for out_idx, global_idx in enumerate(selected_idx):
        loc = locations[global_idx]
        part_groups.setdefault(loc.part_key, []).append((out_idx, loc))

    n_parts = len(part_groups)
    logger.info(
        f"  Writing exp features: {len(selected_idx):,} events from {n_parts:,} parts"
    )

    done_parts = 0
    t0 = time.time()

    _probs_ctx = h5py.File(probs_h5_path, "r") if probs_h5_path else None
    probs_grp  = _probs_ctx[probs_group] if _probs_ctx else None

    # The sig-noise model passes a float padding mask, so batch size decides which hits pass
    # the threshold. Reading probs written at a different batch size than the MC source
    # dataset used silently gives the DA source and target domains different hit selections.
    # See doc/sig_noise_batch_size.md.
    if _probs_ctx is not None:
        file_bs = _probs_ctx.attrs.get("sig_noise_batch_size")
        if file_bs is None:
            logger.warning(
                f"{Path(probs_h5_path).name} carries no sig_noise_batch_size attribute — "
                f"cannot confirm it matches the MC source dataset (256)")
        elif int(file_bs) != batch_size:
            raise SystemExit(
                f"{Path(probs_h5_path).name} was written with batch_size={int(file_bs)}, "
                f"but this build expects {batch_size}. Recompute the probs or fix the "
                f"config — see doc/sig_noise_batch_size.md")

    with h5py.File(h5_path, "r") as h5:
        exp_grp = h5[h5_group]
        for pk, events in part_groups.items():
            ev_starts = exp_grp[f"raw/ev_starts/{pk}/data"][:].astype(np.int64)
            data_raw  = exp_grp[f"raw/data/{pk}/data"][:].astype(np.float32)
            channels  = exp_grp[f"raw/channels/{pk}/data"][:].astype(np.int32)

            if probs_grp is not None:
                prob = probs_grp[f"probs/{pk}/data"][:].astype(np.float32)
            else:
                prob = _run_model(data_raw, ev_starts, model, batch_size, device)
            sig_mask = prob > threshold

            for out_idx, loc in events:
                s, e    = loc.hit_start, loc.hit_end
                ev_mask = sig_mask[s:e]
                dst     = offsets[out_idx]
                n       = offsets[out_idx + 1] - dst
                features[dst:dst + n]  = data_raw[s:e][ev_mask]
                probs_out[dst:dst + n] = prob[s:e][ev_mask]
                chan_out[dst:dst + n]  = channels[s:e][ev_mask]

            done_parts += 1
            if done_parts % 20 == 0 or done_parts == n_parts:
                elapsed = time.time() - t0
                rate    = done_parts / max(elapsed, 1e-3)
                eta     = (n_parts - done_parts) / rate
                logger.info(
                    f"  [{done_parts}/{n_parts}] "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
                )

    if _probs_ctx is not None:
        _probs_ctx.close()

    features.flush()
    probs_out.flush()
    chan_out.flush()
    logger.info(
        f"  Written: {total_sig_hits:,} sig hits, "
        f"{features.nbytes / 1e9:.2f} GB features + "
        f"{probs_out.nbytes / 1e6:.1f} MB probs + "
        f"{chan_out.nbytes / 1e6:.1f} MB channels"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_exp_npy(cfg: dict) -> None:
    """Build exp NPY dataset from exp.h5 using sig-noise model.

    Args:
        cfg: Config dict with keys:
            h5_path_exp      — path to exp.h5
            output_dir       — same output dir as MC dataset
            sig_noise_model  — {device, batch_size}
            sig_noise_threshold
            event_cuts       — {min_hits, min_strings}
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path_exp = cfg.get("h5_path_exp", "data_manager/data/h5datasets/exp.h5")
    threshold   = float(cfg.get("sig_noise_threshold", 0.5))

    cuts_cfg    = cfg.get("event_cuts", {})
    min_hits    = int(cuts_cfg.get("min_hits", 8))
    min_strings = int(cuts_cfg.get("min_strings", 2))

    # exp_full support: top-group name (auto-detect first key if absent),
    # part exclusion (bad runs), and per-part cap (millions of events/part).
    h5_group            = cfg.get("h5_group")
    exclude_parts       = set(cfg.get("exclude_parts", []))
    max_events_per_part = cfg.get("max_events_per_part")
    max_events_per_part = int(max_events_per_part) if max_events_per_part else None
    seed                = int(cfg.get("seed", 42))

    model_cfg    = cfg.get("sig_noise_model", {})
    model_device = model_cfg.get("device", "auto")
    batch_size   = int(model_cfg.get("batch_size", 128))

    # Precomputed sig-noise probs (from predict_exp_h5.py): if given, the builder
    # reads probs / per-event counts instead of running the model → fast I/O.
    probs_h5     = cfg.get("probs_h5")
    probs_group  = cfg.get("probs_group") or h5_group

    if h5_group is None:
        with h5py.File(h5_path_exp, "r") as _h5:
            h5_group = list(_h5.keys())[0]
        probs_group = probs_group or h5_group
    logger.info(f"  h5_group='{h5_group}'  max_events_per_part={max_events_per_part}"
                f"  exclude_parts={sorted(exclude_parts)}  probs_h5={probs_h5}")

    # ── Load sig-noise model (skipped when reading precomputed probs) ─────
    if probs_h5:
        model, device = None, None
        logger.info(f"  Using precomputed probs from {probs_h5} (no model run)")
    else:
        logger.info("\nLoading sig-noise model...")
        from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
            load_model,
        )
        model, _, device = load_model(device=model_device)
        logger.info(f"  Model on {device}, threshold={threshold}")

    t_total = time.time()

    # ── Pass 1: metadata ──────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Pass 1: collecting per-event metadata")
    logger.info(f"{'='*60}")

    _probs_ctx1 = h5py.File(probs_h5, "r") if probs_h5 else None
    _probs_grp1 = _probs_ctx1[probs_group] if _probs_ctx1 else None
    with h5py.File(h5_path_exp, "r") as h5:
        meta = _read_all_exp_metadata(
            h5_file=h5,
            h5_group=h5_group,
            exclude_parts=exclude_parts,
            model=model,
            batch_size=batch_size,
            device=device,
            threshold=threshold,
            min_hits=min_hits,
            min_strings=min_strings,
            probs_grp=_probs_grp1,
            max_events_per_part=max_events_per_part,
            seed=seed,
        )
    if _probs_ctx1 is not None:
        _probs_ctx1.close()

    # Cuts + per-part subsample already applied inside Pass 1.
    n_sig_hits    = meta["n_sig_hits"]
    n_sig_strings = meta["n_sig_strings"]
    locations     = meta["locations"]
    n_selected    = len(n_sig_hits)
    selected      = np.arange(n_selected)
    logger.info(f"\nPass 1 done: {n_selected:,} events selected "
                f"(cut min_hits>={min_hits}, min_strings>={min_strings}"
                f"{f', cap {max_events_per_part}/part' if max_events_per_part else ''})")

    # ── Build output offsets ──────────────────────────────────────────────
    sig_lengths = n_sig_hits[selected].astype(np.int64)
    offsets = np.zeros(n_selected + 1, dtype=np.int64)
    np.cumsum(sig_lengths, out=offsets[1:])

    # ── Pass 2: write filtered features + probs ───────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Pass 2: writing filtered features to {output_dir}")
    logger.info(f"{'='*60}")

    # ── Back-links per event (for train/test split at inference) ─────────
    part_keys_arr = np.array([locations[i].part_key  for i in selected], dtype=object)
    local_ids_arr = np.array([locations[i].event_idx for i in selected], dtype=np.int32)
    # legacy combined name, kept for compatibility
    part_names = np.array([f"{locations[i].part_key}_{locations[i].event_idx}"
                           for i in selected])

    _write_exp_features(
        h5_path=h5_path_exp,
        h5_group=h5_group,
        locations=locations,
        selected_idx=selected,
        offsets=offsets,
        features_path=str(output_dir / "exp_features.npy"),
        probs_path=str(output_dir / "exp_probs.npy"),
        channels_path=str(output_dir / "exp_channels.npy"),
        model=model,
        batch_size=batch_size,
        device=device,
        threshold=threshold,
        probs_h5_path=probs_h5,
        probs_group=probs_group,
    )

    # ── Write per-event arrays ────────────────────────────────────────────
    logger.info("\nWriting per-event arrays...")

    def save(name: str, arr: np.ndarray) -> None:
        path = output_dir / name
        np.save(path, arr)
        logger.info(f"  {name}: {arr.shape} {arr.dtype} ({arr.nbytes / 1e6:.1f} MB)")

    save("exp_offsets.npy",           offsets)
    save("exp_n_sig_hits.npy",        n_sig_hits[selected].astype(np.int32))
    save("exp_n_sig_strings.npy",     n_sig_strings[selected].astype(np.int32))
    save("exp_part_names.npy",        part_names)
    save("exp_h5_part_keys.npy",      part_keys_arr)
    save("exp_h5_local_event_ids.npy", local_ids_arr)

    # ── Dataset info JSON ─────────────────────────────────────────────────
    npy_names = [
        "exp_features.npy", "exp_probs.npy", "exp_offsets.npy",
        "exp_n_sig_hits.npy", "exp_n_sig_strings.npy",
        "exp_channels.npy", "exp_part_names.npy",
        "exp_h5_part_keys.npy", "exp_h5_local_event_ids.npy",
    ]
    total_bytes = sum((output_dir / f).stat().st_size for f in npy_names)

    info = {
        "n_events":          n_selected,
        "n_sig_hits_total":  int(offsets[-1]),
        "total_bytes":       total_bytes,
        "sig_noise_threshold": threshold,
        "event_cuts":        {"min_hits": min_hits, "min_strings": min_strings},
        "h5_path_exp":       h5_path_exp,
        "h5_group":          h5_group,
        "max_events_per_part": max_events_per_part,
        "exclude_parts":     sorted(exclude_parts),
        "seed":              seed,
    }
    info_path = output_dir / "exp_dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"\nTotal: {total_bytes / 1e9:.2f} GB, {n_selected:,} events")
    logger.info(f"Done in {time.time() - t_total:.0f}s")
