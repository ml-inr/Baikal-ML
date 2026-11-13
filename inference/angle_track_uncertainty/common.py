"""Shared data handling for the released angle/track/uncertainty model.

The model expects selected hits (signal/noise gate) in five physical features
``[q, t, x, y, z]``. This module reads our source HDF5 files, applies the release's
selection (``prob >= threshold``, at least 8 hits on at least 2 strings), builds
padded model batches, and maps events to ``event_fk``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import duckdb
import h5py
import numpy as np
import torch

from inference.shared.catalog_query import get_event_fks_exp, get_event_fks_mc

CHANNELS_PER_STRING = 36
STRING_DIVISOR = CHANNELS_PER_STRING
_PART_KEY_RE = re.compile(r"part_s(\d+)_c(\d+)_r(\d+)")

SOURCE_PTYPES: Dict[str, list[str]] = {
    "mc_merged": ["muatm_2020", "nuatm_2020", "nue2_2020"],
    "mc_reco": ["muatm", "nuatm_conv", "nuatm_prompt", "nue2"],
    "exp_full": ["exp_full"],
    "exp_reco": ["exp_reco"],
    "exp": ["exp"],
}

MC_RECO_PTYPE_TO_DATA_CLASS = {
    "muatm": "muatm_2020",
    "nuatm_conv": "nuatm_conv_2020",
    "nuatm_prompt": "nuatm_prompt_2020",
    "nue2": "nue2_2020",
}
MC_RECO_DATA_CLASS_TO_PTYPE = {
    data_class: ptype for ptype, data_class in MC_RECO_PTYPE_TO_DATA_CLASS.items()
}


def _sql_path(path: str | Path) -> str:
    """Quote a filesystem path for an ``ATTACH`` statement."""
    return str(path).replace("'", "''")


def infer_source_name(path: str | Path) -> str:
    """Infer the catalog source name from the file name, then from its groups."""
    stem = Path(path).stem.lower()
    for name in ("mc_merged", "mc_reco", "exp_full", "exp_reco"):
        if name in stem:
            return name
    with h5py.File(path, "r") as h5:
        keys = set(h5.keys())
        if "muatm_2020" in keys:
            return "mc_merged"
        if "muatm" in keys:
            return "mc_reco"
        if "exp_full" in keys:
            return "exp_full"
        if "exp_reco" in keys:
            return "exp_reco"
        if "exp" in keys:
            return "exp"
    raise ValueError(f"Cannot infer source name from {path}")


def data_class_for(ptype: str, source_name: str) -> str:
    if source_name == "mc_reco":
        return MC_RECO_PTYPE_TO_DATA_CLASS.get(ptype, ptype)
    return ptype


def season_for(data_class: str) -> int:
    tail = data_class.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def is_exp_source(source_name: str) -> bool:
    return source_name in {"exp", "exp_reco", "exp_full"}


def parse_exp_part_key(part: str) -> tuple[int, int, int]:
    match = _PART_KEY_RE.match(part)
    if not match:
        raise ValueError(f"Cannot parse exp part key: {part!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _part_sort_key(part: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", part)
    if not match:
        return part, -1
    return re.sub(r"\d+$", "", part), int(match.group(1))


def part_keys(group: h5py.Group) -> list[str]:
    """Part keys in deterministic numeric order (part_2 before part_10)."""
    return sorted(group["raw"]["data"].keys(), key=_part_sort_key)


def read_event_block(
    group: h5py.Group,
    part: str,
    begin: int,
    end: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    int,
]:
    """Read events ``[begin:end)`` of one part.

    Returns ``(ev_starts, raw, channels, ev_ids, header, hit_lo)`` where
    ``ev_starts`` is relative to the first hit of the block, ``hit_lo`` is the
    absolute hit offset of the block inside the part (for the probs file), and
    ``header`` is ``header_prty`` when the source carries it.
    """
    starts_full = np.asarray(group["raw"]["ev_starts"][part]["data"][begin : end + 1],
                             dtype=np.int64)
    lo, hi = int(starts_full[0]), int(starts_full[-1])
    raw = np.asarray(group["raw"]["data"][part]["data"][lo:hi], dtype=np.float32)
    channels = np.asarray(group["raw"]["channels"][part]["data"][lo:hi], dtype=np.int32)

    ev_ids = None
    if "ev_ids" in group and part in group["ev_ids"]:
        ev_ids = np.asarray(group["ev_ids"][part]["data"][begin:end])

    header = None
    if "header_prty" in group and part in group["header_prty"]:
        header = np.asarray(group["header_prty"][part]["data"][begin:end])

    return starts_full - lo, raw, channels, ev_ids, header, lo


def part_has_probs(probs_h5: Optional[h5py.File], ptype: str, part: str) -> bool:
    return (
        probs_h5 is not None
        and ptype in probs_h5
        and "probs" in probs_h5[ptype]
        and part in probs_h5[ptype]["probs"]
    )


def load_norm(
    path: str | Path, group: str = "norm_param"
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return (
            np.asarray(h5[f"{group}/mean"], dtype=np.float32),
            np.asarray(h5[f"{group}/std"], dtype=np.float32),
        )


def probs_window(
    probs_h5: Optional[h5py.File],
    ptype: str,
    part: str,
    hit_lo: int,
    n_hits: int,
) -> Optional[np.ndarray]:
    """Per-hit probabilities for one hit window, or None when not covered.

    Reads only ``[hit_lo : hit_lo + n_hits]``; never materialises a whole part.
    """
    if not part_has_probs(probs_h5, ptype, part):
        return None
    dataset = probs_h5[ptype]["probs"][part]["data"]
    if len(dataset) < hit_lo + n_hits:
        raise ValueError(
            f"{ptype}/{part}: probs have {len(dataset)} hits, need "
            f"{hit_lo + n_hits}"
        )
    return np.asarray(dataset[hit_lo : hit_lo + n_hits], dtype=np.float32)


def validate_probs_ev_starts(
    probs_h5: Optional[h5py.File],
    ptype: str,
    part: str,
    source_starts: np.ndarray,
) -> None:
    """Compare the full per-part event boundaries once, not per window."""
    if not part_has_probs(probs_h5, ptype, part):
        return
    group = probs_h5[ptype]
    if "ev_starts" in group and part in group["ev_starts"]:
        stored = np.asarray(
            group["ev_starts"][part]["data"][:], dtype=np.int64
        )
        if not np.array_equal(stored, np.asarray(source_starts, dtype=np.int64)):
            raise ValueError(
                f"{ptype}/{part}: probs ev_starts do not match the source"
            )


def validate_probs_channels(
    probs_h5: Optional[h5py.File],
    ptype: str,
    part: str,
    source_channels: np.ndarray,
    hit_lo: int,
) -> None:
    """Compare the channels of one hit window; the arrays must be aligned."""
    if not part_has_probs(probs_h5, ptype, part):
        return
    group = probs_h5[ptype]
    if "channels" in group and part in group["channels"]:
        stored = np.asarray(
            group["channels"][part]["data"][
                hit_lo : hit_lo + len(source_channels)
            ],
            dtype=np.int32,
        )
        if not np.array_equal(stored, np.asarray(source_channels, np.int32)):
            raise ValueError(
                f"{ptype}/{part}: probs channels do not match the source"
            )


def release_sn_probs(
    model: torch.nn.Module,
    raw: np.ndarray,
    ev_starts: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run the release SN classifier over a block and return per-hit probabilities.

    The model uses a float padding mask, so batch size is part of the selection.
    Keep the same convention as the release (256 events by default).
    """
    n_events = len(ev_starts) - 1
    first_hit = int(ev_starts[0])
    n_hits = int(ev_starts[-1]) - first_hit
    if n_hits != len(raw):
        raise ValueError(
            f"release_sn_probs: ev_starts span {n_hits} hits, raw has {len(raw)}"
        )
    probs = np.empty(n_hits, dtype=np.float32)
    features = (raw - mean) / std
    for begin in range(0, n_events, batch_size):
        end = min(begin + batch_size, n_events)
        base = int(ev_starts[begin]) - first_hit
        starts = ev_starts[begin : end + 1] - ev_starts[begin]
        lengths = np.diff(starts)
        max_len = int(lengths.max())
        x = np.zeros((end - begin, max_len, 5), dtype=np.float32)
        mask = np.zeros((end - begin, max_len), dtype=bool)
        for row, length in enumerate(lengths):
            x[row, :length] = features[
                base + starts[row] : base + starts[row + 1]
            ]
            mask[row, :length] = True
        with torch.inference_mode():
            logits = model(
                torch.from_numpy(x).to(device),
                torch.from_numpy(mask).to(device),
            )
            batch_probs = torch.sigmoid(logits[..., 1]).float().cpu().numpy()
        for row, length in enumerate(lengths):
            lo, hi = int(starts[row]), int(starts[row + 1])
            probs[base + lo : base + hi] = batch_probs[row, :length]
    return probs


def select_events(
    raw: np.ndarray,
    probs: np.ndarray,
    channels: np.ndarray,
    ev_starts: np.ndarray,
    ev_ids: Optional[np.ndarray],
    *,
    threshold: float,
    min_hits: int,
    min_strings: int,
    channels_per_string: int = CHANNELS_PER_STRING,
) -> list[dict]:
    """Select hits with ``prob >= threshold`` and enforce the 2-string/8-hit cut."""
    selected: list[dict] = []
    n_events = len(ev_starts) - 1
    for index in range(n_events):
        lo, hi = int(ev_starts[index]), int(ev_starts[index + 1])
        keep = probs[lo:hi] >= np.float32(threshold)
        count = int(keep.sum())
        if count < min_hits:
            continue
        event_channels = channels[lo:hi][keep]
        n_strings = int(np.unique(event_channels // channels_per_string).size)
        if n_strings < min_strings:
            continue
        event_id = ""
        if ev_ids is not None:
            value = ev_ids[index]
            event_id = value.decode() if isinstance(value, bytes) else str(value)
        selected.append(
            {
                "index": index,
                "event_id": event_id,
                "n_raw_hits": hi - lo,
                "n_hits": count,
                "n_strings": n_strings,
                "raw": raw[lo:hi][keep],
                "channels": event_channels,
                "probs": probs[lo:hi][keep],
            }
        )
    return selected


def build_model_batch(
    records: list[dict],
    mean: np.ndarray,
    std: np.ndarray,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build padded normalized input, mask, and selected-hit centroids."""
    max_len = min(max_seq_len, max(len(record["raw"]) for record in records))
    x = np.zeros((len(records), max_len, 5), dtype=np.float32)
    mask = np.zeros((len(records), max_len), dtype=bool)
    centers = np.zeros((len(records), 3), dtype=np.float32)
    for row, record in enumerate(records):
        raw = record["raw"]
        centers[row] = raw[:, 2:5].mean(axis=0)
        used = min(len(raw), max_len)
        x[row, :used] = (raw[:used] - mean) / std
        mask[row, :used] = True
    return x, mask, centers


def run_model(
    model: torch.nn.Module,
    x: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.inference_mode():
        direction, offset, log_angle, log_line = model(
            torch.from_numpy(x).to(device),
            torch.from_numpy(mask).to(device),
        )
    return (
        direction.float().cpu().numpy(),
        offset.float().cpu().numpy(),
        log_angle.float().cpu().numpy(),
        log_line.float().cpu().numpy(),
    )


def added_hits_pass_photon_gate(
    raw: np.ndarray,
    probs: np.ndarray,
    ev_starts: np.ndarray,
    event_index: int,
    direction: np.ndarray,
    anchor: np.ndarray,
    *,
    high_threshold: float,
    low_threshold: float,
    max_residual_ns: float,
    refractive_index: float = 1.36,
) -> bool:
    """Upstream 200 ns photon-time gate for one event."""
    lo, hi = int(ev_starts[event_index]), int(ev_starts[event_index + 1])
    probability = probs[lo:hi]
    high = probability >= np.float32(high_threshold)
    added = (probability >= np.float32(low_threshold)) & ~high
    if not high.any() or not added.any():
        return False
    unit = direction / np.linalg.norm(direction)
    delta = raw[lo:hi, 2:5] - anchor
    along = delta @ unit
    transverse = np.linalg.norm(delta - along[:, None] * unit, axis=1)
    photon_factor = np.sqrt(refractive_index**2 - 1.0)
    expected_ns = (along + photon_factor * transverse) / 0.299792458
    residual_ns = raw[lo:hi, 1] - expected_ns
    clock_ns = np.median(residual_ns[high])
    return bool(np.max(np.abs(residual_ns[added] - clock_ns)) < max_residual_ns)


def catalog_event_fks(
    catalog: str,
    source_name: str,
    ptype: str,
    part: str,
    header: Optional[np.ndarray],
    local_indices: np.ndarray,
    absolute_indices: np.ndarray,
) -> dict[int, int]:
    """Map block-local event indices to ``event_fk``; missing events are absent.

    ``local_indices`` address the block's ``header`` rows; ``absolute_indices``
    are the event positions within the part, which is what the MC catalog uses.
    """
    if len(local_indices) == 0:
        return {}
    if not is_exp_source(source_name):
        data_class = data_class_for(ptype, source_name)
        result = get_event_fks_mc(
            catalog,
            source_name,
            data_classes=np.array([data_class] * len(absolute_indices)),
            seasons=np.full(
                len(absolute_indices), season_for(data_class), dtype=np.int32
            ),
            runs=np.array([part] * len(absolute_indices)),
            event_ids=np.asarray(absolute_indices, dtype=np.int64),
        )
    elif header is not None:
        rows = header[local_indices]
        result = get_event_fks_exp(
            catalog,
            source_name,
            seasons=rows[:, 0].astype(np.int32),
            clusters=rows[:, 1].astype(np.int32),
            runs=np.array([str(int(run)) for run in rows[:, 2]]),
            event_ids=rows[:, 3].astype(np.int64),
        )
    else:
        season, cluster, run = parse_exp_part_key(part)
        result = get_event_fks_exp(
            catalog,
            source_name,
            seasons=np.full(len(absolute_indices), season, dtype=np.int32),
            clusters=np.full(len(absolute_indices), cluster, dtype=np.int32),
            runs=np.array([str(run)] * len(absolute_indices)),
            event_ids=np.asarray(absolute_indices, dtype=np.int64),
        )
    return {
        int(local_indices[int(row.query_idx)]): int(row.event_fk)
        for row in result.itertuples(index=False)
    }


def event_set_indices(
    event_set_db: str | Path,
    catalog: str | Path,
    source_name: str,
    query: str = "SELECT event_fk FROM predictions",
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """Resolve an external ``event_fk`` set to source part/local indices.

    ``event_set_db`` is a DuckDB file; ``query`` must return an ``event_fk``
    column. The default query reads the whole ``predictions`` table, which is
    what the nu-classifier prediction DBs contain.

    Returns ``{(ptype, part_key): (local_idx, event_fk)}`` with both arrays
    sorted by ``local_idx``, plus counts for provenance. Events whose
    ``event_fk`` is absent from ``catalog_v2`` or belongs to another source are
    reported as missing rather than guessed.
    """
    if source_name not in SOURCE_PTYPES:
        raise ValueError(f"Unknown source name: {source_name!r}")
    event_set_db = Path(event_set_db)
    catalog = Path(catalog)
    connection = duckdb.connect(str(event_set_db), read_only=True)
    try:
        connection.execute(
            f"ATTACH '{_sql_path(catalog)}' AS cat (READ_ONLY)"
        )
        n_requested = int(
            connection.execute(
                f"""
                WITH target AS ({query})
                SELECT COUNT(DISTINCT t.event_fk)
                FROM target t
                JOIN cat.events e ON e.id = t.event_fk
                WHERE e.source = ?
                """,
                [source_name],
            ).fetchone()[0]
        )
        rows = connection.execute(
            f"""
            WITH target AS ({query})
            SELECT DISTINCT
                e.data_class AS data_class,
                l.part_key  AS part_key,
                l.local_idx AS local_idx,
                t.event_fk  AS event_fk
            FROM target t
            JOIN cat.events e ON e.id = t.event_fk
            JOIN cat.h5_locations l ON l.event_fk = t.event_fk
            WHERE e.source = ?
            ORDER BY 1, 2, 3
            """,
            [source_name],
        ).df()
    finally:
        connection.close()

    stats = {
        "n_event_fk_requested": n_requested,
        "n_event_fk_resolved": int(len(rows)),
        "n_event_fk_missing": int(n_requested - len(rows)),
    }
    if len(rows) == 0:
        return {}, stats

    if source_name == "mc_reco":
        ptypes = rows["data_class"].map(MC_RECO_DATA_CLASS_TO_PTYPE)
    else:
        ptypes = rows["data_class"]
    unknown = sorted(set(ptypes) - set(SOURCE_PTYPES[source_name]))
    if unknown:
        raise ValueError(
            f"{source_name}: event set contains data classes outside "
            f"SOURCE_PTYPES[{source_name!r}]: {unknown}"
        )
    rows = rows.assign(ptype=ptypes)

    result: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (ptype, part), group in rows.groupby(["ptype", "part_key"], sort=False):
        local_idx = group["local_idx"].to_numpy(dtype=np.int64)
        event_fk = group["event_fk"].to_numpy(dtype=np.int64)
        order = np.argsort(local_idx, kind="stable")
        local_idx = local_idx[order]
        event_fk = event_fk[order]
        if len(np.unique(local_idx)) != len(local_idx):
            raise ValueError(
                f"{ptype}/{part}: event set resolves to duplicate local indices"
            )
        result[(str(ptype), str(part))] = (local_idx, event_fk)
    return result, stats


def filter_event_set(
    target: tuple[np.ndarray, np.ndarray] | None,
    begin: int,
    end: int,
    records: list[dict],
) -> tuple[list[dict], int]:
    """Keep ``records`` whose part-local index belongs to the target event set.

    Returns the filtered records and how many target events in this block did
    not pass the track-model hit selection.
    """
    if target is None:
        return records, 0
    local_idx = target[0]
    lo = int(np.searchsorted(local_idx, begin, side="left"))
    hi = int(np.searchsorted(local_idx, end, side="left"))
    if lo == hi:
        return [], 0
    wanted = set((local_idx[lo:hi] - begin).tolist())
    selected = {int(record["index"]) for record in records}
    kept = [record for record in records if int(record["index"]) in wanted]
    return kept, len(wanted - selected)


def track_point_scope(ptype: str, source_name: str) -> str:
    validated = {
        "nuatm_2020", "nue2_2020",            # mc_merged
        "nuatm_conv", "nuatm_prompt", "nue2",  # mc_reco
    }
    if source_name in {"mc_merged", "mc_reco"} and ptype in validated:
        return "validated_single_track_MC"
    return "not_validated_for_possible_multi_track_or_experiment"
