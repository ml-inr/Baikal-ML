"""Per-event tabular feature extraction for the domain classifier.

Given catalog `event_fk`s, resolves their HDF5 location, loads raw hits, runs the
sig-noise model, and computes aggregate statistics over the **signal** hits
(prob >= sn_threshold) — the hits the nu-classifier actually sees.

Works for any source whose h5 has the standard `<group>/raw/{data,channels,
ev_starts}/<part_key>/data` layout (exp_full, muatm_2020, ...).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
import torch

from _common import CATALOG, ROOT  # noqa: E402  (same-dir import via sys.path)

STRING_DIVISOR = 36

FEATURE_COLS = [
    "n_sig_hits", "n_sig_strings",
    "q_max", "q_mean", "q_std", "q_sum", "q_p99",
    "t_min", "t_max", "t_std", "t_range",
    "x_std", "y_std", "z_std", "spatial_std",
]

# Dimensionless shape ratios — often more domain-discriminating than raw scales.
DERIVED_COLS = [
    "q_concentration",   # q_max / q_sum     — dominance of the brightest hit
    "q_cv",              # q_std / q_mean    — charge dispersion (coeff. of variation)
    "q_peak",            # q_max / q_mean    — charge peakedness
    "t_concentration",   # t_std / t_range   — how clustered hits are in time
    "t_per_hit",         # t_range / n_hits  — temporal sparsity
    "hits_per_string",   # n_hits / n_strings
    "verticality",       # z_std / spatial_std
    "xy_aniso",          # x_std / y_std     — horizontal anisotropy
]

FEATURE_COLS_ALL = FEATURE_COLS + DERIVED_COLS


def add_derived_features(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Append dimensionless ratio features (DERIVED_COLS) to a feature frame."""
    out = df.copy()
    out["q_concentration"] = df["q_max"]   / (df["q_sum"]      + eps)
    out["q_cv"]            = df["q_std"]    / (df["q_mean"]     + eps)
    out["q_peak"]          = df["q_max"]    / (df["q_mean"]     + eps)
    out["t_concentration"] = df["t_std"]    / (df["t_range"]    + eps)
    out["t_per_hit"]       = df["t_range"]  / (df["n_sig_hits"] + eps)
    out["hits_per_string"] = df["n_sig_hits"] / (df["n_sig_strings"] + eps)
    out["verticality"]     = df["z_std"]    / (df["spatial_std"] + eps)
    out["xy_aniso"]        = df["x_std"]    / (df["y_std"]      + eps)
    return out


def _compute_stats(hits: np.ndarray, channels: np.ndarray) -> dict:
    n = len(hits)
    if n == 0:
        return {k: np.nan for k in FEATURE_COLS}
    q, t = hits[:, 0], hits[:, 1]
    x, y, z = hits[:, 2], hits[:, 3], hits[:, 4]
    return {
        "n_sig_hits":    float(n),
        "n_sig_strings": float(len(np.unique(channels // STRING_DIVISOR))),
        "q_max":  float(q.max()),  "q_mean": float(q.mean()),
        "q_std":  float(q.std()),  "q_sum":  float(q.sum()),
        "q_p99":  float(np.percentile(q, 99)),
        "t_min":  float(t.min()),  "t_max":  float(t.max()),
        "t_std":  float(t.std()),  "t_range": float(t.max() - t.min()),
        "x_std":  float(x.std()),  "y_std":  float(y.std()),
        "z_std":  float(z.std()),
        "spatial_std": float(np.sqrt(x.var() + y.var() + z.var())),
    }


def resolve_locations(event_fks: np.ndarray, catalog: str | Path = CATALOG) -> pd.DataFrame:
    """event_fk -> (h5_path, part_key, local_idx). Order not preserved."""
    c = duckdb.connect(); c.execute("PRAGMA disable_progress_bar")
    c.execute(f"ATTACH '{catalog}' AS cat (READ_ONLY)")
    c.register("fks", pd.DataFrame({"event_fk": np.asarray(event_fks, dtype=np.int64)}))
    df = c.execute("""
        SELECT f.event_fk, l.h5_path, l.part_key, l.local_idx
        FROM fks f JOIN cat.h5_locations l ON l.event_fk = f.event_fk
    """).df()
    c.close()
    return df


def compute_features(
    event_fks: np.ndarray,
    h5_group: str,
    sn_model,
    sn_device: str,
    sn_predict_flat,
    sn_threshold: float = 0.8,
    amp_clip: float | None = 100.0,
    batch_size: int = 512,
    catalog: str | Path = CATALOG,
) -> pd.DataFrame:
    """Return a feature frame (one row per event_fk) of signal-hit statistics.

    sn_predict_flat: the sig-noise model's `predict_flat` callable.
    amp_clip: clamp hit amplitude (column 0, p.e.) at this value before computing
        charge stats, mirroring the nu-classifier's forward() clip (Q=100 PE).
        Set None to disable (raw charges).
    """
    loc = resolve_locations(event_fks, catalog)

    rows: list[dict] = []
    for (h5_path, part_key), grp_df in loc.groupby(["h5_path", "part_key"], sort=False):
        # Read only the selected events' hit slices — never the whole part
        # (exp_full parts are GB-scale, mc_merged.h5 is ~1 TB). Sort by local_idx
        # for read locality across the gzip chunks.
        grp_df = grp_df.sort_values("local_idx")
        idx = grp_df["local_idx"].to_numpy()
        fks = grp_df["event_fk"].to_numpy()

        hits_list, chan_list = [], []
        with h5py.File(h5_path, "r") as f:
            g = f[h5_group]
            ev_starts = g[f"raw/ev_starts/{part_key}/data"][:].astype(np.int64)
            ds_data = g[f"raw/data/{part_key}/data"]       # lazy datasets
            ds_chan = g[f"raw/channels/{part_key}/data"]
            for li in idx:
                s, e = int(ev_starts[li]), int(ev_starts[li + 1])
                hits_list.append(ds_data[s:e].astype(np.float32))
                chan_list.append(ds_chan[s:e].astype(np.int32))

        flat = np.concatenate(hits_list, axis=0)
        lens = np.array([len(h) for h in hits_list], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        probs = sn_predict_flat(
            sn_model, flat, starts, batch_size=batch_size,
            device=torch.device(sn_device), normalize=True,
        )

        for i, fk in enumerate(fks):
            a, b = int(starts[i]), int(starts[i + 1])
            mask = probs[a:b] >= sn_threshold
            sig_hits = hits_list[i][mask]
            if amp_clip is not None and len(sig_hits):
                sig_hits = sig_hits.copy()
                sig_hits[:, 0] = np.minimum(sig_hits[:, 0], amp_clip)   # mirror model clip
            r = _compute_stats(sig_hits, chan_list[i][mask])
            r["event_fk"] = int(fk)
            rows.append(r)

    return pd.DataFrame(rows)
