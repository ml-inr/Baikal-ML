"""Excess in boxes of the tilt of the strings each event actually used.

Per-run medians hide the spread: within one run string tilts differ by up to a factor eight,
and two regimes coexist -- coherent drift (all strings pushed one way) and incoherent residual
deformation. Averaging over a run mixes both, which is why the run-level test came out flat.

Here every experimental event gets the tilt of the strings it fired, so the excess can be
binned on that. Simulation has exactly zero tilt on all 55 strings, so it contributes a single
acceptance number as the reference.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
PREDS = ROOT / "inference_v2/nu_classifier/preds" / MODEL
EXP_H5 = ROOT / "data_manager/data/h5datasets/exp_full.h5"
PROBS = ROOT / ("data_manager/data/h5datasets/exp_full_probs_"
                "k_nsol_labelneq0_da_hs128_k0p0001.h5")
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"
SN_THR, STRING_DIVISOR = 0.8, 36
MIN_MODULES_PER_STRING = 15


def string_tilts(part: str) -> dict[int, float]:
    """Tilt in milliradians of every string in one run, from module coordinates."""
    with h5py.File(EXP_H5, "r") as f:
        n = min(3_000_000, f[f"exp_full/raw/data/{part}/data"].shape[0])
        data = f[f"exp_full/raw/data/{part}/data"][:n]
        chan = f[f"exp_full/raw/channels/{part}/data"][:n]
    position = {}
    for ch in np.unique(chan):
        m = chan == ch
        if m.sum() >= 30:
            position[int(ch)] = np.median(data[m][:, 2:5], axis=0)
    grouped: dict[int, list] = {}
    for ch, p in position.items():
        grouped.setdefault(ch // STRING_DIVISOR, []).append(p)
    tilts = {}
    for s, points in grouped.items():
        a = np.array(points)
        if len(a) < MIN_MODULES_PER_STRING:
            continue
        z = a[:, 2]
        tx = np.polyfit(z, a[:, 0], 1)[0]
        ty = np.polyfit(z, a[:, 1], 1)[0]
        tilts[s] = 1000.0 * float(np.hypot(tx, ty))
    return tilts


def per_event() -> pd.DataFrame:
    con = duckdb.connect(str(PREDS / "exp_full_thr0p8.duckdb"), read_only=True)
    con.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    rows = con.execute("""
        SELECT l.part_key, l.local_idx, p.score
        FROM predictions p JOIN splits s USING (event_fk)
        JOIN cat.h5_locations l ON l.event_fk = p.event_fk
        WHERE p.n_sn_hits >= 8 AND p.n_sn_strings >= 3 AND NOT s.excluded""").fetchall()
    con.close()
    by_part: dict[str, list] = {}
    for pk, idx, sc in rows:
        by_part.setdefault(pk, []).append((idx, sc))
    print(f"{len(rows):,} experimental events across {len(by_part)} runs")

    out = []
    for i, (part, items) in enumerate(sorted(by_part.items()), 1):
        tilts = string_tilts(part)
        with h5py.File(EXP_H5, "r") as f, h5py.File(PROBS, "r") as pf:
            starts = f[f"exp_full/raw/ev_starts/{part}/data"][:].astype(np.int64)
            chan = f[f"exp_full/raw/channels/{part}/data"][:].astype(np.int32)
            prob = pf[f"exp_full/probs/{part}/data"][:].astype(np.float32)
        for idx, sc in items:
            a, b = int(starts[idx]), int(starts[idx + 1])
            mask = prob[a:b] > SN_THR
            if mask.sum() < 8:
                continue
            used = np.unique(chan[a:b][mask] // STRING_DIVISOR)
            vals = [tilts[s] for s in used if s in tilts]
            if not vals:
                continue
            out.append((sc, float(np.mean(vals)), float(np.max(vals)), len(vals)))
        print(f"  [{i}/{len(by_part)}] {part}  events so far {len(out):,}")
    return pd.DataFrame(out, columns=["score", "tilt_mean", "tilt_max", "n_strings"])


def mc_reference() -> dict[float, float]:
    con = duckdb.connect(str(PREDS / "mc_merged_thr0p8.duckdb"), read_only=True)
    base = "p.n_sn_hits >= 8 AND p.n_sn_strings >= 3 AND NOT s.used_for_labels " \
           "AND s.data_class = 'muatm_2020'"
    out = {}
    for xi in (0.1, 0.5, 0.8):
        out[xi] = con.execute(f"""SELECT count(*) FILTER (WHERE p.score > {xi})::DOUBLE
            / count(*) FROM predictions p JOIN splits s USING (event_fk)
            WHERE {base}""").fetchone()[0]
    con.close()
    return out


if __name__ == "__main__":
    events = per_event()
    events.to_parquet(HERE / "exp_tilt_per_event.parquet")
    ref = mc_reference()
    print(f"\nsimulation acceptance (single geometry, zero tilt): "
          + ", ".join(f"xi>{k}: {100*v:.4f}%" for k, v in ref.items()))
    edges = [0, 1, 2, 3, 4, 5, 6, 8, 12, 20, 100]
    print(f"\n{'tilt of used strings, mrad':>28s} {'events':>10s}"
          + "".join(f"{'xi>'+str(k):>22s}" for k in ref))
    print(f"{'':28s} {'':10s}" + "".join(f"{'acc %':>11s}{'excess':>11s}" for _ in ref))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (events.tilt_mean >= lo) & (events.tilt_mean < hi)
        if m.sum() < 2000:
            continue
        cells = ""
        for xi, mc_rate in ref.items():
            acc = (events.score[m] > xi).mean()
            cells += f"{100*acc:11.4f}{acc/mc_rate:11.2f}"
        print(f"{lo:>13}-{hi:<14} {int(m.sum()):>10,}" + cells)
