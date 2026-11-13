"""Стратифицированная проверка: сколько избытка объясняет сдвиг одного/двух признаков.

Пост-стратификация по бинам признака: MC-события в бине b получают вес
w_b = (вес exp в бине) / (вес MC в бине), так что маргинальное распределение
признака у MC становится как у exp (в популяции score > -2). Затем считается
условная доля хвоста и explained = (f_mc_adj - f_mc_base) / (f_exp - f_mc_base).

Это одномерный/двумерный аналог плотностного перевзвешивания из
`feature_match_excess.py`; веса здесь необучаемые, поэтому результат
устойчивее и прозрачнее (видны использованные бины и перекос весов).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from excess import paths  # noqa: E402

DEFAULT_FEATURES = paths.DATA / "features_signal_expanded_h8s3.parquet"
COUNTS_JSON = (paths.DATA
               / "support_vs_measure_signal_expanded_h8s3_muOnlyDA_e23_tail0.json")
EXP_TAIL = ("exp_tail",)
EXP_LOW = ("exp_low",)
MC_TAIL = ("mc_tail_ref", "mc_tail_hold")
MC_LOW = ("mc_low_ref", "mc_low_hold")

FEATURES = (
    "t_qstd", "t_qmean", "t_span_raw", "fit_length_time",
    "gyration_radius", "track_length_pc1", "diameter_3d",
    "q_t_corr", "logq_t_corr", "t_charge_asym",
    "lc_q_entropy", "lc_t90", "lc_early_q_frac", "lc_ac1",
    "time_std", "log_xy_span",
)
PAIRS = (
    ("t_qstd", "t_qmean"),
    ("t_qmean", "fit_length_time"),
    ("t_qstd", "track_length_pc1"),
    ("t_qstd", "lc_q_entropy"),
)


def _wquantile(x, w, qs):
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    return np.interp(qs, cw / cw[-1], x)


def _edges(values, weights, bins):
    qs = np.linspace(0, 1, bins + 1)
    e = np.unique(_wquantile(values, weights, qs))
    if len(e) < 3:
        return None
    return e


def _bin_index(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1,
                   0, len(edges) - 2)


def evaluate(df, base, cols, bins, n_boot, seed, null_seed=None):
    g = df["grp"].to_numpy()
    is_exp = np.isin(g, EXP_TAIL + EXP_LOW)
    is_mc = np.isin(g, MC_TAIL + MC_LOW)
    sel = is_exp | is_mc
    tail_exp = np.isin(g, EXP_TAIL)[sel]
    tail_mc = np.isin(g, MC_TAIL)[sel]
    b = base[sel]
    x = df.loc[sel, cols].to_numpy(dtype=np.float64)
    finite = np.isfinite(x).all(axis=1)
    if not finite.all():
        sel = np.flatnonzero(sel)
        sel = sel[finite]
        # пересобрать маски после отбрасывания нефинитных строк
        gg = g[sel]
        tail_exp = np.isin(gg, EXP_TAIL)
        tail_mc = np.isin(gg, MC_TAIL)
        is_exp2 = np.isin(gg, EXP_TAIL + EXP_LOW)
        x = df.loc[sel, cols].to_numpy(dtype=np.float64)
        b = base[sel]
    else:
        is_exp2 = is_exp[sel]
    mc_sel = ~is_exp2

    edges = [_edges(x[:, j], b, bins) for j in range(x.shape[1])]
    if any(e is None for e in edges):
        return None
    if null_seed is not None:
        # Внутридоменная перестановка строк: маргиналы признаков у exp и MC
        # сохраняются, но связь признак–скор/хвост разрушается.
        rng = np.random.default_rng(null_seed)
        x = x.copy()
        for dom in (is_exp2, mc_sel):
            pos = np.flatnonzero(dom)
            x[pos] = x[pos][rng.permutation(len(pos))]
    idx = np.zeros(len(x), dtype=np.int64)
    for j, e in enumerate(edges):
        idx = idx * (len(e) - 1) + _bin_index(x[:, j], e)
    n_cells = int(np.prod([len(e) - 1 for e in edges]))

    # Взвешенные счётчики по клеткам.
    e_all = np.bincount(idx[is_exp2], weights=b[is_exp2], minlength=n_cells)
    m_all = np.bincount(idx[mc_sel], weights=b[mc_sel], minlength=n_cells)
    e_tail = np.bincount(idx[is_exp2 & tail_exp],
                         weights=b[is_exp2 & tail_exp], minlength=n_cells)
    m_tail = np.bincount(idx[mc_sel & tail_mc],
                         weights=b[mc_sel & tail_mc], minlength=n_cells)
    ok = (m_all > 0) & (e_all > 0)
    if ok.sum() < 2:
        return None
    r = np.zeros(n_cells)
    r[ok] = e_all[ok] / m_all[ok]

    w_mc = b[mc_sel] * r[idx[mc_sel]]
    f_m0 = float(b[mc_sel & tail_mc].sum() / b[mc_sel].sum())
    f_m = float(w_mc[tail_mc[mc_sel]].sum() / w_mc.sum())
    f_e = float(b[is_exp2 & tail_exp].sum() / b[is_exp2].sum())
    den = f_e - f_m0
    expl = (f_m - f_m0) / den if den > 0 else float("nan")
    ess = float(w_mc.sum() ** 2 / (w_mc ** 2).sum() / len(w_mc))
    max_ratio = float(r[ok].max())

    # Бутстрап: пересобираем клеточные счётчики по событиям.
    rng = np.random.default_rng(seed)
    vals = []
    idx_e, idx_m = idx[is_exp2], idx[mc_sel]
    b_e, b_m = b[is_exp2], b[mc_sel]
    t_e, t_m = tail_exp[is_exp2], tail_mc[mc_sel]
    for _ in range(n_boot):
        ce = rng.multinomial(len(idx_e), np.full(len(idx_e), 1 / len(idx_e)))
        cm = rng.multinomial(len(idx_m), np.full(len(idx_m), 1 / len(idx_m)))
        E = np.bincount(idx_e, weights=ce * b_e, minlength=n_cells)
        M = np.bincount(idx_m, weights=cm * b_m, minlength=n_cells)
        Et = np.bincount(idx_e, weights=ce * b_e * t_e, minlength=n_cells)
        Mt = np.bincount(idx_m, weights=cm * b_m * t_m, minlength=n_cells)
        good = (M > 0) & (E > 0)
        rr = np.zeros(n_cells)
        rr[good] = E[good] / M[good]
        w = cm * b_m * rr[idx_m]
        fm = float((w * t_m).sum() / w.sum())
        fe = float(Et.sum() / E.sum())
        fm0 = float((cm * b_m * t_m).sum() / (cm * b_m).sum())
        vals.append((fm - fm0) / (fe - fm0))
    if len(vals):
        lo, hi = np.quantile(vals, [0.16, 0.84])
    else:
        lo = hi = float("nan")
    return {
        "features": list(cols), "bins": bins, "n_cells": n_cells,
        "n_cells_used": int(ok.sum()),
        "f_mc_base": f_m0, "f_mc_adjusted": f_m, "f_exp": f_e,
        "explained": expl, "explained_68ci": [float(lo), float(hi)],
        "ess_frac": ess, "max_weight_ratio": max_ratio,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--counts", default=str(COUNTS_JSON))
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--null", type=int, default=0,
                    help="within-domain feature permutations as null")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    have = set(df.columns)
    feats = [f for f in FEATURES if f in have]
    pairs = [p for p in PAIRS if all(f in have for f in p)]
    counts = json.loads(Path(args.counts).read_text())["counts"]
    base = np.ones(len(df), dtype=np.float64)
    base[np.isin(df.grp, EXP_LOW)] = (
        counts["exp_low_db"] / counts["exp_low_sample"])
    base[np.isin(df.grp, MC_LOW)] = (
        counts["mc_low"] / counts["mc_low_sample"])

    results = []
    for f in feats:
        r = evaluate(df, base, [f], args.bins, args.bootstrap, 42)
        if r:
            results.append(r)
            print(f"1D {f:22s} explained {r['explained']:+.3f}"
                  f" [{r['explained_68ci'][0]:+.3f},"
                  f" {r['explained_68ci'][1]:+.3f}]"
                  f"  ESS {r['ess_frac']:.3f}"
                  f"  maxratio {r['max_weight_ratio']:.1f}", flush=True)
    for p in pairs:
        r = evaluate(df, base, list(p), 5, args.bootstrap, 42)
        if r:
            results.append(r)
            print(f"2D {p[0]}+{p[1]:<16s} explained {r['explained']:+.3f}"
                  f" [{r['explained_68ci'][0]:+.3f},"
                  f" {r['explained_68ci'][1]:+.3f}]"
                  f"  ESS {r['ess_frac']:.3f}"
                  f"  maxratio {r['max_weight_ratio']:.1f}", flush=True)

    null_results = {}
    if args.null:
        for f in feats:
            vals = []
            for s in range(args.null):
                r = evaluate(df, base, [f], args.bins, 0, 42,
                             null_seed=20_000 + s)
                if r:
                    vals.append(r["explained"])
            if vals:
                null_results[f] = vals
                print(f"null {f:22s} {np.mean(vals):+.3f}"
                      f" +- {np.std(vals):.3f}", flush=True)

    out = Path(args.out) if args.out else (
        paths.DATA / "feature_stratified_excess_h8s3.json")
    payload = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "features": args.features,
        "bins_1d": args.bins, "bins_2d": 5,
        "bootstrap": args.bootstrap,
        "results": results,
        "null": null_results,
        "note": (
            "Пост-стратификация MC к exp по бинам признака(ов) в популяции "
            "score > -2; explained — доля разрыва f_exp - f_mc_base. "
            "ESS и max_weight_ratio показывают перекос весов; при большом "
            "maxratio оценка опирается на малые клетки."),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
