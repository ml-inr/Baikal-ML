"""Сравнение exp и muatm в одинаковых полосах скора.

Две части:
  * counts/rates по полосам (точные, из prediction DB; exp без плохих ранов);
  * признаки из кэша `features_signal_expanded_h8s3.parquet` в тех же полосах:
    медианы exp/muatm и AUC внутри полосы (AUC > 0.5 — exp больше).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from excess import paths  # noqa: E402

MC_MAX = 576_947_380
BAD = ("part_s2020_c02_r0020", "part_s2020_c02_r0249")
PRED = (ROOT / "inference/nu_classifier/preds/"
        "261024_1202_da_nu_classifier_exp_full_E1_weights_z128_k10_"
        "dlr1e-2_lam0p003_delay2_h8s3_muOnlyDA_s42@best_da_model")
LOW = 1.0 / (1.0 + np.exp(2.0))  # sigmoid(-2)

BANDS = [(LOW, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
         (0.8, 0.9), (0.9, 1.0001)]
LABELS = ["0.119-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8",
          "0.8-0.9", "0.9-1.0"]
FEATURES = ["t_qstd", "t_qmean", "t_span_raw", "fit_length_time",
            "lc_t90", "track_length_pc1", "diameter_3d", "q_t_corr",
            "n_hits", "channel_repeat_frac", "log_xy_span"]


def _counts(con, exp):
    rows = []
    for lo, hi in BANDS:
        if exp:
            q = ("""SELECT COUNT(*) FROM predictions p
                    JOIN cat.h5_locations h USING(event_fk)
                    WHERE p.score > ? AND p.score <= ?
                      AND h.part_key NOT IN (?, ?)""")
            n = con.execute(q, [lo, hi, *BAD]).fetchone()[0]
        else:
            n = con.execute(
                "SELECT COUNT(*) FROM predictions "
                "WHERE score > ? AND score <= ? AND event_fk <= ?",
                [lo, hi, MC_MAX]).fetchone()[0]
        rows.append(n)
    if exp:
        total = con.execute(
            """SELECT COUNT(*) FROM predictions p
               JOIN cat.h5_locations h USING(event_fk)
               WHERE h.part_key NOT IN (?, ?)""", list(BAD)).fetchone()[0]
    else:
        total = con.execute(
            "SELECT COUNT(*) FROM predictions WHERE event_fk <= ?",
            [MC_MAX]).fetchone()[0]
    return rows, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(
        paths.DATA / "features_signal_expanded_h8s3.parquet"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    exp = df[df.grp.isin(["exp_tail", "exp_low"])].copy()
    mc = df[df.grp.isin(["mc_tail_ref", "mc_tail_hold",
                         "mc_low_ref", "mc_low_hold"])].copy()
    edges = [b[0] for b in BANDS] + [BANDS[-1][1]]
    for frame in (exp, mc):
        frame["band"] = pd.cut(frame.score, edges, labels=LABELS,
                               right=False, include_lowest=True)

    con = duckdb.connect(str(PRED / "exp_full_thr0p8.duckdb"), read_only=True)
    con.execute(f"ATTACH '{paths.CATALOG}' AS cat (READ_ONLY)")
    e_counts, e_total = _counts(con, True)
    con.close()
    con = duckdb.connect(str(PRED / "mc_merged_thr0p8.duckdb"), read_only=True)
    m_counts, m_total = _counts(con, False)
    con.close()

    stats = []
    for label, (lo, hi), ne, nm in zip(LABELS, BANDS, e_counts, m_counts):
        re, rm = ne / e_total, nm / m_total
        ratio = re / rm if rm else float("nan")
        err = (ratio * np.sqrt(1 / ne + 1 / nm)
               if ratio == ratio else float("nan"))
        stats.append({
            "band": label,
            "exp_n": int(ne), "muatm_n": int(nm),
            "exp_rate": float(re), "muatm_rate": float(rm),
            "exp_over_muatm": float(ratio),
            "ratio_err": float(err),
        })

    label = LABELS
    features = {}
    for f in FEATURES:
        if f not in df.columns:
            continue
        row = {}
        for lab, (lo, hi) in zip(label, BANDS):
            e = exp[(exp.score > lo) & (exp.score <= hi)][f].to_numpy()
            m = mc[(mc.score > lo) & (mc.score <= hi)][f].to_numpy()
            e = e[np.isfinite(e)]
            m = m[np.isfinite(m)]
            if len(e) < 20 or len(m) < 20:
                continue
            y = np.r_[np.ones(len(e)), np.zeros(len(m))]
            auc = float(roc_auc_score(y, np.r_[e, m]))
            row[lab] = {
                "exp_median": float(np.median(e)),
                "muatm_median": float(np.median(m)),
                "auc_exp_gt_muatm": auc,
                "n_exp": int(len(e)), "n_muatm": int(len(m)),
            }
        features[f] = row

    payload = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "populations": {
            "exp": "exp_tail clean (bad runs excluded), score > 0.119",
            "muatm": "mc_tail_ref + mc_tail_hold, score > 0.119",
            "note": "low band 0.119-0.5: counts are population; feature "
                    "medians use the 10k random low samples",
        },
        "totals": {"exp_clean": int(e_total), "muatm": int(m_total)},
        "bands": stats,
        "features": features,
        "note": ("AUC внутри полосы: y=1 для exp, поэтому AUC > 0.5 значит "
                 "exp больше; < 0.5 — exp меньше при том же скоре."),
    }
    out = Path(args.out) if args.out else (
        paths.DATA / "score_band_compare_h8s3.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"{'band':12s} {'exp':>8s} {'muatm':>9s} {'ratio':>7s}")
    for s in stats:
        print(f"{s['band']:12s} {s['exp_n']:8d} {s['muatm_n']:9d} "
              f"{s['exp_over_muatm']:7.3f} +- {s['ratio_err']:.3f}")
    print(f"\n{'feature':20s} " + " ".join(f"{l:>17s}" for l in label))
    for f, row in features.items():
        cells = [f"{row[l]['auc_exp_gt_muatm']:.3f}" if l in row else "-"
                 for l in label]
        print(f"{f:20s} " + " ".join(f"{c:>17s}" for c in cells))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
