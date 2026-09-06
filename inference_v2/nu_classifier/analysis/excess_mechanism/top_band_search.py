"""What sets the top score band apart, beyond the composition already accounted for.

Band-by-band analysis localised the unexplained component: composition explains a constant
1.6-1.9 in every band, but the excess itself is flat-to-falling below score 0.5 and then climbs
to 3.6. The residual therefore sits entirely above 0.5, in roughly 6,000 experimental events.

This script conditions greedily inside that band: start from the best single feature, then add
whichever feature drives the residual closest to one, and stop when nothing helps. If the
residual will not move, the difference is not in the 79 measured quantities.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
PREDS = HERE.parents[1] / "preds" / MODEL
FULL_SIZE = {5: 23_159_967, 6: 7_751, 7: 3_177_637, 8: 3_044}
DROP = {"source", "event_fk", "group_id", "part_key", "cluster", "weight", "score"}
BAND = (0.5, 1.01)


def load() -> pd.DataFrame:
    frame = duckdb.connect(str(HERE / "features.duckdb"), read_only=True).execute(
        "SELECT * FROM features").df()
    score = {}
    for src, db in (("mc", "mc_merged_thr0p8.duckdb"), ("exp", "exp_full_thr0p8.duckdb")):
        con = duckdb.connect(str(PREDS / db), read_only=True)
        con.register("_keys", pd.DataFrame({"event_fk": frame[frame.source == src].event_fk}))
        for fk, s in con.execute("SELECT p.event_fk, p.score FROM predictions p "
                                 "JOIN _keys USING (event_fk)").fetchall():
            score[(src, fk)] = s
        con.close()
    frame["score"] = [score.get((s, k), np.nan) for s, k in zip(frame.source, frame.event_fk)]
    frame["weight"] = frame.group_id.map(
        {k: FULL_SIZE[k] / (frame.group_id == k).sum() for k in FULL_SIZE})
    return frame[frame.n_hits == frame.n_modules]        # events without repeated hits


def residual(mc, exp, in_mc, in_exp, features, n_bins, min_weight=300):
    rate_mc = np.average(in_mc, weights=mc.weight)
    rate_exp = np.average(in_exp, weights=exp.weight)
    key_mc = np.zeros(len(mc), dtype=np.int64)
    key_exp = np.zeros(len(exp), dtype=np.int64)
    for f, k in zip(features, n_bins):
        edges = np.unique(np.nanquantile(mc[f], np.linspace(0.03, 0.97, k)))
        key_mc = key_mc * (len(edges) + 1) + np.digitize(mc[f].fillna(-1e9), edges)
        key_exp = key_exp * (len(edges) + 1) + np.digitize(exp[f].fillna(-1e9), edges)
    predicted = covered = 0.0
    for b in np.unique(key_mc):
        m, e = key_mc == b, key_exp == b
        w_mc, w_exp = mc.weight.values[m], exp.weight.values[e]
        if w_mc.sum() < min_weight or w_exp.sum() < min_weight:
            continue
        predicted += np.average(in_mc[m], weights=w_mc) * w_exp.sum()
        covered += w_exp.sum()
    predicted /= exp.weight.sum()
    if predicted <= 0:
        return np.nan, np.nan, 0.0
    return predicted / rate_mc, rate_exp / predicted, 100 * covered / exp.weight.sum()


if __name__ == "__main__":
    events = load()
    mc = events[events.group_id.isin([5, 6])]
    exp = events[events.group_id.isin([7, 8])]
    in_mc = ((mc.score >= BAND[0]) & (mc.score < BAND[1])).values.astype(float)
    in_exp = ((exp.score >= BAND[0]) & (exp.score < BAND[1])).values.astype(float)
    n_exp = exp.weight.values[in_exp > 0].sum()
    raw = np.average(in_exp, weights=exp.weight) / np.average(in_mc, weights=mc.weight)
    print(f"band score {BAND[0]}-1.0, events without repeated hits")
    print(f"  experimental events in band: {n_exp:,.0f}")
    print(f"  excess in band: x{raw:.2f}\n")

    features = [c for c in events.columns if c not in DROP and events[c].notna().mean() > 0.5]
    chosen, bins = [], []
    for step in range(4):
        best = None
        for f in features:
            if f in chosen:
                continue
            k = [10, 6, 5, 4][len(chosen)]
            try:
                comp, res, cov = residual(mc, exp, in_mc, in_exp, chosen + [f], bins + [k])
            except Exception:
                continue
            if not np.isfinite(res) or cov < 90:
                continue
            if best is None or abs(res - 1) < abs(best[1] - 1):
                best = (comp, res, cov, f, k)
        if best is None:
            break
        comp, res, cov, f, k = best
        chosen.append(f); bins.append(k)
        print(f"step {step+1}: + {f:24s} explained x{comp:.2f}, residual x{res:.2f} "
              f"(covered {cov:.0f}%)")
        if abs(res - 1) < 0.05:
            print("  residual consistent with one -- stopping")
            break
    print(f"\nfinal feature set: {', '.join(chosen)}")
