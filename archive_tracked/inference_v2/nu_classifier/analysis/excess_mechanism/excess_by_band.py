"""Excess explained band by band in classifier score, not cumulatively.

A cumulative cut (score > xi) mixes every band below it, so a component confined to the top of
the score range is diluted and its location cannot be read off. Splitting the score into bands
localises it: each band is an independent population with its own excess and its own
composition correction.

Composition correction: within a band, bin both samples on one feature, apply the simulation's
per-bin band-occupancy to the experimental counts, and sum. The ratio of that prediction to the
simulation's own occupancy is what the feature explains; the rest is the residual.
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
BANDS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3),
         (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
DROP = {"source", "event_fk", "group_id", "part_key", "cluster", "weight", "score"}


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
    assert frame.score.notna().all(), "missing scores"
    frame["weight"] = frame.group_id.map(
        {k: FULL_SIZE[k] / (frame.group_id == k).sum() for k in FULL_SIZE})
    return frame


def explain_band(mc, exp, in_mc, in_exp, feature, n_bins=14, min_weight=400):
    """How much of this band's excess one feature explains via composition."""
    rate_mc = np.average(in_mc, weights=mc.weight)
    rate_exp = np.average(in_exp, weights=exp.weight)
    if rate_mc <= 0 or rate_exp <= 0:
        return np.nan, np.nan
    edges = np.unique(np.nanquantile(mc[feature], np.linspace(0.03, 0.97, n_bins)))
    bin_mc = np.digitize(mc[feature].fillna(-1e9), edges)
    bin_exp = np.digitize(exp[feature].fillna(-1e9), edges)
    predicted = 0.0
    for b in np.unique(bin_mc):
        m, e = bin_mc == b, bin_exp == b
        w_mc, w_exp = mc.weight.values[m], exp.weight.values[e]
        if w_mc.sum() < min_weight or w_exp.sum() < min_weight:
            continue
        predicted += np.average(in_mc[m], weights=w_mc) * w_exp.sum()
    predicted /= exp.weight.sum()
    if predicted <= 0:
        return np.nan, np.nan
    return predicted / rate_mc, rate_exp / predicted


def run(frame: pd.DataFrame, clean_only: bool, top_k: int = 3) -> pd.DataFrame:
    if clean_only:
        frame = frame[frame.n_hits == frame.n_modules]
    mc = frame[frame.group_id.isin([5, 6])]
    exp = frame[frame.group_id.isin([7, 8])]
    features = [c for c in frame.columns if c not in DROP and frame[c].notna().mean() > 0.5]
    rows = []
    for lo, hi in BANDS:
        in_mc = ((mc.score >= lo) & (mc.score < hi)).values.astype(float)
        in_exp = ((exp.score >= lo) & (exp.score < hi)).values.astype(float)
        n_mc = mc.weight.values[in_mc > 0].sum()
        n_exp = exp.weight.values[in_exp > 0].sum()
        if n_mc < 2000 or n_exp < 200:
            continue
        share_mc = np.average(in_mc, weights=mc.weight)
        share_exp = np.average(in_exp, weights=exp.weight)
        scored = []
        for f in features:
            comp, resid = explain_band(mc, exp, in_mc, in_exp, f)
            if np.isfinite(comp):
                scored.append((comp, resid, f))
        scored.sort(reverse=True)
        best = scored[0] if scored else (np.nan, np.nan, "-")
        rows.append({"score band": f"{lo:.2f}-{hi:.2f}",
                     "mc events": int(n_mc), "exp events": int(n_exp),
                     "share mc %": 100 * share_mc, "share exp %": 100 * share_exp,
                     "excess": share_exp / share_mc,
                     "best feature": best[2], "explained": best[0], "residual": best[1],
                     "runners up": ", ".join(f"{f} {c:.2f}" for c, _, f in scored[1:top_k])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    events = load()
    for clean in (False, True):
        title = ("EVENTS WITHOUT REPEATED HITS" if clean else "ALL EVENTS")
        print(f"\n{'='*110}\n{title}\n{'='*110}")
        table = run(events, clean_only=clean)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
