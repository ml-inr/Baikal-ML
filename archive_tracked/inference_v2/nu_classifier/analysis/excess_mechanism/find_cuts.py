"""Search for cuts that remove the experimental excess when applied to both samples.

The excess after a cut C is

    (exp accepted and C / exp passing C) / (mc accepted and C / mc passing C)

with "accepted" meaning score > 0.5, the region where the unexplained part sits. A cut that
drives this to one is what we are looking for.

Two guards, without which the search is meaningless:

* **Held-out validation.** Thousands of candidate cuts are scanned, so some will land near one
  by chance. Cuts are searched on half the parts and runs, then evaluated on the other half.
  Only a cut that survives the move counts.
* **A floor on statistics.** At least 100 accepted experimental events must remain, otherwise
  the ratio is noise.

Cuts use only quantities derived from the hits. No classifier score or embedding enters the
cut itself -- only the definition of "accepted".
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
PREDS = HERE.parents[1] / "preds" / MODEL
FULL_SIZE = {1: 5_162, 2: 525_904, 3: 12_499, 4: 1_360_695,
             5: 23_159_967, 6: 7_751, 7: 3_177_637, 8: 3_044}
ACCEPT = 0.5
MIN_EXP_ACCEPTED = 100          # floor for reporting a ratio at all
RETENTION_LEVELS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
DROP = {"source", "event_fk", "group_id", "part_key", "cluster", "weight", "score"}
N_STEPS = 24


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
    assert frame.score.notna().all()
    frame["weight"] = frame.group_id.map(
        {k: FULL_SIZE[k] / (frame.group_id == k).sum() for k in FULL_SIZE})
    return frame


class Sample:
    """Weighted arrays for one domain, prepared once so the scan stays cheap."""

    def __init__(self, frame: pd.DataFrame, groups: list[int], features: list[str]):
        sub = frame[frame.group_id.isin(groups)]
        self.weight = sub.weight.to_numpy(float)
        self.accepted = (sub.score > ACCEPT).to_numpy()
        self.values = {f: sub[f].to_numpy(float) for f in features}
        self.part = sub.part_key.to_numpy()

    def subset(self, mask: np.ndarray) -> "Sample":
        out = object.__new__(Sample)
        out.weight = self.weight[mask]
        out.accepted = self.accepted[mask]
        out.values = {f: v[mask] for f, v in self.values.items()}
        out.part = self.part[mask]
        return out


def excess(mc: Sample, exp: Sample, keep_mc: np.ndarray, keep_exp: np.ndarray):
    w_mc, w_exp = mc.weight[keep_mc], exp.weight[keep_exp]
    a_mc, a_exp = mc.accepted[keep_mc], exp.accepted[keep_exp]
    n_exp_acc = w_exp[a_exp].sum()
    if w_mc.sum() <= 0 or w_exp.sum() <= 0 or n_exp_acc < MIN_EXP_ACCEPTED:
        return np.nan, 0.0
    rate_mc = np.average(a_mc, weights=w_mc)
    rate_exp = np.average(a_exp, weights=w_exp)
    if rate_mc <= 0:
        return np.nan, 0.0
    return rate_exp / rate_mc, n_exp_acc


def scan(mc: Sample, exp: Sample, features: list[str], base_mc, base_exp):
    """Best single additional cut, ranked by how close it drives the excess to one."""
    out = []
    for f in features:
        v_mc, v_exp = mc.values[f], exp.values[f]
        finite = v_mc[np.isfinite(v_mc)]
        if len(finite) < 1000:
            continue
        for q in np.linspace(0.05, 0.95, N_STEPS):
            t = float(np.quantile(finite, q))
            for sign, keep_m, keep_e in (("<", base_mc & (v_mc < t), base_exp & (v_exp < t)),
                                         (">", base_mc & (v_mc > t), base_exp & (v_exp > t))):
                r, n = excess(mc, exp, keep_m, keep_e)
                if np.isfinite(r):
                    out.append((abs(r - 1), r, n, f, sign, t))
    out.sort()
    return out


def frontier(mc: Sample, exp: Sample, features: list[str], keep_mc, keep_exp, base_n):
    """Every single cut as (retention, excess), so the achievable boundary can be seen."""
    rows = []
    for f in features:
        v_mc, v_exp = mc.values[f], exp.values[f]
        finite = v_mc[np.isfinite(v_mc)]
        if len(finite) < 1000:
            continue
        for q in np.linspace(0.02, 0.98, N_STEPS):
            t = float(np.quantile(finite, q))
            for sign in ("<", ">"):
                km = keep_mc & ((v_mc < t) if sign == "<" else (v_mc > t))
                ke = keep_exp & ((v_exp < t) if sign == "<" else (v_exp > t))
                r, n = excess(mc, exp, km, ke)
                if np.isfinite(r):
                    rows.append({"feature": f, "sign": sign, "threshold": t,
                                 "excess": r, "retention": n / base_n})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    frame = load()
    features = [c for c in frame.columns if c not in DROP and frame[c].notna().mean() > 0.5]
    mc = Sample(frame, [5, 6], features)
    exp = Sample(frame, [7, 8], features)
    all_mc = np.ones(len(mc.weight), bool)
    all_exp = np.ones(len(exp.weight), bool)
    base_r, base_n = excess(mc, exp, all_mc, all_exp)
    print(f"{len(features)} features, no cut: excess x{base_r:.2f}, "
          f"{base_n:,.0f} accepted experimental events\n")

    table = frontier(mc, exp, features, all_mc, all_exp, base_n)
    print("BOUNDARY OF WHAT ONE CUT CAN DO\n")
    print(f"{'retention of accepted exp':>26} {'cuts tried':>11} {'best excess':>12} "
          f"{'the cut':>44}")
    for lo in RETENTION_LEVELS:
        sub = table[table.retention >= lo]
        if sub.empty:
            continue
        best = sub.iloc[(sub.excess - 1).abs().argsort().iloc[0]]
        print(f"{'>= ' + format(100*lo, '.0f') + '%':>26} {len(sub):>11,} "
              f"{best.excess:>12.2f} "
              f"{best.feature + ' ' + best.sign + ' ' + format(best.threshold, '.4g'):>44}")
    print(f"\nlowest excess reachable at all: x{table.excess.min():.2f} "
          f"(retention {table.loc[table.excess.idxmin(), 'retention']:.4f}, "
          f"{table.loc[table.excess.idxmin(), 'feature']} "
          f"{table.loc[table.excess.idxmin(), 'sign']} "
          f"{table.loc[table.excess.idxmin(), 'threshold']:.4g})")
    table.to_parquet(HERE / "cut_frontier.parquet")
    print(f"saved: {HERE / 'cut_frontier.parquet'} ({len(table):,} candidate cuts)")
