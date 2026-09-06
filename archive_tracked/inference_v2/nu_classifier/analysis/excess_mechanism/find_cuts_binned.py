"""Search for an interpretable cut that flattens the excess in every xi band.

Target (set by the user):
  * per-band excess within [0.8, 1.2] for every band with xi >= 0.5
  * at most 15% of muatm lost in the top bands
  * neutrino efficiency reported, not constrained

The cut is applied identically to MC and exp.  Excess is reported under two
normalisations as a cross-check: ``global`` (share of everything passing the
cut) and ``ref`` (share relative to the xi < 0.01 band, where MC and exp
already agree at 0.98).

Everything is validated out of sample: models are fitted on one half of the
MC parts / exp runs and evaluated on the other, never on the events they saw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).parent))
from archive_tracked.inference_v2.nu_classifier.analysis.excess_mechanism.find_cuts import load  # noqa: E402

DROP = {"event_fk", "group_id", "part_key", "source", "weight", "score",
        "cluster", "prob_mean", "prob_min"}
MC_GROUPS, EXP_GROUPS = [5, 6], [7, 8]
NU_GROUPS = {"nuatm": [1, 2], "nue2": [3, 4]}
TOP_GROUPS = [6, 8]                      # exhaustive, weight 1, score > 0.8
BANDS = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
REF_BAND = (0.0, 0.01)
TARGET_LO, TARGET_HI = 0.8, 1.2
MC_KEEP_MIN = 0.85


def halves(df: pd.DataFrame) -> np.ndarray:
    """Assign each row to half 0 or 1 by hashing its MC part / exp run."""
    return pd.util.hash_pandas_object(df.part_key, index=False).to_numpy() % 2


def in_band(df: pd.DataFrame, lo: float, hi: float) -> np.ndarray:
    return ((df.score >= lo) & (df.score < hi)).values


def band_table(
    mc: pd.DataFrame,
    exp: pd.DataFrame,
    keep_mc: np.ndarray,
    keep_exp: np.ndarray,
    norm: str = "global",
) -> tuple[list[float], list[float], list[int]]:
    """Per-band excess, MC retention and surviving exp counts under a cut."""
    w_mc, w_exp = mc.weight.values, exp.weight.values
    if norm == "global":
        n_mc, n_exp = w_mc[keep_mc].sum(), w_exp[keep_exp].sum()
    else:
        r_mc = in_band(mc, *REF_BAND) & keep_mc
        r_exp = in_band(exp, *REF_BAND) & keep_exp
        n_mc, n_exp = w_mc[r_mc].sum(), w_exp[r_exp].sum()
    excess, retention, counts = [], [], []
    for lo, hi in BANDS:
        b_mc, b_exp = in_band(mc, lo, hi), in_band(exp, lo, hi)
        share_mc = w_mc[b_mc & keep_mc].sum() / n_mc
        share_exp = w_exp[b_exp & keep_exp].sum() / n_exp
        excess.append(share_exp / share_mc if share_mc > 0 else np.nan)
        retention.append(np.average(keep_mc[b_mc], weights=w_mc[b_mc]))
        counts.append(int(w_exp[b_exp & keep_exp].sum()))
    return excess, retention, counts


def report(name: str, mc, exp, keep_mc, keep_exp, nu=None) -> None:
    for norm in ("global", "ref"):
        exc, ret, cnt = band_table(mc, exp, keep_mc, keep_exp, norm)
        ok = all(TARGET_LO <= e <= TARGET_HI for e in exc)
        line = "  ".join(f"{e:5.2f}" for e in exc)
        if norm == "global":
            tail = ("  | MC keep " + " ".join(f"{r:.2f}" for r in ret)
                    + f"  | exp {cnt}" + ("   PASS" if ok else ""))
        else:
            tail = "   PASS" if ok else ""
        print(f"{name:34s} {norm:>6s}  {line}{tail}")
    if nu is not None:
        print(f"{'':34s} {'nu':>6s}  " + "  ".join(
            f"{k} {v:.1%}" for k, v in nu.items()))


def main() -> None:
    feats_all = load()
    mc = feats_all[feats_all.group_id.isin(MC_GROUPS)].copy()
    exp = feats_all[feats_all.group_id.isin(EXP_GROUPS)].copy()
    cols = [c for c in feats_all.columns
            if c not in DROP and pd.api.types.is_numeric_dtype(feats_all[c])]
    top = feats_all[feats_all.group_id.isin(TOP_GROUPS)]
    x_top = top[cols].to_numpy(np.float64)
    y_top = (top.group_id == 8).astype(int).to_numpy()
    h_top, h_mc, h_exp = halves(top), halves(mc), halves(exp)

    nu = {n: feats_all[feats_all.group_id.isin(g)].copy()
          for n, g in NU_GROUPS.items()}
    h_nu = {n: halves(d) for n, d in nu.items()}

    print(f"{'model':34s} {'norm':>6s}  " +
          "  ".join(f"{f'{lo}-{hi}':>5s}" for lo, hi in BANDS))
    print(f"{'baseline (no cut)':34s} " + "-" * 40)
    report("baseline", mc, exp, np.ones(len(mc), bool), np.ones(len(exp), bool))

    for depth in (2, 3, 4):
        p_mc, p_exp = np.full(len(mc), np.nan), np.full(len(exp), np.nan)
        p_nu = {n: np.full(len(d), np.nan) for n, d in nu.items()}
        rules = []
        for k in (0, 1):
            tree = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=200, random_state=0)
            tree.fit(np.nan_to_num(x_top[h_top != k], nan=-999),
                     y_top[h_top != k])
            rules.append(export_text(tree, feature_names=cols, max_depth=depth))
            for d, p, h in ((mc, p_mc, h_mc), (exp, p_exp, h_exp)):
                xk = np.nan_to_num(d.loc[h == k, cols].to_numpy(np.float64),
                                   nan=-999)
                p[h == k] = tree.predict_proba(xk)[:, 1]
            for n, d in nu.items():
                xk = np.nan_to_num(d.loc[h_nu[n] == k, cols].to_numpy(np.float64),
                                   nan=-999)
                p_nu[n][h_nu[n] == k] = tree.predict_proba(xk)[:, 1]
        top_mc = in_band(mc, 0.9, 1.01)
        for q in (0.85, 0.80, 0.70):
            thr = np.quantile(p_mc[top_mc], q)
            nu_keep = {}
            for n, d in nu.items():
                hi_nu = (d.score >= 0.8).values
                nu_keep[n] = float((p_nu[n][hi_nu] <= thr).mean())
            report(f"tree depth {depth}, MC keep {q:.2f}", mc, exp,
                   p_mc <= thr, p_exp <= thr, nu_keep)
        print(f"\n--- depth {depth} rules (half 0 model) ---\n{rules[0]}")

    # upper bound on what ANY cut in this feature space can do
    p_mc, p_exp = np.full(len(mc), np.nan), np.full(len(exp), np.nan)
    for k in (0, 1):
        gbm = HistGradientBoostingClassifier(
            max_depth=4, max_iter=300, learning_rate=0.06,
            random_state=0).fit(x_top[h_top != k], y_top[h_top != k])
        for d, p, h in ((mc, p_mc, h_mc), (exp, p_exp, h_exp)):
            p[h == k] = gbm.predict_proba(
                d.loc[h == k, cols].to_numpy(np.float64))[:, 1]
    top_mc = in_band(mc, 0.9, 1.01)
    for q in (0.85, 0.70):
        thr = np.quantile(p_mc[top_mc], q)
        report(f"GBM (upper bound), MC keep {q:.2f}", mc, exp,
               p_mc <= thr, p_exp <= thr)


if __name__ == "__main__":
    main()
