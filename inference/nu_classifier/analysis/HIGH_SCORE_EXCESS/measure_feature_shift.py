"""Где именно exp-хвост отличается от MC: по признакам, а не только по kNN.

Читает таблицу признаков, сохранённую `support_vs_measure.py`, и сравнивает
четыре популяции:

  exp_tail  vs mc_tail_hold   — доменный сдвиг в области высокого скора;
  exp_low   vs mc_low_hold    — контроль: сдвиг там, где избытка нет;
  exp_tail  vs exp_low        — что меняется вместе со скором внутри exp;
  mc_tail_hold vs mc_bg       — что меняется вместе со скором внутри MC.

Одномерная таблица: AUC каждого признака для каждой пары и «хвостовая
специфичность» = (AUC_tail - 0.5) - (AUC_low - 0.5). Признак, который
сдвинут и в хвосте, и в контроле, — глобальный доменный; признак, сдвинутый
только в хвосте, — кандидат в механизм избытка.

Многомерная часть: HistGradientBoosting с 3-фолдовым CV отдельно по группам
признаков. Высокий AUC на exp_tail vs MC — это разделимость популяций, а не
доказательство физики: контроль low показывает, сколько из неё объясняется
глобальным сдвигом домена.

Запуск:
    python .../measure_feature_shift.py
    python .../measure_feature_shift.py --features data/features_...parquet
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from excess import paths, raw_features  # noqa: E402

DEFAULT_FEATURES = (paths.DATA
                    / "features_signal_expanded_h8s3.parquet")
MC_MUATM_MAX = 576_947_380
#: Провенанс отбора, не признаки: группа selection из анализа убрана.
PROVENANCE_COLS = ("n_hits_raw", "n_bright_raw", "n_signal_hits",
                   "n_signal_strings", "signal_fraction")


def _groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Именованные наборы колонок; новые — из raw_features.GROUPS."""
    groups = {k: [c for c in v if c in df.columns]
              for k, v in raw_features.GROUPS.items()}
    groups["strings"] = [c for c in df.columns
                         if c.startswith("string_")
                         or c in ("n_strings_fit", "tilt_frac_gt3")]
    groups["pca_base"] = [
        c for c in ("log_pca_l1", "log_pca_l2", "log_pca_l3",
                    "pca_l2_l1", "pca_l3_l1", "pca_verticality")
        if c in df.columns]
    groups["event_base"] = [
        "n_hits", "n_bright", "n_strings", "n_strings_bright",
        "log_charge_total", "log_charge_bright", "log_brightest",
        "log_charge_mean", "log_charge_std", "log_charge_q95",
        "log_time_span", "log_time_span_bright", "time_std",
        "time_q05", "time_q95", "log_z_span", "log_z_span_bright",
        "log_z_std", "log_x_std", "log_y_std", "log_radial_std",
        "log_xy_span", "hits_per_string_mean", "hits_per_string_max",
        "log_hits_per_string_std",
    ]
    groups["event_base"] = [c for c in groups["event_base"] if c in df.columns]

    # Сводные наборы для многомерной проверки.
    groups["all_no_geometry"] = (
        groups["event_base"] + groups["basic"] + groups["charge"]
        + groups["time"] + groups["lightcurve"]
        + groups["charge_geometry"] + groups["occupancy"])
    groups["all_no_strings"] = (
        groups["all_no_geometry"] + groups["pca_base"]
        + groups["track"] + groups["fits"])
    return groups


def _clean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    return X


def _auc(x: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = x[mask_a], x[mask_b]
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 10 or b.size < 10:
        return float("nan")
    y = np.r_[np.zeros(a.size), np.ones(b.size)]
    return float(roc_auc_score(y, np.r_[a, b]))


def _median_iqr(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return float(np.median(x)), float(q75 - q25)


def univariate(df: pd.DataFrame, features: list[str]) -> list[dict]:
    g = df["grp"]
    m = {name: (g == name).to_numpy() for name in
         ("exp_tail", "exp_low", "mc_tail_ref", "mc_tail_hold",
          "mc_low_hold", "mc_bg")}
    rows = []
    for col in features:
        x = df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(x).any():
            continue
        auc_tail = _auc(x, m["exp_tail"], m["mc_tail_hold"])
        auc_low = _auc(x, m["exp_low"], m["mc_low_hold"])
        auc_exp_score = _auc(x, m["exp_tail"], m["exp_low"])
        auc_mc_score = _auc(x, m["mc_tail_hold"], m["mc_bg"])
        # Отрицательный контроль: половины MC разделены случайно, AUC ~ 0.5.
        auc_mc_halves = _auc(x, m["mc_tail_ref"], m["mc_tail_hold"])
        med_e, iqr_e = _median_iqr(x[m["exp_tail"]])
        med_m, iqr_m = _median_iqr(x[m["mc_tail_hold"]])
        rows.append({
            "feature": col,
            "auc_tail": auc_tail,
            "auc_low": auc_low,
            "tail_specific": (auc_tail - 0.5) - (auc_low - 0.5),
            "tail_specific_abs": abs(auc_tail - 0.5) - abs(auc_low - 0.5),
            "auc_exp_score": auc_exp_score,
            "auc_mc_score": auc_mc_score,
            "auc_mc_halves": auc_mc_halves,
            "median_exp_tail": med_e,
            "iqr_exp_tail": iqr_e,
            "median_mc_tail": med_m,
            "iqr_mc_tail": iqr_m,
        })
    return rows


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        random_state=42)


def _cv_auc(X: np.ndarray, y: np.ndarray,
            folds: int = 3) -> tuple[float, list[float]]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        model = _model().fit(X[tr], y[tr])
        aucs.append(float(roc_auc_score(
            y[te], model.predict_proba(X[te])[:, 1])))
    return float(np.mean(aucs)), aucs


def multivariate(df: pd.DataFrame, groups: dict[str, list[str]]) -> dict:
    """CV AUC разделимости и важности признаков по группам."""
    g = df["grp"]
    pairs = {
        "tail": (df["grp"] == "exp_tail").to_numpy(),
        "low": (df["grp"] == "exp_low").to_numpy(),
    }
    ref = {
        "tail": (df["grp"] == "mc_tail_hold").to_numpy(),
        "low": (df["grp"] == "mc_low_hold").to_numpy(),
    }
    out = {}
    for name, cols in groups.items():
        if len(cols) < 2:
            continue
        X_all = _clean(df[cols].to_numpy(dtype=np.float64))
        rec = {}
        for pair in ("tail", "low"):
            mask = pairs[pair] | ref[pair]
            X = X_all[mask]
            y = pairs[pair][mask].astype(np.int8)
            # Медианы заполняются по всей выборке: это описательная оценка,
            # утечка между фолдами отсутствует (признаки не обучаемые).
            col_med = np.nanmedian(X, axis=0)
            col_med[~np.isfinite(col_med)] = 0.0
            X = np.where(np.isfinite(X), X, col_med)
            auc_mean, auc_folds = _cv_auc(X, y)
            rec[f"auc_{pair}"] = auc_mean
            rec[f"auc_{pair}_folds"] = auc_folds
            if pair == "tail":
                model = _model().fit(X, y)
                imp = permutation_importance(
                    model, X, y, n_repeats=2, random_state=42,
                    scoring="roc_auc", max_samples=0.5)
                order = np.argsort(imp.importances_mean)[::-1][:10]
                rec["top_importance"] = [
                    {"feature": cols[i],
                     "importance": float(imp.importances_mean[i])}
                    for i in order]
        out[name] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    print(f"read {args.features}: {len(df):,} events, {len(df.columns)} columns",
          flush=True)
    print(df["grp"].value_counts().to_string(), flush=True)
    groups = _groups(df)
    all_features = [c for c in df.columns
                    if c not in ("event_fk", "grp", "score") + PROVENANCE_COLS]

    uni = univariate(df, all_features)
    print(f"\nunivariate: {len(uni)} features", flush=True)
    uni_df = pd.DataFrame(uni)
    cols = ["feature", "auc_tail", "auc_low", "tail_specific",
            "auc_exp_score", "auc_mc_score", "auc_mc_halves"]
    print("\nТоп по tail_specific (сдвиг в хвосте сильнее, чем в контроле):")
    print(uni_df.reindex(uni_df["tail_specific"].abs()
                         .sort_values(ascending=False).index)[cols]
          .head(25).to_string(index=False), flush=True)
    print("\nТоп по |AUC_tail-0.5| (просто сильное различие с MC):")
    print(uni_df.reindex((uni_df["auc_tail"] - 0.5).abs()
                         .sort_values(ascending=False).index)[cols]
          .head(25).to_string(index=False), flush=True)

    multi = multivariate(df, groups)
    print("\nМногомерная разделимость (3-fold CV AUC):")
    for name, rec in sorted(multi.items(),
                            key=lambda kv: -max(kv[1].get("auc_tail", 0),
                                                kv[1].get("auc_low", 0))):
        sd_t = float(np.std(rec["auc_tail_folds"]))
        sd_l = float(np.std(rec["auc_low_folds"]))
        print(f"  {name:18s} tail={rec['auc_tail']:.3f}±{sd_t:.3f} "
              f"low={rec['auc_low']:.3f}±{sd_l:.3f} "
              f"diff={rec['auc_tail']-rec['auc_low']:+.3f}", flush=True)

    out = Path(args.out) if args.out else (
        paths.DATA / "feature_shift_signal_expanded_h8s3.json")
    payload = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "features": args.features,
        "groups": {k: len(v) for k, v in groups.items()},
        "univariate": uni,
        "multivariate": multi,
        "note": (
            "auc_tail = exp_tail vs mc_tail_hold; auc_low = exp_low vs "
            "mc_low_hold (контроль глобального доменного сдвига); "
            "tail_specific = (auc_tail-0.5)-(auc_low-0.5). "
            "Многомерный AUC — разделимость популяций, не доказательство "
            "физики; high-контроль показывает вклад глобального сдвига."),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwritten {out}", flush=True)


if __name__ == "__main__":
    main()
