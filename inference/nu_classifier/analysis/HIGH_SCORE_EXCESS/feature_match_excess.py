"""Объясняет ли сдвиг сырых признаков избыток exp в хвосте скора?

Идея: если exp отличается от MC только распределением признаков x, то,
перевзвесив MC к exp по x, мы воспроизведём и долю хвоста. Оценка веса —
отношение плотностей p_exp(x)/p_mc(x), полученное классификатором домена
(HistGradientBoosting, 3 фолда, out-of-fold предсказания), с весами
стратифицированной выборки MC (хвост полный, low/фон — подвыборки).

Два режима, и они отвечают на разные вопросы:

  fit=low   — веса учатся ТОЛЬКО на событиях score <= 0.5 (exp_low vs MC low),
              затем применяются к MC tail+low. Это консервативный тест:
              сдвиг в хвосте не участвует в обучении весов.
  fit=all   — веса учатся на exp(tail+low) vs MC(tail+low). Это медиационный
              тест: насколько различие скора опосредовано признаками x.
              Он не независим — хвост участвует в оценке весов.

Сравниваем условную долю хвоста среди score > -2:
  f = sum_{tail} w / sum_{tail+low} w,   w = base_weight * odds(x).
  explained = (f_mc_adjusted - f_mc_base) / (f_exp - f_mc_base).

Критично смотреть ESS и долю обрезанных весов: при почти идеальной
разделимости домена (strings AUC ~1) веса экстремальны и оценка хрупкая.
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
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from excess import paths  # noqa: E402
import measure_feature_shift as mfs  # noqa: E402

DEFAULT_FEATURES = paths.DATA / "features_signal_expanded_h8s3.parquet"
COUNTS_JSON = (paths.DATA
               / "support_vs_measure_signal_expanded_h8s3_muOnlyDA_e23_tail0.json")
TAIL_GROUPS = ("exp_tail",)
LOW_GROUPS = ("exp_low",)
MC_TAIL = ("mc_tail_ref", "mc_tail_hold")
MC_LOW = ("mc_low_ref", "mc_low_hold")


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.12,
        random_state=42)


def _odds(p: np.ndarray, cap: float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.clip(p / (1.0 - p), 1.0 / cap, cap)


def _fit_cv(X: np.ndarray, y: np.ndarray, w: np.ndarray, cols: list[str],
            folds: int, cap: float):
    """OOF p(exp|x) и обученные фолд-модели."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    p_oof = np.full(len(X), np.nan)
    models = []
    for tr, te in skf.split(X, y):
        model = _model().fit(X[tr], y[tr], sample_weight=w[tr])
        p_oof[te] = model.predict_proba(X[te])[:, 1]
        models.append(model)
    return p_oof, models


def _predict_bagged(models, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.empty(0)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def evaluate_group(df, cols, base, mode, folds, cap, n_boot, seed,
                   null_seed=None):
    """Вернуть f_mc0, f_mc, f_exp, explained, ESS и бутстрап-интервал."""
    g = df["grp"].to_numpy()
    is_exp = np.isin(g, TAIL_GROUPS + LOW_GROUPS)
    is_mc = np.isin(g, MC_TAIL + MC_LOW)
    tail_exp = np.isin(g, TAIL_GROUPS)
    tail_mc = np.isin(g, MC_TAIL)

    if mode == "low":
        fit_mask = (np.isin(g, LOW_GROUPS) | np.isin(g, MC_LOW))
    elif mode == "all":
        fit_mask = is_exp | is_mc
    else:
        raise ValueError(mode)

    X = df[cols].to_numpy(dtype=np.float64)
    fit_idx = np.flatnonzero(fit_mask)
    y = is_exp[fit_idx].astype(np.int8)
    if null_seed is not None:
        # Нулевой контроль: доменные метки перемешаны, различия признаков
        # между exp и MC больше нет; explained должен быть ~0.
        y = np.random.default_rng(null_seed).permutation(y)
    p_oof, models = _fit_cv(X[fit_idx], y, base[fit_idx], cols, folds, cap)

    # Веса для всех MC tail+low: low — OOF, tail — bagged (fit=low).
    mc_idx = np.flatnonzero(is_mc)
    p_mc = np.full(mc_idx.size, np.nan)
    fit_pos = {j: i for i, j in enumerate(fit_idx)}
    for k, j in enumerate(mc_idx):
        if j in fit_pos:
            p_mc[k] = p_oof[fit_pos[j]]
    missing = ~np.isfinite(p_mc)
    if missing.any():
        p_mc[missing] = _predict_bagged(models, X[mc_idx[missing]])
    odds_mc = _odds(p_mc, cap)

    exp_idx = np.flatnonzero(is_exp)
    w_mc = base[mc_idx] * odds_mc
    base_mc = base[mc_idx]
    te_mc = tail_mc[mc_idx]
    te_exp = tail_exp[exp_idx]
    b_exp = base[exp_idx]

    f_m0 = float(base_mc[te_mc].sum() / base_mc.sum())
    f_m = float(w_mc[te_mc].sum() / w_mc.sum())
    f_e = float(b_exp[te_exp].sum() / b_exp.sum())
    denom = f_e - f_m0
    expl = (f_m - f_m0) / denom if denom > 0 else float("nan")
    ess = float(w_mc.sum() ** 2 / (w_mc ** 2).sum() / len(w_mc))
    cap_frac = float(np.mean((odds_mc >= cap) | (odds_mc <= 1.0 / cap)))
    odds_tail = float(odds_mc[te_mc].mean()) if te_mc.any() else float("nan")
    odds_low = float(odds_mc[~te_mc].mean()) if (~te_mc).any() else float("nan")

    # Бутстрап по событиям MC tail+low и exp tail+low (веса фиксированы).
    n_mc, n_exp = len(w_mc), len(b_exp)
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        f_m_bs, f_e_bs = [], []
        for _ in range(n_boot):
            c_mc = rng.multinomial(n_mc, np.full(n_mc, 1.0 / n_mc))
            c_exp = rng.multinomial(n_exp, np.full(n_exp, 1.0 / n_exp))
            f_m_bs.append(float((c_mc * w_mc * te_mc).sum()
                                / (c_mc * w_mc).sum()))
            f_e_bs.append(float((c_exp * b_exp * te_exp).sum()
                                / (c_exp * b_exp).sum()))
        f_m_bs, f_e_bs = np.asarray(f_m_bs), np.asarray(f_e_bs)
        expl_bs = (f_m_bs - f_m0) / (f_e_bs - f_m0)
        lo, hi = np.quantile(expl_bs, [0.16, 0.84])
    else:
        lo = hi = float("nan")

    return {
        "mode": mode,
        "n_features": len(cols),
        "n_fit": int(fit_mask.sum()),
        "n_mc_eval": int(n_mc),
        "n_exp_eval": int(n_exp),
        "f_mc_base": f_m0,
        "f_mc_adjusted": f_m,
        "f_exp": f_e,
        "explained": expl,
        "explained_68ci": [float(lo), float(hi)],
        "ess_frac": ess,
        "capped_frac": cap_frac,
        "odds_tail_mean": odds_tail,
        "odds_low_mean": odds_low,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--counts", default=str(COUNTS_JSON))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--groups", default=None,
                    help="comma-separated subset of feature groups")
    ap.add_argument("--modes", default="low,all")
    ap.add_argument("--null", type=int, default=0,
                    help="number of permuted-domain-label controls per group")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    counts = json.loads(Path(args.counts).read_text())["counts"]
    # Вес стратифицированной выборки: хвост полный, low — подвыборка.
    exp_low_w = counts["exp_low_db"] / counts["exp_low_sample"]
    mc_low_w = counts["mc_low"] / counts["mc_low_sample"]

    base = np.ones(len(df), dtype=np.float64)
    base[np.isin(df.grp, LOW_GROUPS)] = exp_low_w
    base[np.isin(df.grp, MC_LOW)] = mc_low_w

    groups = mfs._groups(df)
    if args.groups:
        keep = set(args.groups.split(","))
        groups = {k: v for k, v in groups.items() if k in keep}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(f"read {args.features}: {len(df):,} events", flush=True)
    print(f"weights: exp_low={exp_low_w:.4f} mc_low={mc_low_w:.4f}", flush=True)
    print(f"groups: {list(groups)}", flush=True)
    results = {}
    for mode in modes:
        for name, cols in groups.items():
            if len(cols) < 2:
                continue
            rec = evaluate_group(df, cols, base, mode, args.folds,
                                 args.cap, args.bootstrap, seed=42)
            results[f"{mode}:{name}"] = rec
            print(f"{mode:4s} {name:16s} f_mc {rec['f_mc_base']:.4f}"
                  f" -> {rec['f_mc_adjusted']:.4f}  f_exp {rec['f_exp']:.4f}"
                  f"  explained {rec['explained']:+.3f}"
                  f" [{rec['explained_68ci'][0]:+.3f},"
                  f" {rec['explained_68ci'][1]:+.3f}]"
                  f"  ESS {rec['ess_frac']:.3f}"
                  f"  cap {rec['capped_frac']:.2f}", flush=True)

    null_results = {}
    if args.null:
        for mode in modes:
            for name, cols in groups.items():
                if len(cols) < 2:
                    continue
                vals = []
                for s in range(args.null):
                    rec = evaluate_group(df, cols, base, mode, args.folds,
                                         args.cap, 0, seed=42,
                                         null_seed=10_000 + s)
                    vals.append(rec["explained"])
                null_results[f"{mode}:{name}"] = vals
                print(f"null {mode:4s} {name:16s} "
                      f"{np.mean(vals):+.3f} +- {np.std(vals):.3f}",
                      flush=True)

    out = Path(args.out) if args.out else (
        paths.DATA / "feature_match_excess_h8s3.json")
    payload = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "features": args.features,
        "counts": counts,
        "weights": {"exp_low": exp_low_w, "mc_low": mc_low_w},
        "folds": args.folds, "cap": args.cap,
        "bootstrap": args.bootstrap,
        "results": results,
        "null": null_results,
        "note": (
            "f_mc_base -> f_mc_adjusted — условная доля хвоста среди score>-2 "
            "до и после перевзвешивания MC к exp по признакам группы; "
            "f_exp — та же доля в exp. explained — какая часть разрыва "
            "f_exp - f_mc_base объяснена. fit=low — веса только по низкому "
            "скору (консервативно); fit=all — включая хвост (медиация). "
            "ESS доля и capped_frac показывают, насколько веса устойчивы."),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
