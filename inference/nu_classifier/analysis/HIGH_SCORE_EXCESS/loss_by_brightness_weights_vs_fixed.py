#!/usr/bin/env python
"""Где весовое обучение потеряло на muatm и выиграло на нейтрино.

Сравнивает невзвешенный focal-лосс (та же функция, что в обучении, но без
per-event весов) двух чекпоинтов на **общих** событиях их mc_merged-баз
(пересечение event_fk внутри класса: пулы различаются только парт-исключением).
Стратификация — по `n_sn_hits` (ось, не зависящая от модели: прокси яркости
события) и по скору FIXED-модели; для каждой полосы — Δлосса с парной ошибкой,
относительное изменение, вклад в общий Δ и сдвиг скоров (P>0.5, P>0.9).

Метки — по классу (nuatm/nue2 -> 1, muatm -> 0), как их писал builder датасета;
класс берётся по границам event_fk, проверенным по catalog_v2 2026-10-11.

Запуск: python inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/loss_by_brightness_weights_vs_fixed.py
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").exists())
PREDS = ROOT / "inference/nu_classifier/preds"

RUNS = {
    "fix": "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model",
    "wgt": "261002_0804_da_nu_classifier_exp_full_E1_lambda0.01_DArepair_weights@best_da_model",
}

#: Границы классов в event_fk (сплошные интервалы текущего catalog_v2; сверять
#: агрегатным запросом при пересборке каталога).
BOUND_NUATM, BOUND_NUE2 = 576_947_380, 643_643_317
#: pos_weight тренера: w[1]/w[0] из class weights обучающей половины.
POS_WEIGHT = 2_249_911 / 2_250_089
GAMMA = 2.0

HIT_EDGES = [5, 8, 12, 20, 40, 80, np.inf]
SCORE_EDGES = [0.0, 0.01, 0.1, 0.5, 0.9, 1.0]
CLASSES = [("muatm_2020", 0), ("nuatm_2020", 1), ("nue2_2020", 2)]


def read(db: Path) -> dict[str, np.ndarray]:
    connection = duckdb.connect(str(db), read_only=True)
    columns = connection.execute(
        "SELECT event_fk, score, n_sn_hits FROM predictions").fetchnumpy()
    connection.close()
    return {key: value for key, value in columns.items()}


def class_code(fk: np.ndarray) -> np.ndarray:
    """0 = muatm, 1 = nuatm, 2 = nue2 по интервалам event_fk."""
    return ((fk > BOUND_NUATM).astype(np.int8)
            + (fk > BOUND_NUE2).astype(np.int8))


def focal_loss(fk: np.ndarray, score: np.ndarray) -> np.ndarray:
    """focal(gamma) с pos_weight тренера, без per-event весов."""
    positive = fk > BOUND_NUATM
    p = score.astype(np.float64)
    p_t = np.where(positive, p, 1.0 - p)
    loss = (1.0 - p_t) ** GAMMA * (-np.log(p_t))
    return np.where(positive, POS_WEIGHT, 1.0) * loss


def band_table(label: str, fk: np.ndarray, score_fix: np.ndarray,
               score_wgt: np.ndarray, n_hits: np.ndarray,
               loss_fix: np.ndarray, loss_wgt: np.ndarray) -> None:
    """Лосс и сдвиг скоров по полосам n_sn_hits; вклад полос в общий Δ."""
    delta = loss_wgt - loss_fix
    total = delta.mean()
    print(f"\n== {label}: общих событий {len(fk):,}, "
          f"Δ(среднее) = {total:+.6f}")
    print(f"{'n_sn_hits':>12s} {'N':>11s} {'loss_fix':>9s} {'loss_wgt':>9s} "
          f"{'Δ':>10s} {'±SE':>7s} {'Δотн.%':>7s} {'доляΔ':>7s} "
          f"{'P>.5 fix':>8s} {'wgt':>6s} {'P>.9 fix':>8s} {'wgt':>6s}")
    for lo, hi in zip(HIT_EDGES[:-1], HIT_EDGES[1:]):
        band = (n_hits >= lo) & (n_hits < hi)
        n = int(band.sum())
        if n == 0:
            continue
        d = delta[band]
        se = d.std() / np.sqrt(n)
        name = f"{int(lo)}-{'inf' if np.isinf(hi) else int(hi)}"
        print(f"{name:>12s} {n:>11,d} {loss_fix[band].mean():>9.6f} "
              f"{loss_wgt[band].mean():>9.6f} {d.mean():>+10.6f} {se:>7.1e} "
              f"{100 * d.mean() / loss_fix[band].mean():>+7.1f} "
              f"{100 * band.sum() * d.mean() / len(fk) / total:>+7.1f} "
              f"{np.mean(score_fix[band] > 0.5):>8.2e} "
              f"{np.mean(score_wgt[band] > 0.5):>6.2e} "
              f"{np.mean(score_fix[band] > 0.9):>8.1e} "
              f"{np.mean(score_wgt[band] > 0.9):>6.1e}")


def score_region_table(label: str, score_fix: np.ndarray,
                       loss_fix: np.ndarray, loss_wgt: np.ndarray) -> None:
    """Та же разница, но полосы — по скору FIXED-модели (общая ось)."""
    delta = loss_wgt - loss_fix
    total = delta.mean()
    print(f"\n-- {label}: Δ по полосам скора FIXED (общая ось)")
    print(f"{'score_fix':>14s} {'N':>11s} {'событ.%':>8s} {'loss_fix':>9s} "
          f"{'loss_wgt':>9s} {'Δ':>10s} {'Δотн.%':>7s} {'доляΔ':>7s}")
    for lo, hi in zip(SCORE_EDGES[:-1], SCORE_EDGES[1:]):
        band = (score_fix >= lo) & (score_fix < hi)
        n = int(band.sum())
        if n == 0:
            continue
        d = delta[band]
        print(f"{lo:>6.2f}-{hi:<6.2f} {n:>11,d} {100 * n / len(score_fix):>7.2f}% "
              f"{loss_fix[band].mean():>9.6f} {loss_wgt[band].mean():>9.6f} "
              f"{d.mean():>+10.6f} "
              f"{100 * d.mean() / loss_fix[band].mean():>+7.1f} "
              f"{100 * band.sum() * d.mean() / len(score_fix) / total:>+7.1f}")


def main() -> None:
    t0 = time.time()
    data = {}
    for tag, run in RUNS.items():
        data[tag] = read(PREDS / run / "mc_merged_thr0p8.duckdb")
        print(f"{tag}: {len(data[tag]['event_fk']):,} событий "
              f"({time.time() - t0:.0f} с)")
        data[tag]["class"] = class_code(data[tag]["event_fk"])

    # -- общие события: сортируем fk и пересекаем ---------------------------
    for tag in RUNS:
        order = np.argsort(data[tag]["event_fk"])
        data[tag] = {key: value[order] for key, value in data[tag].items()}
    common = np.intersect1d(data["fix"]["event_fk"], data["wgt"]["event_fk"],
                            assume_unique=True)
    take = {tag: np.searchsorted(data[tag]["event_fk"], common)
            for tag in RUNS}
    print(f"\nобщих событий: {len(common):,} "
          f"(fix {len(data['fix']['event_fk']):,}, wgt {len(data['wgt']['event_fk']):,}) "
          f"за {time.time() - t0:.0f} с")

    fk = common
    code = class_code(fk)
    loss = {tag: focal_loss(fk, data[tag]["score"][take[tag]]) for tag in RUNS}
    n_hits = data["fix"]["n_sn_hits"][take["fix"]]
    print("контроль: loss(fix).mean() = "
          f"{loss['fix'].mean():.6f}, loss(wgt).mean() = {loss['wgt'].mean():.6f}")

    for name, code_value in CLASSES:
        selected = code == code_value
        print(f"\n{'=' * 78}\nкласс {name}: {int(selected.sum()):,} общих событий")
        band_table(name, fk[selected], data["fix"]["score"][take["fix"]][selected],
                   data["wgt"]["score"][take["wgt"]][selected],
                   n_hits[selected], loss["fix"][selected], loss["wgt"][selected])
        score_region_table(name, data["fix"]["score"][take["fix"]][selected],
                           loss["fix"][selected], loss["wgt"][selected])
    print(f"\nвсего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
