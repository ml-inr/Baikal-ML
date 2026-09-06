"""A human-readable index of what is in ``data/``.

Twenty-odd parquet files with terse names and no catalogue is not a readable
directory.  The provenance sidecars already carry the stage, the row count and
the inputs; what they cannot carry is what the table is *for*.  That sentence
lives here, next to the code, and ``python src/catalogue.py`` prints the index.
"""
from __future__ import annotations

import json
from pathlib import Path

PURPOSE: dict[str, str] = {
    "00_exp_parts": "один экспериментальный ран: кластер, номер, интервал "
                    "времени, число событий, скорость счёта",
    "00_cluster_pairs": "пары ранов разных кластеров, перекрывающиеся по "
                        "настенному времени",
    "00_coincidence_dt": "спектр разностей времён между кластерами, поиск "
                         "совпадений в окне ±100 мкс",
    "00_coincidence_summary": "итог поиска совпадений по каждой паре ранов",
    "00_clock_offsets": "поиск совпадений на любом сдвиге часов, с калибровкой "
                        "чувствительности инъекцией",
    "00_mc_multicluster": "доля многокластерных событий в МК по классам",
    "10_mc_reference": "доля симулированных мюонов в каждой полосе скора — "
                       "знаменатель избытка",
    "10_run_noise": "один ран: шумовая нагрузка тремя способами и избыток по "
                    "полосам",
    "10_run_noise_correlations": "корреляции уровня и наклона профиля с прокси шума",
    "10_run_noise_partial": "те же корреляции по мере снятия конфаундов",
    "20_score_fit": "распределение по скору до и после перевзвешивания потока",
    "20_weights": "требуемый множитель к потоку по ячейкам зенита и энергии",
    "20_smoothness_sweep": "цена гладкости веса: качество подгонки против размаха",
    "20_achievable_bound": "точная верхняя граница усиления при ограниченном "
                           "размахе веса",
    "20_held_out": "расхождение по величинам, не участвовавшим в подгонке",
    "30_loo_residuals": "один выброшенный хит: остаток, расстояние, заряд, "
                        "качество якоря",
    "31_matched_cells": "ширина остатка при согласованной геометрии, exp против МК",
    "31_by_band": "то же по областям скора, рядом с избытком",
    "31_asymmetry": "ранняя и поздняя стороны остатка порознь — каскады или "
                    "калибровка",
    "40_jitter_parts": "скан по размытию времён, по частям",
    "40_jitter_summary": "скан по размытию: доля принятых и наведённый избыток",
    "41_jitter_residuals": "остатки размытой симуляции",
    "41_validation": "ширины остатков: МК, МК с размытием, эксперимент",
    "41_matched_calibration": "какое размытие нужно, чтобы совпасть с данными, "
                              "по ячейкам",
    "50_fp_features": "74 признака на событие для теста 5, без прорежения "
                      "мишени и проверки",
    "51_single_cuts": "одиночные каты: порог, убирающий заданную долю ложных "
                      "срабатываний",
    "51_cut_results": "каждое правило: что осталось от мюонов, эксперимента и "
                      "нейтрино",
    "51_stability": "то же, усреднённое по повторам, с разбросом между ними",
}


def index(data_dir: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(data_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        name = meta["artefact"].replace(".parquet", "")
        rows.append({"file": meta["artefact"], "stage": meta["stage"],
                     "rows": meta["rows"], "columns": len(meta["columns"]),
                     "purpose": PURPOSE.get(name, "— не описан —")})
    return rows


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    rows = index(here / "data")
    width = max(len(r["file"]) for r in rows)
    print(f"{'файл':<{width}}  {'строк':>9}  {'колонок':>7}  строка таблицы — это")
    print("-" * (width + 40))
    for row in rows:
        print(f"{row['file']:<{width}}  {row['rows']:>9,}  {row['columns']:>7}  "
              f"{row['purpose']}")
    missing = [r["file"] for r in rows if r["purpose"].startswith("—")]
    if missing:
        print(f"\nбез описания: {missing}")


if __name__ == "__main__":
    main()
