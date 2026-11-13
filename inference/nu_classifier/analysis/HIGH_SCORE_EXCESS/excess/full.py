"""Данные для ноутбуков full_score_range.

Один запрос собирает кадр событий: фиты BARS + класс/парт из каталога v2 +
скоры из двух prediction-БД (exp_full и mc_merged). Дальше ноутбук к базе не
обращается — весь анализ идёт по этому кадру.

Вынесено из пер-чекпоинтных ноутбуков (2026-10-25); SQL и приведения типов
сохранены дословно.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import reader  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
PRED_ROOT = ROOT / "inference" / "nu_classifier" / "preds"
CATALOG = ROOT / "data_manager" / "catalog_v2.duckdb"
PRED_THRESHOLD_TAG = "thr0p8"

ORDER = ["exp_full", "muatm_2020", "nuatm_2020", "nue2_2020"]
MC_CLASSES = reader.SOURCES["mc_merged"].groups

#: Колонки фита, которые реально используются в full-анализе (ячейки + quality()).
#: Остальные ~29 колонок каталога не материализуются: 44M строк × лишние поля —
#: это и был основной расход времени (~270 с из ~290).
FIT_COLUMNS = (
    "event_fk", "source", "part_key", "local_idx", "block", "run_id",
    "status", "theta_rec", "theta_err", "time_chi2",
    "n_hits", "n_strings", "cov_matrix_status",
    "scf_max_theta", "scf_min_theta",
    "z_dist", "n_triplets", "ev_center_z", "func_value",
    "q_acc_20", "q_tot_20", "t_res_30_med", "log_phit_30_min",
)


def load_events(checkpoint: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Собрать кадр событий для ноутбука full_score_range.

    Возвращает (events, covered). Колонки и типы — как в исходном ноутбуке;
    `covered` — маска событий из частей, покрытых предсказаниями.
    """
    pred_dir = PRED_ROOT / checkpoint
    catalogue = reader.open_fits()

    con = catalogue.connection()
    for source in ("exp_full", "mc_merged"):
        con.execute(
            f"ATTACH '{pred_dir}/{source}_{PRED_THRESHOLD_TAG}.duckdb' "
            f"AS pred_{source} (READ_ONLY)")
    con.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")

    # Скоры живут в двух базах: данные и МС скорились отдельными прогонами.
    con.execute("""
        CREATE VIEW scores AS
        SELECT * FROM pred_exp_full.predictions
        UNION ALL SELECT * FROM pred_mc_merged.predictions
    """)

    fit_columns = ", ".join(f"f.{name}" for name in FIT_COLUMNS)
    print("читаю каталог фитов... (~2 минуты)")
    events = con.execute(f"""
        SELECT {fit_columns},
               m.data_class,
               ok.part_key IS NOT NULL AS scored_part,
               p.score, p.n_sn_hits, p.n_sn_strings
        FROM fitness.fits f
        JOIN {catalogue.manifest_sql()} m USING (event_fk)
        LEFT JOIN (SELECT DISTINCT e.data_class, l.part_key
                   FROM scores s JOIN cat.events e ON e.id = s.event_fk
                   JOIN cat.h5_locations l ON l.event_fk = e.id) ok
          ON ok.data_class = m.data_class AND ok.part_key = f.part_key
        LEFT JOIN scores p USING (event_fk)
    """).df()

    events["data_class"] = events.data_class.astype(
        pd.CategoricalDtype(ORDER, ordered=True))
    for column in ("source", "part_key", "run_id", "status"):
        events[column] = events[column].astype("category")
    # У непроскоренных n_sn_hits — NULL; в float64 это NaN, и любое сравнение с
    # ним даёт False, то есть «не прошёл кат» -- ровно нужная семантика.
    events["n_sn_hits"] = events.n_sn_hits.astype("float64")
    events["n_sn_strings"] = events.n_sn_strings.astype("float64")

    covered = events.scored_part
    print(f"событий {len(events):,}, из них в покрытых частях {int(covered.sum()):,}")
    print("проверка: со скором вне покрытых частей —",
          int((events.score.notna() & ~covered).sum()))
    return events, covered


def attach_truth(events: pd.DataFrame) -> pd.DataFrame:
    """Добавить truth первичной частицы (МК-only) в кадр full-ноутбука.

    Используется ровно тот же примитив читалки, что и в `Reader.stream`:
    `inference.reader.parts.event_scalars` (Handles + `prime_prty` по парту).
    Читаются только маленькие `prime_prty`-массивы, хиты не поднимаются.
    Колонки и их имена — из `reader.PRIME_PRTY_COLUMNS`; у данных (exp_full)
    колонки остаются NaN.

    Строка hits-таблицы — кластер; у многокластерного события truth повторяется,
    как и в reader. Для спектров дедуплицируйте по `event_fk` и используйте
    `prime_event_weight`.
    """
    columns = {f"prime_{name}": np.full(len(events), np.nan, dtype=np.float32)
               for name in reader.PRIME_PRTY_COLUMNS}
    mc_pos = np.flatnonzero(events.data_class.isin(MC_CLASSES).to_numpy())
    if len(mc_pos):
        spec = reader.SOURCES["mc_merged"]
        handles = reader.parts.Handles()
        try:
            mc = (events.iloc[mc_pos][["data_class", "part_key", "local_idx"]]
                  .reset_index(drop=True))
            for (data_class, part_key), block in mc.groupby(
                    ["data_class", "part_key"], observed=True):
                extra = reader.parts.event_scalars(
                    handles, spec, str(data_class), part_key,
                    block["local_idx"].to_numpy(np.int64))
                target = mc_pos[block.index.to_numpy()]
                for name, values in extra.items():
                    columns[name][target] = values
        finally:
            handles.close()
    for name, values in columns.items():
        events[name] = values
    return events


def domain_frames(events: pd.DataFrame) -> dict:
    """Кадры exp/muatm/nuatm/nue2, как их определяли ноутбуки.

    Непроскоренные события отбрасываются (`scored_part`); `logit` считается из
    сохранённого float32-скора.
    """
    events["logit"] = np.log(events["score"] / (1.0 - events["score"]))
    return {
        "exp":   events[(events.data_class == "exp_full") & events.scored_part],
        "muatm": events[(events.data_class == "muatm_2020") & events.scored_part],
        "nuatm": events[(events.data_class == "nuatm_2020") & events.scored_part],
        "nue2":  events[(events.data_class == "nue2_2020") & events.scored_part],
    }


def quality(frame: pd.DataFrame) -> pd.Series:
    """Каты реконструкции: отбор по качеству самого фита, без катов на нейтринность."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            (frame.status == "ok")
            & (frame.n_hits >= 8) & (frame.n_strings >= 2)
            & (frame.cov_matrix_status == 3)
            & (frame.log_phit_30_min > -9)
            & (frame.ev_center_z < 220)
            & (frame.z_dist > 70)
            & (np.log10(frame.func_value / (frame.n_hits - 5)) < 1.1)
            & (frame.n_triplets / frame.n_hits > 0.1)
            & (frame.theta_err < 2.5)
        )


def by_class(mask: pd.Series, data_class: pd.Series) -> pd.Series:
    """Сколько событий каждого класса проходит маску."""
    return mask.groupby(data_class, observed=True).sum()
