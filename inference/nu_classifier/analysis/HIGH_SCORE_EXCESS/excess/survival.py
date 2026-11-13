"""События для кривых выживания скора — без хитов и без фитов.

Отличие от `excess.high`: не читаются ни хиты, ни DQM, ни каталог фитов.
Берутся все доступные проскоренные события: тестовый МК (парты, не входившие
в обучение чекпоинта) и exp (без DA-target и без известных плохих ранов).
Поэтому загрузка лёгкая и не требует кэша.

Логика отбора — предикаты самого reader'а:
  * МК: `NOT_TRAINED` — события, не входившие в обучающий NPY-датасет
    (предсказания и так строились с исключением целых обучающих партов, так что
    на практике это ровно содержимое БД);
  * exp: `NOT_EXCLUDED` — вся проскоренная exp-популяция, включая события,
    бывшие DA-target. Для h8s3-модели DA-target покрывает ~93% чистых exp-событий
    (2.95M из 3.18M): датасет строился с cap 200k/парт, но на 27 партов это
    ~118k/парт, так что cap почти не связывал. Эти события не являются
    held-out по отношению к доменной ветке, зато это полная exp-популяция.

`NOT_TRAINED_PARTS` здесь не используется: `training.parts_json_path` не умеет
разрешать repo-relative путь к parts_json из build_config.yaml (см. ниже).

Предикты разных чекпоинтов могут быть построены с разными катами (5-0 или 8-3);
какой именно — видно по `n_sn_hits`/`n_sn_strings` в кадре.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import reader  # noqa: E402

MC_CLASSES = ("muatm_2020", "nuatm_2020", "nue2_2020")
KEYS = ("exp_full", "muatm", "nuatm", "nue2")
_KEEP = ["score", "n_sn_hits", "n_sn_strings"]


def load_events(checkpoint: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Загрузить проскоренные test-МК и exp события для одного чекпоинта.

    Возвращает ({exp_full, muatm, nuatm, nue2}, отчёт). В кадрах остаются только
    `score`, `n_sn_hits`, `n_sn_strings` и (для МК) `data_class`; `logit`
    добавляется здесь же. Truth не читается: для кривых выживания она не нужна.
    """
    exp = reader.open("exp_full", checkpoint=checkpoint).events(
        where=[reader.NOT_EXCLUDED], columns=_KEEP)
    exp = exp[_KEEP].copy()

    mc = reader.open("mc_merged", checkpoint=checkpoint).events(
        where=[reader.NOT_TRAINED], columns=_KEEP)
    mc = mc[_KEEP + ["data_class"]].copy()

    out: Dict[str, pd.DataFrame] = {"exp_full": exp}
    for cls, name in zip(MC_CLASSES, ("muatm", "nuatm", "nue2")):
        out[name] = mc[mc.data_class == cls].drop(columns="data_class").copy()

    for frame in out.values():
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["logit"] = np.log(frame["score"] / (1.0 - frame["score"]))

    report = pd.DataFrame({
        "класс": list(out),
        "событий": [len(out[k]) for k in out],
        "соскором": [int(out[k]["score"].notna().sum()) for k in out],
        "n_sn_hits>=8": [int((out[k]["n_sn_hits"] >= 8).sum()) for k in out],
    })
    return out, report
