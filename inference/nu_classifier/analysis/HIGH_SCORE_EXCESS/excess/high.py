"""Данные и кэш для ноутбуков high_score_range.

Вынесено из пер-чекпоинтных ноутбуков (2026-10-25). Логика не менялась:
домены, условия отбора, прореживание по партам, DQM-фильтр сигнальных хитов,
признаки из хитов, кэш parquet с манифестом-отпечатком, фит BARS.

Кэш остаётся в `cached_data/<CHECKPOINT>/` (parquet + manifest.json). Старые
манифесты читаются как раньше; в новые дополнительно пишется идентичность
prediction-БД (путь, mtime, размер), чтобы перезапись DuckDB под тем же именем
чекпоинта инвалидировала кэш.
"""
from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import reader  # noqa: E402
from inference.track_fit_alike_bars import (  # noqa: E402
    fit_events, masks, tqdm_progress,
)
from .features import per_event_features, signal_hit_geometry  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
CACHE_ROOT = ANALYSIS_DIR / "cached_data"
PRED_ROOT = ROOT / "inference" / "nu_classifier" / "preds"
PRED_THRESHOLD_TAG = "thr0p8"

#: Метка фильтра в отпечатке кеша: меняется -- кеш пересобирается заново.
DQM_FILTER_TAG = "signal hits on DQM-off channels dropped, v1"
SCORE_CUT, N_WORKERS = 0.5, 16

# Домен -> (источник, условия, знаменатель прореживания по частям).
DOMAINS = {
    "exp_full":   ("exp_full",  [
        #reader.NOT_DA_TARGET,
        reader.NOT_EXCLUDED
        ], 1),
    "muatm":      ("mc_merged", [reader.NOT_TRAINED,
                                 "e.data_class = 'muatm_2020'"], 10),
    "nuatm":      ("mc_merged", [reader.NOT_TRAINED,
                                 "e.data_class = 'nuatm_2020'"], 10),
    "nue2":       ("mc_merged", [reader.NOT_TRAINED,
                                 "e.data_class = 'nue2_2020'"], 10),
}
COLOURS = {"exp_full": "#d62728", "muatm": "#1f77b4",
           "nuatm": "#2ca02c", "nue2": "#ff7f0e"}


def selection(name: str, thinned: bool = True):
    """Условия отбора домена; при `thinned` — с прореживанием по частям."""
    source, conditions, modulus = DOMAINS[name]
    conditions = list(conditions)
    if thinned and modulus > 1:
        # hash() в DuckDB детерминирован между процессами -- проверено, -- так что
        # это воспроизводимая выборка частей, а не случайная при каждом запуске.
        conditions.append(f"hash(l.part_key) % {modulus} = 0")
    return source, conditions


def streaming(r: reader, where, **kwargs):
    """Чанки читалки с полоской прогресса.

    Читалка сама ничего не печатает -- она библиотека, и печать из неё портила бы
    логи и тесты. Вместо этого она зовёт `progress(...)` с описанием состояния, а
    что с ним делать, решает вызывающий. Здесь -- tqdm.
    """
    total = r.count(where=where)
    labels = {"querying": "запрос", "training": "таблица обучения",
              "done": "готово"}
    with tqdm(total=total["events"], unit=" соб", leave=True) as pbar:
        def report(state):
            pbar.set_description(labels.get(
                state.stage, f"часть {state.parts}/{total['parts']}"))
            if state.events:
                pbar.n = state.events
            pbar.refresh()
        yield from r.stream(where=where, progress=report, **kwargs)


def signal_hits_dqm_clean(chunk, source: str) -> np.ndarray:
    """Маска событий: ни один SN-сигнальный хит не лежит на DQM-выключенном канале.

    DQM-маска рана (`masks.dqm_mask_from_solution`) -- та же, что применяет
    реконструкция BARS: каналы с кодами 0/1/4/7/10 она выключает, и их импульсов
    реконструкция не видит вовсе. Наш скоринг маску не знает, поэтому событие, где
    сеть пометила сигналом импульс на выключенном канале, живёт не в той задаче, для
    которой сеть обучалась; такие события выбрасываются целиком.

    Для МС маски нет по построению (BARS реконструирует МС без DQM) -- там сплошь
    True. Локально распакована только ран-уровневая часть маски; посекционная
    (пер-событийная) добавила бы ещё немного, так что фильтр -- нижняя граница.
    """
    n_events = len(chunk.events)
    if source not in ("exp_full", "exp"):
        return np.ones(n_events, dtype=bool)
    part = chunk.events["part_key"].iloc[0]
    solution = masks.solution_path(*masks.part_run(part))
    if solution is None:
        return np.ones(n_events, dtype=bool)      # DQM этого рана не распакован
    allowed = masks.dqm_mask_from_solution(solution)
    hits = chunk.hits
    bad = hits["is_sig"].to_numpy() & ~allowed[hits["channel"].to_numpy()]
    if not bad.any():
        return np.ones(n_events, dtype=bool)
    per_event = np.bincount(hits["event"].to_numpy()[bad], minlength=n_events)
    return per_event == 0


def load(name: str, checkpoint: str, budget_mb: int = 512):
    """События домена с признаками из хитов. Хиты не возвращаются.

    `n_signal_hits` считается здесь заново и обязан совпасть с `n_sn_hits` из базы:
    их писали разные проходы по одним и тем же вероятностям, так что расхождение
    означало бы сбой адресации. `r.verify(chunk)` проверяет ровно это на каждом чанке.
    """
    source, conditions = selection(name)
    r = reader.open(source, checkpoint=checkpoint)
    t0 = time.time()
    frames, read, dropped = [], 0, 0
    for chunk in streaming(r, conditions, budget_mb=budget_mb):
        r.verify(chunk)
        keep = signal_hits_dqm_clean(chunk, source)
        dropped += int((~keep).sum())
        events = chunk.events.reset_index(drop=True)
        # Логит, а не сам скор: сеть бинарная, так что это преактивация сигмоиды.
        events["logit"] = np.log(events["score"] / (1.0 - events["score"]))
        frames.append(pd.concat([events, per_event_features(chunk),
                                 signal_hit_geometry(chunk)],
                                 axis=1)[keep].reset_index(drop=True))
        read += chunk.meta.get("hits_read", 0)
    ev = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return ev, dict(домен=name, событий=len(ev), выброшено_DQM=dropped,
                    хитов_прочитано=read,
                    частей=ev.part_key.nunique() if len(ev) else 0,
                    память_МБ=round(ev.memory_usage(deep=True).sum() / 1e6, 1),
                    секунд=round(time.time() - t0, 1))


def cache_dir(checkpoint: str) -> Path:
    """Каталог кеша этого чекпоинта (создаётся при необходимости)."""
    data = CACHE_ROOT / checkpoint
    data.mkdir(parents=True, exist_ok=True)
    return data


def prediction_identity(checkpoint: str) -> Dict[str, Dict]:
    """Отпечаток prediction-БД: путь, mtime, размер.

    Нужен, чтобы перезапись `preds/<checkpoint>/*.duckdb` (например, после смены
    best-чекпоинта) инвалидировала кеш, хотя имя чекпоинта осталось прежним.
    """
    base = PRED_ROOT / checkpoint
    out: Dict[str, Dict] = {}
    for source in ("exp_full", "mc_merged"):
        path = base / f"{source}_{PRED_THRESHOLD_TAG}.duckdb"
        if path.exists():
            st = path.stat()
            out[source] = {"path": str(path.relative_to(ROOT)),
                           "mtime": int(st.st_mtime), "size": st.st_size}
    return out


def cache_fingerprint(name: str, checkpoint: str,
                      frame: Optional[pd.DataFrame] = None) -> dict:
    """Чем определяется содержимое кеша: чекпоинт, условия отбора, набор колонок."""
    source, conditions = selection(name)
    fingerprint = {"checkpoint": checkpoint, "source": source,
                   "conditions": [str(c) for c in conditions],
                   "filter": DQM_FILTER_TAG,
                   "prediction_db": prediction_identity(checkpoint)}
    if frame is not None:
        fingerprint["columns"] = list(frame.columns)
    return fingerprint


def _check_manifest(name: str, checkpoint: str) -> dict:
    stored = json.loads((cache_dir(checkpoint) / "manifest.json").read_text())[name]
    expected = cache_fingerprint(name, checkpoint)
    differs = []
    for key, value in expected.items():
        if key == "prediction_db":
            # Старые манифесты этого поля не знают -- не наказываем их за это.
            if "prediction_db" in stored and stored["prediction_db"] != value:
                differs.append(key)
            continue
        if stored.get(key) != value:
            differs.append(key)
    if differs:
        raise RuntimeError(
            f"кеш {name} устарел по полям {differs}: "
            f"было {[stored.get(k) for k in differs]}, "
            f"стало {[expected[k] for k in differs]}")
    return stored


def load_cached(name: str, checkpoint: str) -> pd.DataFrame:
    """Кеш домена, если он описывает то же, что посчитал бы `load`."""
    _check_manifest(name, checkpoint)
    return pd.read_parquet(cache_dir(checkpoint) / f"{name}.parquet")


def load_cached_fit(name: str, checkpoint: str) -> pd.DataFrame:
    """Кеш фита домена, если он описывает то же, что посчитал бы `fit_domain`."""
    _check_manifest(name, checkpoint)
    return pd.read_parquet(cache_dir(checkpoint) / f"fit_{name}.parquet")


def channel_mask_for(source: str):
    """DQM-маска каналов. У данных она есть (txt-файл рана), у МК её нет по построению."""
    if source != "exp_full":
        return None
    cache = {}

    def provider(rows):
        part = rows["part_key"].iloc[0]
        if part not in cache:
            cache[part] = masks.dqm_masks(source, part)
        return cache[part]

    return provider


def fit_domain(name: str, ev: pd.DataFrame,
               score_cut: float = SCORE_CUT, workers: int = N_WORKERS) -> pd.DataFrame:
    """Фит BARS для событий домена со скором выше порога."""
    todo = ev.loc[ev["score"] > score_cut,
                  ["source", "data_class", "part_key", "local_idx"]]
    if todo.empty:
        return pd.DataFrame()
    t0 = time.time()
    with tqdm_progress(f"фит {name}") as progress:
        fits = fit_events(todo, n_workers=workers,
                          channel_mask=channel_mask_for(ev["source"].iloc[0]),
                          progress=progress)
    fits = fits.add_prefix("fit_")
    print(f"{name}: {len(todo)} событий за {time.time() - t0:.0f} с, "
          f"сошлось {(fits['fit_status'] == 'ok').mean():.0%}")
    return fits


def with_fit(name: str, EV: dict, FIT: dict) -> pd.DataFrame:
    """EV[name] + колонки фита; выравнивание по индексу, порядок не меняется."""
    return EV[name].join(FIT[name]) if name in FIT else EV[name]


def load_or_build(checkpoint: str,
                  fit_names: Tuple[str, ...] = ("exp_full", "muatm"),
                  force_reload: bool = False,
                  budget_mb: int = 10240,
                  ) -> Tuple[dict, dict, pd.DataFrame, str]:
    """Загрузить домены и фит из кеша или собрать из источника.

    Возвращает (EV, FIT, report, origin), где origin -- "cache" или "source".
    Печать/display остаются в ноутбуке: библиотека молчит.
    """
    data = cache_dir(checkpoint)
    try:
        if force_reload:
            raise FileNotFoundError("force_reload=True")
        EV = {name: load_cached(name, checkpoint) for name in DOMAINS}
        FIT = {name: load_cached_fit(name, checkpoint) for name in fit_names}
        rows = []
        for name, frame in EV.items():
            path = data / f"{name}.parquet"
            rows.append({"домен": name, "событий": len(frame),
                         "колонок": len(frame.columns),
                         "МБ": round(path.stat().st_size / 1e6, 1)})
        return EV, FIT, pd.DataFrame(rows), "cache"
    except (FileNotFoundError, RuntimeError):
        EV, report = {}, []
        for name in DOMAINS:
            EV[name], info = load(name, checkpoint, budget_mb=budget_mb)
            report.append(info)
        FIT = {name: fit_domain(name, EV[name]) for name in fit_names}

        manifest = {}
        for name, frame in EV.items():
            entry = {**cache_fingerprint(name, checkpoint, frame),
                     "events": len(frame),
                     "written": datetime.datetime.now().isoformat(timespec="seconds")}
            manifest[name] = entry
            frame.reset_index(drop=True).to_parquet(
                data / f"{name}.parquet", index=False)
        (data / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False))
        for name, fr in FIT.items():
            if len(fr):
                fr.to_parquet(data / f"fit_{name}.parquet")
        return EV, FIT, pd.DataFrame(report), "source"
