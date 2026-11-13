"""Признаки событий, посчитанные из хитов.

Вынесено из ноутбуков HIGH_SCORE_EXCESS (2026-10-25). Поведение сохранено
дословно: те же формулы, те же имена колонок, тот же отбор ярких хитов.
Хиты никуда не сохраняются — функции работают внутри одного чанка читалки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Порог «яркого» хита. Заряд — единственная величина в событии, которую не
#: трогала ни одна обученная модель, поэтому счётчик ярких хитов служит опорой,
#: независимой от sig-noise сети.
BRIGHT_CHARGE_PE = 5.0


def per_event_features(chunk) -> pd.DataFrame:
    """Свойства события, посчитанные из его хитов.

    Нужны две группировки: по всем хитам и только по ярким. У события может не
    быть ни одного яркого хита, поэтому вторая покрывает не все события, и
    недостающие строки восстанавливаются через reindex — иначе счётчики съедут
    относительно chunk.events на молчаливую величину.
    """
    hits = chunk.hits
    positions = range(len(chunk.events))
    all_hits = hits.groupby("event")
    bright = hits[hits.q > BRIGHT_CHARGE_PE].groupby("event")

    def counted(grouped, value=0):
        return grouped.reindex(positions, fill_value=value).to_numpy()

    def span(grouped, column):
        return (grouped[column].max() - grouped[column].min()).reindex(positions).to_numpy()

    return pd.DataFrame({
        "n_hits":                 counted(all_hits.size()),
        "n_bright_hits":          counted(bright.size()),
        "n_signal_hits":          counted(all_hits["is_sig"].sum()),
        "n_strings":              counted(all_hits["string"].nunique()),
        "n_strings_with_bright":  counted(bright["string"].nunique()),
        "charge_total_pe":        counted(all_hits["q"].sum(), 0.0),
        "charge_in_bright_pe":    counted(bright["q"].sum(), 0.0),
        "brightest_hit_pe":       counted(bright["q"].max(), np.nan),
        "time_span_ns":           span(all_hits, "t"),
        "time_span_of_bright_ns": span(bright, "t"),
        "z_span_of_bright_m":     span(bright, "z"),
    }, index=positions)


def signal_hit_geometry(chunk) -> pd.DataFrame:
    """Наклон z(t), его ошибка и разбросы — по сигнальным хитам, в двух взвешиваниях.

    Всё считается замкнутыми формулами МНК из шести сумм, каждая — один bincount.
    Ни цикла по событиям, ни groupby: 636 тыс. событий в секунду, то есть около
    0.1 % от стоимости чтения чанка, ради которого хиты и поднимаются с диска.

    Наклон в м/нс, так что сравнивать его есть с чем: вертикальный релятивистский
    трек даёт |dz/dt| = 0.2998.

    Вес `q` — не обратная дисперсия, поэтому взвешенная «ошибка наклона» это не
    статистическая ошибка в строгом смысле, а разброс с оглядкой на яркость хитов.
    Колонки с суффиксом `_q` — взвешенные, без суффикса — нет.

    `sigma_horizontal` = sqrt(var x + var y) тождественно равна СКО расстояния от
    центроида в плоскости, так что второй вариант — `sigma_radius`, разброс
    расстояния до оси кластера. Отношения не считаются: их дешевле взять ниже по
    анализу, чем тащить лишнюю колонку через все домены.
    """
    hits = chunk.hits
    signal = hits["is_sig"].to_numpy()
    n_events = len(chunk.events)
    event = hits["event"].to_numpy()[signal]
    t = hits["t"].to_numpy()[signal].astype(np.float64)
    z = hits["z"].to_numpy()[signal].astype(np.float64)
    x = hits["x"].to_numpy()[signal].astype(np.float64)
    y = hits["y"].to_numpy()[signal].astype(np.float64)
    q = hits["q"].to_numpy()[signal].astype(np.float64)
    r = np.hypot(x, y)
    counted = np.bincount(event, minlength=n_events)

    columns = {}
    for tag, w in (("", np.ones(len(event))), ("_q", q)):
        total = lambda v: np.bincount(event, weights=w * v, minlength=n_events)
        Sw = np.bincount(event, weights=w, minlength=n_events)
        St, Sz, Stt, Szz, Stz = total(t), total(z), total(t * t), total(z * z), total(t * z)
        Sx, Sy, Sxx, Syy = total(x), total(y), total(x * x), total(y * y)
        Sr, Srr = total(r), total(r * r)

        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = Sw * Stt - St * St
            # Меньше трёх хитов — прямая проходит точно, разброс не определён.
            # Нулевой знаменатель — все хиты в один момент времени.
            usable = (counted >= 3) & (denominator > 0) & (Sw > 0)
            slope = np.where(usable, (Sw * Stz - St * Sz) / denominator, np.nan)
            intercept = np.where(usable, (Stt * Sz - St * Stz) / denominator, np.nan)
            residual = np.clip(Szz - intercept * Sz - slope * Stz, 0.0, None)
            spread = np.where(usable, np.sqrt(residual / Sw), np.nan)
            error = np.where(usable,
                             np.sqrt(residual / np.maximum(counted - 2, 1) * Sw / denominator),
                             np.nan)
            centred = lambda first, second: np.clip(second / Sw - (first / Sw) ** 2, 0.0, None)
            sigma_z = np.where(Sw > 0, np.sqrt(centred(Sz, Szz)), np.nan)
            sigma_horizontal = np.where(Sw > 0,
                                        np.sqrt(centred(Sx, Sxx) + centred(Sy, Syy)), np.nan)
            sigma_radius = np.where(Sw > 0, np.sqrt(centred(Sr, Srr)), np.nan)

        columns.update({
            f"z_t_slope_m_per_ns{tag}":       slope,
            f"z_t_slope_error_m_per_ns{tag}": error,
            f"z_t_residual_rms_m{tag}":       spread,
            f"sigma_z_m{tag}":                sigma_z,
            f"sigma_horizontal_m{tag}":       sigma_horizontal,
            f"sigma_radius_m{tag}":           sigma_radius,
        })
    return pd.DataFrame({name: values.astype(np.float32)
                         for name, values in columns.items()})


def add_derived_columns(EV: dict) -> None:
    """Добавить производные отношения к событиям каждого домена (in place).

    z_verticality_* — отношение разбросов вдоль z к поперечным; score_zt_baseline —
    сигмоида от измеренного наклона z(t), то есть «насколько событие похоже на
    вертикальный трек» без обучения.
    """
    for name in EV:
        EV[name]["z_verticality_1"] = (
            EV[name]["sigma_z_m"] / EV[name]["sigma_horizontal_m"]
        ).replace([np.inf, -np.inf], np.nan)
        EV[name]["z_verticality_2"] = (
            EV[name]["sigma_z_m_q"] / EV[name]["sigma_horizontal_m_q"]
        ).replace([np.inf, -np.inf], np.nan)
        EV[name]["z_verticality_3"] = (
            EV[name]["sigma_z_m"] / EV[name]["sigma_radius_m"]
        ).replace([np.inf, -np.inf], np.nan)
        EV[name]["z_verticality_4"] = (
            EV[name]["sigma_z_m_q"] / EV[name]["sigma_radius_m_q"]
        ).replace([np.inf, -np.inf], np.nan)
        EV[name]["score_zt_baseline"] = 1 / (1 + np.exp(-EV[name]["z_t_slope_m_per_ns"]))
