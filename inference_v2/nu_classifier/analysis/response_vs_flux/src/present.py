"""Turning artefact tables into something a reader can read.

Stages write parquet with terse column names -- ``mc_share``, ``iqr_ratio``,
``implied_extra_sigma_ns``.  Those are fine inside the code and useless in a
notebook: a reader has no way to know what they mean.  Everything a notebook
displays goes through :func:`show`, which renames the columns, prints what the
rows are, and says which stage produced the file.

The glossary lives here rather than being retyped per notebook, so a term is
defined once and every notebook agrees on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

try:
    from IPython.display import display
except ImportError:                                          # plain python
    display = print


GLOSSARY: dict[str, str] = {
    "muatm": "симулированные атмосферные мюоны — фон, который мы умеем считать",
    "exp": "экспериментальные данные Байкальского телескопа",
    "скор ξ": "выход классификатора от 0 до 1; чем выше, тем больше событие "
              "похоже на нейтрино по мнению сети",
    "сигнальный хит": "срабатывание модуля, которому отдельная сеть sig-noise "
                      "приписала вероятность быть сигналом (а не шумом воды) "
                      "выше 0.8",
    "отбор качества": "не меньше восьми сигнальных хитов не менее чем на трёх "
                      "струнах (в коде это условие называется h8s3, "
                      "n_sn_hits ≥ 8 и n_sn_strings ≥ 3). Применяется к обеим "
                      "выборкам одинаково, иначе сравнивать было бы нечего",
    "избыток": "во сколько раз доля экспериментальных событий в данной области "
               "скора больше такой же доли у симулированных мюонов. Единица "
               "означала бы согласие",
    "полоса": "интервал скора, например 0.8–0.9",
    "ран": "сеанс набора данных одним кластером, от нескольких часов до суток",
    "кластер": "группа струн с собственной электроникой; в сезоне 2020 их семь",
    "струна": "вертикальный трос с оптическими модулями",
    "IQR": "межквартильный размах — ширина распределения, расстояние от 25-го "
           "до 75-го процентиля. Для гауссианы это 1.35 сигмы",
    "остаток": "разность между наблюдённым и предсказанным временем прихода "
               "света на модуль, в наносекундах",
    "якорь": "трек, подогнанный по остальным хитам события; по нему делается "
             "предсказание для выброшенного хита",
    "reco-событие": "событие, на котором штатная реконструкция BARS отработала "
                    "и записала параметры трека. Это другой отбор, чем h8s3, и "
                    "потому независимая проверка избытка",
    "reco_prty": "25 скаляров реконструкции на событие (в МК 31: те же плюс "
                 "шесть векторных). Имена и порядок — в "
                 "inference/shared_utils.py, единицы смешаны: углы трека в "
                 "градусах, scf-углы в радианах",
    "muatm в mc_reco": "фон. Остальные три класса mc_reco — нейтрино, и "
                       "классификатор принимает 99.4–99.9% из них, поэтому "
                       "знаменателем избытка может быть только muatm",
    "ложное срабатывание": "симулированный атмосферный мюон, которому "
                           "классификатор поставил скор выше 0.5. Все "
                           "атмосферные мюоны — фон, поэтому любое их принятие "
                           "ошибочно по определению",
    "обманчивость": "то, чем принятые мюоны отличаются от отвергнутых. Обе "
                    "выборки — одни и те же частицы одного генератора, так что "
                    "различает их только это",
    "селективность": "во сколько раз кат щадит чистые события сильнее, чем "
                     "ложные срабатывания: доля выживших чистых, делённая на "
                     "долю выживших ложных. Единица означала бы, что кат режет "
                     "вслепую",
    "повтор": "один из пяти непересекающихся наборов чистых мюонов, взятых для "
              "балансировки. Кат засчитывается, только если переживает все пять",
    "ячейка": "группа хитов с одинаковой геометрией — одно и то же число хитов, "
              "струн и расстояний. Сравнение внутри ячейки не путает разницу "
              "отклика с разницей состава событий",
}


# Raw column values that end up *inside* a table rather than in its header --
# the proxy names of stage 10, for instance.  Renaming headers is not enough:
# a reader meets `sn_reject_frac` in a cell and is no better off than before.
VALUE_NAMES: dict[str, str] = {
    "trigger_rate_hz": "скорость счёта, Гц",
    "raw_hits_per_event": "сырых хитов на событие",
    "sn_reject_frac": "доля хитов, отвергнутых фильтром",
    "repeat_hits_per_event": "повторных хитов на модуле",
    "q_total_mean": "средний суммарный заряд",
    "quality_frac": "доля событий, прошедших отбор",
    "cluster": "номер кластера",
    "level": "уровень профиля",
    "slope": "наклон профиля",
    "none": "без поправок",
    "both": "с поправкой на оба",
}


def rename_values(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Replace raw identifiers with readable names inside one column."""
    out = frame.copy()
    out[column] = out[column].map(lambda v: VALUE_NAMES.get(v, v))
    return out


def glossary(terms: Iterable[str] | None = None) -> None:
    """Print the definitions a notebook needs, in the order given."""
    keys = list(terms) if terms is not None else list(GLOSSARY)
    width = max(len(k) for k in keys)
    for key in keys:
        text = GLOSSARY[key]
        first, *rest = _wrap(text, 96 - width - 4)
        print(f"{key:>{width}}  —  {first}")
        for line in rest:
            print(f"{'':>{width}}     {line}")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def show(
    frame: pd.DataFrame,
    columns: Mapping[str, str],
    *,
    rows: str,
    source: str,
    note: str | None = None,
    index: bool = False,
    decimals: int = 3,
) -> None:
    """Display a table with readable headers and a caption that explains it.

    ``rows`` says what one row is; ``source`` names the stage and artefact the
    numbers come from, so a reader can find and re-run them; ``note`` carries
    anything else needed to read the table without guessing.
    """
    view = frame[list(columns)].rename(columns=dict(columns))
    print(f"Строка — {rows}.")
    if note:
        for line in _wrap(note, 96):
            print(line)
    print(f"Источник: {source}")
    display(view.round(decimals).style.hide(axis="index")
            if not index else view.round(decimals))


def _plural_rows(n: int) -> str:
    """Russian agreement for 'строка' -- 1 строка, 2 строки, 5 строк."""
    if 11 <= n % 100 <= 14:
        return "строк"
    return {1: "строка", 2: "строки", 3: "строки", 4: "строки"}.get(n % 10, "строк")


def artefact_note(path: Path) -> str:
    """`stage, artefact, rows` for a caption, read from the provenance sidecar."""
    import json
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    rows = int(meta["rows"])
    return (f"стадия {meta['stage']}, файл {meta['artefact']}, "
            f"{rows} {_plural_rows(rows)}")
