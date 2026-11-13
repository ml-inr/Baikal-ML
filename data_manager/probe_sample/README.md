# probe_sample — фиксированная выборка для invariance-пробы

Инструмент **обучения**: фиксированный held-out набор, на котором тренер
(`src/training/da_nu_classifier_trainer_da_repair.py`, `_invariance_probe`) каждые N
эпох измеряет свежим логистическим читателем, сколько доменной информации осталось в
представлении. Через набор ничего не обучается; все события вне обучения
(`NOT_TRAINED` / `NOT_DA_TARGET` / `NOT_EXCLUDED`). Переехал сюда 2026-10-12 из
`inference/nu_classifier/analysis/da_band_probe/` (папка анализа его лишь породила).

Состав после расширения 2026-10-10 (16 008 событий, 134 701 хит):

| группы | что |
|---|---|
| 0, 1 | μatm: 4000 натуральных + 1008 хвостовых (скор FIXED_sn256 >= 0.8) |
| 2, 3 | exp: 4000 натуральных + 1500 хвостовых |
| 4, 5 | ν = nuatm+nue2: 4000 натуральных + 1500 высокоскорных — добавлены, чтобы проба мерила игровую задачу (MC ν+μ против exp с половинным z-флипом) |

`band`: полосы скора (0 = bulk < 0.5, 1 = mid, 2 = хвост >= 0.8), фиксированы при сборке.
`ref_score`: скор FIXED_sn256 (только для фиксации полос).

**Правило неизменности:** первые 10 508 событий (до ν-блоков) побайтово заморожены —
сравнимость с историческими прогонами. Файл **расширять** (сохраняя префикс), не
пересобирать целиком: стрим reader'а не детерминирован, полная пересборка
перерисовывает события (см. NB в `build_probe_sample.py`).

Сборка: `python data_manager/probe_sample/build_probe_sample.py`
(~4 минуты). Потребители: конфиги с `invariance_probe.path` (20 файлов) и
`analysis/da_band_probe/probe_on_checkpoints.py` (`--sample`).

## Префильтр: отдельная выборка

С 2026-11 проба используется и тренером префильтра
(`src/training/da_prefilter_numu_trainer.py`). Его вход — **все сырые хиты**, а не
SN-отфильтрованные, поэтому для него собрана отдельная
`probe_sample_prefilter.npz` (12 000 событий, 678 594 хита, natural-only):

| группа | что |
|---|---|
| 0 | μatm: 4000 натуральных, все хиты |
| 2 | exp: 4000 натуральных, все хиты |
| 4 | ν = nuatm+nue2: 4000 натуральных, все хиты |

Сборка:

```bash
python data_manager/probe_sample/build_probe_sample.py \
    --out data_manager/probe_sample/probe_sample_prefilter.npz \
    --checkpoint <prefilter-run>@best_da_model \
    --hit-mode all --natural-only --strictness parts --skip-da-target-exclusion
```

`--strictness parts` обязателен: у префильтровых NPY нет per-event back-links,
поэтому `NOT_TRAINED` не резолвится, а `NOT_TRAINED_PARTS` — да. Второй флаг нужен,
потому что `target_domain` префильтра — это `h5_path`, а не `npy_dir`; при этом
`exp_full` не пересекается с `exp.h5`, служившим DA-target.
