# ID экспериментов по реконструкции энергии мюона

Единая система коротких ID для матрицы `track_energy_reconstruction`.
Она описывает **способ обучения**, а не конкретный run directory: одинаковый ID
может встречаться у нескольких seed'ов и у баз с разными предобученными mean-моделями.

## Правило

```text
[ID] = [A?] [BASE | B | C | D | H] [-G | -T5]
```

- Префикс `A` — включён domain adaptation (DA). `A` без суффикса — только DA,
  без sigma-головы.
- `B` — joint-обучение mean и sigma, **полный** NLL-градиент, один optimizer.
- `C` — joint-обучение mean и sigma, полный NLL-градиент, **два** optimizer-а
  (control для проверки эффекта раздельного optimizer-а).
- `D` — joint-обучение mean и sigma, sigma-голова получает **detached** features,
  два optimizer-а; encoder учится только от mean-градиента (и DA, если есть префикс `A`).
- `H` — mean/base **заморожен**, обучается только sigma-голова (head-only,
  post-hoc калибровка).
- `G` — Gaussian NLL.
- `T5` — Student-t NLL с ν = 5.
- `BASE` — детерминированная mean-модель без sigma-головы; прежнее обозначение `D0`.

`B`, `C`, `D`, `H` всегда содержат sigma-голову. `A` и `BASE` — не содержат.

Выбор checkpoint: `BASE`, `B`, `C`, `D` и их DA-варианты — по `val_mae_log10`
(mean должен сойтись); `H`/`AH` — по `val_unc_nll`, потому что mean заморожен.
Метрики sigma при этом всегда пишутся отдельно, и их нужно смотреть на выбранном
checkpoint, а не смешивать с критерием выбора.

## Канонические ID

| ID | Что обучается | DA | Sigma-градиент в encoder | Optimizer-ы |
|---|---|---:|---|---|
| `BASE` | mean, Huber | нет | — | один |
| `A` | mean, Huber | да | — | один |
| `B-G` / `B-T5` | mean + sigma, joint full | нет | да | один |
| `AB-G` / `AB-T5` | mean + sigma, joint full | да | да | один |
| `C-G` / `C-T5` | mean + sigma, joint full | нет | да | два |
| `AC-G` / `AC-T5` | mean + sigma, joint full | да | да | два |
| `D-G` / `D-T5` | mean + sigma, joint detached | нет | нет | два |
| `AD-G` / `AD-T5` | mean + sigma, joint detached | да | нет | два |
| `H-G` / `H-T5` | только sigma, base frozen | нет | нет (base frozen) | один (head) |
| `AH-G` / `AH-T5` | только sigma на DA-базе `A` | через базу `A` | нет (base frozen) | один (head) |

`AH-*` — это не «H с DA во время обучения головы»: база `A` уже DA-адаптирована,
затем замораживается. Если разморозить encoder и запустить DA одновременно с
sigma-головой, это будет joint-вариант, а не `AH`.

## Независимая доменная проба

`inference/track_energy_reconstruction/analysis/domain_probe.py` берёт фичи
энкодера на отложенных DA-частях: source — `muatm`, target — `exp_full`.
Проба обучается заново на этих фичах, поэтому не зависит от недообученного
per-epoch MLP в `training_history.csv`. Считаются два разбиения:

- event-level внутри отложенных частей — сравнимо с историей обучения;
- part-disjoint, где целые `part_key` уходят в fit или test — основная проверка.

`da_enabled` в `da_probe.csv` относится к **энкодеру**: `AH-*` наследует
DA-энкодер от `A`, поэтому его проба совпадает с `A` и не является no-DA
контролем.

Текущие MLP AUC (event / part-disjoint; 16 чекпойнтов, один seed модели,
три seed пробы):

| Сравнение | no-DA | DA | Δ event | Δ part |
|---|---:|---:|---:|---:|
| `A` vs `BASE` | 0.669 / 0.665 | 0.606 / 0.599 | −0.064 | −0.066 |
| `AD-G` vs `D-G` | 0.732 / 0.703 | 0.642 / 0.643 | −0.090 | −0.060 |
| `AD-T5` vs `D-T5` | 0.700 / 0.687 | 0.654 / 0.626 | −0.046 | −0.062 |
| `AC-G` vs `C-G` | 0.668 / 0.652 | 0.633 / 0.632 | −0.035 | −0.021 |
| `AC-T5` vs `C-T5` | 0.684 / 0.672 | 0.632 / 0.624 | −0.052 | −0.048 |

Линейная проба даёт то же направление. Это измерение на отложенных событиях,
не сертификат инвариантности: цель не обязана быть AUC = 0.5, а остаточный AUC
может отражать реальные различия MC и данных. `AB-G`/`AB-T5` будут добавлены
после завершения обучений.

## Соответствие старым ID

| Старый ID | Новый ID | Run directory (историческое имя) |
|---|---|---|
| `D0` | `BASE` | `261026_1922_mu_energy_baseline_partsplit_scratch_s42` |
| `A` | `A` | `261026_2057_mu_energy_A_lam0p01_detachedcritic_partsplit_scratch_s42` |
| `B1` | `B-G` | `261026_1931_mu_energy_B1_gaussian_partsplit_scratch_s42` |
| `B2` | `B-T5` | `261026_1931_mu_energy_B2_student_t_nu5_partsplit_scratch_s42` |
| `AB1` | `AB-G` | `261026_1931_mu_energy_AB1_da_lam0p03_gaussian_partsplit_scratch_s42` |
| `AB2` | `AB-T5` | `261026_1931_mu_energy_AB2_da_lam0p03_student_t_nu5_partsplit_scratch_s42` |
| `F1` | `H-G` | новый run |
| `F2` | `H-T5` | новый run |
| `F1-A` / `F2-A` | `AH-G` / `AH-T5` | новый run |
| `J1-G` / `J1-T` | `D-G` / `D-T5` | новый run |
| `J2-G` / `J2-T` | `C-G` / `C-T5` | новый run |

Старые run directories **не переименовываются**: их имена уже попали в
`training_summary.yaml`, checkpoint-ы и отчёты. Новый ID используется в новых
конфигурациях, таблицах и обсуждении; для старых run-ов в скобках указывается
историческое имя.

## Historical caveat: B1/B2/AB1/AB2

The runs `261026_1931_mu_energy_B1...`, `B2...`, `AB1...`, and `AB2...` were
trained before the 2026-10-27 NLL scale correction. Their variance head predicted
`log_var`, while the loss used `exp(log_var)` as the scale instead of
`exp(0.5 * log_var)`. Their **mean** results remain usable as historical mean
models, but their **sigma/NLL/coverage** results are not comparable with the new
`B-*`, `C-*`, `D-*`, and `H-*` runs. Do not use those old sigma metrics as a
reference; retrain `B-G/B-T5` and `AB-G/AB-T5` with the corrected loss if a
joint-full reference is needed.

## Правила для новых run-ов

1. Имя run-а: `<YYMMDD>_<HHMM>_mu_energy_<ID>_<краткий_контекст>_s<seed>`.
   Например: `261027_1200_mu_energy_H-G_d0base_s42`.
2. В YAML явно писать `experiment.id: <ID>` и `experiment.name` с тем же ID.
3. Не использовать `_v2`, `_new`, `_fixed` и подобные суффиксы. Отличие
   выражается ID, seed-ом и контекстом базы.
4. Для `H`/`AH` в `resume` указывать базовый checkpoint и
   `reset_training_state: true`: это инициализация базы, а не продолжение её
   обучения. Для продолжения `BASE` (D0-long) `reset_training_state` не ставить.
5. В `training_summary.yaml` сохраняются `experiment.id`, `sigma_strategy`,
   `resume`-контекст и факторы калибровки; таблицы должны строиться из них, а не
   из догадок по имени каталога.

Очередь запусков и её статус:

```bash
python scripts/run_energy_queue.py \
  --queue src/training/configs/track_energy_reconstruction/energy_queue.yaml
# статус:
trained_models/track_energy_reconstruction/energy_queue_status.json
# логи отдельных запусков:
trained_models/track_energy_reconstruction/queue_logs/<ID>.log
```
