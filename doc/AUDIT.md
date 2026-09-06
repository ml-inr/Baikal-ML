# AUDIT — фактическая карта репозитория

*Составлено 2026-09-06. Только чтение: ни один файл, кроме этого, не создан и не изменён.*

*Дополнено в тот же день: §8 содержит ответы владельца на все 15 вопросов и то, что
удалось доразрешить чтением кода после них. §5.2 и §3.1 снабжены ссылками на эти ответы.*

Это **фактическая** карта, не оценочная. Здесь нет рекомендаций, нет предложений по новой
структуре каталогов и нигде не сказано «это можно удалить». Формулировка «не достижим из
точек входа» — это утверждение о графе импортов, а не о ценности файла.

**Каждое утверждение снабжено основанием.** Где основания нет — написано
«не удалось установить».

---

## 0. Границы обследования и как получены основания

**Что обследовано.** 337 файлов `.py`, 105 `.ipynb`, 57 `.sh`, 18 `.md` в `doc/` + ~30 `.md`
по дереву. Вне обследования оставлены два вендоренных дерева чужого софта:

| Дерево | Что это | Основание |
|---|---|---|
| `bars/` (1.1 ГБ, 12 `.py`) | BARS-фреймворк коллаборации | `.gitignore:/bars`; содержимое — C++/CMake коллаборации |
| `cern_root/` (1.2 ГБ, ~450 `.py`) | установка CERN ROOT | `.gitignore:/cern_root`; `cern_root/lib/{ROOT,cppyy}` — дистрибутив |

**Методы.**

- *Граф импортов* — AST-разбор всех 337 файлов. Резолв учитывает: абсолютные импорты от
  корня репозитория; импорты от каталога скрипта; относительные (`from . import x`);
  форму `from pkg import submodule`; и паттерн `sys.path.insert(0, HERE/"src")` +
  голый `import h5io`, который используют `response_vs_flux/stages/*` и
  `reco_excess/stages/*`.
- *Точки входа* — shebang; блок `if __name__ == "__main__"`; наличие `argparse`; цели
  `python …` из всех 57 `.sh`; команды из `CLAUDE.md` и `.claude/skills/pipelines/SKILL.md`.
- *Дубликаты* — md5 (точные) и Jaccard по 4-строчным шинглам с вырезанными комментариями
  и пустыми строками (близкие).
- *Размеры* — `find -printf '%s'`, `du -sh`.

**Ограничение метода, важное для §2.** Три вида связей граф импортов не видит:

1. `python -m package` (через `__main__.py`) — так документирован запуск NPY-билдера;
2. динамическая загрузка через `importlib.util.spec_from_file_location` — так
   `sig_noise_model_v3.py` грузит `model_simplified.py`, а `sig_noise_model_v2.py` —
   `encoder.py`/`layers.py`;
3. запуск скрипта человеком без `__main__`-гарда (top-level код).
Все три случая ниже помечены отдельно, а не записаны в «недостижимые».

**Ограничение по датам — важное.** В git **16 коммитов**. Первый — 2025-09-05, при этом
коммит `c0e8b70` называется «Shared new model + clearing git history», то есть история
переписывалась. Далее до 2026-06-08 — 14 коммитов. Последний, `5fb2603` **«pre-cleanup
commit» от 2026-09-06 09:38**, сделан уже во время составления этого аудита и добавил
**182 файла разом** (`git show --stat`: «182 files changed, 30220 insertions»).

Отсюда три режима git-даты, и ни один не отражает, когда файл писали:

| Режим | Что означает |
|---|---|
| дата ≤ 2026-06-08 | файл не менялся с прошлого коммита — дата осмысленна |
| дата = **2026-09-06** | файл попал в общий «pre-cleanup commit»; **о возрасте файла не говорит ничего** |
| `(none)` | файл до сих пор вне git |

Поэтому всюду ниже рядом с git-датой приводится **mtime** файловой системы, и это
разные величины. Работа с июня по сентябрь 2026 индивидуальных коммитов не имеет.

**Под контролем версий — 353 файла** (было 184 до `5fb2603`). Из 337 `.py` отслеживаются
**275**. При этом `.gitignore` по-прежнему исключает `*.md`, `*.yaml`, `*.yml`, `*.json`,
`*.sh`, `*.txt`, `*.csv`, `*.png`, `*.ipynb`, `test*.py`, `archive/`, `CLAUDE.md`,
`vision.md`, `conventions.md`, `doc/workflow.md`, `doc/tasklist.md`, `memory`, `my_notes`
(строки 43, 157, 169, 172, 176 и далее), с исключениями `!doc/hdf5_format.md` и
`!gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/README.md`.

Проверено после коммита: `.sh` под git — **0**, `test*.py` — **0**, `.yaml` — 1,
`.md` — 2, `.ipynb` — 1. Следствие остаётся в силе: **все запускающие обёртки, все
конфиги экспериментов, все тесты, все ноутбуки и почти вся документация не
версионируются** — закоммичен был код, но не то, чем его запускают.

Распределение 353 отслеживаемых файлов: `inference_v2` 173, `inference` 65,
`data_manager` 53, `src` 30, `gplotnikov_sig_noise_models` 23, `SharedModels` 4,
по одному — `doc`, `.gitignore`, `energy.C`, `mo_test.py`, `temp_plot_pred.py`.

---

## 1. Точки входа

Всего 189 файлов выглядят исполняемыми. Ниже — по трактам; для каждой строки указано,
**откуда известно**, что это точка входа.

### 1.1 ROOT → HDF5

| Скрипт | Аргументы | Пишет на диск | Основание |
|---|---|---|---|
| `data_manager/root2h5/root2h5.py` | нет argparse; конфиг `root2h5_config.yaml` внутри | `baikal_mc_merged.h5` | `__main__`; SKILL.md: `python root2h5/root2h5.py` |
| `data_manager/root2h5/root2h5_exp.py` | нет argparse | `exp.h5` (legacy, cap 25k) | `__main__`; SKILL.md |
| `data_manager/root2h5/root2h5_exp_full.py` | `--config` | `exp_full.h5` (136 ГБ) | `__main__`+argparse; `run_root2h5_exp_full.sh`; SKILL.md |
| `data_manager/root2h5/root2h5_exp_reco.py` | нет argparse | `exp_reco.h5` | `__main__`; SKILL.md |
| `data_manager/root2h5/root2h5_mc_reco.py` | нет argparse | `baikal_mc_reco.h5` | `__main__`; `run_mc_reco_all.sh` (цикл по частицам, `tee mc_reco_${PARTICLE}.log`) |
| `data_manager/root2h5/archive/root2h5_reco.py` | нет argparse | не удалось установить | `__main__`; в шапке `'''To do: write normal code…'''` |

Обёртки: `run_root2h5_exp_full.sh` (`nohup conda run -n baikal25 … --config`),
`run_mc_reco_all.sh`.

### 1.2 Энергетическая истина (`energy_truth`)

Два каталога — это **стадии одного тракта**, не варианты (см. §3.6).

| Скрипт | Стадия | Основание |
|---|---|---|
| `root2h5/energy_truth/extract_interactions.py` | 1: цепочка взаимодействий из ROOT (на cluster62) | shebang + argparse; docstring |
| `root2h5/energy_truth/run_extract_interactions.py` | батч-драйвер, один `.npz` на входной ROOT | shebang + argparse; docstring |
| `root2h5/energy_truth/read_wout.py` | чтение `.dat`/`.wout` | argparse; `doc/mc_binary_formats.md` |
| `energy_truth/read_root.py` | 1 (новая версия) | shebang; «stage 1 of three (doc/energy_twin_plan.md)» |
| `energy_truth/build_h5.py` | 3: сборка `baikal_mc_merged_energy_truth.h5` (163 ГБ) | shebang + argparse; docstring |
| `energy_truth/validate.py` | 5: валидация готового файла | shebang + argparse; docstring |

Логи стадий лежат рядом: `build_muatm.log` (78 КБ), `build_nuatm.log`, `build_nue2.log`,
`validate_all.log`.

### 1.3 Каталог событий

| Скрипт | Аргументы | Пишет | Основание |
|---|---|---|---|
| `data_manager/catalog_v2/build_mc.py` | через `run_build_mc.sh` | `data_manager/catalog_v2.duckdb` | `__main__`+argparse; `nohup python -u -m data_manager.catalog_v2.build_mc` |
| `catalog_v2/build_mc_reco.py` | — | тот же файл | `run_build_mc_reco.sh` |
| `catalog_v2/build_exp.py` | позиционный `exp\|exp_full\|exp_reco` | тот же файл | `run_build_exp.sh` + SKILL.md |
| `build_catalog_mc_merged.py` | argparse | Parquet в `h5_catalogs/catalogs_mc_merged/` | `__main__`+argparse (v1, см. §3.5) |
| `build_catalog_exp.py`, `build_catalog_exp_reco.py`, `build_catalog_mc_normed.py` | argparse | Parquet | то же |

### 1.4 NPY-датасеты

| Команда | Пишет | Основание |
|---|---|---|
| `python -m data_manager.nu_classifier_ds_builder --config …/default_config.yaml` | `data_manager/datasets/nu_classifier_dataset_*/` | SKILL.md (дословно, с `nohup`); `__main__.py` существует |
| `python data_manager/nu_classifier_ds_builder/build.py` | то же | `__main__`+argparse; SKILL.md |
| `data_manager/nu_classifier_ds_builder/add_theta.py` | добавляет `theta.npy` | shebang + argparse |
| `data_manager/build_exp_nu_classifier_dataset.py` | exp-датасет | `build_exp_full_chain.sh` |
| `data_manager/prefilter_npy_ds_builder/build.py` | `datasets/baikal_mc2020_prefilter/` (48 ГБ) | `__main__`+argparse |

### 1.5 Sig-noise предсказания (`gplotnikov_sig_noise_models/`)

| Скрипт | Дефолты | Пишет | Основание |
|---|---|---|---|
| `k_nsol_labelneq0_da_hs128_k0p0001/predict_mc_h5.py` | `--input baikal_mc_merged.h5 --device auto --batch-size 256 --parts-json` | `{stem}_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5` | argparse; docstring; логи `predict_mc_h5.log` 35 КБ рядом |
| то же `/predict_exp_h5.py` | аналогично | `exp*_probs_*.h5` | argparse |
| то же `/fingerprint_probs_h5.py` | — | `probs_fingerprint_{before,after}.json` | shebang + argparse |
| то же `/validate_exp_probs.py` | — | `validate_exp_probs.log` | argparse |
| `nn_aug_highq_p025/predict_mc_h5.py`, `nn_aug_highq_p0p0002/predict_mc_h5.py` | те же | **тот же самый файл, что и k0p0001** — см. §3.1 | argparse |

### 1.6 Обучение

Пять обёрток в `scripts/` ссылаются на **несуществующие** пути:

| Обёртка | Цель | Состояние цели |
|---|---|---|
| `scripts/train_da_hcut_in_bg.sh` | `src/training/da_hcut_numu_trainer.py` | **нет**; файл в `src/training/archive/` |
| `scripts/train_da_in_bg.sh` | `src/training/da_numu_trainer.py` | **нет**; файл в `src/training/archive/` |
| `scripts/train_da_prefilter_hard_in_bg_upsample.sh` | `…_upsample.py` | **нет**; файл в `archive/260524/` |
| `scripts/train_da_prefilter_hard_in_bg_upsample_big.sh` | `…_upsample.py` | **нет**; там же |
| `scripts/train_da_prefilter_hard_in_bg_upsample_zflip.sh` | `…_upsample_zflip.py` | **нет**; в `archive/260524/` |
| `scripts/train_da_prefilter_soft_in_bg_upsample_zflip.sh` | `…_upsample_zflip.py` | **нет**; там же |

Основание: `for p in …; do [ -f "$p" ]` по всем целям, извлечённым grep'ом из `.sh`.
При этом **все конфиги, на которые они ссылаются, существуют**
(`experiments/da_hcut_numu_baseline.yaml`, `da_numu_baseline.yaml`,
`da_prefilter_numu_hardlabels{,_big,_zflip}.yaml`, `da_prefilter_numu_softlabels_zflip.yaml`).

Рабочие обёртки обучения:

| Обёртка | Тренер | Конфиг |
|---|---|---|
| `scripts/train_in_bg.sh` | `standard_numu_trainer.py` | `experiments/numu_baseline.yaml` |
| `scripts/train_da_nuclassifier_in_bg.sh` | `da_nu_classifier_trainer.py` | `da_nu_classifier_baseline.yaml` |
| `scripts/train_da_nuclassifier_DEBUG.sh` | тот же | `da_nu_classifier_DEBUG.yaml` |
| `scripts/train_da_nuclassifier_hugeds.sh` | тот же | `da_nu_classifier_hugeds.yaml` |
| `scripts/train_da_nuclassifier_withprobs.sh` | тот же | `da_nu_classifier_hugeds_probas.yaml` |
| `scripts/train_da_prefilter_{hard,soft}_in_bg.sh` | `da_prefilter_numu_trainer.py` | `…hardlabels.yaml` / `…softlabels.yaml` |
| `run_domain_sep_{nu_classifier,prefilter}{,_control}.sh` (в корне) | `domain_sep_trainer.py` | 4 `domain_sep_*.yaml` |

Все пишут лог в `experiments/logs/…_$(date).log` и каталог прогона в `experiments/numu/`.
`src/training/sngp_nu_classifier_trainer.py` (`__main__`+argparse,
конфиг `experiments/sngp_nu_classifier_baseline.yaml`) обёртки не имеет.
`src/training/exp_finetuning_trainer.py` вызывается не напрямую, а из
`inference_v2/nu_classifier/exp_finetuning/run_finetune.py`.

### 1.7 Предсказания nu-классификатора и префильтра (`inference_v2/`)

Задокументировано в `inference_v2/README.md` и `SKILL.md`.

| Скрипт | Ключевые дефолты argparse | Пишет |
|---|---|---|
| `nu_classifier/predict_mc.py` | `--threshold 0.8 --min-hits 8 --min-strings 2 --batch-size 512 --sn-batch-size 256 --source mc_merged --output-dir inference_v2/nu_classifier/preds --catalog data_manager/catalog_v2.duckdb` | `preds/{ckpt}/mc_merged_thr0p8.duckdb`, `run_info.json`, строка в `prediction_history.csv` |
| `nu_classifier/predict_exp.py` | те же + `--sample-mode random` | `preds/{ckpt}/exp*_thr0p8.duckdb` |
| `nu_classifier/predict_npy.py` | argparse | тот же `mc_merged_thr{thr}.duckdb`, что и `predict_mc.py` (README: «share the same output DB») |
| `nu_classifier/predict_sngp.py` | argparse | preds SNGP |
| `prefilter/predict_{mc,exp}.py` | argparse | `mc_merged_allhits.duckdb` / `exp_reco_allhits.duckdb` |
| `nu_classifier/compute_scalars.py` | argparse | не удалось установить точный выход без запуска |

Обёрток `.sh` — 14 штук в `inference_v2/nu_classifier/`, включая
`run_batch_predict.sh` (4 шага: npy / mc_merged / mc_reco / exp, распараллеливание по GPU),
`run_chain_FIXED_sn256.sh`, `run_reco_chain_FIXED_sn256.sh`,
`run_e2_early_epochs.sh`, `run_e3b_early_epochs.sh`, `run_e5_vs_e1.sh`,
`run_matched_epoch_infer.sh`, `run_test_infer_exp_full_da.sh`.

В `run_reco_chain_FIXED_sn256.sh` цели `"$SN/predict_mc_h5.py"` и `"$SN/predict_exp_h5.py"`
не резолвятся статически (переменная `$SN`) — вероятно указывают на каталог k0p0001, но
**проверить это чтением скрипта не удалось**.

### 1.8 Fine-tuning

`inference_v2/nu_classifier/exp_finetuning/`: `build_exp_bg.py`, `build_exp_bg_ood.py`,
`run_finetune.py` (все — `__main__`+argparse), обёртки `run_build_exp_bg.sh`,
`run_finetune.sh`, `run_predict_finetuned.sh`. Конфиги `finetune.yaml`,
`finetune_seed32.yaml`, `finetune_E1_ood2p5_rv0p7.yaml`. Пишут в
`exp_finetuning/finetuned_models*/` (см. §4).

### 1.9 Анализ

113 `.py` в 17 подкаталогах `inference_v2/nu_classifier/analysis/`. Из них:

- **С обёрткой `.sh`**: `run_umap.py`, `run_single_analysis.py`, `run_moe_analysis.py`,
  `extract_moe_candidates.py`.
- **Стадийные конвейеры** с `config.yaml` и нумерованными стадиями:
  `reco_excess/stages/{00…05}_*.py` (+ `config.yaml`, `src/`, `notebooks/`) и
  `response_vs_flux/stages/{00,10,20,30,31,40,41,50,51}_*.py` (+ `PROTOCOL.md`,
  `README.md`, `RESULTS.md`). Основание: shebang/`__main__`+argparse и docstring
  вида `nohup python stages/30_split_response.py > /tmp/stage30.log`.
- **Одиночные скрипты с shebang** (`#!/usr/bin/env python`) — 49 штук, в основном в
  `model_comparison/`, `finetune_set/`, `reports/2026-07-0{6,7}/`.
- **Скрипты без гарда, с исполняемым кодом на верхнем уровне** — 14 штук; запускаются
  как `python file.py`. Полный список: `archive/260524/_test.py`,
  `data_manager/nu_classifier_ds_builder/test.py`,
  `data_manager/root2h5/test_mc_reco_small.py`, шесть `test_predictions.py` /
  `test_event_by_event.py` в `gplotnikov_sig_noise_models/*/`,
  `model_comparison/ft_benefit_audit.py`, и четыре файла в `reports/2026-07-06/`
  (`ab_decomposition.py`, `ablate_tail.py`, `zspread_distribution.py`, `zspread_proxy.py`).
  Основание: AST — есть `For`/`With` или ≥3 вызова на верхнем уровне, нет `__main__`
  и нет shebang.

### 1.10 Ноутбуки

**105 `.ipynb`** — самостоятельный класс точек входа. 62 из них импортируют модули
проекта. Все untracked (`.gitignore: *.ipynb`). Крупнейшие скопления: `inference/` (30),
`data_manager/` (25), `inference_v2/` (18), `archive/` (8), `notebooks/` (7).

### 1.11 Планировщик

**Планировщика нет.** `crontab`, systemd-юниты, sbatch/slurm-скрипты не найдены; во всех
57 `.sh` нет ни `#SBATCH`, ни `qsub`. Фоновый запуск делается через `nohup … &` и
(по `doc/`, `SKILL.md`) через tmux/screen. Основание: grep по всем `.sh`, отсутствие
`*.sbatch`/`*.slurm`/`crontab` в дереве.

---

## 2. Достижимость

Из 337 модулей: **254 достижимы транзитивно из точек входа CLI**, ещё **20 —
только из ноутбуков**, **63 — ни оттуда, ни оттуда**.

### 2.1 Достижимо только из ноутбуков

Эти модули живы, но единственный путь к ним — `.ipynb`, который не версионируется:

| Модуль | Кто импортирует |
|---|---|
| `data_manager/catalog_v2/retriever.py` | `data_manager/catalog_v2/test copy.ipynb` |
| `data_manager/constants.py` | `root2df/main.py` → `read_a_root.ipynb`; также строкой в `excess_mechanism/features.py` и в `doc/hdf5_format.md` |
| `data_manager/root2df/{main,internal_root_paths,polars_schema}.py` | `data_manager/root2df/read_a_root.ipynb` |
| `data_manager/data/mc_reco_root/utils.py` | `connect_jinr.ipynb` (как голый `utils`) |
| `inference_v2/shared/{h5_hits,reco_schema}.py` | пакет `HIGH_SCORE_EXCESS/excess/` → `HIGH_SCORE_EXCESS/main.ipynb` |
| `HIGH_SCORE_EXCESS/excess/{__init__,figures,paths,sampling,signal,sources}.py` | `HIGH_SCORE_EXCESS/main.ipynb` (`import excess`) |
| `excess_mechanism/analysis.py` | `excess_mechanism/groups.ipynb` (`import analysis`) |
| `reco_excess/src/present.py`, `response_vs_flux/src/present.py` | соответствующие `notebooks/*.ipynb` |
| `SharedModels/…/src/base_models.py` | `convert_model_to_onnx.ipynb` |
| `src/__init__.py`, `src/data/__init__.py` | ноутбуки |

### 2.2 Ноутбуки ссылаются на модули, которых нет

| Импорт в ноутбуке | Где | Состояние |
|---|---|---|
| `data_manager.root_extractor.main`, `.internal_root_paths` | `data/h5datasets/observe_raw_mc.ipynb`, `data/mc_reco_root/observe.ipynb`, `archive/260524/observe copy.ipynb` | модуля `root_extractor` нет; по именам файлов совпадает с `root2df/` |
| `data_manager.processor` | `archive/260524/draft.ipynb` | нет |
| `data_manager.h5datasets.paths2h5` | `src/data/test_mc_ds.ipynb` | файл есть по пути `data_manager/data/h5datasets/paths2h5.py` — другой путь импорта |

Основание: regex по `code`-ячейкам всех 105 ноутбуков + проверка на существование пути.

### 2.3 Недостижимо ни из CLI, ни из ноутбуков

63 файла, из которых 14 — пустые `__init__.py` (0 байт, md5 `d41d8cd9…`):
`data_manager/{catalog_v2,nu_classifier_ds_builder,prefilter_npy_ds_builder,root2df}/__init__.py`,
`inference/__init__.py`,
`inference_v2/{nu_classifier,nu_classifier/analysis,nu_classifier/exp_finetuning,prefilter,prefilter/analysis,shared}/__init__.py`,
`src/{models,training,utils}/__init__.py`.
и ещё 14 — сами точки входа без `__main__`-гарда (перечислены в §1.9).

Остальные **35** — ниже. `git` — дата последнего коммита, затронувшего файл
(`git log -1 --format=%ad -- <file>`); `mtime` — `stat -c %y`.

Напоминание из §0: git-дата **2026-09-06** означает лишь «попал в общий pre-cleanup
commit» и о возрасте файла не говорит ничего; ориентироваться следует на mtime.

| Файл | git | mtime | git-track | Кто импортирует | Упоминания вне кода |
|---|---|---|---|---|---|
| `data_manager/nu_classifier_ds_builder/__main__.py` | 2026-05-24 | 2026-04-28 | tracked | — | **есть**: SKILL.md запускает `python -m data_manager.nu_classifier_ds_builder` — граф импортов такой запуск не видит |
| `data_manager/prefilter_npy_ds_builder/__main__.py` | 2026-05-24 | 2026-03-26 | tracked | — | нет; аналогичный `-m`-запуск не задокументирован |
| `data_manager/data/h5datasets/paths2h5.py` | (none) | 2025-09-22 | **untracked** | — | нет |
| `data_manager/data/exp_reco_root/utils.py` | (none) | 2026-04-14 | **untracked** | — | нет. Совпадает на 0.58 с `data/mc_reco_root/utils.py` |
| `data_manager/root2h5/eval_tres_reco.py` | 2025-10-03 | 2025-09-16 | tracked | — | нет. Совпадает на 0.81 с `eval_tres.py`, который **достижим** (импортируется `root2h5.py`, `root2h5_exp.py`, `root2h5_mc_reco.py`) |
| `archive/260524/sig_noise_model_v3 copy.py` | (none) | 2026-05-24 | ignored (`archive/`) | — | нет |
| `src/training/exp_finetuning_trainer_legacy_aug.py` | 2026-09-06 | **2026-06-12** | tracked | — | нет. Собственная шапка: «legacy augmentation (pre-fix copy). Kept to reproduce results trained before the augmentation fix» |
| `inference_v2/nu_classifier/analysis/metrics.py` | 2026-06-08 | 2026-05-25 | tracked | — | **есть**: `inference_v2/README.md` документирует его как публичный API (`auc()`, `at_threshold()`, `efficiency_rejection_curve()`) |
| `inference_v2/prefilter/analysis/load.py` | 2026-06-08 | 2026-05-25 | tracked | `prefilter/analysis/__init__.py` (сам недостижим) | README описывает `prefilter/analysis/` как раздел |
| `inference_v2/prefilter/analysis/metrics.py` | 2026-06-08 | 2026-05-25 | tracked | — | то же |
| `inference_v2/prefilter/analysis/plots.py` | 2026-06-08 | 2026-05-25 | tracked | — | то же |
| `…/analysis/model_comparison/ft_benefit_audit.py` | 2026-09-06 | **2026-07-25** | tracked | — | нет (это точка входа с top-level кодом, см. §1.9) |
| `…/analysis/prefilter_vs_nu_classifier/_gen_notebook.py` | 2026-09-06 | **2026-06-15** | tracked | — | нет. Соседний `prefilter_vs_nu_classifier.ipynb` существует — вероятно этим и сгенерирован, **не проверено** |
| **23 файла в `gplotnikov_sig_noise_models/`** | см. ниже | | | | |

23 недостижимых файла sig-noise разложены так:

- `{k0p0001,k0p001,nn_aug_highq_p025,nn_aug_highq_p0p0002}/encoder.py` и `/layers.py`
  (8 файлов) — **грузятся динамически** через `importlib` из `sig_noise_model_v2.py`;
  статически невидимы.
- `{k0p0001,k0p001,k_nsol_labelneq0_da_hs128,hs512_dff512,p025,p0p0002}/model_simplified.py`
  (6) — **грузятся динамически** из `sig_noise_model_v3.py` тем же механизмом.
- `{k0p0001,k0p001,p025,p0p0002}/sig_noise_model_v2.py` (4) — не импортируются ничем;
  живая ветка — `v3`.
- `{k_nsol_labelneq0_da_hs128,k0p001,hs512_dff512,p025,p0p0002}/sig_noise_model_v3.py` (5) —
  недостижимы, потому что весь код проекта импортирует **только** версию из `k0p0001`
  (см. §3.1). Достижима ровно одна: `k0p0001/sig_noise_model_v3.py`.

---

## 3. Кластеры вариантов

### 3.1 `gplotnikov_sig_noise_models/` — 7 каталогов одной модели

Каталоги: `archive/`, `encoder_nl5_nh1_dff512_hs512_bs128/`,
`k_nsol_labelneq0_da_hs128/`, `k_nsol_labelneq0_da_hs128_k0p0001/` (канонический по
`CLAUDE.md`), `k_nsol_labelneq0_da_hs128_k0p001/`, `k_nsol_labelneq0_hs512_dff512/`,
`nn_aug_highq_p025/`, `nn_aug_highq_p0p0002/`.

**По существу различаются двумя вещами**: архитектурой (hidden_size 128 против 512) и
чекпоинтом. Код между каталогами скопирован: `encoder.py`, `layers.py`,
`model_simplified.py` и `sig_noise_model_v2.py` **байт-идентичны** в четырёх каталогах
(md5 `113e6434…`, `40c322e5…`, `8c2ddf9e…`, `c81afb6e…`).

Различие v2 против v3 — не в модели, а в способе сборки:
- `sig_noise_model_v2.py` (251 строка) оборачивает `EncoderDomainAdaptation` из
  `encoder.py`, склеивая синтетический пакет `_snm_pkg` через `importlib`, чтобы
  разрешить относительные импорты; DA-голова отбрасывается обёрткой `_InferenceModel`.
  `DEFAULT_MAX_GPU_HITS = 2_000_000`.
- `sig_noise_model_v3.py` (206 строк) грузит `model_simplified.py` (энкодер + main_head,
  без DA) и читает чекпоинт с `strict=False`. `DEFAULT_MAX_GPU_HITS = 20_000_000`.

**Проверенное расхождение в `nn_aug_highq_p025/` и `nn_aug_highq_p0p0002/`.**
Оба каталога untracked (git status), созданы 2026-07-04. Установлено:

1. `nn_aug_highq_p025/predict_mc_h5.py` и `nn_aug_highq_p0p0002/predict_mc_h5.py` —
   **байт-идентичны** (md5 `1d6ad578…`). Обе строки 33–34:
   `from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3
   import load_model`; строка 75: `model, _, dev = load_model(device=device)` — **без**
   аргумента `checkpoint_path`. То есть оба скрипта грузят чекпоинт `k0p0001`, а не
   лежащий рядом аугментированный.
2. В обоих `MODEL_TAG = "k_nsol_labelneq0_da_hs128_k0p0001"` (строка 42), а выходное имя
   строится как `input.stem + f"_probs_{MODEL_TAG}.h5"` (строка 64). Значит оба пишут в
   `baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5` — файл, который
   существует и весит 189 ГБ.
3. `nn_aug_highq_p025/sig_noise_model_v3.py:23` объявляет
   `DEFAULT_CHECKPOINT = MODEL_DIR / "best_aug_p002.ckpt"`, а в каталоге лежит
   `best_aug_p025.ckpt` (`ls`). Файла `best_aug_p002.ckpt` в этом каталоге нет.
4. `train_config_mc_2020.yaml` в обоих `nn_aug_*` — **байт-копия** конфига из `k0p0001`
   (md5 `ba6562cb…` у всех трёх) с `hidden_size: 128`, тогда как их чекпоинты содержат
   `first_layer.weight` формы **(512, 5)**, 63 тензора, без префикса `encoder.`
   (проверено чтением state_dict через `torch.load(..., weights_only=True)`).
   Для сравнения, `k0p0001/best_mc_2020.ckpt`: `encoder.first_layer.weight` (128, 5),
   69 тензоров. Размеры файлов подтверждают: 31 594 384 Б против 4 063 540 Б.
5. `README.md` в обоих `nn_aug_*` — **байт-копия** README из `k0p0001`
   (md5 `02009e29…` у всех трёх); заголовок гласит
   `# Signal/Noise Hit Classifier — k_nsol_labelneq0_da_hs128_k0p0001`, таблица
   указывает `Hidden size 128`, `DA coefficient k 0.0001`.

Совокупный вывод, который следует из 1–5: **локальные `sig_noise_model_v3.py`,
`model_simplified.py`, `encoder.py`, `layers.py`, конфиг и README в обоих `nn_aug_*`
каталогах не участвуют в работе их же `predict_mc_h5.py`**, а сам `predict_mc_h5.py`
там скорит моделью `k0p0001`. Что при этом было фактически запущено и какие результаты
получены — по коду не установить (вопрос Q1).

**Разрешено ответом владельца (Q1):** чекпоинты `aug_highq` **не использовались ни разу**,
probs-файл посчитан моделью `k0p0001`. Значит дефект **латентный**: ни один существующий
артефакт им не затронут, файл назван корректно. Нереализованным остаётся то, что запуск
любого из двух `predict_mc_h5.py` скорил бы моделью `k0p0001` и переписал бы
канонический probs-файл (189 ГБ).

Прочие точные дубли внутри кластера: `archive/predict_exp.py` == `k0p001/predict_exp.py`
(md5 `c14e753a…`); `test_predictions.py` идентичен в `k0p001`, `k_nsol_labelneq0_da_hs128`,
`hs512_dff512` (md5 `24e2c8e0…`).

### 3.2 `inference/` (v1) против `inference_v2/`

| | `inference/` | `inference_v2/` |
|---|---|---|
| Размер на диске | 999 ГБ (`prefilter_model/` 995 ГБ) | 99 ГБ |
| Формат выхода | HDF5 `mc_preds_{tag}.h5` / `exp_preds_{tag}.h5` со структурой `/{src}/{ptype}/scores/{pk}/data`; Parquet (`make_save_preds_streaming.py`) | per-checkpoint DuckDB `mc_merged_thr{thr}.duckdb` + `run_info.json` + `prediction_history.csv` |
| Ключ события | `event_ids[i]` — индекс внутри part | `event_fk` → `catalog_v2.events.id` |
| Записи в git | 65 файлов | 38 файлов |
| Живые потребители | ~30 ноутбуков (`inference.prefilter_model.utils` — в 12 из них) | скрипты `analysis/`, `reports/`, `reader/` |

**Это смена формата хранения предсказаний, а не смена модели.** `inference/nu_classifier_model/utils.py`
и `inference/prefilter_model/utils.py` — байт-идентичны (md5 `cd4b7b9f…`).

### 3.3 `data_manager/root2df/` против `root2h5/`

`root2df` (4 файла, mtime 2026-03-15, tracked): uproot → **polars** → Parquet,
схема в `polars_schema.py`, пути внутри ROOT в `internal_root_paths.py`.
`root2h5` (10+ файлов): uproot + `multiprocessing.Process/Queue` → **HDF5**.
`root2df` достижим только из `read_a_root.ipynb`; упомянут в `doc/hdf5_format.md`.

### 3.4 Шесть конвертеров `root2h5_*`

Не варианты одного, а **разные источники/схемы**, что заявлено в их шапках:
`root2h5.py` (MC truth) → `root2h5_mc_reco.py` («Based on root2h5.py. Adds reco_prty,
reco_vectors») и `root2h5_exp.py` → `root2h5_exp_reco.py` («Based on root2h5_exp.py. Adds
reco_prty, raw/labels»). `root2h5_exp_full.py` отличается chunked-чтением и
`header_prty` из `BJointHeader`. Мера сходства подтверждает: `root2h5.py` vs
`root2h5_exp.py` — 0.47; `archive/root2h5_reco.py` vs `root2h5.py` — 0.53.

### 3.5 Каталог v1 (Parquet) против v2 (DuckDB)

`build_catalog_{mc_merged,exp,exp_reco,mc_normed}.py` (март 2026) пишут Parquet в
`data_manager/h5_catalogs/` — **219 ГБ** в `catalogs_mc_merged/` плюс 159 МБ / 110 МБ /
502 МБ / 250 МБ в остальных.
`catalog_v2/build_{mc,mc_reco,exp}.py` (май 2026) пишут единый
`data_manager/catalog_v2.duckdb` — **132 ГБ**. Оба набора данных на диске.
Различие по существу — формат и модель ключа: v1 = один Parquet на тип частицы,
v2 = единая таблица `events` с суррогатным `id`, на который ссылается `event_fk`
в предсказаниях.

### 3.6 Два дерева `energy_truth` — это стадии, а не варианты

`data_manager/root2h5/energy_truth/` — извлечение из ROOT/`.wout` и вычисление таргетов
(`extract_interactions.py`, `read_wout.py`, `targets.py`, `targets_vec.py`).
`data_manager/energy_truth/` — сборка и валидация компаньон-файла
(`read_root.py`, `build_h5.py`, `validate.py`, `truth.py`).

Два файла несут **явную шапку об устаревании**, что снимает двусмысленность:
`data_manager/root2h5/create_energy_h5.py` и
`data_manager/root2h5/energy_truth/validate_targets.py` начинаются с
`"""SUPERSEDED (2026-08-21) by data_manager/energy_truth/.` с объяснением, что старая
схема (`targets/`, `interactions/`, `muon/`) больше не существует.
Это единственный кластер в репозитории, где старая версия сама себя маркирует.

### 3.7 Тренеры

11 файлов в `src/training/` + `src/training/archive/` (2) + `archive/260524/` (3).
Меры сходства (шинглы, комментарии вырезаны):

| Пара | Jaccard | Существо различия |
|---|---|---|
| `archive/260524/da_prefilter_numu_trainer_upsample_zflip.py` ↔ `src/training/da_prefilter_numu_trainer.py` | **0.99** | практически один файл; вариант «upsample_zflip» стал основным тренером |
| `archive/260524/…_traintval.py` ↔ `…_upsample.py` | 0.93 | разбиение train/val против upsampling |
| `…_upsample.py` ↔ `src/training/da_prefilter_numu_trainer.py` | 0.73 | без zflip-аугментации |
| `src/training/archive/da_hcut_numu_trainer.py` ↔ `archive/da_numu_trainer.py` | 0.81 | hcut-задача (только сигнальные хиты) против полной |
| `src/training/exp_finetuning_trainer.py` ↔ `_legacy_aug.py` | **0.88** | по собственной шапке `_legacy_aug` — «pre-fix copy», отличается аугментацией |
| `da_nu_classifier_trainer.py` ↔ `da_prefilter_numu_trainer.py` | 0.39 | две разные задачи двухступенчатого пайплайна |

`sngp_nu_classifier_trainer.py` (490 строк) и `domain_sep_trainer.py` (637) — отдельные
задачи, не варианты.

### 3.8 Точные дубли вне sig-noise

| md5 | Файлы |
|---|---|
| `ab65ef0e…` | `reports/2026-07-06/event_unfolding.py` == `reports/2026-07-07/event_unfolding.py` |
| `a548b04e…` | `reco_excess/src/provenance.py` == `response_vs_flux/src/provenance.py` |
| `c12288e7…` | `reco_excess/src/present.py` == `response_vs_flux/src/present.py` |
| `cd4b7b9f…` | `inference/nu_classifier_model/utils.py` == `inference/prefilter_model/utils.py` |

### 3.9 Пары «скрипт и его пересчёт/вариант» в `model_comparison/`

Не дубли по коду, но одна тема: `find_cuts.py` / `find_cuts_binned.py`;
`paper_suppression_curve.py` / `paper_suppression_curve_daexp.py` (Jaccard 0.44) /
`replot_paper_curve.py`; `build_dist_to_nu.py` / `replot_dist_to_nu.py`;
`nu_candidates_display.py` / `nu_candidates_full.py`; `bimodality_full.py`;
`finetune_validation.py` / `working_point_sweep.py` (0.44);
`sngp_exp_eval.py` / `sngp_variance_veto.py` / `sngp_working_point.py`.
Чем именно различаются `*_full`, `replot_*` и `*_daexp` — по коду видно только то, что
это разные срезы/перерисовки; какая версия породила фигуры статьи, **не установлено**
(вопрос Q8).

### 3.10 Каталоги результатов с суффиксами

- `analysis/{moe_results, moe_results_INCLUDED_TRAINSET}`,
  `analysis/{single_results, single_results_INCLUDED_TRAINSET}`
- `exp_finetuning/finetuned_models/` — 7 подкаталогов с суффиксами
  `_FAIL`, `_OLD`, `_OLD_COPY`, `_STRANGE`, `_LARGETAUG_finetuned`, плюс отдельный
  каталог `finetuned_models_LARGETAUG/` с ещё одним прогоном того же чекпоинта.
- `data/h5datasets/`: `exp_reco.h5` / `exp_reco_OLD.h5`;
  `baikal_mc_reco.h5` / `_OLD.h5` / `_OLDNewer.h5`;
  `exp_probs_….h5` / `….DEPRECATED_bs512.h5`;
  `exp_full_probs_….h5` / `….DEPRECATED_bs512.h5`.

---

## 4. Данные и артефакты

Ничего не предлагается делать. Только состав, размер, git-статус, кто пишет, кто читает.

### 4.1 Общий объём

| Каталог | Размер |
|---|---|
| `data_manager/` | 2.6 ТБ |
| `inference/` | 999 ГБ |
| `inference_v2/` | 99 ГБ |
| `experiments/` | 8.9 ГБ |
| `gplotnikov_sig_noise_models/` | 143 МБ |
| `SharedModels/` / `papers/` / `archive/` / `notebooks/` | 5.5 / 4.6 / 3.7 / 1.9 МБ |

### 4.2 HDF5 — 27 файлов, 1746 ГБ

Все под `*.h5` в `.gitignore` ⇒ **ни один не в git**.

| Файл | Размер | mtime | Пишет | Читает |
|---|---|---|---|---|
| `data/h5datasets/baikal_mc_merged.h5` | 956 ГБ | 2026-04-10 | `root2h5/root2h5.py` | `catalog_v2/build_mc.py`, `predict_mc.py`, `nu_classifier_ds_builder/build.py`, `energy_truth/build_h5.py`, `gplotnikov …/predict_mc_h5.py` |
| `…_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5` | 189 ГБ | 2026-08-16 | `k0p0001/predict_mc_h5.py` (и, по §3.1, оба `nn_aug_*/predict_mc_h5.py` писали бы сюда же) | `predict_mc.py --probs-h5`, `nu_classifier_ds_builder`, `inference/nu_classifier_model/predict_mc.py` |
| `…_energy_truth.h5` | 163 ГБ | 2026-08-21 | `energy_truth/build_h5.py` | `energy_truth/validate.py`, `tutorial_energy_truth.ipynb` |
| `exp_full.h5` | 136 ГБ | 2026-06-25 | `root2h5/root2h5_exp_full.py` | `catalog_v2/build_exp.py exp_full`, `predict_exp.py` |
| `exp_reco_full_2020.h5` | 113 ГБ | 2026-04-22 | не удалось установить (в `root2h5_exp_reco.py` имя выхода задаётся конфигом) | не удалось установить |
| `exp_full_probs_….h5` | 55.9 ГБ | 2026-08-16 | `k0p0001/predict_exp_h5.py` | `nu_classifier_ds_builder/exp_builder.py`, `predict_exp.py --probs-h5` |
| `exp_full_probs_….DEPRECATED_bs512.h5` | 55.9 ГБ | 2026-08-16 | то же, прогон при batch 512 | — (помечен в `DEPRECATED.md`) |
| `exp_reco.h5` / `exp_reco_OLD.h5` | 22.0 / 21.5 ГБ | 2026-04-16 | `root2h5_exp_reco.py` | `catalog_v2/build_exp.py exp_reco` |
| `exp_reco_probs_….h5` | 7.5 ГБ | 2026-08-27 | `predict_exp_h5.py` | `predict_exp.py` |
| `baikal_mc_reco.h5` (+`_OLD`, `_OLDNewer`) | 6.6 / 4.4 / 5.8 ГБ | 2026-04 | `root2h5_mc_reco.py` | `catalog_v2/build_mc_reco.py` |
| `baikal_mc_reco_probs_….h5` | 2.5 ГБ | 2026-08-27 | `predict_mc_h5.py` | `predict_mc.py --source mc_reco` |
| `exp.h5`, `exp_probs_*.h5` | 0.27 ГБ и меньше | | `root2h5_exp.py` | `run_batch_predict.sh` (EXP_H5_FILES) |

### 4.3 DuckDB — 105 файлов, 234 ГБ

| Файл | Размер | Пишет | Читает |
|---|---|---|---|
| `data_manager/catalog_v2.duckdb` | **132 ГБ** | `catalog_v2/build_{mc,mc_reco,exp}.py` | `inference_v2/shared/catalog_query.py`, все `predict_*.py` (дефолт `--catalog`), `analysis/*`, `reader/query.py` |
| `preds/{ckpt}/mc_merged_thr0p8.duckdb` × 35 каталогов | до 6.6 ГБ (у `…seed32@best_da_model`) | `predict_mc.py`, `predict_npy.py` (общий файл) | `analysis/load.py`, `reader/`, `reports/*` |
| `preds/{ckpt}/{exp,exp_full,exp_reco,mc_reco}_thr0p8.duckdb` | 1.8 МБ – 256 МБ | `predict_exp.py` / `predict_mc.py` | то же |
| `excess_mechanism/{features,pair,seq}.duckdb` | 331 / 77 / 98 МБ | `excess_mechanism/{build_features,pair_features,seq_features}.py` | `excess_mechanism/{analysis,excess_by_band,find_cuts,…}.py`, `groups.ipynb` |
| `exp_finetuning/finetuned_models*/…/{mc_merged,exp}_thr0p8.duckdb` (16 шт.) | до 486 МБ | `run_predict_finetuned.sh` → `predict_{mc,exp}.py` | `model_comparison/finetune_validation.py` и др. |

Все под `*.duckdb` в `.gitignore`.

**Известная особенность, задокументированная в авто-памяти и подтверждаемая
`inference_v2/README.md`**: `mc_merged_thr*.duckdb` содержит и события из обучения
(записанные `predict_npy.py`), и вне обучения (`predict_mc.py`) — в одной таблице.

### 4.4 Чекпоинты — 1875 файлов, 9.5 ГБ

`experiments/numu/` — 31 каталог прогонов, `experiments/Archive/` — 30.
В каждом: `best_da_model.pth`, `da_checkpoint_epoch_NNN.pth`, `da_config.yaml`,
`normalization_config.yaml`, `da_training_summary.yaml`.
Пишет — соответствующий тренер; читает — `inference_v2/shared/model_utils.py:load_model()`.
`.gitignore` исключает `*.pth`, `*.ckpt`, `experiments/*/checkpoints/`.

Sig-noise чекпоинты (7 шт., 143 МБ в сумме) перечислены в §3.1.

### 4.5 NPY-датасеты — 141 файл, 63 ГБ

| Каталог | Размер | Пишет | Читает |
|---|---|---|---|
| `datasets/baikal_mc2020_prefilter/` | 48 ГБ | `prefilter_npy_ds_builder/build.py` | `src/data/prefilter_npy_dataset/` |
| `datasets/nu_classifier_dataset_h5s0_thr0.8/` | 1.4 ГБ | `nu_classifier_ds_builder/build.py` | `src/data/nu_classifier_dataset/`, `predict_npy.py`, `run_umap.py` |
| `…_exp_full_thr0.8/` и `…_exp_full_thr0.8_DEPRECATED_bs512/` | по 2.2 ГБ | `nu_classifier_ds_builder/exp_builder.py` | DA-таргет в `da_nu_classifier_trainer.py` |
| `…_h5s0_thr0.5`, `…_h5s2_thr0.5`, `…_h8s2_thr0.5`, `nu_classifier_TEST_…` | 0.6–2.1 ГБ | тот же билдер | — |

В каждом каталоге лежит `build_config.yaml` — снимок конфига сборки.

### 4.6 Parquet-каталоги (v1) — 219 ГБ + 1 ГБ

`h5_catalogs/catalogs_mc_merged/` 219 ГБ, `catalogs_exp/` 159 МБ,
`catalogs_exp_reco/` 110 МБ, `catalogs_mc_signoise_normed/` 502 МБ,
`signoise_encoder_nl5…_preds/` 250 МБ. Пишут `build_catalog_*.py`; читают —
`src/data/{numu,prefilter,hcut_numu}_dataset.py` и `usage.ipynb` в каждом каталоге.

### 4.7 Логи и эталонные выходы

- `experiments/train_E1_FIXED_sn256.log` — 5.6 МБ, в корне `experiments/`.
- `experiments/logs/` — логи всех обучений (`.gitignore: *.log`).
- `preds/prediction_history.csv` — append-only журнал каждого запуска предсказаний
  (колонки описаны в `inference_v2/README.md`, включая `is_successful` и `error_msg`;
  неуспешные запуски тоже пишут строку).
- `preds/*.log` — 7 файлов на верхнем уровне `preds/` (`e1_predict_exp_FULL.log` и др.).
- `k0p0001/probs_fingerprint_{before,after}.json` — 270 КБ / 230 КБ, эталонные слепки
  вероятностей; пишет `fingerprint_probs_h5.py`.
- `k0p0001/missing_parts.json` (14 КБ), `predict_*_bs256.log`.

### 4.8 Прочие артефакты

`compare_aug_strategies.ipynb` (251 КБ) и `compare_aug_strategies.png` (113 КБ),
`output.png` (215 КБ) в корне; `my_notes/*.png` (960 КБ, включая два `.excalidraw.png`);
`papers/NNPipelineForBaikalGVD/{main.tex, baikal_ml_highlight.pdf, images/}`
(`.gitignore` исключает `*.tex` и `*.pdf`).
`__pycache__` присутствует внутри `gplotnikov_sig_noise_models/`,
`data_manager/energy_truth/`, `inference_v2/reader/` и др.

---

## 5. Поверхность конфигурации

Гиперпараметры и пути живут **в пяти независимых слоях**.

1. **YAML экспериментов** — 18 файлов в `experiments/` (`da_nu_classifier_*.yaml`,
   `da_prefilter_numu_*.yaml`, `domain_sep_*.yaml`, `numu_baseline.yaml`,
   `sngp_nu_classifier_baseline.yaml`). Плюс снимок `da_config.yaml` +
   `normalization_config.yaml` внутри каждого из 61 каталога прогона.
2. **YAML билдеров и конвертеров** — `nu_classifier_ds_builder/{default,compliment,exp_full,test}_config.yaml`,
   `prefilter_npy_ds_builder/default_config.yaml`,
   `root2h5/root2h5_config{,_exp,_exp_full,_exp_reco,_mc_reco}.yaml`,
   `stats_dict/{default_mc,NuCut5hits_mc}.yaml`,
   `h5_catalogs/catalogs_mc_signoise_normed/norm_params.yaml`.
3. **Переменные в шапках 57 `.sh`** — `CHECKPOINT`, `DEVICE`, `THRESHOLD`, `MIN_HITS`,
   `BATCH_SIZE`, `OUTPUT_DIR`, `MODELS=(…)`. Редактируются на месте перед запуском;
   `inference_v2/README.md` прямо это предписывает: «Edit the variables at the top of the
   shell script».
4. **Дефолты argparse** — приведены в §1.7.
5. **Константы в коде** — `SN_BATCH_SIZE = 256` (`predict_mc.py:62`, `predict_exp.py:68`);
   `THRESHOLD = 0.8` (`test_scoring_equivalence.py:35`);
   `DEFAULT_MAX_GPU_HITS`, `MODEL_DIR`, `DEFAULT_CHECKPOINT`, `MODEL_TAG` в sig-noise;
   `SEED = 42`, `DEVICE = "cuda:0"`, `H5_PATH`, `CATALOG_DIR`, `MODEL_CONFIG` (словарь
   из 15 полей) прямо в `src/data/test_prefilter_reproducibility.py`.

### 5.1 Захардкоженные абсолютные пути

| Путь | Где | Существует |
|---|---|---|
| `/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5` | `src/data/test_prefilter_reproducibility.py:22`; `CLAUDE.md` § Data Location; `.claude/skills/pipelines/SKILL.md` (команда `ls -la …`) | **нет** — каталога `/net` на машине не существует (`ls -d /net`) |
| `/home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5` | `data_manager/build_catalog_mc_normed.py` | не проверено (`/home2` вне доступа) |
| `/home/albert/Baikal2025/inference_v2/nu_classifier/analysis/excess_mechanism` | 4 вхождения в `excess_mechanism/*.py` | да |
| `/home/albert/Baikal2025/data_manager/data/h5datasets/{baikal_mc_merged,…_energy_truth}.h5` | по 2 вхождения | да |
| `/home/albert/miniconda3/envs/baikal25/bin/python` | `run_reco_chain_FIXED_sn256.sh:27` | не проверено |

### 5.2 Установленные противоречия между источниками

| Параметр | Источник A | Источник B | Источник C |
|---|---|---|---|
| `min_hits` / `min_strings` | argparse `predict_mc.py:511-512` и `predict_exp.py:423-424`: **8 / 2** | `run_batch_predict.sh:26-27`: **5 / 0**; `inference/nu_classifier_model/run_predict_mc.sh`: **5 / 0** | `inference_v2/README.md` пример: `cuts={"min_sn_hits": 8, "min_sn_strings": 2}` |
| batch size для sig-noise | константа `SN_BATCH_SIZE = 256` в обоих predict-скриптах | `run_batch_predict.sh` задаёт `BATCH_SIZE=1024` и **не передаёт** `--sn-batch-size` (grep: нет вхождений) | `doc/sig_noise_batch_size.md` и `preds/EXP_PREDS_STALE.md` объясняют, почему это меняет отбор хитов |
| число чекпоинтов в батче | `run_batch_predict.sh` `MODELS=(…)` — **3** записи (все `260531_*_FixedDA`) | `CLAUDE.md`: «8 nu-classifier checkpoints being scored» | `preds/` содержит **35** каталогов |
| путь к MC | `CLAUDE.md`, SKILL.md, тест воспроизводимости: `/net/62/…` | все рабочие скрипты и `run_batch_predict.sh:34`: `data_manager/data/h5datasets/baikal_mc_merged.h5` | — |
| имя модуля каталога | `.claude/settings.json` разрешает `Bash(python -m data_manager.catalog.build_exp *)` | модуля `data_manager.catalog` не существует; есть `data_manager.catalog_v2` | — |
| чекпоинт sig-noise | `nn_aug_highq_p025/sig_noise_model_v3.py:23`: `best_aug_p002.ckpt` | в каталоге лежит `best_aug_p025.ckpt` | — |
| архитектура sig-noise | `nn_aug_highq_*/train_config_mc_2020.yaml`: `hidden_size: 128` | их чекпоинты: `first_layer.weight` (512, 5) | README там же: «Hidden size 128» |
| какие sig-noise модели под git | комментарий `.gitignore:154`: «All model variants except `k_nsol_labelneq0_da_hs128_k0p0001` are excluded» | в списке исключений (строки 155–161) **нет** `nn_aug_highq_p025/` и `nn_aug_highq_p0p0002/` | по факту 11 их `.py` отслеживаются с `5fb2603`; их `train_config_mc_2020.yaml` и `README.md` — нет (перехвачены общими `*.yaml`/`*.md`) |

Последняя строка имеет прямое следствие для §3.1: у двух каталогов, где конфиг не
соответствует чекпоинту, **в git попал код, но не конфиг**, поэтому несоответствие
по истории репозитория не видно.

**Две строки этой таблицы разрешены после ответов владельца (§8):**

* *`min_hits`/`min_strings`* — противоречие источников реально, но неоднозначности в
  данных не создаёт: фактически использованная точка — **5 / 0**, и она восстановима
  для каждого прогона из `run_info.json` (33 из 35 каталогов), 135 логов и
  `prediction_history.csv` (Q10).
* *batch size для sig-noise* — **правки не требует**: `sn_batch_size` не производна от
  `--batch-size`, дефолт 256 применяется к sig-noise независимо
  (`predict_mc.py:303-305` против 329/335), а 1024 относится только к
  nu-классификатору. Остаточная область — исторические прогоны до этой правки (Q11).

### 5.3 Чего в поверхности конфигурации нет

Нет `pyproject.toml`, `setup.py`, `setup.cfg`, `.env`-файла, централизованного модуля
путей. `environment.yml` есть (19 КБ), но `.gitignore` его исключает (`environment.yml`
в секции Conda). `data_manager/constants.py` существует, но достижим только из
`root2df/main.py` (§2.1).

---

## 6. Поверхность верификации

**Отвечая прямо на вопрос «чем проверить, что код после изменений ведёт себя так же»:
единого способа нет, автоматизированного набора тестов нет, но точечные средства есть,
и одно из них построено именно под эту задачу.**

### 6.1 Чего нет

- Нет `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`, `conftest.py`.
- `import pytest` — **0 файлов** во всём репозитории (grep).
- Нет CI: нет `.github/`, `.gitlab-ci.yml`, `Jenkinsfile`.
- Нет команды «прогнать всё». `CLAUDE.md` § Running Tests перечисляет 5 команд, из
  которых `python test_da_training.py` указывает на файл, лежащий в `archive/2025/`.
- Все файлы с «test» в имени **не под git**: `.gitignore` содержит `test*.py`.

### 6.2 Что есть — самопроверяющиеся скрипты

Ассерты по деревьям: `inference_v2` — 54, `gplotnikov_sig_noise_models` — 13,
`data_manager` — 2, `src` — 0, `inference` — 0.

| Скрипт | Что проверяет | Асс. |
|---|---|---|
| `inference_v2/test_scoring_equivalence.py` | **сравнивает две prediction-DuckDB, построенные из одного чекпоинта/probs/parts**: множество `event_fk` — точно, `n_sn_hits`/`n_sn_strings` — точно, `score` — точно (или с `--score-tol`), эмбеддинги — с допуском; отдельно сообщает, сколько событий пересекли порог 0.8. Написан ровно под вопрос «не изменилось ли поведение» | — |
| `inference_v2/reader/tests/test_reader.py` | пакет `reader` (parts, query, spec, training) | 23 |
| `inference_v2/test_run_info.py` | формат `run_info.json` | 14 |
| `response_vs_flux/tests/test_tracks.py` | геометрия треков | 8 |
| `root2h5/energy_truth/test_targets_vec.py` | «Check that targets_vec reproduces targets.py exactly, on real data» — векторная реализация против референсной | argparse |
| `energy_truth/test_truth.py` | «analytic cases, then regression against measured numbers» — аналитика + регрессия на зафиксированных числах | shebang |
| `root2h5/energy_truth/check_bundle_alignment.py` | сильная проверка выравнивания мюонных пучков | shebang |
| `energy_truth/validate.py` | готовый компаньон-файл против исходного HDF5 | shebang |
| `data_manager/test_npy_retrieval.py` | производительность: каталог + h5-retrieval, `--n-events 100000` | документирован в SKILL.md |
| `src/data/test_prefilter_reproducibility.py` | 3 шага обучения дважды с одним seed, сверка лоссов и весов | зависит от несуществующего `/net/…` (§5.1) |
| `src/data/test_numu_dataset.py` | датасет; `doc/tasklist.md` утверждает «4/4 tests passing» | 0 |
| `gplotnikov …/*/test_predictions.py` (6 копий, 3 из них байт-идентичны) | предсказания sig-noise; рядом лежат `test_prob_hist.png`, `test_feature_hist.png` как визуальные эталоны | 13 суммарно |

### 6.3 Эталонные выходы и журналы

- `k0p0001/probs_fingerprint_before.json` (230 КБ) и `…_after.json` (270 КБ) +
  `fingerprint_probs_h5.py`, `validate_exp_probs.py` — слепки для сверки вероятностей.
- `preds/prediction_history.csv` — полный audit trail запусков, включая неуспешные.
- `run_info.json` в каждом каталоге предсказаний; по `EXP_PREDS_STALE.md` в новых
  прогонах он несёт `sn_probs_source` и `sn_batch_size`.
- `experiments/*/da_training_summary.yaml` — итоговые метрики прогона.
- `test_prob_hist.png` / `test_feature_hist.png` в 5 sig-noise каталогах.
- `--check-existing` у обоих predict-скриптов: предупреждает, если повторный прогон даёт
  другие скоры для уже сохранённых событий.

### 6.4 Ноутбуки с зафиксированными числами

105 ноутбуков хранят выводы ячеек, то есть де-факто содержат зафиксированные числа и
графики. Какие из них содержат актуальные результаты, а какие устарели, **из самих
файлов не установить**: даты выполнения ячеек не сверялись с датами входных данных.

---

## 7. Расхождение документации с состоянием

### 7.1 `CLAUDE.md`

| Утверждение | Состояние |
|---|---|
| «Main MC Data: `/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5`» | каталога `/net` не существует; фактический путь `data_manager/data/h5datasets/baikal_mc_merged.h5` |
| «8 nu-classifier checkpoints being scored across mc_merged/mc_reco/exp» | `run_batch_predict.sh` перечисляет 3; в `preds/` 35 каталогов |
| «⏳ Next: UMAP analysis, further representation-space studies, model comparison on exp data» | UMAP пройден; после него — SNGP (`reports/2026-08-10`), fine-tuning, `excess_mechanism`, `exp_excess`, `reco_excess`, `response_vs_flux`, пакет `reader` (сентябрь 2026) |
| «Iterations 1–7 complete» | та же нумерация, что в `doc/tasklist.md` от 2026-03-15; последующая работа в неё не укладывается |
| § Running Tests: `python test_da_training.py` | файл лежит в `archive/2025/test_da_training.py` |
| § Architecture ссылается на `inference/prefilter_model/` и `inference/nu_classifier_model/` как на действующие | оба существуют, но рабочий тракт — `inference_v2/` (тот же `CLAUDE.md` ниже отмечает его как ✅) |

### 7.2 `doc/`

- `doc/PROJECT_STATUS.md` — «*Last updated: 2026-03-15*», «Current Status: **Iteration 4
  Complete**». Отстаёт примерно на полгода.
- `doc/tasklist.md` — тот же горизонт (последняя правка 2026-03-15).
- `doc/workflow.md`, `doc/data_format.md`, `doc/enhanced_training_features.md`,
  `doc/iteration3_summary.md`, `doc/idea.md` — 2025-10…2026-03.
- Актуальны (август 2026): `claims_log.md`, `hdf5_format.md`, `mc_provenance.md`,
  `mc_binary_formats.md`, `mc_energy_truth.md`, `hdf5_energy_truth.md`,
  `energy_twin_plan.md`, `reco_quality_cuts.md`, `sig_noise_batch_size.md`.
- Каталога `docs/` не существует; вся документация в `doc/`. Этот файл положен в `doc/`.

### 7.3 `inference_v2/README.md`

- Описывает `analysis/` как три модуля — `load.py`, `metrics.py`, `plots.py`.
  Фактически там **113 `.py` в 17 подкаталогах**.
- `analysis/metrics.py` документирован как публичный API, но не импортируется ничем (§2.3).
- Схема каталога `preds/` в README перечисляет `mc_merged_thr0p8.duckdb` и
  `exp_reco_thr0p8.duckdb`; фактически встречаются также `exp_thr0p5`, `exp_full_thr0p8`,
  `mc_reco_thr0p8`.

### 7.4 `preds/EXP_PREDS_STALE.md`

Ссылается на `analysis/exp_excess_investigation`; каталог называется
`analysis/exp_excess`. В остальном документ актуален и точен.

### 7.5 Два хранилища памяти, которые разошлись

| | `memory/` в репозитории | авто-память Claude (`~/.claude/projects/…/memory/`) |
|---|---|---|
| Объём | 2 файла (`MEMORY.md`, `project_overview.md`) | ~35 записей |
| Последнее изменение | 2026-04-09 | сентябрь 2026 |
| Содержание | «Completed iterations (1–7)»; «Next steps: Iteration 8 (pending), Iteration 9 (pending)» | результаты SNGP, fine-tuning, диагностика избытка, `reader`, калибровка времени |

`memory/project_overview.md` называет «лучшими»:
`experiments/numu/da_numu_251123_small_moderatelambdmidddnoreg_aug_newlr` (Val AUC 0.9777) и
`experiments/numu/da_hcut5_numu_260206_smallnn_bigds_middleddnoreg_0.2lambda_aug_correct`
(0.9440). **Ни одного из этих каталогов нет ни в `experiments/numu/`, ни в `experiments/Archive/`**
(проверено `ls -d experiments/{numu,Archive}/*251123* *hcut5*` — «No such file or
directory» во всех четырёх случаях). В `experiments/numu/` 31 каталог: 16 прогонов
`2605*/2607*/2608*_da_nu_classifier_*`, 7 прогонов `da_prefilter_numu_2604*`,
`sngp_nu_classifier_baseline`, и несколько служебных.

### 7.6 `.claude/`

`settings.json` в `permissions.allow` содержит
`Bash(python -m data_manager.catalog.build_exp *)` и `python3` — вариант того же;
модуля `data_manager.catalog` нет (есть `catalog_v2`).
`skills/pipelines/SKILL.md` содержит `ls -la /net/62/home3/…` — путь не существует.

### 7.7 Обратный случай: документация, которая точнее кода

Три места фиксируют состояние аккуратно и заслуживают упоминания как контрпример:

- `preds/EXP_PREDS_STALE.md` — начинается со слов «**Corrected note.** An earlier version
  of this file said … That is wrong for most of them, and the truth matters more», далее
  приводит счёт (6 из 69 MC-логов и 24 из 40 exp-логов читали precomputed probs) и даёт
  команду для проверки провенанса конкретного прогона.
- `data_manager/datasets/…_DEPRECATED_bs512/DEPRECATED.md` и
  `exp_finetuning/exp_bg_datasets/DEPRECATED.md` — объясняют, почему артефакт устарел,
  какие прогоны на нём основаны и почему пересборка отложена сознательно.
- Шапки `"""SUPERSEDED (2026-08-21) by data_manager/energy_truth/.` в двух `.py`
  (§3.6) — единственный случай, когда устаревший код маркирует себя сам.

---

## 8. Вопросы и полученные ответы

Ответы владельца получены 2026-09-06, после составления §1–§7. Ниже — исходный вопрос,
ответ, и отдельно то, что удалось **доразрешить чтением кода** после ответа.

Три пометки о статусе используются строго:
**[ответ]** — сказано владельцем; **[проверено]** — установлено чтением кода/данных;
**[решение владельца]** — намерение владельца, зафиксированное как факт о планах, а не
как вывод этого аудита. Аудит по-прежнему нигде не заключает, что что-либо можно удалить.

---

**Q1. `nn_aug_highq_p025/` и `nn_aug_highq_p0p0002/` — чем посчитан probs-файл?**

**[ответ]** Чекпоинты `aug_highq` **нигде не использовались**. Предсказания в probs-файле
сделаны моделью `k_nsol_labelneq0_da_hs128_k0p0001`.

**[следствие]** Дефект из §3.1 — **латентный, не реализованный**. Файл
`baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5` (189 ГБ) назван корректно и
содержит то, что заявлено; ни один результат им не испорчен. Нереализованным остаётся
риск: `predict_mc_h5.py` в обоих каталогах при запуске скорил бы моделью `k0p0001` и
писал бы поверх этого файла (§3.1, пп. 1–2). Несоответствия конфига, чекпоинта и README
внутри самих каталогов сохраняются.

---

**Q2. Стал ли `upsample_zflip` основным тренером префильтра?**

**[ответ]** Да, правильно. Встречный вопрос владельца: «Этого нет в документации?»

**[проверено] Нет, нигде.** Установлено:

* `grep -rniE 'upsample|zflip'` по всем `.md` вне `bars/` и `cern_root/` — **ноль
  совпадений**. Ни `CLAUDE.md`, ни `doc/`, ни `SKILL.md` не упоминают ни upsampling, ни
  z-flip-аугментацию.
* Docstring самого `src/training/da_prefilter_numu_trainer.py` (строки 1–11) описывает
  задачу, домены и мягкие метки, но **не говорит**, что это бывший вариант
  `upsample_zflip`.
* Более того, строка Usage в этом docstring предписывает
  `--config experiments/da_prefilter_numu_baseline.yaml` — **такого файла нет**
  (`ls`: No such file or directory). Существуют `da_prefilter_numu_hardlabels.yaml`,
  `…_hardlabels_big.yaml`, `…_hardlabels_zflip.yaml`, `…_softlabels.yaml`,
  `…_softlabels_zflip.yaml`.

Итого след «вариант стал основным» существует только в виде совпадения файлов на 0.99
(§3.7) и пяти обёрток `scripts/train_*.sh`, указывающих на старые пути (§1.6).

---

**Q3. `inference/` против `inference_v2/` — что породило числа статьи?**

**[ответ]** В `inference/` делался анализ модели **prefilter**, его результаты отражены
в `main.tex`.
**[решение владельца]** Переделать этот анализ инструментарием `inference_v2` через
`reader`, чтобы не хранить массив данных, и после этого убрать текущий `inference/` из
проекта.

**[замечание аудита]** `inference/` — 999 ГБ, из них `inference/prefilter_model/` 995 ГБ;
к нему привязаны ~30 ноутбуков, 12 из которых импортируют
`inference.prefilter_model.utils` (§2.1, §3.2). Числа prefilter в `main.tex` до
переделки воспроизводятся только этим трактом.

---

**Q4. Parquet-каталоги v1 против `catalog_v2.duckdb`?**

**[ответ]** `h5_catalogs/*` в Parquet — первая версия того, что затем реализовано в
`catalog_v2.duckdb`. Ничего актуального из них не читает.
**[решение владельца]** Убрать.

**[замечание аудита]** Объём — 219 ГБ в `catalogs_mc_merged/` плюс ~1 ГБ в остальных
(§4.6). Статические ссылки на `CATALOG_DIR` остаются в `src/data/numu_dataset.py`,
`prefilter_dataset.py`, `hcut_numu_dataset.py` и в `usage.ipynb` каждого каталога;
из них `hcut_numu_dataset.py` относится к закрываемой ветке (Q14).

---

**Q5. `data_manager/root2df/`?**

**[ответ]** Инструментом давно не пользуюсь.
**[решение владельца]** Заархивировать.

Согласуется с §2.1: достижим только из `read_a_root.ipynb`. Отдельно остаётся, что три
ноутбука импортируют несуществующий `data_manager.root_extractor` — по именам файлов это
прежнее имя `root2df` (§2.2).

---

**Q6 и Q7. `exp_finetuning_trainer` против `_legacy_aug`; семь каталогов
`finetuned_models*` с суффиксами `_FAIL`/`_OLD`/`_OLD_COPY`/`_STRANGE`?**

**[ответ]** Проводить процедуру fine-tuning для статьи было ошибкой.
**[решение владельца]** Всё, что касается текущих fine-tuned моделей, отправить в архив.

**[замечание аудита]** Затрагивает: `src/training/exp_finetuning_trainer.py` и
`_legacy_aug.py`; `inference_v2/nu_classifier/exp_finetuning/` целиком (билдеры фона,
`run_finetune.py`, три `finetune*.yaml`, 8 каталогов моделей, 16 DuckDB до 486 МБ);
`analysis/finetune_set/` (7 скриптов); `model_comparison/{finetune_validation,
ft_benefit_audit}.py`. В §7.7 отмечено, что `exp_bg_datasets/DEPRECATED.md` уже описывал
сознательную отсрочку пересборки этого набора.

---

**Q8. `paper_suppression_curve*` / `replot_*` / `*_full` — что породило фигуры?**

**[ответ]** Не помню, давно не смотрел этот анализ.
**[решение владельца]** В архив; для статьи делать заново.

Остаётся зафиксированным как **не установленное**: какой именно скрипт породил фигуры,
лежащие сейчас в `papers/NNPipelineForBaikalGVD/`.

---

**Q9. Чем получены отчёты `reports/2026-07-08` и `2026-07-09` без единого `.py`?**

**[ответ]** Отчёты писались агентами.

Объясняет наблюдение §3.9/§1.9: в этих двух каталогах есть `REPORT.md`, `summary.md`,
`figures/`, `tables/`, но нет кода. Воспроизводимость таких отчётов из репозитория
**не обеспечена**: связь «число в отчёте → скрипт, который его посчитал» отсутствует.

---

**Q10. `min_hits`/`min_strings`: 5/0 или 8/2?**

**[ответ]** Не помню; в каталогах `preds` всё залогировано.

**[проверено] Ответ подтверждается — параметры восстановимы из трёх независимых
источников:**

| Источник | Охват |
|---|---|
| `preds/*/run_info.json` | **33 из 35** каталогов; поля `min_hits`, `min_strings`, `threshold` есть и в старом плоском формате, и в новом (`runs.{source}`) |
| `preds/*/*.log` | **135** файлов со строкой вида `min_hits=5  min_strings=0` |
| `preds/prediction_history.csv` | **125** строк, **35** уникальных чекпоинтов, отдельные колонки `min_hits`, `min_strings` |

**[проверено] Фактически использованная рабочая точка — 5/0, а не argparse-дефолт 8/2.**
Подсчёт по логам: `5 / 0` — 51 вхождение, `8 / 2` — 2, `8 / 3` — 2. Из четырёх записей
`run_info.json` нового формата все четыре — `min_hits=5, min_strings=0`.

Отбор h8s3 применяется, таким образом, ниже по течению как срез, а не на этапе
предсказания (кроме двух прогонов с `8 / 3`). Противоречие в §5.2 остаётся
противоречием **источников конфигурации**, но не создаёт неоднозначности в данных.

---

**Q11. `BATCH_SIZE=1024` в `run_batch_predict.sh` и `--sn-batch-size`?**

Критерий владельца: **[ответ]** «если скрипт загружает сигнальность хитов из probs-h5,
batch size не важен; если скорит на лету — надо указывать batch size как для модели
`k_nsol_labelneq0_da_hs128_k0p0001`» (то есть 256).

**[проверено] По этому критерию текущий код уже корректен на обоих маршрутах:**

* `run_batch_predict.sh` **не передаёт** `--probs-h5` ни в одном из четырёх шагов
  (grep: ноль вхождений; полный список аргументов шага `mc_merged` — строки 163–178) ⇒
  маршрут «на лету».
* Но `sn_batch_size` — **самостоятельный параметр, а не производная от `--batch-size`**:
  `predict_mc.py:303-305` вызывает `predict_flat(..., batch_size=sn_batch_size)`, тогда
  как `--batch-size` используется только для nu-классификатора (строки 329, 335).
  Дефолт — `SN_BATCH_SIZE = 256` (строка 62), и `run_batch_predict.sh` его не
  переопределяет. То есть sig-noise считается при 256, а 1024 относится только к
  nu-классификатору.
* На маршруте с probs-файлом фактическое значение не угадывается, а извлекается:
  `predict_mc.py:462` пишет в `run_info` `probs_batch_size(probs_h5) if probs_h5 else
  sn_batch_size`.

**Остаточная область — только историческая**: предсказания, сделанные до этой правки,
считали sig-noise при 1024. Именно это, со счётом «6 из 69 MC-логов и 24 из 40 exp-логов
читали precomputed probs», описано в `preds/EXP_PREDS_STALE.md` (§7.7). Правка в самом
`run_batch_predict.sh` по критерию владельца **не требуется**.

---

**Q12. `_OLD`, `_OLDNewer`, `.DEPRECATED_bs512`?**

**[ответ]** Чем отличаются — не знаю.
**[решение владельца]** Старые версии уже не нужны.

Чем `baikal_mc_reco_OLDNewer.h5` (5.8 ГБ) отличается от `_OLD.h5` (4.4 ГБ) и от текущего
`baikal_mc_reco.h5` (6.6 ГБ), **установить не удалось** — ни по коду, ни по датам
(все три mtime 2026-04-16…04-28).

---

**Q13. Репозиторный `memory/`?**

**[ответ]** Очень устаревшая память; модели, названные там лучшими, скорее всего удалены.
Если агент ею не пользовался — она бесполезна.

**[проверено]** Не пользовался: в этой сессии применялась авто-память Claude
(`~/.claude/projects/…/memory/`, ~35 записей до сентября 2026), репозиторный `memory/`
прочитан только как объект аудита. Обе модели из `memory/project_overview.md`
отсутствуют и в `experiments/numu/`, и в `experiments/Archive/` (§7.5).

---

**Q14. Ветка hcut?**

**[ответ]** hcut была одним из вариантов prefilter; явного выигрыша не дала.
**[решение владельца]** Закрывать.

Затрагивает `src/data/hcut_numu_dataset.py` (1145 строк),
`src/training/archive/da_hcut_numu_trainer.py` (1439), `experiments/da_hcut_numu_baseline.yaml`,
`scripts/train_da_hcut_in_bg.sh` (уже неработоспособна, §1.6) и ноутбуки
`da_numu_report_hcut_{2026,regime}.ipynb`, `notebooks/test_signal_numu_ds.ipynb`.

---

**Q15. `inference_v2/reader/`?**

**[ответ]** Новый инструмент на замену всем старым; на него планируется перевести весь
анализ статьи.

Согласуется с §2.1: самый свежий код репозитория (mtime 2026-08-31…09-03), 23 ассерта в
собственном тесте, подробный `tutorial.ipynb`, при этом ни один `.py`-скрипт его пока не
использует — только ноутбуки. Вместе с ответом на Q3 это означает, что `reader` —
целевой путь чтения и для переделываемого prefilter-анализа.

---

### 8.1 Что осталось неустановленным после ответов

| Вопрос | Что именно не установлено |
|---|---|
| Q8 | какой скрипт породил фигуры в `papers/NNPipelineForBaikalGVD/` |
| Q12 | чем `baikal_mc_reco_OLDNewer.h5` отличается от `_OLD.h5` и от текущего |
| Q9 | связь «число в `reports/2026-07-08`, `2026-07-09` → породивший его код» |
| §1.7 | на что указывает `$SN` в `run_reco_chain_FIXED_sn256.sh` |
| §4.2 | кто пишет и кто читает `exp_reco_full_2020.h5` (113 ГБ) |
| §5.1 | существует ли `/home2/ivkhar/Baikal/data/normed/…` (каталог вне доступа) |

### 8.2 Зафиксированные решения владельца

Перечислено как факт о намерениях на 2026-09-06, не как вывод аудита.
В архив или из проекта предполагается вынести: `inference/` — после переделки
prefilter-анализа на `inference_v2` + `reader` (Q3); Parquet-каталоги `h5_catalogs/`
(Q4); `data_manager/root2df/` (Q5); всё, относящееся к текущему fine-tuning (Q6, Q7);
скрипты анализа `model_comparison/paper_*`, `replot_*`, `*_full` (Q8); старые версии
HDF5 `_OLD`, `_OLDNewer`, `.DEPRECATED_bs512` (Q12); репозиторный `memory/` (Q13);
ветка hcut (Q14). Целевой инструмент чтения — `inference_v2/reader/` (Q15).

Правок по итогам ответов **не требуют**: `run_batch_predict.sh` в части
`--sn-batch-size` (Q11) и probs-файл `k0p0001` (Q1) — оба уже корректны.

---

*Составлено только на чтении: изменений в рабочем дереве не сделано, git-команд,
меняющих состояние, не выполнялось. `git status --porcelain` после составления — пусто.
Сам этот файл git не показывает как новый: `.gitignore:172 *.md` его **игнорирует**
(проверено `git check-ignore -v doc/AUDIT.md`), исключения сделаны только для
`doc/hdf5_format.md` и README канонической sig-noise модели.*

*Коммит `5fb2603` «pre-cleanup commit» (2026-09-06 09:38, 182 файла) сделан не в рамках
этого аудита; его влияние на git-даты и на состав отслеживаемых файлов учтено в §0.*
