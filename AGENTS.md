# AGENTS.md

Guidance for AI coding agents working in this repository. Describes the state as of
**2026-11-01**, after the cleanup recorded in `doc/AUDIT.md`.

This is the single instruction file. `CLAUDE.md` only points here; coding standards are in
`conventions.md`. Read this file in full before starting work, whichever file name your
harness looks for. References below to skills, permissions or hook configuration belong to
particular harnesses (Claude Code, DeepSeek Harness); the rules they carry apply anyway.

## What this project is

Neutrino detection for **Baikal-GVD**, an underwater Cherenkov telescope. The task is
binary classification of detector events: neutrino signal against a vastly larger
atmospheric-muon background. An event is a variable-length sequence of hits, each hit a
5-vector `[Q, t, x, y, z]` — charge, time, and cluster-centred coordinates. Typical
event: 35–200+ hits.

Selection runs in **two stages**, trained separately:

| Stage | Input | Purpose |
|---|---|---|
| **prefilter** | all raw hits | cuts EAS background ×10–20 |
| **nu-classifier** | hits passing the sig-noise model (thr 0.8) | high-purity neutrino selection |

Both are transformer encoders trained with **domain adaptation** (DANN: gradient reversal
+ domain discriminator), because the model must transfer from Monte Carlo to real data.

## Environment

```bash
conda activate baikal25
python -c "import torch, numpy, h5py, duckdb; print('OK')"
```

Python 3.10+, PyTorch, h5py, uproot/awkward (ROOT reading), DuckDB (event catalogue and
prediction storage), NumPy, PyYAML. Environment file: `environment.yml`.

Training logs go to Python `logging` and TensorBoard simultaneously:
`trained_models/<task>/<run>/tensorboard/`, viewed with `bash scripts/tb.sh <task>/<run>`.

## Live tree

```
src/                       модели, датасеты, тренеры, конфиги обучения
  models/                  base_models.py (трансформер), domain_discriminator.py (GRL)
  data/                    nu_classifier_dataset/, prefilter_npy_dataset/,
                           mu_energy_dataset/ — memmap NPY
  training/                da_nu_classifier_trainer.py (+ _da_repair, _ensemble),
                           da_prefilter_numu_trainer.py, mu_energy_trainer.py,
                           non_neural_energy_baseline.py
    configs/{prefilter,classifier,track_energy_reconstruction}/   шаблоны запуска

data_manager/              всё, что превращает ROOT в обучаемые массивы
  root2h5/                 ROOT → HDF5 (MC, exp, exp_full, reco); mc_binary/ (read_wout.py)
  nu_classifier_ds_builder/  HDF5 → NPY для классификатора
  prefilter_npy_ds_builder/  HDF5 → NPY для префильтра
  mu_energy_ds_builder/    HDF5 → NPY для регрессии энергии мюона (h8s3)
  probe_sample/            фиксированная выборка для DA invariance-пробы
  catalog_v2/              сборка catalog_v2.duckdb
  energy_truth/            сборка и валидация энергетического компаньона
  data/, datasets/         сами файлы (вне git)

inference/                 предсказания и чтение
  nu_classifier/           predict_{npy,mc,exp}.py + обёртки run_*.sh
    analysis/HIGH_SCORE_EXCESS/   анализ избытка высокого скора
  prefilter/               predict_{mc,exp}.py + обёртки
  sig_noise/               predict_{mc,exp}_h5.py + обёртки; h5-вероятности рядом с источником
  angle_track_uncertainty/ predict.py -> DuckDB; upstream CSV; подготовка selected-HDF5
  track_energy_reconstruction/  predict_full.py, predict_non_neural_energy.py,
                           preds/, analysis/ (domain_probe, sigma_vs_binned)
  shared/                  model_utils, catalog_query, reco_schema, run_info, history
  reader/                  единый потоковый доступ к источникам и предсказаниям
  track_fit_alike_bars/    порт фита BARS v2.3.2 + валидация против самого BARS

trained_models/            результаты обучения (вне git, кроме снимков конфигов)
  prefilter/<run>/         best_da_model.pth, da_config.yaml, tensorboard/, logs/
  classifier/<run>/        то же
  track_energy_reconstruction/<run>/   энергетические чекпоинты (mean и sigma-головы)
  sig_noise/<run>/         внешняя модель сигнал/шум (не наша): чекпоинт + код
  angle_track_uncertainty/<run>/  внешняя angle/track/uncertainty модель: чекпоинт + код
doc/                       справочники по данным и журнал утверждений
  energy_training_schemes/ сборщики фигур и pptx про энергетическую задачу
  learning/                учебные материалы: DANN shortcut, toy_da_transfer
scripts/                   обёртки запуска обучения и очередей
```

## Two archives — the difference matters

| | `archive/` | `archive_tracked/` |
|---|---|---|
| Under git | no | **yes** |
| What | старые черновики и эксперименты | код, снятый с эксплуатации по решениям `doc/AUDIT.md` §8.2 |

Both are **default-deny** in `.gitignore`: only `.py`, `.sh`, `.yaml`, `.yml`, `.md`,
`.ipynb` are tracked. A data file dropped into either stays out of git regardless of
extension — including `.dat`, `.wout`, `.tar`, `.bin`, which have no rule of their own.

Check what a path would do before relying on it, and judge by the **exit code**, not by
whether `-v` printed something (it prints negating rules too):

```bash
git check-ignore -q --no-index <path> && echo IGNORED || echo TRACKED
```

## The pipelines

### Prefilter

```
baikal_mc_merged.h5
  └─ data_manager/prefilter_npy_ds_builder/build.py
       labels from MC truth: raw/labels != 0 → n_signal_hits → soft label
       min(1, n_sig/h_saturate) for neutrinos, 0 for muons
  └─ data_manager/datasets/baikal_mc2020_prefilter/{hard_nu_labels,soft_nu_labels_hsat9}/
  └─ src/data/prefilter_npy_dataset/   source = NPY, target = exp.h5 (all labels 0)
  └─ src/training/da_prefilter_numu_trainer.py --config src/training/configs/prefilter/…
  └─ trained_models/prefilter/<run>/
```

Canonical run: `da_prefilter_numu_260429_0155_Qclip_softlabels_PlateauLR_lambda0.05_MC2M_dmodel128_zflipBothMCExp_SoftFocalLoss`.

Inference: `inference/prefilter/predict_{mc,exp}.py` — **all raw hits, no sig-noise
step** → `preds/<run>@<ckpt>/{mc_merged,exp_reco}_allhits.duckdb`.

Note the prefilter has no way to exclude its own training events at inference: its NPY
dataset carries no h5 back-links (the nu-classifier's does, via `h5_part_keys.npy`).

### Nu-classifier

```
baikal_mc_merged.h5  +  …_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5
  └─ data_manager/nu_classifier_ds_builder/   hits with sig-noise prob > 0.8
  └─ data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8/
  └─ src/data/nu_classifier_dataset/
  └─ src/training/da_nu_classifier_trainer.py --config src/training/configs/classifier/…
  └─ trained_models/classifier/<run>/
```

Canonical run: `260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32`.

Inference: `inference/nu_classifier/predict_{npy,mc,exp}.py` →
`preds/<run>@<ckpt>/{mc_merged,exp_reco,exp_full}_thr0p8.duckdb`.

**Sig-noise batch size is part of the hit selection, not a speed knob.** The model takes a
float padding mask, so batch size changes which hits pass the threshold. `--sn-batch-size`
is a separate parameter, default **256**, applied independently of `--batch-size` (which
only affects the classifier). See `doc/sig_noise_batch_size.md`.

External dependency: `trained_models/sig_noise/k_nsol_labelneq0_da_hs128_k0p0001/sig_noise_model_v3.py`.
The old `gplotnikov_sig_noise_models/...` import path still works through a
compatibility shim.

### Energy reconstruction

Third line, started in October 2026: regress the muon energy at entry into the cluster.

```
baikal_mc_merged.h5 + …_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5
  └─ data_manager/mu_energy_ds_builder/   h8s3 cut (n_sig_hits ≥ 8, n_sig_strings ≥ 3),
                                          single-track events, target e_at_entry[:, 1]
  └─ data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M[_partsplit]/
  └─ src/training/mu_energy_trainer.py --config src/training/configs/track_energy_reconstruction/…
  └─ trained_models/track_energy_reconstruction/<run>/
```

`muatm` is dropped (too few events left after h8s3 and the single-track cut); the dataset is
`nuatm` + `nue2` only, flat in `log10(E)` through `flat_selection`, and stores **train+val
only** — the test complement is not in the dataset. Experiment IDs `A/BASE/B/C/D/H` with
`-G`/`-T5` describe the training regime, not a run directory; see
`doc/track_energy_experiment_ids.md`.

Inference: `inference/track_energy_reconstruction/predict_full.py` streams the full h8s3
source and writes `preds/<run>@<ckpt>/mc_merged_h8s3_thr0p8.duckdb`. For the part-split
dataset the final test is a separate set of new MC parts, never the flat complement.
`pred_sigma_dex` is the aleatoric scale of `log10(E/GeV)`, not an OOD score.

### Angle / track reconstruction

External `release_v3` model (G. Plotnikov): per event, a direction, a point on an
infinite track line, and marginal angular / transverse-distance uncertainties, from
SN-selected hits. It is neither the energy regressor nor the BARS fit. Bundle and
model code: `trained_models/angle_track_uncertainty/release_v3/`; adapter:
`inference/angle_track_uncertainty/predict.py` (DuckDB keyed by `event_fk`). It
supports the canonical project SN probs (default, `sn_mode=canonical`) and the
release's frozen `signal_noise.ckpt` (`sn_mode=release`); the two modes write
separate DBs. Calibrated radii are filled only for unchanged release-SN p>=0.70 MC
rows on CPU; canonical, experimental and gate rows get NULL radii. The angle/track
backbone uses a boolean padding mask (batch-independent up to float rounding), but
the bundled release SN uses a float mask and is batch-composition dependent.
Validated only for single-track MC (`nuatm`/`nue2`); experimental and
multi-track-muon point/uncertainty outputs are not validated. See
`doc/angle_track_uncertainty.md`.

### Shared

`data_manager/catalog_v2.duckdb` (132 GB) maps every event to an `event_fk`, the key all
prediction tables reference. Built by `data_manager/catalog_v2/build_{mc,mc_reco,exp}.py`.

`inference/reader/` is the newest and intended way to read sources and predictions
without materialising anything; see `inference/reader/tutorial.ipynb`.

## Data

All under `data_manager/`, none in git.

| Path | Size | What |
|---|---|---|
| `data/h5datasets/baikal_mc_merged.h5` | 956 GB | MC: `muatm_2020`, `nuatm_2020`, `nue2_2020` |
| `data/h5datasets/…_probs_k_nsol…k0p0001.h5` | 189 GB | per-hit sig-noise probabilities for the above |
| `data/h5datasets/baikal_mc_merged_energy_truth.h5` | 163 GB | energy-truth companion |
| `data/h5datasets/exp_full.h5` | 136 GB | full-stat exp, has `header_prty` with physical event ids |
| `data/h5datasets/exp_reco_full_2020.h5` | 113 GB | exp with reconstruction |
| `data/h5datasets/exp_reco.h5` | 22 GB | exp with reconstruction, smaller sample |
| `data/h5datasets/exp.h5` | 0.66 GB | legacy exp, ~650K events — the DA target domain |
| `catalog_v2.duckdb` | 132 GB | event catalogue |
| `datasets/` | 64 GB | memory-mapped NPY sets: classifier, prefilter, `mu_energy_dataset_h8s3_thr0.8_train1M[_partsplit]` |

`nue2` is **astrophysical ν_μ with an E⁻² spectrum**, not electron neutrinos — the name
misleads and older comments in the repo get this wrong.

## Reference documentation

These describe **data**, not code layout, and stay valid across refactors:

- `doc/hdf5_format.md` — layout of every HDF5 file
- `doc/mc_provenance.md` — where the MC comes from, generator → `.dat` → `.wout` → ROOT → h5.
  Read before trusting any field's units; BARS comments disagree with the data.
- `doc/mc_binary_formats.md` — word layout of `.dat`/`.wout`, verified against ROOT
- `doc/mc_energy_truth.md`, `doc/hdf5_energy_truth.md`, `doc/energy_twin_plan.md` — what
  energy ground truth exists and how the companion file is built
- `doc/reco_quality_cuts.md` — BARS quality cuts, mixed units, known faults
- `doc/track_fit_alike_bars.md` — the ported BARS track fit: what it reproduces, where the
  DQM channel mask comes from, what disagrees and why
- `doc/sig_noise_batch_size.md` — why batch size defines the hit selection
- `doc/angle_track_uncertainty.md` — the external angle/track/uncertainty release:
  bundle, input selection, DuckDB schema and validation limits
- `doc/track_energy_experiment_ids.md` — energy experiment IDs `A/BASE/B/C/D/H` + `-G`/`-T5`
  and the checkpoint-selection rule per ID
- `doc/energy_training_schemes/README.md` — energy figures and decks, and the facts each one
  encodes; every number is traced to a dataset report or a prediction DB
- `doc/simgvd_review.md` — review of the simGVD → MCREAD → BARS chain with locations,
  statuses and proposed fixes
- `doc/learning/` — teaching notes: DANN encoder shortcut and `toy_da_transfer`, each with
  the scripts that build the figures and the PDF
- `doc/claims_log.md` — every factual claim with its evidence and status
  (`verified` / `consistent` / `assumed` / `refuted`)
- `doc/AUDIT.md` — map of the repository **before** the 2026-09 cleanup, plus the
  decisions that drove it. Historical: its paths no longer resolve.

## Verifying claims

Research claims here have repeatedly been documented as established and turned out wrong.
Every one of those failures shared a cause: the check tested **internal consistency** — one
field against another from the same file, or a pattern against an already-formed
hypothesis — instead of anchoring to something outside the data. More checking does not fix
this; different checking does.

Before writing a factual claim into `doc/`:

- **Anchor externally.** A claim about units or semantics is not established until checked
  against something outside the dataset: physics that holds independently, a production
  whose name states its range, an independent implementation, or the code that writes the
  format. Comparing two fields of the same file is `consistent`, never `verified`.
- **Read the specification before decoding data.** If a parser, writer or schema exists,
  read it first. Reverse-engineering by pattern-matching is a last resort, and it produced
  a day of wrong conclusions when `Task_MAIN.cpp` was two steps away.
- **Separate "consistent with" from "proves".** State which rival hypotheses a test can and
  cannot distinguish. `Reg ≤ E_primary` proves the two share a unit, not which unit.
- **Name the falsifying observation before collecting evidence.** "If this is true, X; if I
  see Y, it is dead." Skipping this turns a search into a search for confirmation.
- **Isolate before editing code.** Reproduce the failure in the smallest setting that shows
  it — one element vs a batch, one file vs a run — rather than guessing at a fix.
- **Mark confidence in the text.** Write `verified` / `consistent` / `assumed` where a
  reader would otherwise have to trust the tone, and log it in `doc/claims_log.md`.
- **Record refutations, don't overwrite them.** When a claim flips, say so in place and add
  the bad argument to the log — the failure modes repeat.

**Confidence is part of the answer.** Reporting a result without saying how well it is
established is an incomplete report, not a concise one.

## Working rules

Coding standards are in `conventions.md`: Python 3.10+, PEP 8 at 88 columns, type hints on
signatures, config from YAML, seeds set explicitly, device handled explicitly.

Two rules specific to how this repository has gone wrong before:

**Do not create variant-named files.** No `_v2`, `_new`, `_fixed`, `_final`, `_OLD`,
`copy`. Edit the existing file. If a genuinely separate version is needed, ask first —
`doc/AUDIT.md` §3 catalogues what happens otherwise: seven near-identical copies of one
sig-noise model, two byte-identical predict scripts wired to the wrong checkpoint, and
`_OLD`/`_OLDNewer` HDF5 pairs nobody can now tell apart.

**Every figure or number that reaches a paper or a report must have a generating script in
the repository and a provenance record.** `doc/AUDIT.md` Q8/Q9 records the cost of skipping
this: for the figures currently in `papers/` it is no longer possible to say which script
produced them, and the reports of 2026-07-08 and 2026-07-09 contain no code at all. A
provenance helper exists as a model:
`archive_tracked/inference_v2/nu_classifier/analysis/reco_excess/src/provenance.py`.

Never do without asking: implement a solution, make an architectural decision, add
complexity, or move to the next stage of work.

## Commands

Detailed pipeline commands live in `.claude/skills/pipelines/SKILL.md` — read that file
before running or changing a pipeline. Claude Code may also load it automatically as a
skill; other harnesses must open it explicitly. The essentials:

```bash
# обучение (правьте переменные в шапке обёртки)
bash scripts/train_da_prefilter_soft_in_bg.sh
bash scripts/train_da_nuclassifier_in_bg.sh

# предсказания
bash inference/nu_classifier/run_predict_mc.sh
bash inference/prefilter/run_predict_exp.sh
bash inference/angle_track_uncertainty/run_predict_mc.sh   # angle/track/uncertainty

# энергия: тестовые части лучшим continuation-чекпоинтом
GPU=3 SUBSET=test bash inference/track_energy_reconstruction/run_predict_best_continuation.sh

# энергия: собрать презентацию из скриптов (не править pptx руками)
python doc/energy_training_schemes/build_full_deck.py

# фит а-ля BARS: самопроверки, поэтапная сверка и валидация против BARS
python inference/track_fit_alike_bars/tests/test_fit.py            # 9 проверок
python inference/track_fit_alike_bars/tests/stage_diagnostics.py   # по стадиям, на ROOT
python -m inference.track_fit_alike_bars.validate_vs_bars --source exp_reco --n-parts 200

# самопроверяющиеся скрипты
python inference/reader/tests/test_reader.py       # 16 проверок (1 пропускается)
python inference/test_run_info.py                  # 14 assert
python data_manager/energy_truth/test_truth.py        # аналитика + регрессия

# сравнить два прогона предсказаний (эталон поведения)
python inference/test_scoring_equivalence.py --baseline <dir> --candidate <dir>

# tensorboard
bash scripts/tb.sh classifier/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32
```

There is no automated test suite: no `pytest`, no CI, no single command that checks
everything. The scripts above are what exists.

## State as of 2026-11-01

Both original pipelines are trained and operational; predictions for the canonical
checkpoints are in `inference/*/preds/`. `inference/reader/` is the intended way to read
sources and predictions; the older per-analysis loading code is archived.

The **muon-energy line** is the main work in progress. The dataset
`mu_energy_dataset_h8s3_thr0.8_train1M` and its `_partsplit` variant are built; deterministic
and heteroscedastic checkpoints (`BASE`, `A`, `B`, `C`, `D`, `H`, with `-G` / `-T5`) are
trained; the evaluated ones are scored into `inference/track_energy_reconstruction/preds/`.
The current validation path is the part-split test on new MC parts, not the flat-selection
complement. The line is presented in
`doc/energy_training_schemes/energy_training_schemes.pptx`; a non-neural HGB baseline from
clipped charge and hit-count features is the bar the network must clear. The learned per-event sigma is aleatoric: it beats a per-bin residual scale on
Gaussian NLL, not on coverage.

DA repair: `src/training/da_nu_classifier_trainer_da_repair.py` and
`data_manager/probe_sample/` measure every N epochs how much domain information a fresh
logistic reader can still extract from the encoder. The "discriminator goes blind while the
domains stay separable" question is open; the probe AUCs live in the trainer logs, not yet
in a report.

`inference/track_fit_alike_bars/` is a Python port of the BARS v2.3.2 muon reconstruction,
written to cut on reconstruction quality before the next stage of the analysis. It
reproduces BARS' hit selection event for event once given the DQM channel mask BARS used
(`masks.py`, which reads it from the reco ROOT files or from the run's DQM solution file),
and it stops short of BARS only at the minimiser: `n_calls`, `covMatrixStatus` and the last
digits of the fit are not comparable between ROOT's Minuit2 and iminuit. The package is
self-checked and its agreement with BARS is measured in `validation/`; see
`doc/track_fit_alike_bars.md`.

Known open points, all recorded in `doc/AUDIT.md`:

- the experimental excess above MC at high classifier score is not closed;
  `doc/claims_log.md` carries the current status of each claim about it
- the working tree is far ahead of the last commit (2026-09-06): ~300 uncommitted files,
  including the whole energy line. Run `git status` before assuming a path is tracked
- `src/training/domain_sep_trainer.py` remains live while all four of its configs and all
  its runs are archived
- SNGP classes (`RandomFeatureGPHead`, `NuMuSNGPModel`) remain in
  `src/models/base_models.py` though the SNGP line is archived; they are inert unless
  `model_type: numu_sngp` is used
- `src/data/test_numu_dataset.py` imports `data_manager.h5datasets.paths2h5`, which does
  not resolve — the file is at `data_manager/data/h5datasets/paths2h5.py`
