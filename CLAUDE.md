# CLAUDE.md

Guidance for Claude Code working in this repository. Describes the state as of
**2026-09-06**, after the cleanup recorded in `doc/AUDIT.md`.

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
  data/                    nu_classifier_dataset/, prefilter_npy_dataset/ — memmap NPY
  training/                da_nu_classifier_trainer.py, da_prefilter_numu_trainer.py
    configs/{prefilter,classifier,track_energy_reconstruction}/   шаблоны запуска

data_manager/              всё, что превращает ROOT в обучаемые массивы
  root2h5/                 ROOT → HDF5 (MC, exp, exp_full, reco); + energy_truth/
  nu_classifier_ds_builder/  HDF5 → NPY для классификатора
  prefilter_npy_ds_builder/  HDF5 → NPY для префильтра
  catalog_v2/              сборка catalog_v2.duckdb
  energy_truth/            сборка и валидация энергетического компаньона
  data/, datasets/         сами файлы (вне git)

inference_v2/              предсказания и чтение
  nu_classifier/           predict_{npy,mc,exp}.py + обёртки run_*.sh
  prefilter/               predict_{mc,exp}.py + обёртки
  shared/                  model_utils, catalog_query, reco_schema, run_info, history
  reader/                  единый потоковый доступ к источникам и предсказаниям

trained_models/            результаты обучения (вне git, кроме снимков конфигов)
  prefilter/<run>/         best_da_model.pth, da_config.yaml, tensorboard/, logs/
  classifier/<run>/        то же
  track_energy_reconstruction/   пустая, задел под будущую задачу

gplotnikov_sig_noise_models/   внешняя модель сигнал/шум (не наша)
doc/                       справочники по данным и журнал утверждений
scripts/                   обёртки запуска обучения
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

## The two pipelines

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

Inference: `inference_v2/prefilter/predict_{mc,exp}.py` — **all raw hits, no sig-noise
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

Inference: `inference_v2/nu_classifier/predict_{npy,mc,exp}.py` →
`preds/<run>@<ckpt>/{mc_merged,exp_reco,exp_full}_thr0p8.duckdb`.

**Sig-noise batch size is part of the hit selection, not a speed knob.** The model takes a
float padding mask, so batch size changes which hits pass the threshold. `--sn-batch-size`
is a separate parameter, default **256**, applied independently of `--batch-size` (which
only affects the classifier). See `doc/sig_noise_batch_size.md`.

External dependency: `gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/sig_noise_model_v3.py`.

### Shared

`data_manager/catalog_v2.duckdb` (132 GB) maps every event to an `event_fk`, the key all
prediction tables reference. Built by `data_manager/catalog_v2/build_{mc,mc_reco,exp}.py`.

`inference_v2/reader/` is the newest and intended way to read sources and predictions
without materialising anything; see `inference_v2/reader/tutorial.ipynb`.

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
| `datasets/` | 59 GB | memory-mapped NPY training sets |

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
- `doc/sig_noise_batch_size.md` — why batch size defines the hit selection
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

Detailed pipeline commands live in the `pipelines` skill (`.claude/skills/pipelines/`),
loaded on demand. The essentials:

```bash
# обучение (правьте переменные в шапке обёртки)
bash scripts/train_da_prefilter_soft_in_bg.sh
bash scripts/train_da_nuclassifier_in_bg.sh

# предсказания
bash inference_v2/nu_classifier/run_predict_mc.sh
bash inference_v2/prefilter/run_predict_exp.sh

# самопроверяющиеся скрипты
python inference_v2/reader/tests/test_reader.py       # 23 assert
python inference_v2/test_run_info.py                  # 14 assert
python data_manager/energy_truth/test_truth.py        # аналитика + регрессия
python data_manager/root2h5/energy_truth/test_targets_vec.py

# сравнить два прогона предсказаний (эталон поведения)
python inference_v2/test_scoring_equivalence.py --baseline <dir> --candidate <dir>

# tensorboard
bash scripts/tb.sh classifier/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32
```

There is no automated test suite: no `pytest`, no CI, no single command that checks
everything. The scripts above are what exists.

## State as of 2026-09-06

Both pipelines are trained and operational; predictions for the canonical checkpoints are
in `inference_v2/*/preds/`. `inference_v2/reader/` is new and intended to replace the older
per-analysis loading code, which has been archived.

Known open points, all recorded in `doc/AUDIT.md`:

- the experimental excess above MC at high classifier score is not closed;
  `doc/claims_log.md` carries the current status of each claim about it
- `src/training/domain_sep_trainer.py` remains live while all four of its configs and all
  its runs are archived
- SNGP classes (`RandomFeatureGPHead`, `NuMuSNGPModel`) remain in
  `src/models/base_models.py` though the SNGP line is archived; they are inert unless
  `model_type: numu_sngp` is used
- `src/data/test_numu_dataset.py` imports `data_manager.h5datasets.paths2h5`, which does
  not resolve — the file is at `data_manager/data/h5datasets/paths2h5.py`
