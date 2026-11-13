# track_energy_reconstruction

Prediction storage for muon-energy checkpoints, organized like
`inference/nu_classifier/preds/`.

Each checkpoint gets one directory:

```
preds/<training-run>@<checkpoint-stem>/
    mc_merged_h8s3_thr0p8.duckdb   # table `predictions`
    run_info.json                  # checkpoint, source, cuts, provenance, counts
    predict_full_<timestamp>.log
preds/prediction_history.csv       # one row per prediction attempt (same columns as the classifier)
```

The checkpoint stem is `best_model` for the best-validation checkpoint, so the
canonical directory is e.g. `261023_2037_..._cont_s42@best_model`.

## What is predicted

The training dataset
(`data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M`) contains only the
flat-selected **train+val** population. The test complement is not stored in it.

For the part-split dataset
(`mu_energy_dataset_h8s3_thr0.8_train1M_partsplit`) the training data contains
only the train parts. Its final test is a separate set of **new MC parts**, not
the flat-selection complement inside the same parts; do not use the old
`predict_full.py --subset test` complement as the final part-level test.

`predict_full.py` therefore streams the **full h8s3 MC source** from the main HDF5
and the stored sig-noise probabilities, replays the builder's event cuts and the
`prob > threshold` hit selection, and writes the model prediction for every source
event. No catalog lookup of hits is needed; the catalog is used once to attach
`event_fk` to every selected event.

Every source event is marked with the provenance of the checkpoint's training
dataset:

| `split` | meaning |
|---|---|
| `train` | present in the training dataset and in its train split; `used_for_labels=true` |
| `val` | present in the training dataset and in its val split |
| `test` | not in the training dataset: the held-out complement (and, for older datasets, the old flat complement) |

`--subset all` writes all events; `--subset test` writes only the held-out ones.
The DB is keyed by `event_fk` with `INSERT OR IGNORE`, so either subset can be run
first and the other appended later without duplicating events.

The model output is z-normalized. The checkpoint stores the transform
(`target_normalization.mean/std`), the feature normalization and `max_hits`;
predictions are converted back to `log10(E/GeV)` and `E` in GeV before writing.

## Table schema

```
predictions(
    event_fk BIGINT PRIMARY KEY,
    pred_log10e FLOAT,
    pred_energy_gev DOUBLE,
    pred_sigma_dex FLOAT,     -- heteroscedastic sigma, dex; NULL for deterministic models
    true_log10e FLOAT,
    true_energy_gev DOUBLE,
    energy_status INTEGER,
    particle_type INTEGER,
    n_sig_hits INTEGER,
    n_sig_strings INTEGER,
    theta FLOAT,              -- MC primary zenith angle (0° vertical)
    split VARCHAR,            -- train / val / test
    flat_weight FLOAT,        -- flat-selection weight; NaN for test
    used_for_labels BOOLEAN,  -- true for train
    h5_part_key VARCHAR,
    h5_local_event_id INTEGER
)
```

`train` predictions are in-sample; `val` and `test` are held out. Always filter by
`split` in analysis; do not average over the whole table. For test events, `theta`
and the truth columns come from the same full-source pass, so the DB is
self-contained without joining the training NPY arrays.

Older prediction DBs predate the `theta` and `pred_sigma_dex` columns;
`open_energy_db` adds them in place. Their existing rows keep both `NULL`.
`pred_sigma_dex` is filled only for checkpoints with
`model.heteroscedastic.enabled: true`; deterministic checkpoints write `NULL`.
The sigma is the conditional aleatoric scale of `log10(E/GeV)`, not an OOD score.

## Usage

```bash
# Predict the test complement with the best continuation checkpoint
GPU=3 SUBSET=test bash inference/track_energy_reconstruction/run_predict_best_continuation.sh

# Or all events (train + val + test)
GPU=3 SUBSET=all bash inference/track_energy_reconstruction/run_predict_best_continuation.sh

# Equivalent direct call, with an optional smoke limit or part filter
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -m inference.track_energy_reconstruction.predict_full \
    --checkpoint trained_models/track_energy_reconstruction/<run>/best_model.pth \
    --npy-dir data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M \
    --device cuda:0 --batch-size 512 --subset test \
    [--limit 2000] [--parts part_1006,part_4000] [--check-existing]
```

`--npy-dir` must be the training dataset that produced the checkpoint: it supplies
the split labels, flat weights and `event_fk` values. A checkpoint trained before
the dataset refactor has no YAML `train_split`; the predictor then reads its legacy
`split_*_indices.npy` files. For such a checkpoint use
`data_manager/datasets/mu_energy_dataset_h8s3_thr0.8`, not the new dataset.

The script is incremental (`INSERT OR IGNORE` keyed by `event_fk`), so a smoke run,
a `--subset test` run and a later `--subset all` run can share one DB; the history
records each attempt and `run_info.json` records the latest one.

## Verification

For the best continuation checkpoint, the held-out validation MAE computed from
the DB should reproduce the training-side `val_mae_log10` (0.49028 at epoch 143),
because inference uses the same feature normalization, target inverse transform and
`max_hits` as training:

```python
import duckdb
con = duckdb.connect("inference/track_energy_reconstruction/preds/<run>@best_model/mc_merged_h8s3_thr0p8.duckdb", read_only=True)
print(con.execute("""
    SELECT split, COUNT(*), AVG(ABS(pred_log10e - true_log10e)) AS mae
    FROM predictions GROUP BY split ORDER BY split
""").df())
```

`val` should give ~0.490 dex. `train` will be lower (in-sample). For a checkpoint
trained on the `train1M` dataset, `test` is the complement of the flat selection and
has no sparse-tail events, so its MAE is not directly comparable to the weighted
training metric. The legacy checkpoint in this directory was trained on the old
part subset; its full-source `test` also contains tail events from the additional
2020 parts.

The streaming implementation was checked event-for-event against the old
dataset-based predictor on `part_1006`: 6,832/6,832 `event_fk` matched, splits and
`n_sig_hits` agreed, and predictions differed by at most 4.8e-6 in `log10E`
(float32 rounding).

## Non-neural baseline

The handcrafted-feature baseline is deliberately separate from the neural
predictor. It uses one row per event, charge clipped at 100 p.e. (the same input
clip as the neural model), and no hit sequence:

```
data_manager/mu_energy_ds_builder/build_non_neural_features.py
src/data/mu_energy_dataset/non_neural_features.py
src/training/non_neural_energy_baseline.py
inference/track_energy_reconstruction/predict_non_neural_energy.py
```

For the part-split dataset
(`mu_energy_dataset_h8s3_thr0.8_train1M_partsplit`), the feature sidecar lives
next to the hit-level arrays:

```
non_neural_features.npy
non_neural_features.yaml
```

Train and evaluate the three baseline models with:

```bash
python -m data_manager.mu_energy_ds_builder.build_non_neural_features \
    --dataset-dir data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M_partsplit

python -m src.training.non_neural_energy_baseline \
    --config src/training/configs/track_energy_reconstruction/non_neural_energy_baseline.yaml
```

Apply a saved model to new MC parts (never used for tuning) with:

```bash
python -m inference.track_energy_reconstruction.predict_non_neural_energy \
    --run-dir trained_models/track_energy_reconstruction/<run> \
    --model hgb --npy-dir <new_parts_dir> --split-name test_new
```

The analysis notebook is
`analysis/non_neural_energy_baseline.ipynb`. Its last block accepts arbitrary
neural-network prediction paths keyed by `event_fk`; no checkpoint is hardcoded.

## Analysis

The current test-part analysis is
`analysis/predictions_analysis.ipynb`.  It reads the DuckDBs listed in its
`RUN_SPECS` table and writes `analysis/reports/prediction_manifest.csv`,
`summary_metrics.csv`, `class_metrics.csv`, `calibration_metrics.csv` and figures
under `analysis/reports/figures/`.  It evaluates only `split = 'test'`.

The non-training domain probe is `analysis/domain_probe.py`.  It refits a
standardised linear probe and a larger MLP (three seeds) on the held-out DA parts,
instead of relying on the under-trained per-epoch MLP in `training_history.csv`.
It reports two splits: event-level inside the held-out parts (comparable with the
training history) and part-disjoint, where whole part keys go to either fit or
test.  The part-disjoint AUC is the primary check because events from one part
share detector conditions.

```bash
python -m inference.track_energy_reconstruction.analysis.domain_probe \
    --device cuda:0
# writes analysis/reports/da_probe.csv
```

`da_enabled` in that CSV means **the encoder** was trained with DA, including a
frozen-head run such as `AH-*` that inherits a DA encoder from `A`.  The probe
measures encoder features, so an `AH-*` row is not a no-DA control.

Add newly finished checkpoints to `RUN_SPECS` in `analysis/domain_probe.py` and to
`RUN_SPECS` in the notebook before re-running them.

The current probe (16 checkpoints; event split / part-disjoint MLP AUC):

| contrast | no-DA (event / part) | DA (event / part) | Δ MLP event | Δ MLP part |
|---|---:|---:|---:|---:|
| `A` vs `BASE` | 0.669 / 0.665 | 0.606 / 0.599 | −0.064 | −0.066 |
| `AD-G` vs `D-G` | 0.732 / 0.703 | 0.642 / 0.643 | −0.090 | −0.060 |
| `AD-T5` vs `D-T5` | 0.700 / 0.687 | 0.654 / 0.626 | −0.046 | −0.062 |
| `AC-G` vs `C-G` | 0.668 / 0.652 | 0.633 / 0.632 | −0.035 | −0.021 |
| `AC-T5` vs `C-T5` | 0.684 / 0.672 | 0.632 / 0.624 | −0.052 | −0.048 |

`AB-G` / `AB-T5` rows remain pending until those runs finish.

The linear probe shows the same direction.  These are held-out measurements, not
a certificate of invariance: the target is not necessarily AUC = 0.5, and a
remaining AUC can reflect legitimate feature differences between MC and data.

### Per-event sigma versus a binned residual scale

`analysis/sigma_vs_binned.py` asks whether the learned per-event sigma carries
information beyond a scale that depends only on predicted energy.  It fits the
baseline per-bin residual RMS and one global scale for the learned sigma on one
half of the test parts, then scores both with Gaussian NLL on the other half:

```bash
python -m inference.track_energy_reconstruction.analysis.sigma_vs_binned \
    --ids D-G,AC-G,B-G,H-G
# writes analysis/reports/sigma_vs_binned.csv
```

On the current Gaussian checkpoints, per-event sigma wins the proper score, not
just coverage:

| model | NLL, per-event | NLL, E_pred-bin | Δ [95% part-bootstrap CI] | within-bin Spearman(sigma, abs err) |
|---|---:|---:|---:|---:|
| `D-G` | 0.811 | 0.902 | −0.091 [−0.103, −0.082] | 0.31 |
| `AC-G` | 0.792 | 0.913 | −0.121 [−0.136, −0.111] | 0.35 |
| `B-G` | 0.794 | 0.923 | −0.129 [−0.143, −0.119] | 0.34 |
| `H-G` | 0.846 | 0.916 | −0.070 [−0.079, −0.064] | 0.29 |

Both are calibrated on the fit half; held-out 68/95% coverage is ~0.683/0.951
for both.  The gain is therefore in sharpness/discrimination inside an energy
bin, not in global coverage.  This is an information test after a common
recalibration, not a check of the training-time raw calibration factors.  The
comparison is Gaussian-only (Student-t scale is not a Gaussian bin RMS) and uses
one split-half realization; the CI resamples whole test parts.

A stronger conditional baseline may use only **observables**: hit count,
signal-string count, charge summaries, time-spread summaries and similar
quantities computed identically on MC and data.  The `theta` column in the
prediction DB is the MC primary zenith angle — truth, not an observable — and
must never be a training feature or an operational selection variable.  A
reconstructed direction is admissible only if the same reconstruction is
applied to both MC and data and is validated as such; it is not the stored
`theta`.

Per-checkpoint detail notebooks live in `analysis/by_checkpoint/`. They are thin
wrappers over `analysis/reporting.py` and contain the 2D residual-vs-energy and
prediction-vs-truth histograms, true- and predicted-energy bins, per-bin
MAE/bias, per-class tables and the converged DA probe.  A separate
`## Sigma analysis` block adds coverage versus true and predicted energy,
global and energy-binned PIT, sigma-versus-predicted-energy, and the
signed/absolute error ranking (2D histograms plus global and within-bin
correlations).  Its tables are `sigma_ranking.csv`, `pit_by_true_energy.csv`
and `pit_by_predicted_energy.csv`; figures are under
`analysis/reports/by_checkpoint/<run>@best_model/figures/`.

