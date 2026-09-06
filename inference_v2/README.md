# inference_v2

Structured prediction pipeline for Baikal-GVD neutrino detection models.

---

## Directory layout

```
inference_v2/
├── nu_classifier/          # High-purity neutrino selection (sig-noise-filtered hits)
│   ├── predict_npy.py      # Fast: reads pre-built NPY dataset
│   ├── predict_mc.py       # Arbitrary MC events from h5 (on-the-fly sig-noise)
│   ├── predict_exp.py      # Exp data from h5 (on-the-fly sig-noise)
│   ├── run_predict_npy.sh
│   ├── run_predict_mc.sh
│   ├── run_predict_exp.sh
│   ├── analysis/           # Importable API for notebooks and research scripts
│   │   ├── load.py         # load_preds(), compare_checkpoints(), load_history()
│   │   ├── metrics.py      # auc(), at_threshold(), efficiency_rejection_curve()
│   │   └── plots.py        # plot_score_dist(), plot_efficiency_rejection(), plot_roc()
│   └── preds/
│       ├── prediction_history.csv        # append-only run log
│       └── {experiment_dir}@{checkpoint_stem}/   # e.g. 260509_1015_...seed27_lr1e4@best_da_model
│           ├── mc_merged_thr0p8.duckdb   # predictions table
│           ├── exp_reco_thr0p8.duckdb
│           └── run_info.json
│
├── prefilter/              # Background prefilter (all raw hits, no sig-noise step)
│   ├── predict_mc.py
│   ├── predict_exp.py
│   ├── run_predict_mc.sh
│   ├── run_predict_exp.sh
│   ├── analysis/
│   └── preds/
│       └── {experiment_dir}@{checkpoint_stem}/
│           ├── mc_merged_allhits.duckdb
│           ├── exp_reco_allhits.duckdb
│           └── run_info.json
│
└── shared/                 # Utilities shared across tasks
    ├── model_utils.py      # load_model(), load_sn_model(), predict_scores()
    ├── catalog_query.py    # DuckDB catalog JOINs, append_predictions_*()
    ├── history.py          # prediction_history.csv writer
    └── metrics.py          # auc_score(), at_threshold(), efficiency_rejection_curve()
```

---

## Predictions storage

Each checkpoint gets its own subdirectory under `preds/`. Predictions are stored in
per-source DuckDB files:

| Task | Source | File |
|---|---|---|
| nu_classifier | mc_merged or mc_reco | `mc_merged_thr{thr}.duckdb` |
| nu_classifier | exp or exp_reco | `exp_reco_thr{thr}.duckdb` |
| prefilter | mc_merged or mc_reco | `mc_merged_allhits.duckdb` |
| prefilter | exp or exp_reco | `exp_reco_allhits.duckdb` |

**Schema (nu_classifier):**
```sql
predictions(event_fk BIGINT PRIMARY KEY, score FLOAT, n_sn_hits INTEGER, n_sn_strings INTEGER)
```
- `event_fk` — references `catalog_v2.events.id`
- `n_sn_hits` / `n_sn_strings` — hits/strings passing the sig-noise model (not MC ground truth)

**Schema (prefilter):**
```sql
predictions(event_fk BIGINT PRIMARY KEY, score FLOAT, n_hits INTEGER)
```

All scripts use `INSERT OR IGNORE` — safe to re-run on overlapping subsets.
Add `--check-existing` to warn if a re-run produces different scores for already-stored events.

---

## Running predictions

Edit the variables at the top of the shell script, then run from the project root.
The checkpoint path is specified as `EXPERIMENT_DIR/CHECKPOINT_FILE.pth`:

```bash
# Nu-classifier — fastest, uses pre-built NPY dataset (training set only)
bash inference_v2/nu_classifier/run_predict_npy.sh

# Nu-classifier — arbitrary MC events from raw h5 (e.g. test set not in NPY)
bash inference_v2/nu_classifier/run_predict_mc.sh

# Nu-classifier — experimental data
bash inference_v2/nu_classifier/run_predict_exp.sh

# Prefilter — MC
bash inference_v2/prefilter/run_predict_mc.sh

# Prefilter — experimental data
bash inference_v2/prefilter/run_predict_exp.sh
```

The preds output directory is derived automatically from the checkpoint path:
```
experiments/numu/260509_1015_..._seed27_lr1e4/best_da_model.pth
  → preds/260509_1015_..._seed27_lr1e4@best_da_model/

experiments/numu/260509_1015_..._seed27_lr1e4/da_checkpoint_epoch_045.pth
  → preds/260509_1015_..._seed27_lr1e4@da_checkpoint_epoch_045/
```

To process only specific parts, pass `--parts part_0,part_1,part_2` to any predict script.

---

## Run history

Every script invocation appends a row to `preds/prediction_history.csv`:

| column | description |
|---|---|
| `timestamp` | ISO datetime of run start |
| `checkpoint` | checkpoint dir name |
| `script` | which predict script was run |
| `source` | data source (mc_merged, exp_reco, …) |
| `db_file` | output duckdb path |
| `threshold` | sig-noise threshold (NaN for prefilter) |
| `min_hits` / `min_strings` | quality cuts applied |
| `n_events_new` | rows inserted in this run |
| `n_events_skipped` | rows already present (INSERT OR IGNORE) |
| `is_successful` | False if script raised an exception |
| `error_msg` | exception message on failure |

Failed runs still get a row, so the history is a full audit trail.

```python
from inference_v2.nu_classifier.analysis import load_history
df = load_history("inference_v2/nu_classifier/preds")
```

---

## Analysis in notebooks

```python
from inference_v2.nu_classifier.analysis import load_preds, compare_checkpoints
from inference_v2.nu_classifier.analysis.plots import plot_efficiency_rejection, plot_roc

PREDS = "inference_v2/nu_classifier/preds"

# Load predictions for one checkpoint, joined with catalog metadata
df = load_preds(
    f"{PREDS}/260509_1015_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed27_lr1e4@best_da_model",
    source="mc_merged",
    thr=0.8,
    catalog_path="data_manager/catalog_v2.duckdb",
    cuts={"min_sn_hits": 8, "min_sn_strings": 2},
)
# df columns: event_fk, score, n_sn_hits, n_sn_strings, data_class, season, cluster, run, event_id

# Compare best model vs a specific epoch checkpoint
df_all = compare_checkpoints(
    [
        f"{PREDS}/260509_1015_..._seed27_lr1e4@best_da_model",
        f"{PREDS}/260509_1015_..._seed27_lr1e4@da_checkpoint_epoch_045",
    ],
    source="mc_merged", thr=0.8,
)
# df_all has an extra 'checkpoint' column

# Physics properties (theta, energy) are NOT in the catalog.
# Retrieve them from baikal_mc_merged.h5 via h5_locations if needed.
```

Drop new research scripts directly into `analysis/`:
```
nu_classifier/analysis/compare_seasons.py
nu_classifier/analysis/energy_threshold_study.py
```

---

## Key notes

- **`predict_npy.py` and `predict_mc.py` share the same output DB** (`mc_merged_thr{thr}.duckdb`)
  because they score the same physical mc_merged events via different read paths.
  Run `predict_npy.py` for the training set (fast), `predict_mc.py` for out-of-training events.

- **mc_reco h5 group names** (`muatm`, `nuatm_conv`, `nuatm_prompt`) differ from catalog
  `data_class` values (`muatm_2020`, `nuatm_conv_2020`, `nuatm_prompt_2020`).
  The mapping is in `shared/catalog_query.py:MC_RECO_PTYPE_TO_DATA_CLASS`.

- **exp_reco catalog lookup** uses `header_prty` columns `[season, cluster, run, event_id_in_run]`.
  Plain `exp.h5` (no `header_prty`) uses `(part_key, local_index)` as the key — ensure
  the catalog was built for that source first (`bash data_manager/catalog_v2/run_build_exp.sh`).
