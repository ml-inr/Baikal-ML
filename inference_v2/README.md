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
│   ├── compute_scalars.py  # Per-event scalars derived from stored predictions
│   ├── run_predict_*.sh    # Wrappers; edit the variables at the top
│   ├── run_batch_predict.sh# npy / mc_merged / mc_reco / exp across several checkpoints
│   ├── analysis/           # Only HIGH_SCORE_EXCESS/ remains; see "Analysis" below
│   ├── reports/<date>/     # Frozen research reports with their generating scripts
│   └── preds/
│       ├── prediction_history.csv        # append-only run log
│       └── {run}@{checkpoint_stem}/      # e.g. 260816_2250_..._FIXED_sn256@best_da_model
│           ├── mc_merged_thr0p8.duckdb
│           ├── exp_reco_thr0p8.duckdb
│           └── run_info.json
│
├── prefilter/              # Background prefilter (all raw hits, no sig-noise step)
│   ├── predict_mc.py
│   ├── predict_exp.py
│   ├── run_predict_{mc,exp}.sh
│   ├── analysis/           # load.py, metrics.py, plots.py
│   └── preds/{run}@{checkpoint_stem}/{mc_merged,exp_reco}_allhits.duckdb
│
├── reader/                 # Streaming access to sources and predictions — start here
│   ├── tutorial.ipynb
│   └── tests/test_reader.py
│
└── shared/                 # Utilities shared across tasks
    ├── model_utils.py      # load_model(), load_sn_model(), predict_scores()
    ├── catalog_query.py    # DuckDB catalog JOINs, append_predictions_*()
    ├── history.py          # prediction_history.csv writer
    ├── run_info.py         # run_info.json reader/writer
    ├── reco_schema.py      # reco_prty column names and converter configs
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
trained_models/classifier/260509_1015_..._seed27_lr1e4/best_da_model.pth
  → preds/260509_1015_..._seed27_lr1e4@best_da_model/

trained_models/classifier/260509_1015_..._seed27_lr1e4/da_checkpoint_epoch_045.pth
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
import pandas as pd
df = pd.read_csv("inference_v2/nu_classifier/preds/prediction_history.csv")
```

The writer is `inference_v2/shared/history.py:append_history_row`; there is no reader
helper — it is a plain CSV on purpose.

---

## Analysis

Use **`inference_v2/reader/`**. It streams events and hits from any source joined with any
checkpoint's predictions, without materialising anything:

```python
from inference_v2 import reader

reader.checkpoints()                       # which scored checkpoints exist
r = reader.open("exp_reco", checkpoint="260816_2250_..._FIXED_sn256@best_da_model")

ev = r.events(where=[reader.H8S3])         # event table; no hit is read
for chunk in r.stream(where=[reader.H8S3], budget_mb=512):
    chunk.events, chunk.hits, chunk.parts, chunk.meta
```

Predicates compose: `H8S3`, `NOT_TRAINED`, `NOT_DA_TARGET`, `NOT_EXCLUDED`, or raw
`Sql(...)`. `reader.sources()` prints every known source with the evidence behind its
exclusions. Worked examples: `reader/tutorial.ipynb`.

The run log is read with:

```python
from inference_v2.shared.history import ...      # writer lives here
```

> **A note on where analysis code goes.** An earlier version of this file advised dropping
> research scripts straight into `analysis/`. That produced 113 files in 17 subdirectories,
> most of them unreachable from any entry point — see `doc/AUDIT.md` §1.9 and §2. They have
> been archived. Prefer `reader/` plus a dated directory under `reports/` that contains both
> the script and its output, so a number in a report can always be traced to the code that
> produced it.

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
