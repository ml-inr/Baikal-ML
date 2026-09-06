---
name: pipelines
description: How to run this project's data and inference pipelines — ROOT→HDF5 conversion, NPY dataset builders, DuckDB catalog builds, nu-classifier and prefilter predictions, UMAP analysis. Use when running or modifying any of these steps.
---

# Data and inference pipelines

Command reference moved out of CLAUDE.md so it loads only when a pipeline is actually being
run. Paths are relative to the project root.

### Data Processing
```bash
# Process ROOT files to HDF5 (if needed)
cd data_manager && python root2h5/root2h5.py          # MC data
cd data_manager && python root2h5/root2h5_exp.py      # Experimental data (legacy, 25k-capped)
cd data_manager && python root2h5/root2h5_exp_reco.py # Exp reco data
# Full-statistics exp (chunked, physical IDs via BJointHeader) → exp_full.h5
bash data_manager/root2h5/run_root2h5_exp_full.sh     # or: python root2h5/root2h5_exp_full.py --config root2h5/root2h5_config_exp_full.yaml

# Build NPY datasets for nu-classifier training (HDF5 → memory-mapped NPY)
python data_manager/nu_classifier_ds_builder/build.py  # adjust config inside

# Build DuckDB catalog_v2 (data_manager/catalog_v2.duckdb)
bash data_manager/catalog_v2/run_build_mc.sh              # MC merged (~100 min, 651M rows)
bash data_manager/catalog_v2/run_build_mc_reco.sh         # MC reco (~minutes, incremental)
bash data_manager/catalog_v2/run_build_exp.sh exp        # Exp (legacy 25k-capped)
bash data_manager/catalog_v2/run_build_exp.sh exp_full   # Full-stat exp (~133M, physical event_id)
bash data_manager/catalog_v2/run_build_exp.sh exp_reco   # Exp reco

# Rebuild NPY dataset with h5 back-links (adds h5_part_keys.npy, h5_local_event_ids.npy)
nohup python -u -m data_manager.nu_classifier_ds_builder \
    --config data_manager/nu_classifier_ds_builder/default_config.yaml \
    > data_manager/nu_classifier_ds_builder/build_th0.8.log 2>&1 &

# Performance test: catalog lookup + h5 retrieval
python -m data_manager.test_npy_retrieval --n-events 100000

# Check data file existence
ls -la data_manager/data/h5datasets/baikal_mc_merged.h5
```

### Running Inference (inference_v2/)
```bash
# Batch predictions across the checkpoints listed in MODELS=() inside the script
# (4 steps: npy / mc_merged / mc_reco / exp)
bash inference_v2/nu_classifier/run_batch_predict.sh [npy|mc_merged|mc_reco|exp|all]

# Single-model shortcut (edit script to change checkpoint/device)
bash inference_v2/nu_classifier/run_predict_mc.sh    # MC predictions
bash inference_v2/nu_classifier/run_predict_exp.sh   # Exp predictions
bash inference_v2/prefilter/run_predict_mc.sh        # Prefilter MC
bash inference_v2/prefilter/run_predict_exp.sh       # Prefilter exp
```

### Reading sources and predictions

`inference_v2/reader/` is the intended way to read any source joined with any checkpoint's
predictions, streaming rather than materialising. Worked examples in
`inference_v2/reader/tutorial.ipynb`.

```python
from inference_v2 import reader
reader.checkpoints()                       # which scored checkpoints exist
r = reader.open("exp_reco", checkpoint="260816_2250_..._FIXED_sn256@best_da_model")
ev = r.events(where=[reader.H8S3])
```

The UMAP tooling that used to live at `analysis/run_umap.py` was archived in the 2026-09
cleanup; see `archive_tracked/inference_v2/nu_classifier/analysis/`.

