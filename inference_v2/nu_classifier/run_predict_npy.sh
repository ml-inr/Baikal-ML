#!/usr/bin/env bash
# Run nu-classifier predictions from pre-built NPY dataset.
# Edit the variables below before running.

set -euo pipefail

CHECKPOINT="experiments/numu/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32/best_da_model.pth"
NPY_DIR="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
MIN_HITS=5
MIN_STRINGS=0
BATCH_SIZE=1024
DEVICE="cuda:2"
OUTPUT_DIR="inference_v2/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"
SAVE_EMBEDDINGS=true  # uncomment to store encoder embeddings alongside scores

EXTRA_ARGS=()
[ "${SAVE_EMBEDDINGS:-false}" = "true" ] && EXTRA_ARGS+=(--save-embeddings)

python inference_v2/nu_classifier/predict_npy.py \
    --checkpoint   "$CHECKPOINT" \
    --npy-dir      "$NPY_DIR" \
    --min-hits     "$MIN_HITS" \
    --min-strings  "$MIN_STRINGS" \
    --batch-size   "$BATCH_SIZE" \
    --device       "$DEVICE" \
    --output-dir   "$OUTPUT_DIR" \
    --catalog      "$CATALOG" \
    "${EXTRA_ARGS[@]}"
