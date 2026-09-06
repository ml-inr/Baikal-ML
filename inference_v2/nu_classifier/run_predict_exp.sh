#!/usr/bin/env bash
# Run nu-classifier predictions for experimental data (on-the-fly sig-noise filtering).

set -euo pipefail

CHECKPOINT="experiments/numu/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32/best_da_model.pth"
#EXP_H5="data_manager/data/h5datasets/exp_reco.h5"
#EXP_H5="data_manager/data/h5datasets/exp.h5"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0
BATCH_SIZE=1024
DEVICE="cuda:4"
OUTPUT_DIR="inference_v2/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"

# Optional — comment out to disable
SKIP_DONE_PARTS=true
# SAVE_EMBEDDINGS=true  # uncomment to store encoder embeddings alongside scores

EXTRA_ARGS=()
[ "${SKIP_DONE_PARTS:-false}"  = "true" ] && EXTRA_ARGS+=(--skip-done-parts)
[ "${SAVE_EMBEDDINGS:-false}"  = "true" ] && EXTRA_ARGS+=(--save-embeddings)

python inference_v2/nu_classifier/predict_exp.py \
    --checkpoint   "$CHECKPOINT" \
    --exp-h5       "$EXP_H5" \
    --threshold    "$THRESHOLD" \
    --min-hits     "$MIN_HITS" \
    --min-strings  "$MIN_STRINGS" \
    --batch-size   "$BATCH_SIZE" \
    --device       "$DEVICE" \
    --output-dir   "$OUTPUT_DIR" \
    --catalog      "$CATALOG" \
    "${EXTRA_ARGS[@]}"
