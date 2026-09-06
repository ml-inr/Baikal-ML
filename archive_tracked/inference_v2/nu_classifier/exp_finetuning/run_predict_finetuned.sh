#!/usr/bin/env bash
# Run mc_merged + exp predictions for a finetuned nu-classifier checkpoint.
# Predictions land directly inside the finetuned model dir so that
# run_single_analysis.sh can point CHECKPOINT_DIR there without any path gymnastics.
#
# Usage: bash inference_v2/nu_classifier/exp_finetuning/run_predict_finetuned.sh
# Run from project root.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_NAME="260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model_finetuned"

# .pth file to load — use best_finetuned_model.pth or latest_finetuned_checkpoint.pth
CHECKPOINT_PTH="inference_v2/nu_classifier/exp_finetuning/finetuned_models/${CHECKPOINT_NAME}/latest_finetuned_checkpoint.pth"

# Output dir: predictions go into OUTPUT_DIR/CHECKPOINT_NAME/
OUTPUT_DIR="inference_v2/nu_classifier/exp_finetuning/finetuned_models"

MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
EXP_H5="data_manager/data/h5datasets/exp.h5"
CATALOG="data_manager/catalog_v2.duckdb"

THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0
BATCH_SIZE=1024
MIN_BATCH_EVENTS=32768
DEVICE="cuda:1"

MAX_EVENTS_PER_PTYPE=1000000
SKIP_DONE_PARTS=false

# ── MC merged ─────────────────────────────────────────────────────────────────

echo "=== MC merged predictions ==="

MC_EXTRA=()
[ -n "${MAX_EVENTS_PER_PTYPE:-}" ] && MC_EXTRA+=(--max-events-per-ptype "$MAX_EVENTS_PER_PTYPE")
[ "${SKIP_DONE_PARTS:-false}" = "true" ] && MC_EXTRA+=(--skip-done-parts)

python inference_v2/nu_classifier/predict_mc.py \
    --checkpoint       "$CHECKPOINT_PTH"     \
    --checkpoint-name  "$CHECKPOINT_NAME"    \
    --mc-h5            "$MC_H5"              \
    --source           "mc_merged"           \
    --threshold        "$THRESHOLD"          \
    --min-hits         "$MIN_HITS"           \
    --min-strings      "$MIN_STRINGS"        \
    --batch-size       "$BATCH_SIZE"         \
    --min-batch-events "$MIN_BATCH_EVENTS"   \
    --device           "$DEVICE"             \
    --output-dir       "$OUTPUT_DIR"         \
    --catalog          "$CATALOG"            \
    "${MC_EXTRA[@]}"

# ── Exp ───────────────────────────────────────────────────────────────────────

echo "=== Exp predictions ==="

EXP_EXTRA=()
[ "${SKIP_DONE_PARTS:-false}" = "true" ] && EXP_EXTRA+=(--skip-done-parts)

python inference_v2/nu_classifier/predict_exp.py \
    --checkpoint       "$CHECKPOINT_PTH"     \
    --checkpoint-name  "$CHECKPOINT_NAME"    \
    --exp-h5           "$EXP_H5"             \
    --threshold        "$THRESHOLD"          \
    --min-hits         "$MIN_HITS"           \
    --min-strings      "$MIN_STRINGS"        \
    --batch-size       "$BATCH_SIZE"         \
    --device           "$DEVICE"             \
    --output-dir       "$OUTPUT_DIR"         \
    --catalog          "$CATALOG"            \
    "${EXP_EXTRA[@]}"

echo "=== Done. Predictions in ${OUTPUT_DIR}/${CHECKPOINT_NAME}/ ==="
