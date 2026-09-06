#!/usr/bin/env bash
# Build exp-background NPY dataset for fine-tuning.
# Selects exp events with score < SCORE_THR from the specified checkpoint's
# prediction DB, runs the sig-noise model, writes exp_bg_*.npy files.
#
# Usage: bash inference_v2/nu_classifier/exp_finetuning/run_build_exp_bg.sh
# Run from project root.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT="260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model"

PREDS_DIR="inference_v2/nu_classifier/preds/${CHECKPOINT}"
EXP_H5="data_manager/data/h5datasets/exp.h5"
CATALOG="data_manager/catalog_v2.duckdb"

SCORE_THR="0.1"    # select events with score < this
SN_THR="0.8"       # must match DB filename suffix
MIN_HITS="5"
MIN_STRINGS="0"
BATCH_SIZE="512"
DEVICE="cuda:2"

# Output: inference_v2/nu_classifier/exp_finetuning/exp_bg_datasets/{CHECKPOINT}_lt{SCORE_THR}/
# (default, set --output-dir to override)

# ── Launch ────────────────────────────────────────────────────────────────────

python inference_v2/nu_classifier/exp_finetuning/build_exp_bg.py \
    --preds-dir       "$PREDS_DIR"    \
    --exp-h5          "$EXP_H5"       \
    --catalog         "$CATALOG"      \
    --score-threshold "$SCORE_THR"    \
    --sn-threshold    "$SN_THR"       \
    --min-hits        "$MIN_HITS"     \
    --min-strings     "$MIN_STRINGS"  \
    --batch-size      "$BATCH_SIZE"   \
    --device          "$DEVICE"       \
    "$@"
