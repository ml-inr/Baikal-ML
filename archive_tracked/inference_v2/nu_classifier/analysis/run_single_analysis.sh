#!/usr/bin/env bash
# Single-checkpoint score analysis: score distributions by class and
# suppression-vs-efficiency curves swept over 4 quality cuts (h5s0, h5s2, h8s3, h10s3).
#
# Usage: bash inference_v2/nu_classifier/analysis/run_single_analysis.sh
# Run from project root.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

CATALOG="data_manager/catalog_v2.duckdb"
NPY_DIR="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8" # To exclude train set

# Option A — standard checkpoint from PREDS_DIR (comment out CHECKPOINT_DIR below)
#PREDS_DIR="inference_v2/nu_classifier/preds"
#CHECKPOINT="260507_2111_da_nu_classifier_h5s0_lambda0.001_thr0.8@best_da_model"
#CHECKPOINT="260507_2119_da_nu_classifier_h5s0_lambda0.0_thr0.8@best_da_model"
#CHECKPOINT="260507_2121_da_nu_classifier_h5s0_lambda0.01_thr0.8@best_da_model"
#CHECKPOINT="260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed27@best_da_model"
#CHECKPOINT="260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model"
#CHECKPOINT="260531_1720_da_nu_classifier_h5s0_lambda0.01_thr0.8_FixedDA@best_da_model"

# Option B — full path to any preds dir (finetuned models, custom paths, etc.)
# When set, overrides PREDS_DIR + CHECKPOINT above.
CHECKPOINT_DIR="inference_v2/nu_classifier/exp_finetuning/finetuned_models_LARGETAUG/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model_finetuned"

SOURCE="mc_merged"
THRESHOLD="0.8"

# Classes used for suppression-vs-efficiency curve.
SIGNAL_CLASSES=("nuatm_2020" "nue2_2020")
BG_CLASSES=("muatm_2020")

# Extra sources to overlay on the score distribution plot (no metrics computed).
# Set to empty array to disable: OVERLAY_SOURCES=()
# Options: exp  exp_reco
OVERLAY_SOURCES=("exp")

# ── Launch ────────────────────────────────────────────────────────────────────

if [ -n "${CHECKPOINT_DIR:-}" ]; then
    CKPT_ARGS=(--checkpoint-dir "$CHECKPOINT_DIR")
else
    CKPT_ARGS=(--preds-dir "$PREDS_DIR" --checkpoint "$CHECKPOINT")
fi

python inference_v2/nu_classifier/analysis/run_single_analysis.py \
    "${CKPT_ARGS[@]}"                          \
    --source          "$SOURCE"                 \
    --threshold       "$THRESHOLD"              \
    --signal-classes  "${SIGNAL_CLASSES[@]}"    \
    --bg-classes      "${BG_CLASSES[@]}"        \
    --overlay-sources "${OVERLAY_SOURCES[@]}"   \
    --catalog         "$CATALOG"                \
    --npy-dir         "$NPY_DIR"               \
    "$@"
