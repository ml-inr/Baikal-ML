#!/usr/bin/env bash
# Extract exp candidate events where ALL listed models predict score > SCORE_THR.
# Loads raw hits from exp.h5, runs sig-noise model for hit masks, writes one JSON per event.
#
# Usage: bash inference_v2/nu_classifier/analysis/run_extract_candidates.sh
# Run from project root.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

PREDS_DIR="inference_v2/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"
EXP_H5="data_manager/data/h5datasets/exp.h5"

CHECKPOINTS=(
    "260507_2111_da_nu_classifier_h5s0_lambda0.001_thr0.8@best_da_model"
    "260507_2119_da_nu_classifier_h5s0_lambda0.0_thr0.8@best_da_model"
    "260507_2121_da_nu_classifier_h5s0_lambda0.01_thr0.8@best_da_model"
    "260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed27@best_da_model"
    "260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model"
    "260509_1015_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed27_lr1e4@best_da_model"
)

SCORE_THR="0.95"   # all models must exceed this score
SN_THR="0.8"       # sig-noise threshold for hit mask and DB filename lookup
DEVICE="cuda:2"       # sig-noise model device (cpu is fine for a handful of events)

# ── Launch ────────────────────────────────────────────────────────────────────

python inference_v2/nu_classifier/analysis/extract_moe_candidates.py \
    --preds-dir       "$PREDS_DIR"              \
    --checkpoints     "${CHECKPOINTS[@]}"        \
    --exp-h5          "$EXP_H5"                 \
    --score-threshold "$SCORE_THR"              \
    --sn-threshold    "$SN_THR"                 \
    --device          "$DEVICE"                 \
    --catalog         "$CATALOG"                \
    "$@"
