#!/usr/bin/env bash
# MoE score analysis: inner-join + average predictions from multiple checkpoints,
# then plot score distributions by class and suppression-vs-efficiency curves
# swept over 4 quality cuts (h5s0, h5s2, h8s3, h10s3).
#
# Usage: bash inference_v2/nu_classifier/analysis/run_moe_analysis.sh
# Run from project root.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

PREDS_DIR="inference_v2/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"
NPY_DIR="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8" # To exclude train set

# Checkpoints to ensemble (names of subdirs under PREDS_DIR, thr0.8 group).
# All must have {SOURCE}_thr{THRESHOLD_TAG}.duckdb present.
CHECKPOINTS=(
    # "260503_1101_da_nu_classifier_h5s0_lambda0.0@best_da_model" # signoise 0.5
    # "260503_1101_da_nu_classifier_h5s0_lambda0.001@best_da_model"  # signoise 0.5
    # "260503_1103_da_nu_classifier_h5s0_lambda0.001_drout0.3@best_da_model"  # signoise 0.5
    "260507_2111_da_nu_classifier_h5s0_lambda0.001_thr0.8@best_da_model"  # signoise 0.8
    #"260507_2119_da_nu_classifier_h5s0_lambda0.0_thr0.8@best_da_model"  # signoise 0.8
    "260507_2121_da_nu_classifier_h5s0_lambda0.01_thr0.8@best_da_model"  # signoise 0.8
    "260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed27@best_da_model"  # signoise 0.8
    "260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model"  # signoise 0.8
    "260509_1015_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed27_lr1e4@best_da_model"  # signoise 0.8
)

SOURCE="mc_merged"
THRESHOLD="0.8"

# Classes used for suppression-vs-efficiency curve.
# Events from other classes (exp) are shown in score dist but excluded from metrics.
SIGNAL_CLASSES=("nuatm_2020" "nue2_2020")
BG_CLASSES=("muatm_2020")

# Extra sources to overlay on the score distribution plot (no metrics computed).
# Set to empty array to disable: OVERLAY_SOURCES=()
# Options: exp  exp_reco
OVERLAY_SOURCES=("exp")

# ── Launch ────────────────────────────────────────────────────────────────────

python inference_v2/nu_classifier/analysis/run_moe_analysis.py \
    --preds-dir       "$PREDS_DIR"              \
    --checkpoints     "${CHECKPOINTS[@]}"        \
    --source          "$SOURCE"                  \
    --threshold       "$THRESHOLD"               \
    --signal-classes  "${SIGNAL_CLASSES[@]}"     \
    --bg-classes      "${BG_CLASSES[@]}"         \
    --overlay-sources "${OVERLAY_SOURCES[@]}"    \
    --catalog         "$CATALOG"                 \
    --npy-dir         "$NPY_DIR"                 \
    "$@"
