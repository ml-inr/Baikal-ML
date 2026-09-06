#!/usr/bin/env bash
# ── Nu-classifier MC prediction ───────────────────────────────────────────────
# Edit the variables below, then run:
#   bash inference/nu_classifier_model/run_predict_mc.sh
# from the project root (or anywhere — the script resolves the root itself).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/../.."   # project root

# ── Required ──────────────────────────────────────────────────────────────────
CHECKPOINT="experiments/numu/260509_1015_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed27_lr1e4/best_da_model.pth"

# ── Optional (defaults shown) ─────────────────────────────────────────────────
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
PROBS_H5="data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
PARTS_JSON="data_manager/nu_classifier_ds_builder/testds_parts.json"
THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0
BATCH_SIZE=512
DEVICE="cuda:1"
OUTPUT_DIR="inference/nu_classifier_model/results"

# ─────────────────────────────────────────────────────────────────────────────
python inference/nu_classifier_model/predict_mc.py \
    --checkpoint   "$CHECKPOINT"   \
    --mc-h5        "$MC_H5"        \
    --probs-h5     "$PROBS_H5"     \
    --parts-json   "$PARTS_JSON"   \
    --threshold    "$THRESHOLD"    \
    --min-hits     "$MIN_HITS"     \
    --min-strings  "$MIN_STRINGS"  \
    --batch-size   "$BATCH_SIZE"   \
    --device       "$DEVICE"       \
    --output-dir   "$OUTPUT_DIR"
