#!/usr/bin/env bash
# ── Nu-classifier Exp prediction ──────────────────────────────────────────────
# Edit the variables below, then run:
#   bash inference/nu_classifier_model/run_predict_exp.sh
# from the project root (or anywhere — the script resolves the root itself).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/../.."   # project root

# ── Required ──────────────────────────────────────────────────────────────────
CHECKPOINT="experiments/numu/260509_1015_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed27_lr1e4/best_da_model.pth"

# ── Optional (defaults shown) ─────────────────────────────────────────────────
EXP_H5="data_manager/data/h5datasets/exp.h5"
THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0
BATCH_SIZE=512
DEVICE="cuda:2"
OUTPUT_DIR="inference/nu_classifier_model/results"

# ─────────────────────────────────────────────────────────────────────────────
python inference/nu_classifier_model/predict_exp.py \
    --checkpoint   "$CHECKPOINT"   \
    --exp-h5       "$EXP_H5"       \
    --threshold    "$THRESHOLD"    \
    --min-hits     "$MIN_HITS"     \
    --min-strings  "$MIN_STRINGS"  \
    --batch-size   "$BATCH_SIZE"   \
    --device       "$DEVICE"       \
    --output-dir   "$OUTPUT_DIR"
