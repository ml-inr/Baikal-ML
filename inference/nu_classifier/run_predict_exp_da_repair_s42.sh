#!/usr/bin/env bash
# Full-statistics nu-classifier scoring of exp_full with the UNWEIGHTED DA-repair model
# (261002_0535_..._DArepair, seed 42, best_da_model.pth).
#
# Mirrors run_predict_exp_weights.sh in everything that affects the hit selection
# (same stored sig-noise probabilities, threshold, cuts, encoder geometry) so the two
# outputs are directly comparable: weighted vs unweighted DA-repair sibling, same seed 42.
#
# Resumable via --skip-done-parts. Run under nohup.
set -euo pipefail
cd /home/albert/Baikal2025

export CUDA_DEVICE_ORDER=PCI_BUS_ID

CHECKPOINT="trained_models/classifier/261002_0535_da_nu_classifier_exp_full_E1_lambda0.01_DArepair/best_da_model.pth"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
PROBS_H5="data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"

THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0

BATCH_SIZE=1024
DEVICE="cuda:1"
OUTPUT_DIR="inference/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"

python -u inference/nu_classifier/predict_exp.py \
    --checkpoint   "$CHECKPOINT" \
    --exp-h5       "$EXP_H5" \
    --probs-h5     "$PROBS_H5" \
    --probs-group  "exp_full" \
    --threshold    "$THRESHOLD" \
    --min-hits     "$MIN_HITS" \
    --min-strings  "$MIN_STRINGS" \
    --batch-size   "$BATCH_SIZE" \
    --device       "$DEVICE" \
    --output-dir   "$OUTPUT_DIR" \
    --catalog      "$CATALOG" \
    --mark-da-target-npy data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8 \
    --skip-done-parts \
    --save-embeddings
