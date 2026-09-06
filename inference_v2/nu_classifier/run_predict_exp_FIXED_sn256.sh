#!/usr/bin/env bash
# Full-statistics nu-classifier scoring of exp_full with the FIXED model.
#
# The experimental counterpart of run_predict_mc_FIXED_sn256.sh, and it must stay in step
# with it: both read stored sig-noise probabilities written at batch 256, so MC and data
# receive the same hit selection. That is the whole point — see doc/sig_noise_batch_size.md.
#
# Resumable via --skip-done-parts. Run under nohup.

set -euo pipefail
cd /home/albert/Baikal2025

CHECKPOINT="experiments/numu/260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256/best_da_model.pth"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
PROBS_H5="data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"

THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0

BATCH_SIZE=1024
DEVICE="cuda:0"
OUTPUT_DIR="inference_v2/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"

python -u inference_v2/nu_classifier/predict_exp.py \
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
    --skip-done-parts \
    --save-embeddings
