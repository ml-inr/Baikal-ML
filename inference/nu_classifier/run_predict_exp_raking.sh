#!/usr/bin/env bash
# Full-statistics nu-classifier scoring of exp_full with the RAKING champion
# (261021_2009_..._raking, best checkpoint = epoch 25 by val_loss).
#
# Same stored sig-noise probabilities and the same cuts as the weights-line wrappers
# (run_predict_exp_weights.sh), so this output stays directly comparable with them.
# The training target events are marked via --mark-da-target-npy.
#
# Resumable via --skip-done-parts. Run under nohup.
set -euo pipefail
cd /home/albert/Baikal2025

export CUDA_DEVICE_ORDER=PCI_BUS_ID     # so that cuda:5 below = GPU 5 in nvidia-smi

CHECKPOINT="trained_models/classifier/261021_2009_da_nu_classifier_exp_full_E1_weights_z128_k10_dlr1e-2_lam0p003_delay2_s42_raking/best_da_model.pth"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
PROBS_H5="data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"

THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0

BATCH_SIZE=1024
DEVICE="cuda:5"
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
