#!/usr/bin/env bash
# Full-statistics nu-classifier scoring of exp_full with the RAKING lambda=0 control
# (261021_2009_..._z128_lam0_delay2_s42_raking, best checkpoint = epoch 18 by val_loss).
#
# The matched no-DA control of run_predict_exp_raking.sh: same raking weights, same
# data, same cuts; the only difference is that the domain branch never acts (lambda=0).
# Same stored sig-noise probabilities as the champion wrapper, so the two are directly
# comparable. Training target events are marked via --mark-da-target-npy.
#
# Resumable via --skip-done-parts. Run under nohup.
set -euo pipefail
cd /home/albert/Baikal2025

export CUDA_DEVICE_ORDER=PCI_BUS_ID     # so that cuda:5 below = GPU 5 in nvidia-smi

CHECKPOINT="trained_models/classifier/261021_2009_da_nu_classifier_exp_full_E1_weights_z128_lam0_delay2_s42_raking/best_da_model.pth"
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
