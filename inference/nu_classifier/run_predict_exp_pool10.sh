#!/usr/bin/env bash
# Full-statistics nu-classifier scoring of exp_full with the discriminator-pool model
# (261002_1101_..._pool10, best checkpoint = epoch 31).
#
# Identical to run_predict_exp_FIXED_sn256.sh in everything that affects the hit selection:
# same stored sig-noise probabilities, same threshold and event cuts, same encoder geometry.
# That is deliberate - it keeps this output directly comparable with the FIXED line
# (see doc/sig_noise_batch_size.md).
#
# The only difference from the single-discriminator runs is the domain branch (a pool of 10
# discriminators), which inference does not use at all: the score comes from the encoder and
# the classifier head, and they live in `base_model_state_dict`, so the checkpoint loads into
# the standard architecture unchanged.
#
# Resumable via --skip-done-parts. Run under nohup.
set -euo pipefail
cd /home/albert/Baikal2025

export CUDA_DEVICE_ORDER=PCI_BUS_ID     # so that cuda:5 below = GPU 5 in nvidia-smi

CHECKPOINT="trained_models/classifier/261002_1101_da_nu_classifier_exp_full_E1_lambda0.01_pool10/best_da_model.pth"
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
    --skip-done-parts \
    --save-embeddings
