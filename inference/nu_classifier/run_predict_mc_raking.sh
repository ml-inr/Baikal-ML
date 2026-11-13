#!/usr/bin/env bash
# Full-statistics nu-classifier scoring of mc_merged with the RAKING champion
# (261021_2009_..._raking, best checkpoint = epoch 25 by val_loss).
#
# The MC counterpart of run_predict_exp_raking.sh and it must stay in step with it:
# both read the stored sig-noise probabilities, so MC and data get the same hit
# selection. See doc/sig_noise_batch_size.md for why that coupling matters.
# Training events are excluded via the NPY dataset back-links.
#
# Resumable via --skip-done-parts. Run under nohup.
set -euo pipefail
cd /home/albert/Baikal2025

export CUDA_DEVICE_ORDER=PCI_BUS_ID     # so that cuda:5 below = GPU 5 in nvidia-smi

CHECKPOINT="trained_models/classifier/261021_2009_da_nu_classifier_exp_full_E1_weights_z128_k10_dlr1e-2_lam0p003_delay2_s42_raking/best_da_model.pth"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
PROBS_H5="data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
SOURCE="mc_merged"

THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0

BATCH_SIZE=1024
MIN_BATCH_EVENTS=32768
DEVICE="cuda:5"
OUTPUT_DIR="inference/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"

python -u inference/nu_classifier/predict_mc.py \
    --checkpoint       "$CHECKPOINT" \
    --mc-h5            "$MC_H5" \
    --probs-h5         "$PROBS_H5" \
    --source           "$SOURCE" \
    --threshold        "$THRESHOLD" \
    --min-hits         "$MIN_HITS" \
    --min-strings      "$MIN_STRINGS" \
    --batch-size       "$BATCH_SIZE" \
    --min-batch-events "$MIN_BATCH_EVENTS" \
    --device           "$DEVICE" \
    --output-dir       "$OUTPUT_DIR" \
    --catalog          "$CATALOG" \
    --npy-dir-to-exclude data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8 \
    --skip-done-parts \
    --save-embeddings
