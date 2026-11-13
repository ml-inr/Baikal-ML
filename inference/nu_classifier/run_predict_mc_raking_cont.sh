#!/usr/bin/env bash
# Full-statistics nu-classifier scoring of mc_merged with the RAKING continuation
# checkpoint e45 (261023_0631_..._raking_cont, da_checkpoint_epoch_045.pth).
#
# e45 is the deepest saved checkpoint of the continuation: save_every=5 and val_loss
# did not improve after e25, so e46-e49 were never written and there is no new
# best_da_model.pth in this run. Same data, weights and cuts as the champion wrapper
# (run_predict_mc_raking.sh), so the two outputs are directly comparable.
# Training events are excluded via the NPY dataset back-links.
#
# Resumable via --skip-done-parts. Run under nohup.
set -euo pipefail
cd /home/albert/Baikal2025

export CUDA_DEVICE_ORDER=PCI_BUS_ID     # so that cuda:1 below = GPU 1 in nvidia-smi

CHECKPOINT="trained_models/classifier/261023_0631_da_nu_classifier_exp_full_E1_weights_z128_k10_dlr1e-2_lam0p003_delay2_s42_raking_cont/da_checkpoint_epoch_045.pth"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
PROBS_H5="data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
SOURCE="mc_merged"

THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0

BATCH_SIZE=1024
MIN_BATCH_EVENTS=32768
DEVICE="cuda:1"
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
