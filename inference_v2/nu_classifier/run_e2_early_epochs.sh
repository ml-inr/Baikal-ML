#!/usr/bin/env bash
# Overfitting-onset probe for E2 (spectral norm): score E2 at epochs {1,3,5,10}
# on out-of-training MC + exp held-out, to see the exp neutrino-like excess
# (excess(thr) = exp_frac(>thr) / mc_muatm_frac(>thr)) vs training epoch.
# EXP subsample uses --sample-mode contiguous (one file-offset window per part;
# unbiased since a part is one short run) -> only that window's gzip chunks
# decompress, much faster than scattered random reads.
# NOTE: this exp event set differs from the E3b (random-sampled) probe; do not
# mix the two in one plot. Within E2 it is consistent (deterministic window/part).
# Usage: bash run_e2_early_epochs.sh [cuda:0]
set -uo pipefail
cd /home/albert/Baikal2025

DEVICE="${1:-cuda:0}"
RUN="260705_0708_da_nu_classifier_exp_full_E2_lambda0.01"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
MC_TRAIN_NPY="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
EXP_TRAIN_NPY="data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8"
EXP_PROBS_H5="data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
THR=0.8; MINH=5; MINS=0
MC_PER_PTYPE=700000
EXP_PER_PART=70000

for ep in 001 003 005 010; do
  CKPT="experiments/numu/${RUN}/da_checkpoint_epoch_${ep}.pth"
  [[ -f "$CKPT" ]] || { echo "MISSING $CKPT"; continue; }
  echo "###################################################################"
  echo "### $RUN  epoch $ep  (device=$DEVICE)"
  echo "###################################################################"

  echo "--- MC (out-of-training, ~${MC_PER_PTYPE}/ptype) ---"
  python inference_v2/nu_classifier/predict_mc.py \
      --checkpoint "$CKPT" --mc-h5 "$MC_H5" \
      --npy-dir-to-exclude "$MC_TRAIN_NPY" \
      --max-events-per-ptype "$MC_PER_PTYPE" \
      --threshold "$THR" --min-hits "$MINH" --min-strings "$MINS" \
      --device "$DEVICE" || echo "MC FAILED for $RUN@$ep"

  echo "--- EXP (out-of-training, ~${EXP_PER_PART}/part, contiguous slices) ---"
  python inference_v2/nu_classifier/predict_exp.py \
      --checkpoint "$CKPT" --exp-h5 "$EXP_H5" \
      --exclude-training-npy "$EXP_TRAIN_NPY" \
      --probs-h5 "$EXP_PROBS_H5" --probs-group exp_full \
      --max-events-per-part "$EXP_PER_PART" --sample-mode contiguous \
      --threshold "$THR" --min-hits "$MINH" --min-strings "$MINS" \
      --device "$DEVICE" || echo "EXP FAILED for $RUN@$ep"
done
echo "ALL DONE"
