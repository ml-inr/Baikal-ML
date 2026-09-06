#!/usr/bin/env bash
# Matched-working-point TEST inference for the E1/E2/E3b comparison.
# Instead of each run's `best_da_model` (which sit at very different epochs:
# E1=23, E2=43, E3b=37), we score checkpoints matched to E1 @ epoch 10 on
# val_loss (primary) and F1 (control) — see analysis/model_comparison.
#   anchor : E1  @ epoch 10
#   E2     @ epoch 22 (val_loss/AUC match)  and  epoch 28 (F1 match)  -> bracket
#   E3b    @ epoch 10 (val_loss/AUC match ~= F1 match at 8)
# Same cuts / probs h5 as the round-1 test. Runs sequentially on one GPU.
# Usage: bash run_matched_epoch_infer.sh [cuda:0]
set -uo pipefail
cd /home/albert/Baikal2025

DEVICE="${1:-cuda:0}"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
MC_TRAIN_NPY="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
EXP_TRAIN_NPY="data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8"
EXP_PROBS_H5="data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
THR=0.8; MINH=5; MINS=0
MC_PER_PTYPE=700000
EXP_PER_PART=70000

# "run_dir:epoch"
JOBS=(
  "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01:010"
  "260705_0708_da_nu_classifier_exp_full_E2_lambda0.01:022"
  "260705_0708_da_nu_classifier_exp_full_E2_lambda0.01:028"
  "260705_0721_da_nu_classifier_exp_full_E3b_lambda0.01:010"
)

for job in "${JOBS[@]}"; do
  run="${job%%:*}"; ep="${job##*:}"
  CKPT="experiments/numu/${run}/da_checkpoint_epoch_${ep}.pth"
  [[ -f "$CKPT" ]] || { echo "MISSING $CKPT"; continue; }
  echo "###################################################################"
  echo "### $run  epoch $ep  (device=$DEVICE)"
  echo "###################################################################"

  echo "--- MC (out-of-training, ~${MC_PER_PTYPE}/ptype) ---"
  python inference_v2/nu_classifier/predict_mc.py \
      --checkpoint "$CKPT" --mc-h5 "$MC_H5" \
      --npy-dir-to-exclude "$MC_TRAIN_NPY" \
      --max-events-per-ptype "$MC_PER_PTYPE" \
      --threshold "$THR" --min-hits "$MINH" --min-strings "$MINS" \
      --device "$DEVICE" || echo "MC FAILED for $run@$ep"

  echo "--- EXP (out-of-training, ~${EXP_PER_PART}/part) ---"
  python inference_v2/nu_classifier/predict_exp.py \
      --checkpoint "$CKPT" --exp-h5 "$EXP_H5" \
      --exclude-training-npy "$EXP_TRAIN_NPY" \
      --probs-h5 "$EXP_PROBS_H5" --probs-group exp_full \
      --max-events-per-part "$EXP_PER_PART" \
      --threshold "$THR" --min-hits "$MINH" --min-strings "$MINS" \
      --device "$DEVICE" || echo "EXP FAILED for $run@$ep"
done
echo "ALL DONE"
