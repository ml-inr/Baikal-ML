#!/usr/bin/env bash
# Out-of-training TEST inference for the three exp_full DA models (E1/E2/E3b).
# For each checkpoint: ~2M out-of-training MC (mc_merged) + ~2M out-of-training
# exp (exp_full) events, scored with the same cuts (min_hits=5, min_strings=0,
# SN thr 0.8). Training events are excluded via the NPY back-links.
# Runs sequentially on one GPU. Usage: bash run_test_infer_exp_full_da.sh [cuda:0]

set -uo pipefail
cd /home/albert/Baikal2025

DEVICE="${1:-cuda:0}"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
MC_TRAIN_NPY="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
EXP_TRAIN_NPY="data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8"
EXP_PROBS_H5="data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
THR=0.8; MINH=5; MINS=0
MC_PER_PTYPE=700000     # ~2.1M over 3 ptypes
EXP_PER_PART=70000      # ~2.03M over 29 parts

RUNS=(
  "260705_0702_da_nu_classifier_exp_full_E1_lambda0.01"
  "260705_0708_da_nu_classifier_exp_full_E2_lambda0.01"
  "260705_0721_da_nu_classifier_exp_full_E3b_lambda0.01"
)

for run in "${RUNS[@]}"; do
  CKPT="experiments/numu/${run}/best_da_model.pth"
  echo "###################################################################"
  echo "### $run  (device=$DEVICE)"
  echo "###################################################################"

  echo "--- MC (out-of-training, ~${MC_PER_PTYPE}/ptype) ---"
  python inference_v2/nu_classifier/predict_mc.py \
      --checkpoint "$CKPT" --mc-h5 "$MC_H5" \
      --npy-dir-to-exclude "$MC_TRAIN_NPY" \
      --max-events-per-ptype "$MC_PER_PTYPE" \
      --threshold "$THR" --min-hits "$MINH" --min-strings "$MINS" \
      --device "$DEVICE" || echo "MC FAILED for $run"

  echo "--- EXP (out-of-training, ~${EXP_PER_PART}/part) ---"
  python inference_v2/nu_classifier/predict_exp.py \
      --checkpoint "$CKPT" --exp-h5 "$EXP_H5" \
      --exclude-training-npy "$EXP_TRAIN_NPY" \
      --probs-h5 "$EXP_PROBS_H5" --probs-group exp_full \
      --max-events-per-part "$EXP_PER_PART" \
      --threshold "$THR" --min-hits "$MINH" --min-strings "$MINS" \
      --device "$DEVICE" || echo "EXP FAILED for $run"
done

echo "ALL DONE"
