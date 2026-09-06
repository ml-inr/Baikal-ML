#!/usr/bin/env bash
# E5 (horizon θ-loss) vs E1 baseline. E5@ep5 (latest) vs two E1 references:
#   E1@ep5  = same training budget (fair ablation, horizon on/off)
#   E1@ep1  = matched on val_f1/AUC (E5's aggregate metrics are deflated by the
#             deliberate horizon abstention, so a naive match picks an early E1).
# MC out-of-training (all 3 ptypes, for AUC/efficiency) + exp held-out (probs h5,
# contiguous sampling). Runs sequentially. Usage: bash run_e5_vs_e1.sh [cuda:3]
set -uo pipefail
cd /home/albert/Baikal2025
DEVICE="${1:-cuda:3}"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
MC_TRAIN_NPY="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
EXP_TRAIN_NPY="data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8"
EXP_PROBS_H5="data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
THR=0.8; MINH=5; MINS=0; MC_PER_PTYPE=700000; EXP_PER_PART=70000
E5=experiments/numu/260707_0226_da_nu_classifier_exp_full_E5_horizon_lambda0.01
E1=experiments/numu/260705_0702_da_nu_classifier_exp_full_E1_lambda0.01
CKPTS=("$E5/da_checkpoint_epoch_005.pth" "$E1/da_checkpoint_epoch_005.pth" "$E1/da_checkpoint_epoch_001.pth")

for CKPT in "${CKPTS[@]}"; do
  echo "############## $CKPT ($DEVICE) ##############"
  echo "--- MC (out-of-training) ---"
  python inference_v2/nu_classifier/predict_mc.py --checkpoint "$CKPT" --mc-h5 "$MC_H5" \
      --npy-dir-to-exclude "$MC_TRAIN_NPY" --max-events-per-ptype "$MC_PER_PTYPE" \
      --threshold "$THR" --min-hits "$MINH" --min-strings "$MINS" --device "$DEVICE" \
      || echo "MC FAILED $CKPT"
  echo "--- EXP (out-of-training, contiguous) ---"
  python inference_v2/nu_classifier/predict_exp.py --checkpoint "$CKPT" --exp-h5 "$EXP_H5" \
      --exclude-training-npy "$EXP_TRAIN_NPY" --probs-h5 "$EXP_PROBS_H5" --probs-group exp_full \
      --max-events-per-part "$EXP_PER_PART" --sample-mode contiguous \
      --threshold "$THR" --min-hits "$MINH" --min-strings "$MINS" --device "$DEVICE" \
      || echo "EXP FAILED $CKPT"
done
echo "ALL DONE"
