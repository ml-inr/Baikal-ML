#!/usr/bin/env bash
# Reconstructed samples end to end: sig-noise probabilities, then classifier scores.
#
# Why the probabilities are precomputed rather than run on the fly.  The sig-noise
# batch size defines which hits survive (doc/sig_noise_batch_size.md), and pinning
# --sn-batch-size is not enough: predict_mc.py accumulates events across parts
# before each GPU launch, so the *composition* of a batch depends on which parts
# the run happened to select and in what order.  Two runs over different part
# subsets would then disagree about the hits of the same event.  Precomputing
# freezes the numbers once.
#
# It also makes the reco samples comparable with mc_merged and exp_full, whose
# scores come from stored probabilities at batch 256.  On-the-fly reco scores
# would be produced by a different procedure, and the reco excess could not be
# quoted next to the 2.87 measured on the others.
#
# Both steps are resumable and both write next to their input.
#
# Usage:
#   nohup bash inference_v2/nu_classifier/run_reco_chain_FIXED_sn256.sh > /tmp/reco.log 2>&1 &

set -euo pipefail
cd /home/albert/Baikal2025

export CUDA_DEVICE_ORDER=PCI_BUS_ID     # so cuda:N means what nvidia-smi shows
DEVICE="cuda:5"
PY=/home/albert/miniconda3/envs/baikal25/bin/python
SN=gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001

CHECKPOINT="experiments/numu/260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256/best_da_model.pth"
H5DIR=data_manager/data/h5datasets
MC_RECO_H5="$H5DIR/baikal_mc_reco.h5"
EXP_RECO_H5="$H5DIR/exp_reco.h5"
TAG=k_nsol_labelneq0_da_hs128_k0p0001
MC_RECO_PROBS="$H5DIR/baikal_mc_reco_probs_${TAG}.h5"
EXP_RECO_PROBS="$H5DIR/exp_reco_probs_${TAG}.h5"

SN_BATCH=256             # NOT a speed knob: it defines the hit selection
THRESHOLD=0.8
MIN_HITS=5               # matches the mc_merged and exp_full runs of this checkpoint;
MIN_STRINGS=0            # the analysis cut h8s3 is tighter and is applied later
BATCH_SIZE=1024          # classifier batch -- speed only, the model is batch-independent
MIN_BATCH_EVENTS=32768   # mc_reco has ~25,000 tiny parts

echo "########## 1/4  sig-noise probabilities for mc_reco"
"$PY" "$SN/predict_mc_h5.py" --input "$MC_RECO_H5" \
      --device "$DEVICE" --batch-size "$SN_BATCH"

echo "########## 2/4  sig-noise probabilities for exp_reco"
"$PY" "$SN/predict_exp_h5.py" --input "$EXP_RECO_H5" \
      --device "$DEVICE" --batch-size "$SN_BATCH"

echo "########## 3/4  classifier scores for mc_reco"
"$PY" inference_v2/nu_classifier/predict_mc.py \
    --checkpoint "$CHECKPOINT" --source mc_reco --mc-h5 "$MC_RECO_H5" \
    --probs-h5 "$MC_RECO_PROBS" --threshold "$THRESHOLD" \
    --min-hits "$MIN_HITS" --min-strings "$MIN_STRINGS" \
    --batch-size "$BATCH_SIZE" --min-batch-events "$MIN_BATCH_EVENTS" \
    --sn-batch-size "$SN_BATCH" --device "$DEVICE" --skip-done-parts

echo "########## 4/4  classifier scores for exp_reco"
"$PY" inference_v2/nu_classifier/predict_exp.py \
    --checkpoint "$CHECKPOINT" --exp-h5 "$EXP_RECO_H5" \
    --probs-h5 "$EXP_RECO_PROBS" --threshold "$THRESHOLD" \
    --min-hits "$MIN_HITS" --min-strings "$MIN_STRINGS" \
    --batch-size "$BATCH_SIZE" --sn-batch-size "$SN_BATCH" \
    --device "$DEVICE" --skip-done-parts

echo "########## done"
