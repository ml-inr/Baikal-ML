#!/usr/bin/env bash
# Predict the best muon-energy continuation checkpoint on the full h8s3 source.
#
# Usage (from the repository root):
#   GPU=3 SUBSET=test bash inference/track_energy_reconstruction/run_predict_best_continuation.sh
#
# SUBSET=all|test (default all).  NPY_DIR must be the training dataset that
# produced CHECKPOINT: it supplies the split and flat weights.  This checkpoint
# predates the dataset refactor, so the default is the old dataset and the
# predictor reads its legacy split files.
#
# Override CHECKPOINT / NPY_DIR / OUTPUT_DIR / SUBSET via the environment.
set -euo pipefail

cd "$(dirname "$0")/../.."

CHECKPOINT="${CHECKPOINT:-trained_models/track_energy_reconstruction/261023_2037_mu_energy_h8s3_d128_l4_flat1M_znorm_zflip_noise_fixedsched_cont_s42/best_model.pth}"
NPY_DIR="${NPY_DIR:-data_manager/datasets/mu_energy_dataset_h8s3_thr0.8}"
OUTPUT_DIR="${OUTPUT_DIR:-inference/track_energy_reconstruction/preds}"
GPU="${GPU:-3}"
SUBSET="${SUBSET:-all}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" \
python -m inference.track_energy_reconstruction.predict_full \
    --checkpoint "${CHECKPOINT}" \
    --npy-dir "${NPY_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --subset "${SUBSET}" \
    --device cuda:0 \
    --batch-size 512
