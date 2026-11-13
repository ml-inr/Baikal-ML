#!/usr/bin/env bash
# Sig-noise probabilities for MC. Batch size 256 defines the hit selection
# (doc/sig_noise_batch_size.md); it is not a speed knob.

set -euo pipefail
cd "$(dirname "$0")/../.."

INPUT="data_manager/data/h5datasets/baikal_mc_merged.h5"
DEVICE="auto"
BATCH_SIZE=256          # defines which hits pass the 0.8 cut

# Extra arguments are passed through, e.g. --append --parts-json <path>.
python -u inference/sig_noise/predict_mc_h5.py \
    --input      "$INPUT" \
    --device     "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    "$@"
