#!/usr/bin/env bash
# Sig-noise probabilities for experimental data. Batch size 256 defines the hit
# selection (doc/sig_noise_batch_size.md); it is not a speed knob.

set -euo pipefail
cd "$(dirname "$0")/../.."

INPUT="data_manager/data/h5datasets/exp_full.h5"
DEVICE="auto"
BATCH_SIZE=256          # must match the MC probs file

# Extra arguments are passed through, e.g. --group <name> --exclude-parts ...
python -u inference/sig_noise/predict_exp_h5.py \
    --input      "$INPUT" \
    --device     "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    "$@"
