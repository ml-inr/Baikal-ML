#!/usr/bin/env bash
# Run prefilter predictions for experimental data (all raw hits, no sig-noise filtering).

set -euo pipefail

CHECKPOINT="experiments/numu/da_prefilter_numu_260429_0155_Qclip_softlabels_PlateauLR_lambda0.05_MC2M_dmodel128_zflipBothMCExp_SoftFocalLoss/best_da_model.pth"
EXP_H5="data_manager/data/h5datasets/exp_reco.h5"
BATCH_SIZE=512
DEVICE="auto"
OUTPUT_DIR="inference_v2/prefilter/preds"
CATALOG="data_manager/catalog_v2.duckdb"

SKIP_DONE_PARTS=true
SAVE_EMBEDDINGS=false

# Build optional args from defined variables
EXTRA_ARGS=()
[ "${SKIP_DONE_PARTS:-false}"  = "true" ] && EXTRA_ARGS+=(--skip-done-parts)
[ "${SAVE_EMBEDDINGS:-false}"  = "true" ] && EXTRA_ARGS+=(--save-embeddings)

python inference_v2/prefilter/predict_exp.py \
    --checkpoint  "$CHECKPOINT" \
    --exp-h5      "$EXP_H5" \
    --batch-size  "$BATCH_SIZE" \
    --device      "$DEVICE" \
    --output-dir  "$OUTPUT_DIR" \
    --catalog     "$CATALOG" \
    "${EXTRA_ARGS[@]}"
