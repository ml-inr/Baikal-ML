#!/usr/bin/env bash
# Run prefilter predictions for MC data (all raw hits, no sig-noise filtering).

set -euo pipefail

CHECKPOINT="experiments/numu/da_prefilter_numu_260429_0155_Qclip_softlabels_PlateauLR_lambda0.05_MC2M_dmodel128_zflipBothMCExp_SoftFocalLoss/best_da_model.pth"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
SOURCE="mc_merged"
BATCH_SIZE=1024
MIN_BATCH_EVENTS=32768
DEVICE="auto"
OUTPUT_DIR="inference_v2/prefilter/preds"
CATALOG="data_manager/catalog_v2.duckdb"

# Optional filters — comment out a line to disable that filter
PTYPES="muatm_2020,nuatm_2020,nue2_2020"
MAX_EVENTS_PER_PTYPE=10000000
# PARTS="part_0,part_1,part_2"
SKIP_DONE_PARTS=true
SAVE_EMBEDDINGS=false

# Build optional args from defined variables
EXTRA_ARGS=()
[ -n "${PTYPES:-}"                 ] && EXTRA_ARGS+=(--ptypes                "$PTYPES")
[ -n "${PARTS:-}"                  ] && EXTRA_ARGS+=(--parts                 "$PARTS")
[ -n "${MAX_EVENTS_PER_PTYPE:-}"   ] && EXTRA_ARGS+=(--max-events-per-ptype  "$MAX_EVENTS_PER_PTYPE")
[ "${SKIP_DONE_PARTS:-false}"  = "true" ] && EXTRA_ARGS+=(--skip-done-parts)
[ "${SAVE_EMBEDDINGS:-false}"  = "true" ] && EXTRA_ARGS+=(--save-embeddings)

python inference_v2/prefilter/predict_mc.py \
    --checkpoint       "$CHECKPOINT" \
    --mc-h5            "$MC_H5" \
    --source           "$SOURCE" \
    --batch-size       "$BATCH_SIZE" \
    --min-batch-events "$MIN_BATCH_EVENTS" \
    --device           "$DEVICE" \
    --output-dir       "$OUTPUT_DIR" \
    --catalog          "$CATALOG" \
    "${EXTRA_ARGS[@]}"
