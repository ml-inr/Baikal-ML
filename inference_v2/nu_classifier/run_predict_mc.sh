#!/usr/bin/env bash
# Run nu-classifier predictions from mc_merged HDF5 (on-the-fly sig-noise filtering).

set -euo pipefail

CHECKPOINT="experiments/numu/260531_1720_da_nu_classifier_h5s0_lambda0.01_thr0.8_FixedDA/best_da_model.pth"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
SOURCE="mc_merged"
#MC_H5="data_manager/data/h5datasets/baikal_mc_reco.h5"
#SOURCE="mc_reco"
THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0
BATCH_SIZE=1024
MIN_BATCH_EVENTS=32768   # accumulate parts until this many events before GPU inference
DEVICE="cuda:1"
OUTPUT_DIR="inference_v2/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"

# Optional filters — comment out a line to disable that filter
#PTYPES="muatm_2020,nuatm_2020,nue2_2020"
MAX_EVENTS_PER_PTYPE=20_000_000
# PARTS="part_0,part_1,part_2"
# NPY_DIR_TO_EXCLUDE not applicable for mc_reco (training data is from mc_merged)
SKIP_DONE_PARTS=true   # set to false to reprocess already-scored parts
SAVE_EMBEDDINGS=false  # uncomment to store encoder embeddings alongside scores

# Build optional args from defined variables
EXTRA_ARGS=()
[ -n "${PTYPES:-}"                 ] && EXTRA_ARGS+=(--ptypes                "$PTYPES")
[ -n "${PARTS:-}"                  ] && EXTRA_ARGS+=(--parts                 "$PARTS")
[ -n "${MAX_EVENTS_PER_PTYPE:-}"   ] && EXTRA_ARGS+=(--max-events-per-ptype  "$MAX_EVENTS_PER_PTYPE")
[ -n "${NPY_DIR_TO_EXCLUDE:-}"     ] && EXTRA_ARGS+=(--npy-dir-to-exclude    "$NPY_DIR_TO_EXCLUDE")
[ "${SKIP_DONE_PARTS:-false}"   = "true" ] && EXTRA_ARGS+=(--skip-done-parts)
[ "${SAVE_EMBEDDINGS:-false}"   = "true" ] && EXTRA_ARGS+=(--save-embeddings)

python inference_v2/nu_classifier/predict_mc.py \
    --checkpoint      "$CHECKPOINT" \
    --mc-h5           "$MC_H5" \
    --source          "$SOURCE" \
    --threshold       "$THRESHOLD" \
    --min-hits        "$MIN_HITS" \
    --min-strings     "$MIN_STRINGS" \
    --batch-size      "$BATCH_SIZE" \
    --min-batch-events "$MIN_BATCH_EVENTS" \
    --device          "$DEVICE" \
    --output-dir   "$OUTPUT_DIR" \
    --catalog      "$CATALOG" \
    "${EXTRA_ARGS[@]}"
