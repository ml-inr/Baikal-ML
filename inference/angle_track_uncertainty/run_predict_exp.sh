#!/usr/bin/env bash
# Angle/track/uncertainty predictions for experimental data.
# Experimental point/uncertainty outputs are NOT validated; see the README.

set -euo pipefail
cd "$(dirname "$0")/../.."

SOURCE="data_manager/data/h5datasets/exp_full.h5"
SOURCE_NAME="exp_full"
CHECKPOINT="trained_models/angle_track_uncertainty/release_v3/angle_track_uncertainty.pt"
SN_MODE="canonical"        # canonical | release
PROBS_H5=""                # empty = sibling {stem}_probs_k_nsol...h5
DEVICE="auto"
THREADS=8
BATCH_SIZE=256
SN_BATCH_SIZE=256
OUTPUT_DIR="inference/angle_track_uncertainty/preds"
CATALOG="data_manager/catalog_v2.duckdb"

PARTS=""
LIMIT=""
GATE="false"
EVENT_SET_DB=""             # optional: DuckDB with predictions(event_fk)
EVENTS_PER_CHUNK=""         # e.g. 32768 for sparse event sets

EXTRA_ARGS=()
[ -n "${PROBS_H5:-}" ] && EXTRA_ARGS+=(--probs-h5 "$PROBS_H5")
[ -n "${PARTS:-}"    ] && EXTRA_ARGS+=(--parts "$PARTS")
[ -n "${LIMIT:-}"    ] && EXTRA_ARGS+=(--limit "$LIMIT")
[ -n "${EVENT_SET_DB:-}" ] && EXTRA_ARGS+=(--event-set-db "$EVENT_SET_DB")
[ -n "${EVENTS_PER_CHUNK:-}" ] && EXTRA_ARGS+=(--events-per-chunk "$EVENTS_PER_CHUNK")
if [ "${GATE:-false}" = "true" ]; then
    EXTRA_ARGS+=(--conditional-low-threshold 0.50 --max-added-photon-residual-ns 200)
fi

python -u -m inference.angle_track_uncertainty.predict \
    --source         "$SOURCE" \
    --source-name    "$SOURCE_NAME" \
    --checkpoint     "$CHECKPOINT" \
    --sn-mode        "$SN_MODE" \
    --device         "$DEVICE" \
    --threads        "$THREADS" \
    --batch-size     "$BATCH_SIZE" \
    --sn-batch-size  "$SN_BATCH_SIZE" \
    --output-dir     "$OUTPUT_DIR" \
    --catalog        "$CATALOG" \
    "${EXTRA_ARGS[@]}"
