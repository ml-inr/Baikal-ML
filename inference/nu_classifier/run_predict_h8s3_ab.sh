#!/usr/bin/env bash
# Predictions for the four 8-3 / mu-only A/B models (best checkpoint at run time).
#
# The event cut is 8-3 (min_hits=8, min_strings=3), NOT the historical 5-0:
# the DuckDB contains only the evaluation population.  n_sn_hits/n_sn_strings
# are still stored, so stricter subsets can be taken later.
#
# Per model: exp_full first, then mc_merged (only if exp reported Done).
# Outputs: preds/<run>@best_da_model/{exp_full,mc_merged}_thr0p8.duckdb
# Re-running after best_da_model changes requires deleting those DBs first
# (the scripts resume with --skip-done-parts).
#
# Usage: bash inference/nu_classifier/run_predict_h8s3_ab.sh <cell>
set -euo pipefail
cd /home/albert/Baikal2025
export CUDA_DEVICE_ORDER=PCI_BUS_ID

CELL="${1:?укажи ячейку}"
RUN="261024_1202_da_nu_classifier_exp_full_E1_weights_${CELL}"

case "$CELL" in
  z128_k10_dlr1e-2_lam0p003_delay2_h5s0_muOnlyDA_s42)
    SRC_NPY="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
    EXP_NPY="data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8"
    DEVICE="cuda:5" ;;
  z128_k10_dlr1e-2_lam0p003_delay2_h8s3_s42)
    SRC_NPY="data_manager/datasets/nu_classifier_dataset_h8s3_thr0.8"
    EXP_NPY="data_manager/datasets/nu_classifier_dataset_exp_full_h8s3_thr0.8"
    DEVICE="cuda:2" ;;
  z128_k10_dlr1e-2_lam0p003_delay2_h8s3_muOnlyDA_s42)
    SRC_NPY="data_manager/datasets/nu_classifier_dataset_h8s3_thr0.8"
    EXP_NPY="data_manager/datasets/nu_classifier_dataset_exp_full_h8s3_thr0.8"
    DEVICE="cuda:3" ;;
  z128_lam0_delay2_h8s3_s42)
    SRC_NPY="data_manager/datasets/nu_classifier_dataset_h8s3_thr0.8"
    EXP_NPY="data_manager/datasets/nu_classifier_dataset_exp_full_h8s3_thr0.8"
    DEVICE="cuda:1" ;;
  *)
    echo "unknown cell: $CELL" >&2
    exit 2 ;;
esac

CHECKPOINT="trained_models/classifier/${RUN}/best_da_model.pth"
[ -f "$CHECKPOINT" ] || { echo "no checkpoint: $CHECKPOINT" >&2; exit 1; }

MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
MC_PROBS="data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
EXP_PROBS="data_manager/data/h5datasets/exp_full_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"

THRESHOLD=0.8
MIN_HITS=8
MIN_STRINGS=3
BATCH_SIZE=1024
MIN_BATCH_EVENTS=32768
OUTPUT_DIR="inference/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"
LOG_EXP="inference/nu_classifier/logs/predict_exp_${CELL}.out"
LOG_MC="inference/nu_classifier/logs/predict_mc_${CELL}.out"

echo "[ab] ${CELL} on ${DEVICE}: exp_full ..."
python -u inference/nu_classifier/predict_exp.py \
    --checkpoint   "$CHECKPOINT" \
    --exp-h5       "$EXP_H5" \
    --probs-h5     "$EXP_PROBS" \
    --probs-group  "exp_full" \
    --threshold    "$THRESHOLD" \
    --min-hits     "$MIN_HITS" \
    --min-strings  "$MIN_STRINGS" \
    --batch-size   "$BATCH_SIZE" \
    --device       "$DEVICE" \
    --output-dir   "$OUTPUT_DIR" \
    --catalog      "$CATALOG" \
    --mark-da-target-npy "$EXP_NPY" \
    --skip-done-parts \
    --save-embeddings > "$LOG_EXP" 2>&1

if ! grep -q "^Done:" "$LOG_EXP"; then
    echo "[ab] ABORT: exp failed — see $LOG_EXP" >&2
    tail -20 "$LOG_EXP" >&2
    exit 1
fi
echo "[ab] ${CELL}: exp done; mc_merged ..."

python -u inference/nu_classifier/predict_mc.py \
    --checkpoint       "$CHECKPOINT" \
    --mc-h5            "$MC_H5" \
    --probs-h5         "$MC_PROBS" \
    --source           "mc_merged" \
    --threshold        "$THRESHOLD" \
    --min-hits         "$MIN_HITS" \
    --min-strings      "$MIN_STRINGS" \
    --batch-size       "$BATCH_SIZE" \
    --min-batch-events "$MIN_BATCH_EVENTS" \
    --device           "$DEVICE" \
    --output-dir       "$OUTPUT_DIR" \
    --catalog          "$CATALOG" \
    --npy-dir-to-exclude "$SRC_NPY" \
    --skip-done-parts \
    --save-embeddings > "$LOG_MC" 2>&1

if ! grep -q "^Done:" "$LOG_MC"; then
    echo "[ab] ABORT: mc failed — see $LOG_MC" >&2
    tail -20 "$LOG_MC" >&2
    exit 1
fi
echo "[ab] ${CELL}: DONE"
grep "^Done:" "$LOG_EXP" "$LOG_MC"
