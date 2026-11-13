#!/usr/bin/env bash
# Exact scoring commands for the new prefilter pair (provenance record).
#
# The predictions were made 2026-11-11 with these commands.  The databases
# already exist; rerunning is only needed to reproduce them.  MC is scored on
# the held-out canonical part lists in scored_parts/ with per-class caps
# (10M muatm, 3M nuatm, all 1.29M nue2) and --exclude-trained-parts, which drops
# every h5 part the training NPY dataset was built from.  exp_full is complete.
#
# Devices: DA on cuda:0, control on cuda:3 by default; override with
#   DA_GPU=cuda:N CT_GPU=cuda:M bash run_predictions.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$ROOT"

DA="trained_models/prefilter/da_prefilter_numu_261108_1943_Qclip_hardlabels_FocalLoss_muOnlyDA_k10_refresh5_probe/best_da_model.pth"
CONTROL="trained_models/prefilter/da_prefilter_numu_261109_1845_Qclip_hardlabels_FocalLoss_lambda0_control_k10_refresh5_probe/best_da_model.pth"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
CATALOG="data_manager/catalog_v2.duckdb"
OUT="inference/prefilter/preds"
PARTS="$HERE/scored_parts"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
DA_GPU="${DA_GPU:-cuda:0}"
CT_GPU="${CT_GPU:-cuda:3}"

score_model() {
    local checkpoint="$1"
    local gpu="$2"

    python inference/prefilter/predict_mc.py \
        --checkpoint "$checkpoint" --mc-h5 "$MC_H5" --source mc_merged \
        --ptypes muatm_2020 --parts "$(cat "$PARTS/muatm_2020.txt")" \
        --max-events-per-ptype 10000000 --exclude-trained-parts \
        --batch-size 1024 --min-batch-events 32768 --device "$gpu" \
        --output-dir "$OUT" --catalog "$CATALOG" --probs-h5 auto

    python inference/prefilter/predict_mc.py \
        --checkpoint "$checkpoint" --mc-h5 "$MC_H5" --source mc_merged \
        --ptypes nuatm_2020 --parts "$(cat "$PARTS/nuatm_2020.txt")" \
        --max-events-per-ptype 3000000 --exclude-trained-parts \
        --batch-size 1024 --min-batch-events 32768 --device "$gpu" \
        --output-dir "$OUT" --catalog "$CATALOG" --probs-h5 auto

    python inference/prefilter/predict_mc.py \
        --checkpoint "$checkpoint" --mc-h5 "$MC_H5" --source mc_merged \
        --ptypes nue2_2020 --parts "$(cat "$PARTS/nue2_2020.txt")" \
        --max-events-per-ptype 1500000 --exclude-trained-parts \
        --batch-size 1024 --min-batch-events 32768 --device "$gpu" \
        --output-dir "$OUT" --catalog "$CATALOG" --probs-h5 auto

    python inference/prefilter/predict_exp.py \
        --checkpoint "$checkpoint" --exp-h5 "$EXP_H5" \
        --batch-size 512 --chunk-events 200000 --device "$gpu" \
        --output-dir "$OUT" --catalog "$CATALOG" --probs-h5 auto \
        --skip-done-parts
}

score_model "$DA" "$DA_GPU"
score_model "$CONTROL" "$CT_GPU"
