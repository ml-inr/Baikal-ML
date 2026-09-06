#!/usr/bin/env bash
# Batch nu-classifier predictions for multiple checkpoints.
#
# Four steps, each independently selectable:
#   npy        — predict_npy.py on mc_merged training data (fast, sequential)
#   mc_merged  — predict_mc.py --source mc_merged, excluding training parts (parallel)
#   mc_reco    — predict_mc.py --source mc_reco (parallel)
#   exp        — predict_exp.py on each configured EXP_H5_FILES (parallel)
#
# Usage (from project root):
#   bash inference_v2/nu_classifier/run_batch_predict.sh [npy|mc_merged|mc_reco|exp|all]
#   No argument → runs all four steps in sequence.
#
# Tip: run in a tmux/screen session; per-model logs go to preds/{ckpt_name}/*.log

set -euo pipefail
STEP="${1:-all}"

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

DEVICES=("cuda:1" "cuda:2" "cuda:4")   # GPUs to spread parallel steps across

BATCH_SIZE=1024
MIN_BATCH_EVENTS=32768       # accumulate parts until this many events (small-part throughput fix)
MIN_HITS=5
MIN_STRINGS=0
MAX_EVENTS_PER_PTYPE=1000000
OUTPUT_DIR="inference_v2/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"
SKIP_DONE_PARTS=true         # set false to reprocess all parts from scratch

# Data paths
MC_MERGED_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
MC_RECO_H5="data_manager/data/h5datasets/baikal_mc_reco.h5"

# Exp h5 files to score — each file is processed in a separate parallel wave.
# Comment out a line to skip that source.
EXP_H5_FILES=(
    #"data_manager/data/h5datasets/exp_reco.h5"
    "data_manager/data/h5datasets/exp.h5"
)

# NPY datasets — threshold auto-detected from dataset_info.json
NPY_DIR_THR05="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.5"
NPY_DIR_THR08="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"

# ── Model list: "experiment_dir  sn_threshold" ────────────────────────────────
# sn_threshold: sig-noise threshold used when building the training NPY data.
# Controls hit filtering in mc_merged, mc_reco, and exp steps.
MODELS=(
    "260531_1700_da_nu_classifier_h5s0_lambda0.1_thr0.8_FixedDA         0.8"
    "260531_1720_da_nu_classifier_h5s0_lambda0.01_thr0.8_FixedDA        0.8"
    "260531_1722_da_nu_classifier_h5s0_lambda0.0_thr0.8_FixedDA         0.8"
)

# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

N_DEV="${#DEVICES[@]}"
N_MOD="${#MODELS[@]}"

log() { echo "[batch $(date '+%H:%M:%S')] $*"; }

# Per-GPU slot tracking: at most 1 process per GPU at a time.
# _slot_pids[i] holds the PID of the job currently running on DEVICES[i], or "".
declare -a _slot_pids
_slot_failed=0

_init_slots() {
    _slot_failed=0
    _slot_pids=()
    for ((i=0; i<N_DEV; i++)); do _slot_pids[$i]=""; done
}

# Wait for the previous job on slot dev_idx, then clear it.
_wait_slot() {
    local dev_idx=$1 label=$2
    local pid="${_slot_pids[$dev_idx]:-}"
    [[ -z "$pid" ]] && return
    wait "$pid" || { log "  FAILED pid=$pid ($label)"; _slot_failed=$((_slot_failed + 1)); }
    _slot_pids[$dev_idx]=""
}

# Wait for all remaining slot jobs; report summary.
_drain_slots() {
    local label=$1
    for ((i=0; i<N_DEV; i++)); do _wait_slot "$i" "$label"; done
    if [[ $_slot_failed -gt 0 ]]; then
        log "WARNING: $_slot_failed $label job(s) failed"
        _slot_failed=0
        return 1
    fi
    log "  All $label jobs done OK"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — predict_npy.py  (sequential, device = DEVICES[0])
# ══════════════════════════════════════════════════════════════════════════════

run_npy() {
    log "=== STEP npy: predict_npy.py for $N_MOD models (sequential) ==="
    local dev="${DEVICES[0]}"
    local i=0
    for entry in "${MODELS[@]}"; do
        local exp_dir thr
        read -r exp_dir thr <<< "$entry"
        local ckpt="experiments/numu/${exp_dir}/best_da_model.pth"
        local tag="[$((i+1))/$N_MOD] $exp_dir"
        if [[ ! -f "$ckpt" ]]; then
            log "$tag — SKIP: checkpoint not found"
            i=$((i + 1)); continue
        fi
        local npy_dir
        [[ "$thr" == "0.5" ]] && npy_dir="$NPY_DIR_THR05" || npy_dir="$NPY_DIR_THR08"

        # thr0.5 datasets pre-date the h5 back-link files required by predict_npy.py
        if [[ ! -f "${npy_dir}/h5_part_keys.npy" ]]; then
            log "$tag — SKIP npy: ${npy_dir} missing h5_part_keys.npy (old dataset, no back-links)"
            i=$((i + 1)); continue
        fi
        log "$tag  thr=$thr  npy=$(basename "$npy_dir")  dev=$dev"
        python inference_v2/nu_classifier/predict_npy.py \
            --checkpoint  "$ckpt" \
            --npy-dir     "$npy_dir" \
            --min-hits    "$MIN_HITS" \
            --min-strings "$MIN_STRINGS" \
            --batch-size  "$BATCH_SIZE" \
            --device      "$dev" \
            --output-dir  "$OUTPUT_DIR" \
            --catalog     "$CATALOG"
        i=$((i + 1))
    done
    log "=== npy step done ==="
}

# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — predict_mc.py --source mc_merged  (out-of-training events, parallel)
# ══════════════════════════════════════════════════════════════════════════════

run_mc_merged() {
    log "=== STEP mc_merged: predict_mc.py --source mc_merged for $N_MOD models (1 per GPU) ==="
    _init_slots
    local i=0
    for entry in "${MODELS[@]}"; do
        local exp_dir thr
        read -r exp_dir thr <<< "$entry"
        local ckpt="experiments/numu/${exp_dir}/best_da_model.pth"
        local tag="[$((i+1))/$N_MOD] $exp_dir"
        if [[ ! -f "$ckpt" ]]; then
            log "$tag — SKIP: checkpoint not found"
            i=$((i + 1)); continue
        fi
        local dev_idx=$((i % N_DEV))
        local dev="${DEVICES[$dev_idx]}"
        local npy_dir
        [[ "$thr" == "0.5" ]] && npy_dir="$NPY_DIR_THR05" || npy_dir="$NPY_DIR_THR08"

        _wait_slot "$dev_idx" "mc_merged"   # ensure GPU is free before launching
        log "$tag  thr=$thr  dev=$dev  &"

        local args=(
            --checkpoint           "$ckpt"
            --mc-h5                "$MC_MERGED_H5"
            --source               mc_merged
            --threshold            "$thr"
            --min-hits             "$MIN_HITS"
            --min-strings          "$MIN_STRINGS"
            --max-events-per-ptype "$MAX_EVENTS_PER_PTYPE"
            --batch-size           "$BATCH_SIZE"
            --min-batch-events     "$MIN_BATCH_EVENTS"
            --device               "$dev"
            --output-dir           "$OUTPUT_DIR"
            --catalog              "$CATALOG"
        )
        # Only exclude training parts when the NPY dataset has h5 back-link files.
        # Older thr0.5 datasets were built without them — skip exclusion to avoid crash.
        if [[ -f "${npy_dir}/h5_part_keys.npy" ]]; then
            args+=(--npy-dir-to-exclude "$npy_dir")
        else
            log "  NOTE: $npy_dir missing h5_part_keys.npy — mc_merged covers all events (incl. training)"
        fi
        [[ "${SKIP_DONE_PARTS}" == "true" ]] && args+=(--skip-done-parts)

        python inference_v2/nu_classifier/predict_mc.py "${args[@]}" &
        _slot_pids[$dev_idx]=$!
        i=$((i + 1))
    done
    _drain_slots "mc_merged"
    log "=== mc_merged step done ==="
}

# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 — predict_mc.py --source mc_reco  (parallel)
# ══════════════════════════════════════════════════════════════════════════════

run_mc_reco() {
    log "=== STEP mc_reco: predict_mc.py --source mc_reco for $N_MOD models (1 per GPU) ==="
    _init_slots
    local i=0
    for entry in "${MODELS[@]}"; do
        local exp_dir thr
        read -r exp_dir thr <<< "$entry"
        local ckpt="experiments/numu/${exp_dir}/best_da_model.pth"
        local tag="[$((i+1))/$N_MOD] $exp_dir"
        if [[ ! -f "$ckpt" ]]; then
            log "$tag — SKIP: checkpoint not found"
            i=$((i + 1)); continue
        fi
        local dev_idx=$((i % N_DEV))
        local dev="${DEVICES[$dev_idx]}"

        _wait_slot "$dev_idx" "mc_reco"   # ensure GPU is free before launching
        log "$tag  thr=$thr  dev=$dev  &"

        local args=(
            --checkpoint           "$ckpt"
            --mc-h5                "$MC_RECO_H5"
            --source               mc_reco
            --threshold            "$thr"
            --min-hits             "$MIN_HITS"
            --min-strings          "$MIN_STRINGS"
            --max-events-per-ptype "$MAX_EVENTS_PER_PTYPE"
            --batch-size           "$BATCH_SIZE"
            --min-batch-events     "$MIN_BATCH_EVENTS"
            --device               "$dev"
            --output-dir           "$OUTPUT_DIR"
            --catalog              "$CATALOG"
        )
        [[ "${SKIP_DONE_PARTS}" == "true" ]] && args+=(--skip-done-parts)

        python inference_v2/nu_classifier/predict_mc.py "${args[@]}" &
        _slot_pids[$dev_idx]=$!
        i=$((i + 1))
    done
    _drain_slots "mc_reco"
    log "=== mc_reco step done ==="
}

# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — predict_exp.py  (per h5 file, all models in parallel per file)
# ══════════════════════════════════════════════════════════════════════════════

run_exp() {
    log "=== STEP exp: predict_exp.py for $N_MOD models (1 per GPU) ==="
    for exp_h5 in "${EXP_H5_FILES[@]}"; do
        if [[ ! -f "$exp_h5" ]]; then
            log "  SKIP missing file: $exp_h5"
            continue
        fi
        log "  Source: $exp_h5 — launching up to $N_DEV jobs at a time"
        _init_slots
        local i=0
        for entry in "${MODELS[@]}"; do
            local exp_dir thr
            read -r exp_dir thr <<< "$entry"
            local ckpt="experiments/numu/${exp_dir}/best_da_model.pth"
            local tag="  [$((i+1))/$N_MOD] $exp_dir"
            if [[ ! -f "$ckpt" ]]; then
                log "$tag — SKIP: checkpoint not found"
                i=$((i + 1)); continue
            fi
            local dev_idx=$((i % N_DEV))
            local dev="${DEVICES[$dev_idx]}"

            _wait_slot "$dev_idx" "exp($(basename "$exp_h5"))"   # ensure GPU is free
            log "$tag  thr=$thr  dev=$dev  &"

            local args=(
                --checkpoint    "$ckpt"
                --exp-h5        "$exp_h5"
                --threshold     "$thr"
                --min-hits      "$MIN_HITS"
                --min-strings   "$MIN_STRINGS"
                --batch-size    "$BATCH_SIZE"
                --device        "$dev"
                --output-dir    "$OUTPUT_DIR"
                --catalog       "$CATALOG"
            )
            [[ "${SKIP_DONE_PARTS}" == "true" ]] && args+=(--skip-done-parts)

            python inference_v2/nu_classifier/predict_exp.py "${args[@]}" &
            _slot_pids[$dev_idx]=$!
            i=$((i + 1))
        done
        _drain_slots "exp($(basename "$exp_h5"))"
    done
    log "=== exp step done ==="
}

# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

log "Starting batch predictions — step=$STEP  models=$N_MOD  gpus=${DEVICES[*]}"
case "$STEP" in
    npy)        run_npy ;;
    mc_merged)  run_mc_merged ;;
    mc_reco)    run_mc_reco ;;
    exp)        run_exp ;;
    all)        run_npy; run_mc_merged; run_mc_reco; run_exp ;;
    *)
        echo "Usage: $0 [npy|mc_merged|mc_reco|exp|all]" >&2
        exit 1
        ;;
esac

log "=== BATCH DONE: step=$STEP ==="
