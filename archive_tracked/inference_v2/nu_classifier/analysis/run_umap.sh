#!/usr/bin/env bash
# UMAP embedding analysis for a single nu-classifier checkpoint.
# Samples out-of-training MC events + exp events split by score bucket,
# fits 2D/3D UMAP projections, saves PNG + interactive HTML plots.
#
# Usage: bash inference_v2/nu_classifier/analysis/run_umap.sh [--reuse-coords]
# Run from project root.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

#!!!
#CHECKPOINT="experiments/numu/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32/best_da_model.pth"
CHECKPOINT="inference_v2/nu_classifier/exp_finetuning/finetuned_models/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model_finetuned/latest_finetuned_checkpoint.pth"

NPY_DIR="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
EXP_H5="data_manager/data/h5datasets/exp.h5"

#!!!
#PREDS_DIR="inference_v2/nu_classifier/preds"
PREDS_DIR="inference_v2/nu_classifier/exp_finetuning/finetuned_models/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model_finetuned"

CATALOG="data_manager/catalog_v2.duckdb"

THRESHOLD=0.8
N_PER_CLASS=10000   # out-of-training MC events per particle type
N_EXP_HIGH=5000     # exp events with score > threshold
N_EXP_LOW=5000      # exp events with score <= threshold

MIN_HITS=8
MIN_STRINGS=2
BATCH_SIZE=512
DEVICE="cuda:1"

UMAP_NEIGHBORS=15
UMAP_MIN_DIST=0.1
SEED=42

# ── Launch ────────────────────────────────────────────────────────────────────

python inference_v2/nu_classifier/analysis/run_umap.py \
    --checkpoint    "$CHECKPOINT"   \
    --npy-dir       "$NPY_DIR"      \
    --mc-h5         "$MC_H5"        \
    --exp-h5        "$EXP_H5"       \
    --preds-dir     "$PREDS_DIR"    \
    --catalog       "$CATALOG"      \
    --threshold     "$THRESHOLD"    \
    --n-per-class   "$N_PER_CLASS"  \
    --n-exp-high    "$N_EXP_HIGH"   \
    --n-exp-low     "$N_EXP_LOW"    \
    --min-hits      "$MIN_HITS"     \
    --min-strings   "$MIN_STRINGS"  \
    --batch-size    "$BATCH_SIZE"   \
    --device        "$DEVICE"       \
    --umap-neighbors "$UMAP_NEIGHBORS" \
    --umap-min-dist "$UMAP_MIN_DIST" \
    --seed          "$SEED"         \
    "$@"
# Pass-through: --reuse-coords skips UMAP refit if umap_coords.npz exists.
