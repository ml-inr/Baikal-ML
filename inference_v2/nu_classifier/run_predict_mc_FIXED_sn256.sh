#!/usr/bin/env bash
# Full-statistics nu-classifier scoring of mc_merged with the FIXED model.
#
# Differences from run_predict_mc.sh, all deliberate:
#   * PROBS_H5 is set, so the hit selection comes from the stored probabilities
#     (written at sig-noise batch 256) instead of being recomputed on the fly at the
#     classifier's batch size. That coupling is what made earlier runs irreproducible —
#     see doc/sig_noise_batch_size.md.
#   * no MAX_EVENTS_PER_PTYPE cap: every part that has probabilities is scored
#     (10,100 muatm + 2,000 nuatm + 400 nue2 of the 22,404 parts in the h5).
#   * embeddings are stored, so representation-space analysis needs no second pass.
#
# Resumable: --skip-done-parts means an interrupted run continues where it stopped.
# Run under nohup — a run of this length must survive the terminal that started it.

set -euo pipefail
cd /home/albert/Baikal2025

CHECKPOINT="experiments/numu/260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256/best_da_model.pth"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
PROBS_H5="data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5"
SOURCE="mc_merged"

# Loose event cut at scoring time; n_sn_hits / n_sn_strings are stored per event, so the
# h8s3 selection is applied later in SQL. Keeps the run comparable with earlier ones.
THRESHOLD=0.8
MIN_HITS=5
MIN_STRINGS=0

BATCH_SIZE=1024          # classifier batch — speed only, the model is batch-independent
MIN_BATCH_EVENTS=32768
DEVICE="cuda:0"
OUTPUT_DIR="inference_v2/nu_classifier/preds"
CATALOG="data_manager/catalog_v2.duckdb"

python -u inference_v2/nu_classifier/predict_mc.py \
    --checkpoint       "$CHECKPOINT" \
    --mc-h5            "$MC_H5" \
    --probs-h5         "$PROBS_H5" \
    --source           "$SOURCE" \
    --threshold        "$THRESHOLD" \
    --min-hits         "$MIN_HITS" \
    --min-strings      "$MIN_STRINGS" \
    --batch-size       "$BATCH_SIZE" \
    --min-batch-events "$MIN_BATCH_EVENTS" \
    --device           "$DEVICE" \
    --output-dir       "$OUTPUT_DIR" \
    --catalog          "$CATALOG" \
    --skip-done-parts \
    --save-embeddings
