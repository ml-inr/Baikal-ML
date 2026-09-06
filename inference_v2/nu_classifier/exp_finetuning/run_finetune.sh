#!/usr/bin/env bash
# Supervised fine-tuning: MC NPY + exp-background NPY, no DANN.
#
# Prerequisites:
#   1. Run run_build_exp_bg.sh to create the exp-background dataset.
#   2. Set paths below to match your experiment.
#
# Usage: bash inference_v2/nu_classifier/exp_finetuning/run_finetune.sh
# Run from project root.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT="260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model"

PRETRAINED="experiments/numu/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32/best_da_model.pth"
MC_NPY_DIR="data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
EXP_BG_NPY_DIR="inference_v2/nu_classifier/exp_finetuning/exp_bg_datasets/${CHECKPOINT}_lt0p1"
OUTPUT_DIR="inference_v2/nu_classifier/exp_finetuning/finetuned_models"
EXP_NAME="${CHECKPOINT}_finetuned"
DEVICE="cuda:2"

CONFIG_PATH="inference_v2/nu_classifier/exp_finetuning/finetune.yaml"

# ── Write config ──────────────────────────────────────────────────────────────

cat > "$CONFIG_PATH" << YAML
pretrained_checkpoint: ${PRETRAINED}
mc_npy_dir:   ${MC_NPY_DIR}
exp_bg_npy_dir: ${EXP_BG_NPY_DIR}
output_dir:   ${OUTPUT_DIR}
experiment_name: ${EXP_NAME}
device: ${DEVICE}

seed: 42
max_hits: 500
mc_train_split: 0.9
bg_train_split: 0.8

epochs: 10
batch_size: 512
validate_every: 1
save_every: 1
save_epoch_checkpoints: true
log_every: 50

learning_rate: 5e-5
weight_decay:  0.01
scheduler: plateau
scheduler_params:
  mode: min
  patience: 3
  factor: 0.5
  min_lr: 1.0e-7

classification_loss: focal
focal_gamma: 2.0
class_weights: true

early_stopping:
  enabled: true
  monitor: loss
  mode: min
  patience: 8
  min_delta: 1.0e-4

tensorboard: true
save_config: true

dataloader:
  num_workers: 0
  pin_memory: false
  augmentation:
    rotation_enabled: true
    noise_std: [0.05, 5.0, 1.0, 1.0, 2.0]
YAML

echo "Config written to ${CONFIG_PATH}"

# ── Launch ────────────────────────────────────────────────────────────────────

python inference_v2/nu_classifier/exp_finetuning/run_finetune.py \
    --config "$CONFIG_PATH" \
    "$@"
