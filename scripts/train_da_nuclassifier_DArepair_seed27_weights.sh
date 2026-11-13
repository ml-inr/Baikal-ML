#!/usr/bin/env bash
# Two ablations of the DA-repair run (261002_0535), launched together on the two
# GPUs the session was given (cuda:2 and cuda:3):
#
#   1. seed27 — identical recipe, seed 27 instead of 42. Measures how much of the
#      DArepair-vs-FIXED probe trajectory is seed noise.
#   2. weights — identical recipe and seed 42, plus per-event weights that flatten
#      ln E and cos(theta) within each class (clipped at 20x the mean; built by
#      data_manager/nu_classifier_ds_builder/add_weights.py). Measures whether the
#      exp excess survives removing the "bright / near-horizon ⇒ neutrino" prior.
#
# Note on the weights: full flattening of a 5-decade energy spectrum is not
# achievable at a usable ESS — the weights are clipped, and the realized
# ESS/flatness are recorded in the dataset's weights_info.json (ESS ~3-5% for the
# MC classes, i.e. the run supervises mostly the tail-enriched part of the sample).
set -euo pipefail
cd /home/albert/Baikal2025

# Device ids in the configs are cuda:2 / cuda:3; PCI_BUS_ID makes them mean the
# same devices nvidia-smi numbers. CUDA_VISIBLE_DEVICES is deliberately NOT set,
# so the ids the configs carry are the ones the session named.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# The copy of the trainer carrying the repair and the invariance probe;
# da_nu_classifier_trainer.py stays the untouched baseline.
TRAINER=src/training/da_nu_classifier_trainer_da_repair.py

nohup python -u "$TRAINER" \
  --config src/training/configs/classifier/da_nu_classifier_exp_full_E1_DArepair_seed27.yaml \
  > trained_models/classifier/logs/train_E1_DArepair_seed27.log 2>&1 &
echo "seed27  started PID $! ; log: trained_models/classifier/logs/train_E1_DArepair_seed27.log"

nohup python -u "$TRAINER" \
  --config src/training/configs/classifier/da_nu_classifier_exp_full_E1_DArepair_weights.yaml \
  > trained_models/classifier/logs/train_E1_DArepair_weights.log 2>&1 &
echo "weights started PID $! ; log: trained_models/classifier/logs/train_E1_DArepair_weights.log"
