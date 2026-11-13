#!/usr/bin/env bash
# DA-repair control run: FIXED_sn256 settings + the discriminator-degeneration fix.
#
# Why the fix exists: the FIXED_sn256 discriminator ended unable to separate the
# domains at all (val AUC ~0.52) while a fresh probe on the same frozen features
# separated them at 0.70-0.78 — the adversarial game had left it uninformative.
# Diagnosis and numbers: inference/nu_classifier/analysis/da_band_probe/README.md
# (sections "Результат 3" and "Результат 4").
#
# The run adds an independent invariance probe to the epoch metrics; whether it
# is the repair or the game itself that keeps the discriminator alive is read
# from `probe_auc_*` against `val_domain_auc` in the history CSV.
set -euo pipefail
cd /home/albert/Baikal2025

# GPU: the session's device is cuda:0 (config `data.device`), pinned by PCI bus id.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

CONFIG="src/training/configs/classifier/da_nu_classifier_exp_full_E1_DArepair.yaml"
LOG="trained_models/classifier/logs/train_E1_DArepair.log"

# The DA-repair variant of the trainer; `da_nu_classifier_trainer.py` stays the
# untouched baseline that trained the FIXED_sn256 line.
nohup python -u src/training/da_nu_classifier_trainer_da_repair.py --config "$CONFIG" > "$LOG" 2>&1 &
echo "started PID $! ; log: $LOG"
