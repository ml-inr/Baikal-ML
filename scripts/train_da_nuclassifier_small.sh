#!/usr/bin/env bash
# Маленькая модель (2 слоя, d_model 64, z=64) с критиком как в эталоне (MLP [32,16]+LN)
# — проверка предложения владельца об упрощении архитектуры. Единственная переменная
# относительно эталона 261002_0535 (DArepair) — размер энкодера; веса не используются.
# Предыдущий запуск (261010_1642, линейный критик) — отрицательный результат, см.
# inference/nu_classifier/analysis/small_model_critic_diag/.
#
# Тренер — тот же вариант с починкой DA (da_nu_classifier_trainer_da_repair.py):
# baseline и ensemble-вариант не трогаются.
#
# GPU: конфиг называет cuda:1 — это Tesla P100 №1 в nvidia-smi, но только при
# CUDA_DEVICE_ORDER=PCI_BUS_ID (без неё порядок карт другой).
set -euo pipefail
cd /home/albert/Baikal2025
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # против фрагментации на 16 ГБ

CONFIG="src/training/configs/classifier/da_nu_classifier_exp_full_E1_small_d64_l2.yaml"
LOG="trained_models/classifier/logs/train_E1_small_d64_l2.log"
nohup python -u src/training/da_nu_classifier_trainer_da_repair.py --config "$CONFIG" > "$LOG" 2>&1 &
echo "started PID $! ; log: $LOG"
