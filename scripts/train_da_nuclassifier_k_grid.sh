#!/usr/bin/env bash
# K-ячейки полного обучения (tenth): K=10 внутренних шагов критика на батч.
# Запуск:  bash scripts/train_da_nuclassifier_k_grid.sh <cell>
# Ячейки: z128_k10_dlr1e-2_lam0p01_delay8_s42
#         z128_k10_dlr1e-2_lam0p03_delay8_s42
#         z128_k10_dlr1e-2_lam0p01_delay8_s42_inoise0
# Раскладка карт зашита в конфигах (data.device). CUDA_DEVICE_ORDER обязателен.
set -euo pipefail
cd /home/albert/Baikal2025
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

CELL="${1:?укажи ячейку: z128_k10_*}"
CONFIG="src/training/configs/classifier/da_nu_classifier_exp_full_E1_weights_tenth_${CELL}.yaml"
[ -f "$CONFIG" ] || { echo "нет конфига $CONFIG" >&2; exit 1; }
LOG="trained_models/classifier/logs/train_E1_weights_tenth_${CELL}.log"

nohup python -u src/training/da_nu_classifier_trainer_da_repair.py --config "$CONFIG" > "$LOG" 2>&1 &
echo "cell ${CELL}: PID $! ; log: $LOG"
