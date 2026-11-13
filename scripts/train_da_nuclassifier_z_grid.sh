#!/usr/bin/env bash
# Сетка «1/10 данных»: размерность z (d_model) × lr дискриминатора × λ-контроль.
#
# Запуск одной ячейки:  bash scripts/train_da_nuclassifier_z_grid.sh <cell>
# Ячейки: z128 | z128_dlr3e-3 | z8 | z8_dlr3e-3 | z3 | z3_dlr3e-3 | z3_lam0
#
# Раскладка по картам зашита в конфигах (PCI-индексы: 0 и 2 — RTX Ada по две
# ячейки, 3 — Ada, 1 — P100 под λ-контроль). CUDA_DEVICE_ORDER обязателен:
# без него у torch другой порядок карт (см. doc-память проекта).
set -euo pipefail
cd /home/albert/Baikal2025
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Лимит потоков на процесс: при 8-10 параллельных прогонах проба (18 logreg-фитов)
# и колейт данных иначе устраивают давку 10x64 потоков на CPU.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

CELL="${1:?укажи ячейку: z128|z128_dlr3e-3|z8|z8_dlr3e-3|z3|z3_dlr3e-3|z3_lam0}"
CONFIG="src/training/configs/classifier/da_nu_classifier_exp_full_E1_weights_tenth_${CELL}.yaml"
[ -f "$CONFIG" ] || { echo "нет конфига $CONFIG" >&2; exit 1; }
LOG="trained_models/classifier/logs/train_E1_weights_tenth_${CELL}.log"

nohup python -u src/training/da_nu_classifier_trainer_da_repair.py --config "$CONFIG" > "$LOG" 2>&1 &
echo "cell ${CELL}: PID $! ; log: $LOG"
