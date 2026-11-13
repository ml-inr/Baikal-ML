#!/bin/bash
# K-эксперимент, большая λ (идея владельца 15.10): K внутренних шагов критика при
# λ=0.1 — одновременно «правильный» критик (градиент энкодера вблизи оптимума
# критика) и сильный эффект DA. Сравнение с готовыми K=1: onset_lam0p1_dlr0p01.csv
# (v3) и K=5/K=10 при λ=0.01 (текущие). λ=0.1, dlr=1e-2, 5000 батчей, чекпойнт эпохи 5.
# Карта: cuda:5 (A100, свободна; там же идёт K=5·λ=0.01). Запуск: bash run_grid_k_lam0p1.sh
set -u
cd /home/albert/Baikal2025
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
OUT=inference/nu_classifier/analysis/da_onset_geometry
for K in 3 5 10; do
  key="lam0p1_dlr0p01_k${K}"
  nohup python -u "$OUT/monitor_onset.py" --lam 0.1 --dlr 0.01 --k-steps "$K" \
      --n-batches 5000 --device cuda:5 > "$OUT/log_${key}.log" 2>&1 &
  echo "$key -> cuda:5  PID $!"
  sleep 45
done
echo "три руки λ=0.1 (K=3,5,10) запущены"
