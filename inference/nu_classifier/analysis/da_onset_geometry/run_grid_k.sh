#!/bin/bash
# K-эксперимент (опрос владельца 15.10): K внутренних шагов критика на один шаг
# энкодера при λ=0.01, dlr=1e-2. Дизайн: K шагов того же вида; градиент энкодера —
# один раз, после последнего шага («догнавший» критик). Сравнение с готовой рукой
# K=1: onset_lam0p01_dlr0p01.csv из партии v3. 5000 батчей на руку.
# Запуск: bash run_grid_k.sh   (логи — log_lam0p01_dlr0p01_k{5,10}.log)
set -u
cd /home/albert/Baikal2025
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
OUT=inference/nu_classifier/analysis/da_onset_geometry
nohup python -u "$OUT/monitor_onset.py" --lam 0.01 --dlr 0.01 --k-steps 5 \
    --n-batches 5000 --device cuda:2 > "$OUT/log_lam0p01_dlr0p01_k5.log" 2>&1 &
echo "K=5  -> cuda:2  PID $!"
sleep 60
nohup python -u "$OUT/monitor_onset.py" --lam 0.01 --dlr 0.01 --k-steps 10 \
    --n-batches 5000 --device cuda:3 > "$OUT/log_lam0p01_dlr0p01_k10.log" 2>&1 &
echo "K=10 -> cuda:3  PID $!"
