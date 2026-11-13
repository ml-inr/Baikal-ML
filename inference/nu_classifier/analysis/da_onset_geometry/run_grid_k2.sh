#!/bin/bash
# Вторая волна K-оптимизации (добро владельца 16.10): границы по K и λ + длинный горизонт.
#   Короткие (5000 батчей) на cuda:0 (RTX 5000 Ada, свободна):
#     K=20·λ0.01·dlr1e-2; λ0.03·K{5,10}·dlr1e-2; K=10·λ0.01·dlr1e-3; λ0.1·K=10
#     (последняя — перезапись CSV уже обученной руки с новой колонкой inner_val_acc).
#   Длинный горизонт (20000 батчей ≈ 20 эпох) на cuda:5 (A100, свободна):
#     K=10·λ0.01·dlr1e-2 → сторожатся как onset_lam0p01_dlr0p01_k10_n20000.csv.
# Монитор с этой волны логирует inner_val_acc — точность критика по внутренним
# шагам k (диагностика «лечимости» критика; элементы = после 0..K-1 шагов + финал).
# cuda:1-4 заняты чужими задачами. Запуск: bash run_grid_k2.sh
set -u
cd /home/albert/Baikal2025
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
OUT=inference/nu_classifier/analysis/da_onset_geometry
# длинный горизонт — первым, на A100
nohup python -u "$OUT/monitor_onset.py" --lam 0.01 --dlr 0.01 --k-steps 10 \
    --n-batches 20000 --device cuda:5 > "$OUT/log_lam0p01_dlr0p01_k10_n20000.log" 2>&1 &
echo "длинный K=10·λ0.01 ×20000 -> cuda:5  PID $!"
sleep 60
for cell in "0.01 0.01 20" "0.03 0.01 5" "0.03 0.01 10" "0.01 0.001 10" "0.1 0.01 10"; do
  read -r lam dlr k <<< "$cell"
  key=$(python3 -c "print((f'lam{$lam:g}_dlr{$dlr:g}'+f'_k{$k}').replace('.','p'))")
  nohup python -u "$OUT/monitor_onset.py" --lam "$lam" --dlr "$dlr" --k-steps "$k" \
      --n-batches 5000 --device cuda:0 > "$OUT/log_${key}.log" 2>&1 &
  echo "$key -> cuda:0  PID $!"
  sleep 45
done
echo "вторая волна запущена (1 длинная + 5 коротких)"
