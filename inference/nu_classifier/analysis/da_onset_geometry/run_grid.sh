#!/bin/bash
# Партия онсет-диагностики v3: 16 рук (λ×dlr + контроль) × 5000 батчей,
# по процессу на руку. Карты: cuda:2/3 (Ada), cuda:5 (A100), cuda:1 (P100);
# cuda:0 и cuda:4 заняты чужими задачами. Логи — log_<рука>.log рядом с CSV.
# Запуск: bash run_grid.sh
set -u
cd /home/albert/Baikal2025
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
OUT=inference/nu_classifier/analysis/da_onset_geometry
N_BATCHES=${N_BATCHES:-5000}
DEVICES=(cuda:2 cuda:3 cuda:5 cuda:1)
CELLS=()
for lam in 0.001 0.003 0.01 0.03 0.1; do
  for dlr in 1e-3 1e-2 3e-2; do CELLS+=("$lam $dlr"); done
done
CELLS+=("0.0 1e-2")   # контроль
i=0
for cell in "${CELLS[@]}"; do
  read -r lam dlr <<< "$cell"
  dev=${DEVICES[$((i % ${#DEVICES[@]}))]}
  key=$(python3 -c "print(f'lam{$lam:g}_dlr{$dlr:g}'.replace('.','p'))")
  nohup python -u "$OUT/monitor_onset.py" --lam "$lam" --dlr "$dlr" \
      --n-batches "$N_BATCHES" --device "$dev" > "$OUT/log_${key}.log" 2>&1 &
  echo "  $key -> $dev  PID $!"
  i=$((i+1))
  sleep 60
done
echo "все 16 рук запущены ($N_BATCHES батчей на руку)"
