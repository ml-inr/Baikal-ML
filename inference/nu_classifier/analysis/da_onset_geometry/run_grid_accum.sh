#!/bin/bash
# Накопление градиентов игры (идея владельца 16.10, опрос пройден):
# λ=0.01, dlr=1e-2, K=10 (чемпион) × accum N=5 и N=10, 5000 батчей.
# Семантика: критик и λ-градиент энкодера копятся N батчей и применяются раз в N
# (усреднение /N); классовая ветка — каждый батч как обычно; внутренние K-1
# адаптационные шаги критика — немедленные. Сравнение с чемпионом
# (onset_lam0p01_dlr0p01_k10.csv: fresh_end 0.729, dom_acc 0.635).
# Ждёт освобождения cuda:0 (волна-2 ~3 ч), затем запускает обе руки.
set -u
cd /home/albert/Baikal2025
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
OUT=inference/nu_classifier/analysis/da_onset_geometry
while pgrep -f "monitor_onset[.]py.*cuda:0" >/dev/null 2>&1; do sleep 300; done
echo "cuda:0 свободна, запускаю accum-руки: $(date)"
for N in 5 10; do
  nohup python -u "$OUT/monitor_onset.py" --lam 0.01 --dlr 0.01 --k-steps 10 --accum "$N" \
      --n-batches 5000 --device cuda:0 > "$OUT/log_lam0p01_dlr0p01_k10_a${N}.log" 2>&1 &
  echo "K=10·accum=$N -> cuda:0  PID $!"
  sleep 45
done
echo "accum-руки запущены"
