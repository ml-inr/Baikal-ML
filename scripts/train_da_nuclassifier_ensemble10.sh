#!/usr/bin/env bash
# Пул из 10 доменных дискриминаторов: остаются ли члены пула разными в живой игре.
#
# Зачем: один критик читает одно направление в пространстве фич, и энкодер может
# удовлетворить его переупаковкой признаков вместо удаления домена. Ансамбль
# должен закрыть этот путь — но только если его члены продолжают различаться.
# На замороженных фичах они сходятся к одной функции; вопрос в том, удерживает ли
# их друг от друга живая игра. Прогон каждую эпоху пишет в историю:
#   pool_cos_mean/min/max — косинусы между читающими направлениями членов;
#   pool_disagree         — доля событий, где члены расходятся в метке домена;
#   pool_auc_min/max      — разброс доменных AUC по членам;
#   pool_sel_share        — как часто агрегат "min" выбирал одного и того же члена.
#
# GPU: конфиг называет cuda:1, и это ровно Tesla P100 №1 в nvidia-smi — но только
# потому, что ниже выставлен CUDA_DEVICE_ORDER=PCI_BUS_ID. Без него PyTorch
# сортирует карты «сначала быстрые», и cuda:1 — это уже другая, занятая карта
# (проверено 2026-10-02: без переменной P100 оказался cuda:5).
#
# Измерено на P100 (2026-10-02): 0.27 с/батч -> ~70 мин чистого обучения на эпоху,
# 4 ГБ видеопамяти из 16.
set -euo pipefail
cd /home/albert/Baikal2025

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Переменная длина событий -> аллокатор фрагментирует память: на P100
# (16 ГБ) запас уходил за 20 минут. expandable_segments это лечит.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="src/training/configs/classifier/da_nu_classifier_exp_full_E1_ensemble10.yaml"
LOG="trained_models/classifier/logs/train_E1_pool10.log"

# Вариант тренера с пулом; da_repair и baseline не трогаются.
nohup python -u src/training/da_nu_classifier_trainer_ensemble.py --config "$CONFIG" > "$LOG" 2>&1 &
echo "started PID $! ; log: $LOG"
