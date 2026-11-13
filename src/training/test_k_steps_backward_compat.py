"""Регрессия обратной совместимости для training.k_steps.

k_steps — число внутренних шагов критика на батч (n_critic). При k_steps=1 и при
отсутствии ключа обучение обязано быть прежним байт-в-байт: эталон снят на коде
ДО добавления k_steps (18.10.2026).

Тест гоняет тренер на урезанном наборе (2000 событий на домен, CPU, 2 эпохи,
сид 42) и сверяет метрики эпох и контрольные суммы параметров:
  1) конфиг без ключа k_steps — должен совпасть с эталоном;
  2) конфиг с k_steps: 1      — должен совпасть с эталоном;
  3) конфиг с k_steps: 2      — должен отличаться (ветка K жива) и не падать.

Запуск: python src/training/test_k_steps_backward_compat.py
"""
from __future__ import annotations

import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from src.training.da_nu_classifier_trainer_da_repair import (  # noqa: E402
    NuClassifierDomainAdaptationTrainer,
)

BASE_CFG = REPO / (
    "trained_models/classifier/"
    "261013_1823_da_nu_classifier_exp_full_E1_weights_tenth_z128_dlr1e-2_lam0p01_delay8_s42/"
    "da_config.yaml"
)

# Эталон снят до добавления k_steps (18.10.2026), режим capture.
# (epoch, classification_loss, domain_loss, total_loss, domain_accuracy,
#  domain_utilization, accuracy, auc, domain_auc, sum(p), sum(p^2))
REF: list = [
    (0, 0.16554836, 0.79543844, 0.9609868, 0.51308002, 0.91183036,
     0.50869305, 0.47441703, 0.51439794, 1264.445665, 2719.701399),
    (1, 0.15972851, 0.78076573, 0.94049424, 0.53486454, 0.91183036,
     0.50389688, 0.5043671, 0.54434841, 1265.109566, 2724.837857),
]


def build_cfg(k_steps):
    cfg = copy.deepcopy(yaml.safe_load(open(BASE_CFG)))
    cfg["data"]["device"] = "cpu"
    cfg["data"]["source_domain"]["max_events"] = 2000
    cfg["data"]["target_domain"]["max_events"] = 2000
    cfg["logging"]["output_dir"] = "/tmp/k_compat"
    cfg["logging"]["tensorboard"] = False
    cfg["logging"]["save_config"] = False
    cfg["training"]["epochs"] = 2
    if k_steps is not None:
        cfg["training"]["k_steps"] = k_steps
    return cfg


def run(k_steps=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    t = NuClassifierDomainAdaptationTrainer(build_cfg(k_steps))
    t.prepare_data()
    t.prepare_model()
    out = []
    for e in range(2):
        t.current_epoch = e
        m = t.train_epoch()
        s1 = float(sum(p.detach().double().sum() for p in t.da_model.parameters()))
        s2 = float(sum((p.detach().double() ** 2).sum() for p in t.da_model.parameters()))
        out.append((e,
                    round(float(m["classification_loss"]), 8),
                    round(float(m["domain_loss"]), 8),
                    round(float(m["total_loss"]), 8),
                    round(float(m["domain_accuracy"]), 8),
                    round(float(m["domain_utilization"]), 8),
                    round(float(m["accuracy"]), 8),
                    round(float(m["auc"]), 8),
                    round(float(m["domain_auc"]), 8),
                    round(s1, 6), round(s2, 6)))
    return out


def main() -> None:
    res_none = run(None)
    if not REF:
        res_none_2 = run(None)
        print("детерминизм (два одинаковых прогона совпали):", res_none == res_none_2)
        print("ЭТАЛОН (вставить в REF):")
        for r in res_none:
            print("   ", r)
        return
    res_k1 = run(1)
    assert res_none == REF, "конфиг без k_steps разошёлся с эталоном!"
    assert res_k1 == REF, "k_steps=1 разошёлся с эталоном!"
    print("K=1 и отсутствие ключа: совпадение с эталоном — OK")
    res_k2 = run(2)
    assert res_k2 != REF, "k_steps=2 не изменил обучение — ветка K не работает!"
    print("K=2: обучение отличается — ветка K жива — OK")


if __name__ == "__main__":
    main()
