#!/usr/bin/env python
"""Может ли критик обучиться на замороженных фичах — и как быстро игра его снова слепит.

Мотив: `Val/Domain_AUC` сидит на ~0.54, тогда как свежий линейный пробинг на тех же
фичах даёт ~0.72. Две гипотезы:

* **H1 (погоня за движущейся целью)** — на неподвижных фичах критик обучение берёт,
  но в игре энкодер уводит цель быстрее, чем критик успевает;
* **H2 (критик не берёт и снимок)** — ёмкость/оптимизация/instance_noise не дают
  обучиться даже на замороженном энкодере.

Фаза A — это в точности процедура `_refresh_discriminator` (энкодер заморожен,
критик дообучается своим оптимизатором на свежих батчах), но с мерой AUC на
фиксированном eval-наборе после каждой порции шагов. Контроль — логистический
пробинг на тех же замороженных фичах: планка, которую критик обязан взять, если
верна H1.

Фаза B — настоящий шаг обучения (классификация + доменный лосс с GRL, все три
оптимизатора) после refresh'а: AUC критика мерится на свежем eval каждые 100 шагов,
то есть видно, за сколько шагов игра снова ослепляет критика.

Запуск: python inference/nu_classifier/analysis/da_band_probe/critic_fit_and_chase.py <run_dir> [cuda:5]
"""
from __future__ import annotations

import copy
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.training.da_nu_classifier_trainer_da_repair import (  # noqa: E402
    NuClassifierDomainAdaptationTrainer,
)

LAMBDA = 0.01          # λ в конце обучения (max_lambda конфига)
REFRESH_STEPS = 300    # размер починки из конфига
GAME_STEPS = 1500      # сколько шагов игры прогоняем в фазе B
MEASURE_EVERY = 100


def build_trainer(run: str, device: str) -> NuClassifierDomainAdaptationTrainer:
    run_dir = ROOT / "trained_models/classifier" / run
    config = yaml.safe_load(open(run_dir / "da_config.yaml"))
    config["data"]["device"] = device
    config["logging"]["tensorboard"] = False
    config["logging"]["save_model_summary"] = False
    config["logging"]["output_dir"] = "/tmp/critic_fit_logs"
    trainer = NuClassifierDomainAdaptationTrainer(config)
    trainer.prepare_data()
    trainer.prepare_model()
    checkpoint = torch.load(run_dir / "best_da_model.pth", map_location="cpu",
                            weights_only=False)
    trainer.da_model.base_model.load_state_dict(checkpoint["base_model_state_dict"])
    trainer.da_model.domain_discriminator.load_state_dict(
        checkpoint["domain_discriminator_state_dict"])
    trainer.da_model.to(trainer.device)
    print(f"[{run}] чекпоинт эпохи {checkpoint.get('epoch')}", flush=True)
    return trainer


def move_batch(trainer, batch):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(trainer.device)
    return batch


@torch.no_grad()
def frozen_eval(trainer, n_pairs: int, seed: int, offset: int = 0):
    """Фичи текущего энкодера на фиксированно перевёрнутых парах val-батчей.

    `offset` пропускает первые батчи загрузчика, чтобы обучающая выборка
    контрольного пробинга не пересекалась с eval-набором.
    """
    torch.manual_seed(seed)
    model = trainer.da_model
    was_training = model.training
    model.eval()
    source_iter = iter(trainer.source_val_loader)
    target_iter = iter(trainer.target_val_loader)
    for _ in range(offset):
        next(source_iter), next(target_iter)
    features = {"src": [], "tgt": []}
    for _ in range(n_pairs):
        source_batch = move_batch(trainer, next(source_iter))
        target_batch = move_batch(trainer, next(target_iter))
        src_dom, tgt_dom, *_ = trainer._build_domain_batches(source_batch, target_batch)
        features["src"].append(model.get_features(src_dom))
        features["tgt"].append(model.get_features(tgt_dom))
    if was_training:
        model.train()
    return {"src": torch.cat(features["src"]), "tgt": torch.cat(features["tgt"])}


@torch.no_grad()
def critic_auc(trainer, features) -> float:
    """AUC дискриминатора на замороженных фичах; eval-режим, как в validate_epoch."""
    model = trainer.da_model
    was_training = model.training
    model.eval()
    logits = torch.cat([model.domain_discriminator(features["src"]),
                        model.domain_discriminator(features["tgt"])]).cpu().numpy().ravel()
    if was_training:
        model.train()
    labels = np.r_[np.zeros(len(features["src"])), np.ones(len(features["tgt"]))]
    return roc_auc_score(labels, logits)


def logistic_probe_auc(train_features, eval_features) -> float:
    """Свежий логистический пробинг: фит на одном снимке, AUC на другом."""
    to_xy = lambda f: (np.r_[f["src"].cpu().numpy(), f["tgt"].cpu().numpy()],
                       np.r_[np.zeros(len(f["src"])), np.ones(len(f["tgt"]))])
    x_train, y_train = to_xy(train_features)
    x_eval, y_eval = to_xy(eval_features)
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(x_train), y_train)
    return roc_auc_score(y_eval, clf.decision_function(scaler.transform(x_eval)))


def game_step(trainer, source_batch, target_batch) -> None:
    """Один шаг обучения — тело цикла train_epoch, слово в слово."""
    model = trainer.da_model
    source_batch = move_batch(trainer, source_batch)
    target_batch = move_batch(trainer, target_batch)

    source_class_logits, _ = model.classification_forward(source_batch)
    classification_loss = trainer._calculate_classification_loss(
        source_class_logits, source_batch["labels"].float(),
        theta=source_batch.get("theta"), sample_weight=source_batch.get("weights"))

    trainer.optimizer_feature.zero_grad()
    trainer.optimizer_classifier.zero_grad()
    trainer.optimizer_discriminator.zero_grad()
    classification_loss.backward()

    src_dom, tgt_dom, n_source, n_target, domain_bs = trainer._build_domain_batches(
        source_batch, target_batch)
    src_logits, _ = model.domain_forward(src_dom)
    tgt_logits, _ = model.domain_forward(tgt_dom)
    domain_loss = (
        nn.functional.binary_cross_entropy_with_logits(
            src_logits, torch.zeros(domain_bs, 1, device=trainer.device))
        + nn.functional.binary_cross_entropy_with_logits(
            tgt_logits, torch.ones(domain_bs, 1, device=trainer.device))
    ) / 2
    utilization = (domain_bs / n_source + domain_bs / n_target) / 2
    (domain_loss / utilization).backward()

    trainer.optimizer_feature.step()
    trainer.optimizer_classifier.step()
    trainer.optimizer_discriminator.step()


def main() -> None:
    run = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:5"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    trainer = build_trainer(run, device)
    model = trainer.da_model
    if len(sys.argv) > 3:                      # контроль: шум входа критика
        model.domain_discriminator.instance_noise = float(sys.argv[3])
        print(f"[{run}] instance_noise переопределён: "
              f"{model.domain_discriminator.instance_noise}", flush=True)
    if len(sys.argv) > 4:                      # контроль: lr дискриминатора
        for group in trainer.optimizer_discriminator.param_groups:
            group["lr"] = float(sys.argv[4])
        print(f"[{run}] lr дискриминатора переопределён: "
              f"{trainer.optimizer_discriminator.param_groups[0]['lr']}", flush=True)
    initial_discriminator = copy.deepcopy(model.domain_discriminator.state_dict())
    model.eval()

    eval_set = frozen_eval(trainer, n_pairs=4, seed=100)
    train_snapshot = frozen_eval(trainer, n_pairs=8, seed=200, offset=4)
    print(f"\n[{run}] старт: критик на замороженном eval — AUC "
          f"{critic_auc(trainer, eval_set):.4f}", flush=True)
    print(f"[{run}] контрольный логистический пробинг — AUC "
          f"{logistic_probe_auc(train_snapshot, eval_set):.4f}", flush=True)

    # -- Фаза A: refresh с замерами -----------------------------------------
    print(f"\n[{run}] фаза A — замороженный энкодер, дообучение критика "
          f"(+instance_noise {model.domain_discriminator.instance_noise})")
    total = 0
    for chunk in (50, 50, 100, 200, 400, 800):
        trainer._refresh_discriminator(steps=chunk)
        total += chunk
        print(f"    шагов {total:>4d}: AUC {critic_auc(trainer, eval_set):.4f}",
              flush=True)

    # -- Фаза B: игра после refresh'а ---------------------------------------
    model.domain_discriminator.load_state_dict(initial_discriminator)
    trainer._refresh_discriminator(steps=REFRESH_STEPS)
    model.set_lambda(LAMBDA)
    game_eval = frozen_eval(trainer, n_pairs=4, seed=100)
    print(f"\n[{run}] фаза B — игра, λ={LAMBDA}; после refresh {REFRESH_STEPS} шагов: "
          f"AUC {critic_auc(trainer, game_eval):.4f}", flush=True)
    source_iter = itertools.cycle(trainer.source_train_loader)
    target_iter = itertools.cycle(trainer.target_train_loader)
    for step in range(1, GAME_STEPS + 1):
        trainer.da_model.set_lambda(LAMBDA)
        game_step(trainer, next(source_iter), next(target_iter))
        if step % MEASURE_EVERY == 0:
            game_eval = frozen_eval(trainer, n_pairs=4, seed=100)
            print(f"    шагов игры {step:>4d}: критик AUC "
                  f"{critic_auc(trainer, game_eval):.4f}", flush=True)
    print(f"\n[{run}] всего {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
