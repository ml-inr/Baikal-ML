"""Разложение разницы auc_game_all против auc_all: ν и z-флип по отдельности.

Задача игры отличается от исторической пробы (μ-vs-exp) двумя вещами сразу:
на стороне MC добавлены нейтрино (группы 4,5) и включён z-флип-протокол.
Здесь каждый эффект измеряется отдельно, свежим logreg (3-fold, как в пробе),
на одном и том же чекпойнте и одних и тех же фичах образца пробы.

Срезы:
  A  μ vs exp, чистое          (историческая задача)
  B  μ vs exp, обе стороны зеркальны
  C  ν vs exp, чистое
  D  ν vs exp, обе стороны зеркальны
  E  MC(μ+ν) vs exp, чистое    (эффект добавления ν)
  F  MC vs exp в протоколе игры (половинный флип источника, зеркало цели) — сверка
     с probe_auc_game_all из лога

Запуск: python .../decompose_game_vs_hist.py [--checkpoint PATH]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
logging.basicConfig(level=logging.ERROR)

from src.training.da_nu_classifier_trainer_da_repair import (  # noqa: E402
    NuClassifierDomainAdaptationTrainer,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RUN = "trained_models/classifier/261010_1755_da_nu_classifier_exp_full_E1_small_d64_l2"
EMB = Path(__file__).parent / "probe_emb_latest.npz"


def cv_auc(X, y):
    aucs = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(aucs))


def oof_scores(X, y):
    """Out-of-fold decision values (3-fold, тот же протокол, что cv_auc)."""
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        s[te] = clf.decision_function(sc.transform(X[te]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=f"{RUN}/latest_da_checkpoint.pth")
    args = ap.parse_args()

    if EMB.exists():
        d = np.load(EMB)
        emb, emb_f, group, band = d["emb"], d["emb_f"], d["group"], d["band"]
        print("эмбеддинги: кэш")
    else:
        cfg = yaml.safe_load(open(f"{RUN}/da_config.yaml"))
        cfg["data"]["device"] = "cpu"
        cfg["data"]["source_domain"]["max_events"] = 20000
        cfg["data"]["target_domain"]["max_events"] = 20000
        cfg["logging"]["output_dir"] = "/tmp/diag_small_out"
        cfg["logging"]["tensorboard"] = False
        with NuClassifierDomainAdaptationTrainer(cfg) as t:
            t.prepare_data()
            t.prepare_model()
            ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            t.da_model.base_model.load_state_dict(ck["base_model_state_dict"])
            t.da_model.eval()
            sample = t._probe_sample()
            emb = t._probe_embeddings(sample["hits"], sample["offsets"])
            emb_f = t._probe_embeddings(sample["hits"], sample["offsets"],
                                        flip_z=True)
            group, band = sample["group"], sample["band"]
        np.savez_compressed(EMB, emb=emb, emb_f=emb_f, group=group, band=band)
        print(f"эмбеддинги: посчитаны с {args.checkpoint}")

    mu = np.flatnonzero(group <= 1)
    nu = np.flatnonzero(group >= 4)
    exp = np.flatnonzero((group >= 2) & (group <= 3))

    def task(idx_a, idx_b, flipped=False):
        E = emb_f if flipped else emb
        X = np.concatenate([E[idx_a], E[idx_b]])
        y = np.concatenate([np.zeros(len(idx_a)), np.ones(len(idx_b))]).astype(int)
        return cv_auc(X, y)

    A = task(mu, exp, False)
    B = task(mu, exp, True)
    C = task(nu, exp, False)
    D = task(nu, exp, True)
    E = task(np.concatenate([mu, nu]), exp, False)

    rng = np.random.default_rng(42)
    mc = np.concatenate([mu, nu])
    flip_sel = rng.choice(mc, size=len(mc) // 2, replace=False)
    is_fl = np.zeros(len(group), bool)
    is_fl[flip_sel] = True
    src_rows = np.where(is_fl[mc, None], emb_f[mc], emb[mc])
    tgt_rows = np.concatenate([emb[exp], emb_f[exp]], axis=0)
    F = cv_auc(np.concatenate([src_rows, tgt_rows]),
               np.concatenate([np.zeros(len(src_rows)), np.ones(len(tgt_rows))]).astype(int))

    print(f"μ({len(mu)}) vs exp({len(exp)}), чистое  (A): {A:.4f}")
    print(f"μ vs exp, зеркало                      (B): {B:.4f}")
    print(f"ν({len(nu)}) vs exp, чистое            (C): {C:.4f}")
    print(f"ν vs exp, зеркало                      (D): {D:.4f}")
    print(f"MC(μ+ν) vs exp, чистое                 (E): {E:.4f}")
    print(f"MC vs exp, протокол игры               (F): {F:.4f}  "
          f"(сверка с auc_game_all в логе)")
    print(f"--- вклад добавления ν (E-A): {E - A:+.4f}; "
          f"вклад флипа (F-E): {F - E:+.4f}")

    # --- локализация цены флип-протокола: кто смешан, источник или цель ---
    tgt_pool = np.concatenate([emb[exp], emb_f[exp]], axis=0)
    src_clean = emb[mc]
    y0, y1 = np.zeros(len(mc), int), np.ones(len(tgt_pool), int)

    G1 = cv_auc(np.concatenate([src_clean, emb[exp]]), np.concatenate([y0, np.ones(len(exp), int)]))
    G2 = cv_auc(np.concatenate([src_rows, emb[exp]]), np.concatenate([y0, np.ones(len(exp), int)]))
    G3 = cv_auc(np.concatenate([src_clean, tgt_pool]), np.concatenate([y0, y1]))
    G4 = cv_auc(np.concatenate([src_rows, tgt_pool]), np.concatenate([y0, y1]))
    print(f"G1 MC чистый vs exp чистый:              {G1:.4f}  (= E)")
    print(f"G2 MC СМЕШАН vs exp чистый:              {G2:.4f}")
    print(f"G3 MC чистый vs exp СМЕШАН(удвоен):      {G3:.4f}")
    print(f"G4 MC смешан vs exp смешан:              {G4:.4f}  (= F)")
    print(f"--- цена смешения источника (G2-G1): {G2-G1:+.4f}; "
          f"цели (G3-G1): {G3-G1:+.4f}")

    # где сидит потеря в G4: разбор по частям на out-of-fold скорах
    X4 = np.concatenate([src_rows, tgt_pool])
    y4 = np.concatenate([y0, y1])
    sc = oof_scores(X4, y4)
    n_mc, n_exp = len(mc), len(exp)
    mu_s, nu_s = slice(0, len(mu)), slice(len(mu), n_mc)
    exp_clean = slice(n_mc, n_mc + n_exp)
    exp_flip = slice(n_mc + n_exp, n_mc + 2 * n_exp)
    auc_sub = lambda a, b: roc_auc_score(
        np.concatenate([np.zeros(len(a)), np.ones(len(b))]),
        np.concatenate([sc[a], sc[b]]))
    i_mu = np.arange(n_mc)[slice(0, len(mu))]
    i_nu = np.arange(n_mc)[slice(len(mu), n_mc)]
    i_ec = np.arange(n_mc, n_mc + n_exp)
    i_ef = np.arange(n_mc + n_exp, n_mc + 2 * n_exp)
    mc_flip_mask = is_fl[mc]
    i_mu_c, i_mu_f = i_mu[~mc_flip_mask[:len(mu)]], i_mu[mc_flip_mask[:len(mu)]]
    i_nu_c, i_nu_f = i_nu[~mc_flip_mask[len(mu):]], i_nu[mc_flip_mask[len(mu):]]
    print("--- разбор G4: AUC подмножеств тем же направлением (out-of-fold) ---")
    print(f"  μ(чистые)   vs exp: {auc_sub(i_mu_c, np.arange(n_mc, len(y4))):.4f}")
    print(f"  μ(зеркало)  vs exp: {auc_sub(i_mu_f, np.arange(n_mc, len(y4))):.4f}")
    print(f"  ν(чистые)   vs exp: {auc_sub(i_nu_c, np.arange(n_mc, len(y4))):.4f}")
    print(f"  ν(зеркало)  vs exp: {auc_sub(i_nu_f, np.arange(n_mc, len(y4))):.4f}")


if __name__ == "__main__":
    main()
