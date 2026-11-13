"""Диагностика «пола 0.70» честной пробы: что в нём домен, а что класс.

Вопрос владельца: в exp физически почти только ШАЛы (≈ muatm), поэтому доменный
разрыв надо мерить на СОГЛАСОВАННОЙ популяции (muatm vs exp), а ν-часть игровой
пробы — это классовое содержание, которое DA убирать не должен.

Проба (probe_sample.npz) смешана: группы 0/1 = μ natural/tail (score>=0.8),
2/3 = exp natural/tail, 4/5 = ν natural/high-score. Поэтому:
  * natural-анализ — только группы 0, 2, 4 (несмещённые популяции);
  * tail-анализ — группы 1, 3, 5 (контекст, хвосты высокого скора).

На чекпойнтах эталонного прогона λ=0.003 (best@e53, e35) и стартовом t0 (epoch 5
базового прогона 261013_1823) меряем:
  A. линейный читатель на игровой конструкции (пол, эталон);
  B. MLP 64x32 на той же игре — нелинейная читаемость;
  C. устойчивость: сиды флипов {42,7,123}, половинная подвыборка;
  D. |cos(домен-ось, класс-ось)|;
  E. НАТУРАЛЬНЫЕ разрывы (чистые): μ_nat vs exp_nat, ν_nat vs exp_nat (+tail-версии);
  G. разложение ИГРОВОЙ пробы по классу на natural-популяции: μ_nat vs exp_nat,
     ν_nat vs exp_nat (+tail-версии) — сколько «пола» это домен, сколько класс.

Запуск: python inference/nu_classifier/analysis/da_floor_probe/measure.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from src.training.da_nu_classifier_trainer_da_repair import (  # noqa: E402
    NuClassifierDomainAdaptationTrainer,
)

RUN = ("trained_models/classifier/"
       "261018_1930_da_nu_classifier_exp_full_E1_weights_tenth_z128_k10_dlr1e-2_"
       "lam0p003_delay8_s42")
T0_CKPT = ("trained_models/classifier/"
           "261013_1823_da_nu_classifier_exp_full_E1_weights_tenth_z128_dlr1e-2_lam0p01_"
           "delay8_s42/da_checkpoint_epoch_005.pth")
CKPTS = {"t0 (e5 базы)": T0_CKPT,
         "best(e53)": f"{RUN}/best_da_model.pth",
         "e35": f"{RUN}/da_checkpoint_epoch_035.pth"}
SEED = 42


def build_trainer(ckpt_path: str) -> NuClassifierDomainAdaptationTrainer:
    cfg = yaml.safe_load(open(f"{RUN}/da_config.yaml"))
    cfg["data"]["device"] = "cpu"
    cfg["logging"]["output_dir"] = "/tmp/da_floor_probe"
    cfg["logging"]["tensorboard"] = False
    t = NuClassifierDomainAdaptationTrainer(cfg)
    t.prepare_data()
    t.prepare_model()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    t.da_model.base_model.load_state_dict(ck["base_model_state_dict"])
    return t


def embeddings(t, s, flip=False):
    t.da_model.eval()
    with torch.no_grad():
        return t._probe_embeddings(s["hits"], s["offsets"], flip_z=flip)


def game_frames(zc, zf, mc, exp, flip_seed=SEED):
    rng = np.random.default_rng(flip_seed)
    flip = rng.choice(mc, size=len(mc) // 2, replace=False)
    src = zc[mc].copy()
    src[np.isin(mc, flip)] = zf[flip]
    tgt = np.concatenate([zc[exp], zf[exp]])
    X = np.concatenate([src, tgt])
    y = np.concatenate([np.zeros(len(src)), np.ones(len(tgt))]).astype(int)
    return X, y


def two_fold_auc(X, y, fit_predict, seed=SEED):
    aucs = []
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(X, y):
        aucs.append(fit_predict(X[tr], y[tr], X[te], y[te]))
    return float(np.mean(aucs))


def linear_fp(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.decision_function(sc.transform(Xte)))


def mlp_fp(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                        early_stopping=True, n_iter_no_change=20,
                        random_state=SEED).fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])


def fisher_dir(X1, X0, reg=1e-6):
    m1, m0 = X1.mean(0), X0.mean(0)
    C = 0.5 * (np.cov(X1.T) + np.cov(X0.T))
    return np.linalg.solve(C + reg * np.eye(C.shape[0]), m1 - m0)


def cos01(a, b):
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def auc_of(X_a, X_b, fp=linear_fp):
    y = np.r_[np.zeros(len(X_a)), np.ones(len(X_b))].astype(int)
    return two_fold_auc(np.concatenate([X_a, X_b]), y, fp)


def main() -> None:
    for name, ckpt in CKPTS.items():
        t = build_trainer(ckpt)
        s = t._probe_sample()
        g = s["group"]
        mu, nu = np.flatnonzero(g <= 1), np.flatnonzero(g >= 4)
        exp = np.flatnonzero((g >= 2) & (g <= 3))
        mc = np.concatenate([mu, nu])
        zc = embeddings(t, s, flip=False)
        zf = embeddings(t, s, flip=True)
        X, y = game_frames(zc, zf, mc, exp)

        # natural/tail индексы (0/1 = μ nat/tail, 2/3 = exp nat/tail, 4/5 = ν nat/hi)
        i_nat = {k: np.flatnonzero(g == k) for k in (0, 2, 4)}
        i_tail = {k: np.flatnonzero(g == k) for k in (1, 3, 5)}

        lin = auc_of(X[y == 0], X[y == 1])
        mlp = auc_of(X[y == 0], X[y == 1], mlp_fp)
        seeds = []
        for s_ in (42, 7, 123):
            Xs, ys = game_frames(zc, zf, mc, exp, flip_seed=s_)
            seeds.append(auc_of(Xs[ys == 0], Xs[ys == 1]))
        rng = np.random.default_rng(SEED)
        half = rng.choice(len(X), size=len(X) // 2, replace=False)
        lin_half = auc_of(X[half][y[half] == 0], X[half][y[half] == 1])
        w_dom = fisher_dir(X[y == 1], X[y == 0])
        w_cls = fisher_dir(zc[nu], zc[mu])

        # E. натуральные разрывы на чистых событиях (согласованные популяции)
        E_mu_nat = auc_of(zc[i_nat[0]], zc[i_nat[2]])
        E_nu_nat = auc_of(zc[i_nat[4]], zc[i_nat[2]])
        E_mu_tail = auc_of(zc[i_tail[1]], zc[i_tail[3]])
        E_nu_tail = auc_of(zc[i_tail[5]], zc[i_tail[3]])

        # G. разложение игровой пробы по классу (строки источника = порядок mc;
        #    цель — exp natural/tail в двух копиях: оригинал + зеркало)
        src = X[y == 0]
        tgt = X[y == 1]
        g_src = g[mc]
        g_tgt = np.concatenate([g[exp], g[exp]])

        def game_subset(g_a, g_b):
            return auc_of(src[np.isin(g_src, g_a)], tgt[np.isin(g_tgt, g_b)])

        G_mu_nat = game_subset([0], [2])
        G_nu_nat = game_subset([4], [2])
        G_mu_tail = game_subset([1], [3])
        G_nu_tail = game_subset([5], [3])

        print(f"=== {name}")
        print(f"  A. линейный (игра, все):           {lin:.4f}")
        print(f"  B. MLP (игра, все):                {mlp:.4f}   ({mlp-lin:+.4f})")
        print(f"  C. сиды флипов 42/7/123:           {seeds[0]:.4f} / {seeds[1]:.4f} / {seeds[2]:.4f}"
              f"  | половина: {lin_half:.4f}")
        print(f"  D. |cos(домен-ось, класс-ось)|:    {cos01(w_dom, w_cls):.4f}")
        print(f"  E. чистые: μ_nat {E_mu_nat:.4f} | ν_nat {E_nu_nat:.4f}"
              f"   (tail: μ {E_mu_tail:.4f} | ν {E_nu_tail:.4f})")
        print(f"  G. игра:   μ_nat {G_mu_nat:.4f} | ν_nat {G_nu_nat:.4f}"
              f"   (tail: μ {G_mu_tail:.4f} | ν {G_nu_tail:.4f})")
        print()


if __name__ == "__main__":
    main()
