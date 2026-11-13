"""Эффективная размерность пространства признаков чекпойнта (порт /tmp-диагностик).

Объединяет два октябрьских скрипта (diag_dim.py, diag_pc_flip.py), которые жили
только в /tmp; числа из них цитируются в шапке конфига маленькой модели, а
генерирующего скрипта в репозитории не было. Считается:

  1) спектр PCA (participation ratio, компонент на 90/95/99% дисперсии, топ-10)
     для источника (mc: mu+nu) и для всего облака (mc+exp), по ЧИСТЫМ событиям;
  2) нелинейная оценка размерности two-NN (Facco 2017) на mc / всё / exp;
  3) читаемость классовой (nu vs mu) и доменной (mc vs exp) задач при усечении
     до k компонент PCA — свежий logreg, честный сплит 4000/4000, только
     центрирование (в PCA-базисе стандартизация усилила бы хвостовой шум);
     домен считается и в чистом виде, и в z-flip-версии (задача тренера:
     каждому событию флип с p=1/2, как в _build_domain_batches);
  4) углы PC1 с классовым и доменными направлениями, нормы этих направлений.

Фичи (6x8000x d) кэшируются в <script_dir>/<run_name>_feats.npz — в git не попадают.

Запуск:
    python inference/nu_classifier/analysis/feature_dimensionality/measure_dim.py \
        --run trained_models/classifier/<run>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from src.models.base_models import create_model            # noqa: E402
from sklearn.linear_model import LogisticRegression        # noqa: E402
from sklearn.metrics import roc_auc_score                  # noqa: E402

SRC = "data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8"
TGT = "data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8"


def gather(fp, op, n, seed, mask=None):
    """n случайных событий (массивы хитов) из memmap-датасета."""
    feats = np.load(fp, mmap_mode="r")
    offs = np.load(op)
    g = np.random.default_rng(seed)
    ev = []
    while len(ev) < n:
        i = int(g.integers(0, len(offs) - 1))
        if mask is not None and not mask[i]:
            continue
        a = int(offs[i])
        b = min(int(offs[i + 1]), a + MAX_HITS)
        ev.append(np.asarray(feats[a:b], np.float32))
    return ev


def collate(ev, flips=None):
    """Паддинг + нормировка; flips[j] -> негация z (сырого; mean_z=0)."""
    evs = []
    for j, e in enumerate(ev):
        e2 = e.copy()
        if flips is not None and flips[j]:
            e2[:, 4] = -e2[:, 4]
        evs.append(e2)
    L = max(len(e) for e in evs)
    X = np.zeros((len(evs), L, 5), np.float32)
    M = np.zeros((len(evs), L), bool)
    for i, e in enumerate(evs):
        X[i, :len(e)] = (e - MEANS) / STDS
        M[i, :len(e)] = True
    return {"features": torch.tensor(X),
            "lengths": torch.tensor(M.sum(1)),
            "mask": torch.tensor(M)}


def two_nn(X, frac=0.9, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(3000, len(X)), replace=False)
    S = X[idx]
    d = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    d.sort(axis=1)
    mu_ = np.sort(d[:, 1] / d[:, 0])
    mu_ = mu_[:int(frac * len(mu_))]
    return (len(mu_) - 1) / np.sum(np.log(mu_))


def probe(A, B, k=None, V=None, seed=0):
    """Свежий logreg: обучение на половине, AUC на второй. Только центрирование."""
    if k is not None:
        A, B = A @ V[:, :k], B @ V[:, :k]
    r = np.random.default_rng(seed)
    A, B = A[r.permutation(len(A))], B[r.permutation(len(B))]
    sa, sb = len(A) // 2, len(B) // 2
    X = np.vstack([A[:sa], B[:sb]])
    y = np.r_[np.zeros(sa), np.ones(sb)]
    m = X.mean(0)
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(X - m, y)
    Xte = np.vstack([A[sa:], B[sb:]])
    yte = np.r_[np.zeros(len(A) - sa), np.ones(len(B) - sb)]
    assert len(np.unique(yte)) == 2, "в тесте один класс"
    return roc_auc_score(yte, clf.predict_proba(Xte - m)[:, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    cache = Path(args.cache) if args.cache else \
        Path(__file__).parent / f"{run.name}_feats.npz"
    n = args.n

    cfg = yaml.safe_load(open(run / "da_config.yaml"))
    global MEANS, STDS, MAX_HITS
    st = yaml.safe_load(open("data_manager/stats_dict/default_mc.yaml"))
    MEANS, STDS = np.array(st["means"], np.float32), np.array(st["stds"], np.float32)
    MAX_HITS = cfg["data"]["max_hits"]

    ck_epoch = None
    if cache.exists():
        d = np.load(cache)
        Zmu, Znu, Zex = d["Zmu"], d["Znu"], d["Zex"]
        Fmu, Fnu, Fex = d["Fmu"], d["Fnu"], d["Fex"]
        print(f"фичи: кэш {cache}")
    else:
        typ = np.load(f"{SRC}/particle_types.npy")
        mu = gather(f"{SRC}/features.npy", f"{SRC}/offsets.npy", n, 1, typ == 0)
        nu = gather(f"{SRC}/features.npy", f"{SRC}/offsets.npy", n, 2, typ > 0)
        ex = gather(f"{TGT}/exp_features.npy", f"{TGT}/exp_offsets.npy", n, 3)
        ck = torch.load(run / "best_da_model.pth", map_location="cpu",
                        weights_only=False)
        ck_epoch = ck.get("epoch")
        # amp_clip конвертируем в нормированные единицы, как тренер:
        # (raw - mean_Q)/std_Q. Без конверсии клип 100 в нормированных единицах
        # не срабатывает, и фичи отличаются от того, что модель видит вперёд.
        # Конвенция обязательна: с ней эталонный чекпойнт воспроизводит
        # записанные ранее числа (PR 1.4, угол PC1-класс 0.6°, 0.517@1PC).
        mc_cfg = dict(cfg["model"])
        if mc_cfg.get("amp_clip") is not None:
            k = mc_cfg["amp_clip"]
            mc_cfg["amp_clip"] = float((k - MEANS[0]) / STDS[0])
        model = create_model(mc_cfg)
        model.load_state_dict(ck["base_model_state_dict"])
        model.eval()
        rng = np.random.default_rng(3)
        f_mu, f_nu, f_ex = (rng.random(n) < 0.5 for _ in range(3))

        def Z(ev, fl=None):
            with torch.no_grad():
                return model.get_feature_representation(collate(ev, fl)).numpy()

        Zmu, Znu, Zex = Z(mu), Z(nu), Z(ex)
        Fmu, Fnu, Fex = Z(mu, f_mu), Z(nu, f_nu), Z(ex, f_ex)
        np.savez_compressed(cache, Zmu=Zmu, Znu=Znu, Zex=Zex,
                            Fmu=Fmu, Fnu=Fnu, Fex=Fex)
        print(f"фичи: посчитаны с {run/'best_da_model.pth'} и закэшированы в {cache}")

    print(f"чекпойнт: epoch={ck_epoch} | "
          f"фичи {Zmu.shape[1]}-мерные, событий на класс {len(Zmu)}")
    Zsrc = np.vstack([Zmu, Znu])
    Zall = np.vstack([Zmu, Znu, Zex])

    # 1) спектр
    for tag, X in (("источник (mc)", Zsrc), ("всё вместе", Zall)):
        Xc = X - X.mean(0)
        ev = np.clip(np.linalg.eigvalsh(np.cov(Xc.T))[::-1], 0, None)
        pr = (ev.sum() ** 2) / (ev ** 2).sum()
        cum = np.cumsum(ev) / ev.sum()
        n90, n95, n99 = (int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99))
        print(f"\n{tag}: participation ratio = {pr:.1f} из {X.shape[1]}")
        print(f"   компонент на 90/95/99%: {n90}/{n95}/{n99}")
        print("   первые 10 долей (%): " + " ".join(f"{100*e/ev.sum():.1f}" for e in ev[:10]))

    # 2) two-NN
    print("\nвнутренняя размерность two-NN:")
    print(f"   источник: {two_nn(Zsrc):.1f} | всё: {two_nn(Zall):.1f} | exp: {two_nn(Zex):.1f}")

    # 3) усечение PCA
    V = np.linalg.svd(Zall - Zall.mean(0), full_matrices=False)[2].T
    print(f"\n{'компонент':>10}{'класс (чистые)':>16}{'домен чистые':>15}{'домен z-flip':>15}")
    for k in (None, 1, 2, 3, 4, 8, 16, 32, 64):
        kk = "все" if k is None else k
        a_cls = probe(Znu, Zmu, k, V)
        a_dom_c = probe(Zsrc, Zex, k, V)
        a_dom_f = probe(np.vstack([Fmu, Fnu]), Fex, k, V)
        print(f"{kk:>10}{a_cls:>16.3f}{a_dom_c:>15.3f}{a_dom_f:>15.3f}")

    # 4) углы PC1 с направлениями
    u1 = V[:, 0]

    def ang(a, b):
        c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

    d_cls = Znu.mean(0) - Zmu.mean(0)
    d_dom_c = Zex.mean(0) - Zsrc.mean(0)
    d_dom_f = Fex.mean(0) - np.vstack([Fmu, Fnu]).mean(0)
    print(f"\nугол(PC1, класс) = {ang(u1, d_cls):.1f}° | "
          f"угол(PC1, домен чистые) = {ang(u1, d_dom_c):.1f}° | "
          f"угол(PC1, домен z-flip) = {ang(u1, d_dom_f):.1f}°")
    print(f"|Δcls| = {np.linalg.norm(d_cls):.2f} | "
          f"|Δdom чистые| = {np.linalg.norm(d_dom_c):.2f} | "
          f"|Δdom z-flip| = {np.linalg.norm(d_dom_f):.2f}")


if __name__ == "__main__":
    main()
