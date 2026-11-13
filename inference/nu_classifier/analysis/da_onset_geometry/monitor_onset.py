"""Включение DA-игры крупным планом: эволюция геометрии фич по батчам (v3).

Правила (уточнение владельца 15.10): ДОМЕННЫЕ метрики — только игровая конструкция
(источник = MC, μ+ν вперемешку, половинный z-flip, фиксированный сид 42; цель = exp,
удвоенная зеркалом). КЛАССОВЫЕ метрики — на ЧИСТЫХ событиях: классификатор видит
source-батчи БЕЗ z-флипа (аугментации rotation в плоскости x-y и малый координатный
шум в обучении включены, но вертикальную ось не трогают; z-flip — только у доменной
игры). Чистые доменные метрики v2 убраны в v2_400_batches/.

Идея: взять чистый чекпойнт ДО выключателя λ (эпоха 5, λ=0), дообучить N батчей
с разными (λ, D-LR) и мониторить пространство фич на фиксированных 16 008
событиях пробы. Шаг обучения повторяет train_epoch тренера дословно. Каждая рука —
ОТДЕЛЬНЫЙ процесс (партия: run_grid.sh), с одного чекпойнта, один порядок батчей
(посев 42 перед каждой эпохой; source/target shuffle идут из глобального RNG,
дропаут детерминирован по батчам).

Метрики (направления — замкнутая форма Σ⁻¹Δμ, без обучения, в исходных осях):
  каждый батч  — cls_loss, dom_loss и dom_acc (точность живого критика на его же
                 доменных батчах; 0.5 = критик мёртв, «кто выигрывает»);
  каждые 10    — геометрия на игровой конструкции: d_ratio/ang_d (центроид против
                 t0), auc_dom (лучшая линейная домен-ось, in-sample — мера
                 пересечения), cov_ratio (tr Σ_tgt / tr Σ_src — форма), w0_game
                 (читатель t0 на текущей игре), auc_cls, углы |cos|: класс↔PCA1,
                 домен↔класс, домен↔PCA1;
  каждые 50    — честный свежий читатель (логрег 2-fold) + его направление
                 (cos к класс-оси, к w0, к PCA1), поворот PCA-8 против t0.

Оговорки честности: auc_dom/auc_cls — in-sample (годятся для ЭВОЛЮЦИИ, смещение
мало и одинаково по батчам); fresh_game — 2-fold, честен. Проба невзвешенная.

Запуск одной руки:
  python .../monitor_onset.py --lam 0.01 --dlr 0.01 --n-batches 5000 --device cuda:2
  --k-steps N — K внутренних шагов критика на один шаг энкодера (n_critic): K-1 шагов
  на отсоединённых фичах (энкодер не трогаем), финальный шаг с графом — градиент
  энкодера один раз, от «догнавшего» критика; по умолчанию 1 (как в партии v3).
  --accum M — накопление градиентов игры (критик + λ-градиент энкодера) за M батчей,
  применение раз в M (усреднение /M); классовая ветка учится каждый батч, как обычно
  (её шаг энкодера отдельный); внутренние K-1 адаптационных шагов критика — немедленные.
Пересборка картинки из CSV:
  python .../monitor_onset.py --plot-only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from src.training.da_nu_classifier_trainer_da_repair import (  # noqa: E402
    NuClassifierDomainAdaptationTrainer,
)

HERE = Path(__file__).resolve().parent
BASE_RUN = ("trained_models/classifier/"
            "261013_1823_da_nu_classifier_exp_full_E1_weights_tenth_z128_dlr1e-2_lam0p01_delay8_s42")
CKPT = f"{BASE_RUN}/da_checkpoint_epoch_005.pth"
DEFAULT_N = 5000
GEOM_EVERY = 10
FRESH_EVERY = 50          # кратно GEOM_EVERY
CSV_EVERY = 100           # промежуточная запись CSV (страховка от падения)
SEED = 42

# партия v3: λ × D-LR + контроль
LAMS = [0.001, 0.003, 0.01, 0.03, 0.1]
DLRS = [1e-3, 1e-2, 3e-2]
ARMS = [(lam, dlr) for lam in LAMS for dlr in DLRS] + [(0.0, 1e-2)]

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRAY = "#b9b8b2"
LAM2C = dict(zip(LAMS, CAT))
DLR_LS = {1e-3: ":", 1e-2: "-", 3e-2: "--"}
DLR_MK = {1e-3: "o", 1e-2: "s", 3e-2: "^"}
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def arm_key(lam: float, dlr: float, k_steps: int = 1,
            n_batches: int = DEFAULT_N, accum: int = 1) -> str:
    key = f"lam{lam:g}_dlr{dlr:g}".replace(".", "p")
    if k_steps != 1:
        key += f"_k{k_steps}"
    if accum != 1:
        key += f"_a{accum}"
    if n_batches != DEFAULT_N:
        key += f"_n{n_batches}"
    return key


def _dom_loss(s_dl, t_dl, dbs, device):
    return (torch.nn.functional.binary_cross_entropy_with_logits(
                s_dl, torch.zeros(dbs, 1, device=device))
            + torch.nn.functional.binary_cross_entropy_with_logits(
                t_dl, torch.ones(dbs, 1, device=device))) / 2


def _val_acc(s_dl, t_dl, idx):
    """Точность критика на валидной половине батча (диагностика внутренних шагов)."""
    s = s_dl.detach().squeeze(1)[idx]
    t = t_dl.detach().squeeze(1)[idx]
    return 0.5 * (float((s < 0).float().mean()) + float((t > 0).float().mean()))


def build_trainer(device: str) -> NuClassifierDomainAdaptationTrainer:
    cfg = yaml.safe_load(open(f"{BASE_RUN}/da_config.yaml"))
    cfg["data"]["device"] = device
    cfg["logging"]["output_dir"] = "/tmp/da_onset_v3"
    cfg["logging"]["tensorboard"] = False
    t = NuClassifierDomainAdaptationTrainer(cfg)
    t.prepare_data()
    t.prepare_model()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    t.da_model.base_model.load_state_dict(ck["base_model_state_dict"])
    t.da_model.domain_discriminator.load_state_dict(ck["domain_discriminator_state_dict"])
    t.optimizer_feature.load_state_dict(ck["optimizer_feature_state_dict"])
    t.optimizer_classifier.load_state_dict(ck["optimizer_classifier_state_dict"])
    t.optimizer_discriminator.load_state_dict(ck["optimizer_discriminator_state_dict"])
    print(f"чекпойнт загружен: {CKPT} (epoch {ck.get('epoch')})")
    return t


def prep_probe(t):
    s = t._probe_sample()
    g = s["group"]
    mu = np.flatnonzero(g <= 1)
    nu = np.flatnonzero(g >= 4)
    exp = np.flatnonzero((g >= 2) & (g <= 3))
    mc = np.concatenate([mu, nu])      # источник игры: μ+ν вперемешку
    return s, mc, mu, nu, exp


def embeddings(t, s, flip=False):
    t.da_model.eval()
    with torch.no_grad():
        z = t._probe_embeddings(s["hits"], s["offsets"], flip_z=flip)
    t.da_model.train()
    return z


def game_frames(zc, zf, mc, mu, nu, exp):
    """Игровая конструкция тренера: половина mc z-флипнута (сид 42), exp зеркалена.

    Возвращает src (MC, смешанные ориентации), tgt (exp ×2), маски is_mu/is_nu
    (по строкам src) и склейку X со сторонами side.
    """
    rng = np.random.default_rng(SEED)
    flip = rng.choice(mc, size=len(mc) // 2, replace=False)
    src = zc[mc].copy()
    src[np.isin(mc, flip)] = zf[flip]
    tgt = np.concatenate([zc[exp], zf[exp]])
    is_mu = np.isin(mc, mu)
    is_nu = np.isin(mc, nu)
    X = np.concatenate([src, tgt])
    side = np.concatenate([np.zeros(len(src)), np.ones(len(tgt))]).astype(int)
    return src, tgt, is_mu, is_nu, X, side


def frozen_reader(X0, y0):
    sc = StandardScaler().fit(X0)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X0), y0)
    return sc, clf


def auc_dir(sc, clf, X, y):
    return roc_auc_score(y, clf.decision_function(sc.transform(X)))


def orig_dir(sc, clf):
    return clf.coef_[0] / sc.scale_


def auc_axis(w, X1, X0):
    y = np.concatenate([np.ones(len(X1)), np.zeros(len(X0))]).astype(int)
    return float(roc_auc_score(y, np.concatenate([X1, X0]) @ w))


def pca1(X):
    Xc = X - X.mean(0)
    _, V = np.linalg.eigh(Xc.T @ Xc)
    return V[:, -1]


def pca_angle(U0, X, k=8):
    U1 = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:k].T
    s = np.linalg.svd(U0.T @ U1, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s[-1], -1, 1))))


def cos01(a, b):
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fresh_game_auc(X, y):
    aucs = []
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=SEED).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(aucs))


def batch_stream(loader):
    """Бесконечный поток батчей; посев перед каждой эпохой — руки синхронны."""
    while True:
        torch.manual_seed(SEED)
        for batch in loader:
            yield batch


def run_arm(t, s, mc, mu, nu, exp, lam, dlr, n_batches, k_steps=1, accum=1):
    # откат к чекпойнту
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    t.da_model.base_model.load_state_dict(ck["base_model_state_dict"])
    t.da_model.domain_discriminator.load_state_dict(ck["domain_discriminator_state_dict"])
    for opt, key in ((t.optimizer_feature, "optimizer_feature_state_dict"),
                     (t.optimizer_classifier, "optimizer_classifier_state_dict"),
                     (t.optimizer_discriminator, "optimizer_discriminator_state_dict")):
        opt.load_state_dict(ck[key])
        for gr in opt.param_groups:
            gr["lr"] = dlr if opt is t.optimizer_discriminator else gr["lr"]
    t.da_model.train()
    t.da_model.set_lambda(lam)

    src_stream = batch_stream(t.source_train_loader)
    tgt_stream = batch_stream(t.target_train_loader)

    if accum > 1:
        enc_params = [p for g in t.optimizer_feature.param_groups for p in g["params"]]
        disc_params = [p for g in t.optimizer_discriminator.param_groups
                       for p in g["params"]]
        bufE = {id(p): torch.zeros_like(p) for p in enc_params}
        bufD = {id(p): torch.zeros_like(p) for p in disc_params}

    # t0-метрики (игровая конструкция)
    zc0 = embeddings(t, s)
    zf0 = embeddings(t, s, flip=True)
    src0, tgt0, is_mu, is_nu, X0, side0 = game_frames(zc0, zf0, mc, mu, nu, exp)
    d0 = tgt0.mean(0) - src0.mean(0)
    sc0, clf0 = frozen_reader(X0, side0)
    w0_dir = orig_dir(sc0, clf0)
    U0 = np.linalg.svd(src0 - src0.mean(0), full_matrices=False)[2][:8].T
    eye = np.eye(zc0.shape[1])
    print(f"t0: свежий читатель на игре (2-fold) = {fresh_game_auc(X0, side0):.4f}")

    rows = []
    for b in range(n_batches):
        sb = next(src_stream)
        tb = next(tgt_stream)
        for bb in (sb, tb):
            for k in bb:
                if isinstance(bb[k], torch.Tensor):
                    bb[k] = bb[k].to(t.device)

        # === шаг тренера ===
        t.da_model.set_lambda(lam)

        if accum == 1:
            # дословная копия train_epoch: класс и игра в общий backward, три step
            src_logits, _ = t.da_model.classification_forward(sb)
            cls_loss = t._calculate_classification_loss(
                src_logits, sb["labels"].float(), theta=sb.get("theta"),
                sample_weight=sb.get("weights"))
            t.optimizer_feature.zero_grad()
            t.optimizer_classifier.zero_grad()
            t.optimizer_discriminator.zero_grad()
            cls_loss.backward()
        else:
            # класс — как обычно: свой forward/backward и свой шаг энкодера
            src_logits, _ = t.da_model.classification_forward(sb)
            cls_loss = t._calculate_classification_loss(
                src_logits, sb["labels"].float(), theta=sb.get("theta"),
                sample_weight=sb.get("weights"))
            t.optimizer_feature.zero_grad()
            t.optimizer_classifier.zero_grad()
            cls_loss.backward()
            t.optimizer_feature.step()
            t.optimizer_classifier.step()
            t.optimizer_feature.zero_grad()
            t.optimizer_classifier.zero_grad()

        src_db, tgt_db, n_src, n_tgt, dbs = t._build_domain_batches(sb, tb)
        util = (dbs / n_src + dbs / n_tgt) / 2
        idx_val = np.random.default_rng(SEED).permutation(dbs)[: dbs // 2]

        # K-1 шагов критика на отсоединённых фичах (энкодер не трогаем);
        # кривая точности по шагам — диагностика «лечимости» критика (inner_val_acc)
        acc_curve = []
        for _ in range(k_steps - 1):
            with torch.no_grad():
                feat_s = t.da_model.get_features(src_db)
                feat_t = t.da_model.get_features(tgt_db)
            s_log = t.da_model.domain_discriminator(feat_s)
            t_log = t.da_model.domain_discriminator(feat_t)
            dom_loss = _dom_loss(s_log, t_log, dbs, t.device)
            acc_curve.append(_val_acc(s_log, t_log, idx_val))
            t.optimizer_discriminator.zero_grad()
            (dom_loss / util).backward()
            t.optimizer_discriminator.step()

        # финальный шаг с графом: энкодер видит «догнавшего» критика
        t.optimizer_discriminator.zero_grad()
        s_dl, _ = t.da_model.domain_forward(src_db)
        t_dl, _ = t.da_model.domain_forward(tgt_db)
        dom_loss = _dom_loss(s_dl, t_dl, dbs, t.device)
        (dom_loss / util).backward()

        with torch.no_grad():
            acc_s = float((s_dl.squeeze(1) < 0).float().mean())
            acc_t = float((t_dl.squeeze(1) > 0).float().mean())
            acc_curve.append(_val_acc(s_dl, t_dl, idx_val))

        if accum == 1:
            t.optimizer_feature.step()
            t.optimizer_classifier.step()
            t.optimizer_discriminator.step()
        else:
            # копим градиенты игры (энкодер: λ-часть; критик: финальный градиент);
            # внутренние адаптационные шаги критика выше — немедленные (как в чемпионе)
            with torch.no_grad():
                for p in enc_params:
                    if p.grad is not None:
                        bufE[id(p)].add_(p.grad)
                for p in disc_params:
                    if p.grad is not None:
                        bufD[id(p)].add_(p.grad)
            t.optimizer_feature.zero_grad()
            t.optimizer_discriminator.zero_grad()
            if (b + 1) % accum == 0:
                for p in enc_params:
                    p.grad = bufE[id(p)] / accum
                for p in disc_params:
                    p.grad = bufD[id(p)] / accum
                t.optimizer_feature.step()
                t.optimizer_discriminator.step()
                with torch.no_grad():
                    for p in enc_params:
                        bufE[id(p)].zero_()
                    for p in disc_params:
                        bufD[id(p)].zero_()
                t.optimizer_feature.zero_grad()
                t.optimizer_discriminator.zero_grad()
        row = dict(batch=b, cls_loss=float(cls_loss), dom_loss=float(dom_loss),
                   dom_acc=0.5 * (acc_s + acc_t))

        # === геометрия (игровая конструкция) ===
        if (b + 1) % GEOM_EVERY == 0:
            zc = embeddings(t, s)
            zf = embeddings(t, s, flip=True)
            src, tgt, _, _, X, side = game_frames(zc, zf, mc, mu, nu, exp)
            m_src, m_tgt = src.mean(0), tgt.mean(0)
            C_src, C_tgt = np.cov(src.T), np.cov(tgt.T)
            d = m_tgt - m_src
            w_dom = np.linalg.solve(0.5 * (C_src + C_tgt) + 1e-6 * eye, d)
            # класс-ось — ЧИСТЫЕ μ/ν: классификатор в обучении видит чистые события
            w_cls = np.linalg.solve(
                0.5 * (np.cov(zc[nu].T) + np.cov(zc[mu].T)) + 1e-6 * eye,
                zc[nu].mean(0) - zc[mu].mean(0))
            p1 = pca1(X)                                   # игровое облако (домен-углы)
            p1c = pca1(np.concatenate([zc[mc], zc[exp]]))  # чистое облако (класс-ось)
            row.update(
                d_ratio=float(np.linalg.norm(d) / (np.linalg.norm(d0) + 1e-12)),
                ang_d=float(np.degrees(np.arccos(np.clip(
                    d @ d0 / (np.linalg.norm(d) * np.linalg.norm(d0) + 1e-12), -1, 1)))),
                auc_dom=auc_axis(w_dom, tgt, src),
                auc_cls=auc_axis(w_cls, zc[nu], zc[mu]),
                cos_cls_pca1=cos01(w_cls, p1c),
                cos_dom_cls=cos01(w_dom, w_cls),
                cos_dom_pca1=cos01(w_dom, p1),
                cov_ratio=float(np.trace(C_tgt) / (np.trace(C_src) + 1e-12)),
                w0_game=auc_dir(sc0, clf0, X, side))
            # val-точность критика после 0..K-1 внутренних шагов + финальный forward
            row["inner_val_acc"] = "|".join(f"{a:.3f}" for a in acc_curve)
        if (b + 1) % FRESH_EVERY == 0:
            scf = StandardScaler().fit(X)
            clff = LogisticRegression(max_iter=2000).fit(scf.transform(X), side)
            w_fresh = orig_dir(scf, clff)
            row["fresh_game"] = fresh_game_auc(X, side)
            row["pca_angle"] = pca_angle(U0, src)
            row["cos_freshdom_cls"] = cos01(w_fresh, w_cls)
            row["cos_freshdom_w0"] = cos01(w_fresh, w0_dir)
            row["cos_freshdom_pca1"] = cos01(w_fresh, p1)
        rows.append(row)

        if (b + 1) % CSV_EVERY == 0:
            pd.DataFrame(rows).to_csv(
                HERE / f"onset_{arm_key(lam, dlr, k_steps, n_batches, accum)}.csv",
                index=False)
        if (b + 1) % 100 == 0:
            print(f"  λ={lam:g} dlr={dlr:g} | батч {b+1}: dom_acc={row['dom_acc']:.3f} "
                  f"dom_loss={row['dom_loss']:.3f}"
                  + (f" d_ratio={row['d_ratio']:.3f} auc_dom={row['auc_dom']:.3f}"
                     f" fresh={row.get('fresh_game', float('nan')):.3f}"
                     if "d_ratio" in row else ""))
    return rows


def _line(axx, df: pd.DataFrame, col: str, **kw) -> None:
    """Линия по непустым точкам колонки (matplotlib 3.10 не рисует NaN-разрывы)."""
    if col not in df:
        return
    dd = df.dropna(subset=[col])
    if len(dd):
        axx.plot(dd.batch, dd[col], **kw)


def plot(all_rows: dict) -> None:
    plt.rcParams.update({"font.size": 10, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK})
    fig, ax = plt.subplots(2, 4, figsize=(21, 9))
    for (lam, dlr), rows in all_rows.items():
        df = pd.DataFrame(rows)
        c = GRAY if lam == 0 else LAM2C[lam]
        ls = "-" if lam == 0 else DLR_LS[dlr]
        lbl = "λ=0 (контроль)" if lam == 0 else f"λ={lam:g}·dlr={dlr:g}"

        _line(ax[0, 0], df, "d_ratio", color=c, ls=ls, lw=1.3)
        _line(ax[0, 1], df, "auc_dom", color=c, ls=ls, lw=1.3)
        dff = df.dropna(subset=["fresh_game"])
        if len(dff):
            ax[0, 1].plot(dff.batch, dff.fresh_game, color=c, ls=ls,
                          marker=DLR_MK[dlr], ms=3.5, lw=0.8, mfc="none", alpha=0.9)
        _line(ax[0, 2], df, "cov_ratio", color=c, ls=ls, lw=1.3)
        _line(ax[0, 3], df, "w0_game", color=c, ls=ls, lw=1.3)
        _line(ax[1, 0], df, "cos_cls_pca1", color=c, ls=ls, lw=1.3)
        _line(ax[1, 1], df, "cos_dom_cls", color=c, ls=ls, lw=1.3)
        dfcf = df.dropna(subset=["cos_freshdom_cls"])
        if len(dfcf):
            ax[1, 1].plot(dfcf.batch, dfcf.cos_freshdom_cls, color=c, ls=ls,
                          marker=DLR_MK[dlr], ms=3.5, lw=0.8, mfc="none", alpha=0.9)
        dfcw = df.dropna(subset=["cos_freshdom_w0"])
        if len(dfcw):
            ax[1, 2].plot(dfcw.batch, dfcw.cos_freshdom_w0, color=c, ls=ls,
                          marker=DLR_MK[dlr], ms=3.5, lw=0.8, mfc="none", alpha=0.9)
        dfacc = df.assign(dom_acc_s=df.dom_acc.rolling(50, min_periods=10).mean())
        _line(ax[1, 3], dfacc, "dom_acc_s", color=c, ls=ls, lw=1.3)

    # легенда: λ — цвет (сплошные), dlr — стиль
    for lam in [0.0] + LAMS:
        c = GRAY if lam == 0 else LAM2C[lam]
        ttl = "λ=0 (контроль)" if lam == 0 else f"λ={lam:g}"
        ax[0, 0].plot([], [], color=c, lw=2, label=ttl)
    for dlr in DLRS:
        ax[0, 0].plot([], [], color=INK2, ls=DLR_LS[dlr], marker=DLR_MK[dlr],
                      ms=4, lw=1.2, label=f"dlr={dlr:g}")
    ax[0, 0].legend(fontsize=8, ncol=2, framealpha=0.9, loc="lower left")

    titles = [
        ("Центроид игры: |Δμ|/|Δμ₀| (источник MC флип-микс, цель exp×2)", (0.4, 1.4)),
        ("Пересечение: AUC домен-оси (линии, Fisher)\n+ честный читатель (точки, 2-fold)", None),
        ("Форма: tr Σ_tgt / tr Σ_src", None),
        ("Старый читатель w₀ на игре (линейка t0)", (0.3, 1.0)),
        ("Класс-ось (чистые μ/ν, как в обучении) против PCA1 чистого облака: |cos|", (0, 1)),
        ("Домен-ось (игра) против класс-оси (чистые): |cos|\n(линии — Fisher; точки — свежая, логрег)", (0, 1)),
        ("Смена лучшей домен-оси: |cos(свежая, w₀)|", (0, 1)),
        ("Живой критик: точность на своих батчах (скольз. 50)", (0.4, 1.0)),
    ]
    for a, (ttl, ylim) in zip(ax.ravel(), titles):
        a.grid(True, color="#e8e7e3", lw=0.6)
        a.set_xlabel("батч")
        a.set_title(ttl, loc="left", fontsize=9.5)
        if ylim:
            a.set_ylim(*ylim)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Включение DA-игры: геометрия фич по батчам — только игровая конструкция "
                 "(чекпойнт эпохи 5, 1/10, z128; 16 рук)", y=1.0)
    fig.tight_layout()
    out = HERE / "da_onset_geometry.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("картинка:", out)


K_FAM = [
    ((0.01, 0.01, 1, 1), "λ=0.01·K=1", "#2a78d6", "-"),
    ((0.01, 0.01, 5, 1), "λ=0.01·K=5", "#2a78d6", "--"),
    ((0.01, 0.01, 10, 1), "λ=0.01·K=10", "#2a78d6", ":"),
    ((0.1, 0.01, 1, 1), "λ=0.1·K=1", "#eb6834", "-"),
    ((0.1, 0.01, 3, 1), "λ=0.1·K=3", "#eb6834", "--"),
    ((0.1, 0.01, 5, 1), "λ=0.1·K=5", "#eb6834", "-."),
    ((0.1, 0.01, 10, 1), "λ=0.1·K=10", "#eb6834", ":"),
    ((0.01, 0.01, 10, 5), "λ=0.01·K=10·accum5", "#e87ba4", "-"),
    ((0.01, 0.01, 10, 10), "λ=0.01·K=10·accum10", "#e87ba4", "--"),
]


def plot_k() -> None:
    """Сравнение K-семьи: честный метр, живой критик, старая ось."""
    plt.rcParams.update({"font.size": 10, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for (lam, dlr, k, accum), lbl, c, ls in K_FAM:
        f = HERE / f"onset_{arm_key(lam, dlr, k, DEFAULT_N, accum)}.csv"
        if not f.exists():
            print(f"нет CSV: {f.name} — пропускаю")
            continue
        df = pd.read_csv(f)
        dff = df.dropna(subset=["fresh_game"])
        if len(dff):
            ax[0].plot(dff.batch, dff.fresh_game, color=c, ls=ls, marker="o",
                       ms=3, lw=1.2, mfc="none", label=lbl)
        ax[1].plot(df.batch, df.dom_acc.rolling(50, min_periods=10).mean(),
                   color=c, ls=ls, lw=1.3, label=lbl)
        _line(ax[2], df, "w0_game", color=c, ls=ls, lw=1.3)
    for a, (ttl, ylim) in zip(ax, [
            ("Честный читатель на игре (2-fold): изъятие", (0.70, 0.78)),
            ("Живой критик: точность на своих батчах (скольз. 50)", (0.45, 0.75)),
            ("Старый читатель w₀ на игре (линейка t0)", (0.3, 0.9))]):
        a.grid(True, color="#e8e7e3", lw=0.6)
        a.set_xlabel("батч")
        a.set_title(ttl, loc="left", fontsize=10)
        if ylim:
            a.set_ylim(*ylim)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(fontsize=8, ncol=2)
    fig.suptitle("K-шаги критика (λ=0.01 и 0.1, dlr=1e-2, 5000 батчей): "
                 "правильный критик + сильное DA?", y=1.02)
    fig.tight_layout()
    out = HERE / "da_onset_k_family.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("картинка:", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float)
    ap.add_argument("--dlr", type=float)
    ap.add_argument("--n-batches", type=int, default=DEFAULT_N)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--k-steps", type=int, default=1)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--plot-k", action="store_true")
    args = ap.parse_args()

    if args.plot_k:
        plot_k()
        return

    if args.plot_only:
        all_rows = {}
        for lam, dlr in ARMS:
            f = HERE / f"onset_{arm_key(lam, dlr)}.csv"
            if f.exists():
                all_rows[(lam, dlr)] = pd.read_csv(f).to_dict("records")
            else:
                print(f"нет CSV: {f.name} — пропускаю")
        plot(all_rows)
        return

    assert args.lam is not None and args.dlr is not None, "--lam и --dlr обязательны"
    key = arm_key(args.lam, args.dlr, args.k_steps, args.n_batches, args.accum)
    print(f"рука {key}: λ={args.lam:g} dlr={args.dlr:g} K={args.k_steps} "
          f"accum={args.accum} n={args.n_batches} dev={args.device}")
    t = build_trainer(args.device)
    s, mc, mu, nu, exp = prep_probe(t)
    rows = run_arm(t, s, mc, mu, nu, exp, args.lam, args.dlr, args.n_batches,
                   k_steps=args.k_steps, accum=args.accum)
    pd.DataFrame(rows).to_csv(HERE / f"onset_{key}.csv", index=False)
    print(f"рука {key} завершена, CSV записан")


if __name__ == "__main__":
    main()
