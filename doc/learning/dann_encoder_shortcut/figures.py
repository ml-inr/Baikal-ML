#!/usr/bin/env python
"""Геометрические иллюстрации к конспекту «DANN: обход дискриминатора».

Все рисунки строятся здесь и только здесь — PDF их только вставляет. Данные
для last-рисунка (траектория нашего прогона) читаются из настоящих файлов
репозитория, синтетика — с фиксированными зёрнами.

Запуск:
    python doc/learning/dann_encoder_shortcut/figures.py
Пишет PNG в doc/learning/dann_encoder_shortcut/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
ROOT = HERE.parents[2]          # корень репозитория

SRC, TGT, ACC = "#1f77b4", "#d62728", "#2ca02c"
GRAY = "#666666"


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def _arrow(ax, xy_from, xy_to, color=GRAY, lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=14,
                                 lw=lw, ls=ls, color=color, shrinkA=0, shrinkB=0))


def _box(ax, x, y, w, h, text, fc="#f2f2f2", ec=GRAY, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


# ── 1. Архитектура DANN ──────────────────────────────────────────────────
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.6); ax.axis("off")

    _box(ax, 0.15, 1.5, 1.5, 0.8, "событие $x$\n(хиты)", fc="#eef4fb")
    _box(ax, 2.3, 1.5, 1.9, 0.8, "энкодер\n$z = E(x)$", fc="#eef4fb")
    _box(ax, 5.1, 2.4, 2.2, 0.8, "классификатор\n$L_{cls}$ (ν / μ)", fc="#e8f6e8")
    _box(ax, 5.1, 0.5, 2.2, 0.8, "дискриминатор\n$L_{dom}$ (MC / exp)", fc="#fdeceb")

    ax.add_patch(FancyBboxPatch((4.35, 1.35), 0.45, 1.0, boxstyle="round,pad=0.02",
                                fc="#fff3cd", ec="#c9a227", lw=1.2))
    ax.text(4.575, 1.85, "GRL\n×(−λ)", ha="center", va="center", fontsize=9)

    _arrow(ax, (1.65, 1.9), (2.3, 1.9))
    _arrow(ax, (4.2, 1.9), (4.35, 1.9))
    _arrow(ax, (4.8, 1.9), (5.1, 2.8))
    _arrow(ax, (4.8, 1.9), (5.1, 0.9))
    ax.text(5.35, 1.95, "вперёд: тождественно\nназад: $\\nabla \\to -\\lambda\\nabla$",
            fontsize=8.5, color="#7a6000", va="center")

    ax.text(5.2, 3.35, "$L = L_{cls} - \\lambda\\, L_{dom}$", fontsize=13, color="black")
    ax.text(0.15, 0.45,
            "энкодер учится:  (а) разделять ν и μ по меткам источника,\n"
            "(б) делать z неразличимым для дискриминатора (домен)",
            fontsize=8.5, color=GRAY, va="top")
    _save(fig, "fig_architecture.png")


# ── 2. Цепочка отказа ────────────────────────────────────────────────────
def fig_failure_chain() -> None:
    steps = [
        ("D и E обновляются одновременно", "#f2f2f2"),
        ("D отстаёт: его $w$ устарел на один-два шага", "#fdeceb"),
        ("градиент в E от домена лежит в 1 направлении $\\propto w$", "#fdeceb"),
        ("E сдвигает $z$ ровно так, чтобы $w^\\top z$ потерял доменную зависимость", "#fdeceb"),
        ("доменный лосс падает, D выглядит мёртвым", "#f2f2f2"),
        ("но домен жив: свежий логрег на тех же $z$ даёт AUC 0.70", "#e8f6e8"),
        ("лечение: не давать $w$ устаревать (refresh)\nи завести несколько независимых $w$ (ансамбль)", "#e8f6e8"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.set_xlim(0, 9.2); ax.set_ylim(0, 6.6); ax.axis("off")
    h, gap = 0.62, 0.28
    for i, (text, fc) in enumerate(steps):
        y = 6.2 - i * (h + gap)
        _box(ax, 0.5, y, 8.2, h, text, fc=fc, fs=9.5)
        if i < len(steps) - 1:
            _arrow(ax, (4.6, y), (4.6, y - gap), lw=1.4)
    _save(fig, "fig_failure_chain.png")


# ── 3. 1D: оптимальный дискриминатор ─────────────────────────────────────
def fig_optimal_d_1d() -> None:
    x = np.linspace(-6, 6, 600)
    def g(mu, s): return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.4), height_ratios=[1.25, 1])
    for col, (mu_t, title) in enumerate([(2.6, "домены различаются"),
                                         (-1.0, "домены совпали")]):
        ps, pt = g(-1.0, 1.4), g(mu_t, 1.4)
        d = pt / (ps + pt)
        ax = axes[0, col]
        ax.plot(x, ps, color=SRC, lw=2, label="p(z | MC)")
        ax.plot(x, pt, color=TGT, lw=2, label="p(z | exp)")
        ax.set_ylim(0, 0.33); ax.set_yticks([]); ax.set_xticks([])
        ax.set_title(title, fontsize=10.5)
        if col == 0:
            ax.legend(fontsize=8.5, loc="upper left", frameon=False)

        ax2 = axes[1, col]
        ax2.plot(x, d, color="black", lw=2)
        ax2.axhline(0.5, color=GRAY, lw=1, ls=":")
        ax2.set_ylim(-0.05, 1.05); ax2.set_yticks([0, 0.5, 1])
        ax2.set_xlabel("$w^\\top z$", fontsize=10)
        if col == 0:
            ax2.set_ylabel("$D^*(z)$ — оптимум", fontsize=10)
            i = int(np.argmax(np.abs(d - 0.5)))
            ax2.annotate("$D^*=\\dfrac{p_{MC}}{p_{MC}+p_{exp}}$",
                         xy=(x[i], d[i]), xytext=(x[i] + 0.4, d[i] - 0.35),
                         fontsize=10, arrowprops=dict(arrowstyle="-", color=GRAY, lw=1))
        else:
            ax2.text(0.05, 0.60, "$D^* \\equiv 1/2$,  лосс $= \\ln 2$\n"
                                 "(никого ни с чем не спутать)",
                     transform=ax2.transAxes, fontsize=9.5)
    fig.tight_layout()
    _save(fig, "fig_optimal_d_1d.png")


# ── 4. 2D: дискриминатор читает одно направление ─────────────────────────
def fig_readout_2d() -> None:
    rng = np.random.default_rng(0)
    a = np.array([1.9, 0.0])
    src = rng.normal(0, 0.55, (220, 2))
    tgt = rng.normal(0, 0.55, (220, 2)) + a
    w = np.array([1.0, 0.22]); w = w / np.linalg.norm(w)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    ax = axes[0]
    ax.scatter(src[:, 0], src[:, 1], s=9, color=SRC, alpha=0.55, label="MC")
    ax.scatter(tgt[:, 0], tgt[:, 1], s=9, color=TGT, alpha=0.55, label="exp")
    _arrow(ax, (0, 0), 2.7 * w, color="black", lw=2)
    ax.text(2.75 * w[0], 2.75 * w[1], "$w$", fontsize=13)
    lim = np.array([-2.2, 3.6])
    # граница решений: w^T z = const, проведена через середину облака
    c = 0.5 * (src.mean(0) + tgt.mean(0)) @ w
    t = np.array([-3.0, 3.0])
    perp = np.array([-w[1], w[0]])
    p0 = c * w + t[0] * perp
    p1 = c * w + t[1] * perp
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GRAY, ls="--", lw=1.4)
    ax.text(c * w[0] - 1.55, 1.55, "граница решений\n(перпендикулярна $w$)",
            fontsize=8.5, color=GRAY, ha="left", va="center")
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("$z_1$", fontsize=11); ax.set_ylabel("$z_2$", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", frameon=False)

    ax = axes[1]
    ps = src @ w; pt = tgt @ w
    bins = np.linspace(-2.2, 3.9, 40)
    ax.hist(ps, bins=bins, color=SRC, alpha=0.6, label="MC")
    ax.hist(pt, bins=bins, color=TGT, alpha=0.6, label="exp")
    ax.axvline(c, color=GRAY, ls="--", lw=1.4)
    ax.set_yticks([]); ax.set_xlabel("проекция $w^\\top z$", fontsize=11)
    ax.text(0.02, 0.9, "всё, что видит дискриминатор", transform=ax.transAxes,
            fontsize=9.5, color=GRAY)
    fig.tight_layout()
    _save(fig, "fig_readout_2d.png")


# ── 5. Главный рисунок: обход через поворот ──────────────────────────────
def fig_nullspace() -> None:
    rng = np.random.default_rng(1)
    a = np.array([2.0, 0.0])
    h_src = rng.normal(0, 0.42, (240, 2))
    h_tgt = rng.normal(0, 0.42, (240, 2)) + a
    R = np.array([[0.0, -1.0], [1.0, 0.0]])           # поворот на 90 градусов
    z_src, z_tgt = h_src @ R.T, h_tgt @ R.T

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    def scatter(ax, p, q):
        ax.scatter(p[:, 0], p[:, 1], s=8, color=SRC, alpha=0.5)
        ax.scatter(q[:, 0], q[:, 1], s=8, color=TGT, alpha=0.5)
        ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4)
        ax.set_xlabel("$z_1$", fontsize=10); ax.set_ylabel("$z_2$", fontsize=10)

    def readout_arrow(ax, direction, color, label, y_at=-2.75, length=2.2):
        """Стрелка «вот это направление читает D» — ниже облака, не мешая точкам."""
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        base = np.array([-0.9 * d[0], y_at])
        _arrow(ax, base, base + length * d, color=color, lw=2)
        ax.plot([-3.2 * d[0], 3.2 * d[0]], [-3.2 * d[1], 3.2 * d[1]],
                ls=":", lw=0.9, color=color, alpha=0.7)
        tip = base + length * d
        ax.text(tip[0] + 0.15, tip[1] - 0.42, label, fontsize=9, color=color)

    def threshold(ax, x, label=None):
        ax.plot([x, x], [-3.4, 3.4], ls="--", lw=1.3, color=GRAY)
        if label:
            ax.text(x + 0.12, 2.55, label, fontsize=8.5, color=GRAY)

    ax = axes[0]
    scatter(ax, h_src, h_tgt)
    readout_arrow(ax, (1.0, 0.0), "black", "читаемое $w$ (и порог по нему)")
    threshold(ax, 1.0, "порог D")
    ax.set_title("до: доменный сигнал\nвдоль читаемого $w$", fontsize=10)

    ax = axes[1]
    scatter(ax, z_src, z_tgt)
    readout_arrow(ax, (1.0, 0.0), "black", "тот же $w$")
    threshold(ax, 0.0, "тот же порог")
    ax.set_title("после одного шага E:\nдля старого $w$ домены неразличимы", fontsize=10)
    ax.text(0.5, -0.32, "AUC $\\approx$ 0.5   (лосс $\\approx \\ln 2$)",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=GRAY)

    ax = axes[2]
    scatter(ax, z_src, z_tgt)
    _arrow(ax, (2.45, -2.4), (2.45, 2.3), color=ACC, lw=2)
    ax.text(2.30, -2.75, "новый $w'$: его находит\nсвежий логрег", fontsize=9, color=ACC,
            ha="left", va="center")
    ax.set_title("те же фичи, свежий логрег:\nдомен на месте", fontsize=10)
    ax.text(0.5, -0.32, "AUC $\\approx$ 1.0", transform=ax.transAxes, ha="center",
            fontsize=9.5, color=ACC)
    fig.tight_layout()
    _save(fig, "fig_nullspace.png")


# ── 6. «Комната для пряток»: d − M ───────────────────────────────────────
def fig_room() -> None:
    fig = plt.figure(figsize=(11.5, 4.0))
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    for k, (M, desc) in enumerate([
            (1, "M = 1: спрятаться можно\nв плоскости $(d-M=2)$"),
            (2, "M = 2: остаётся прямая $(d-M=1)$"),
            (3, "M = 3: спрятаться негде $(d-M=0)$")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 40)
        R = 1.0
        xs = R * np.outer(np.cos(u), np.sin(v))
        ys = R * np.outer(np.sin(u), np.sin(v))
        zs = R * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color="#dbe9f6", alpha=0.25, linewidth=0)
        # «Комната для пряток» — пересечение ядер прочитанных направлений:
        # ker(w_1) для M=1 это плоскость, ker(w_1)∩ker(w_2) — прямая, и т.д.
        if M == 1:
            yy, zz = np.meshgrid(np.linspace(-0.85, 0.85, 2), np.linspace(-0.85, 0.85, 2))
            ax.plot_surface(np.zeros_like(yy), yy, zz, color=ACC, alpha=0.55, linewidth=0)
        elif M == 2:
            t = np.linspace(-1.0, 1.0, 2)
            ax.plot(0 * t, 0 * t, t, color=ACC, lw=5, alpha=0.9)
        else:
            ax.scatter([0], [0], [0], color=ACC, s=45)
        for w in [(1, 0, 0), (0, 1, 0), (0, 0, 1)][:M]:
            ax.plot([0, 1.0 * w[0]], [0, 1.0 * w[1]], [0, 1.0 * w[2]],
                    color="black", lw=2)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.set_title(desc, fontsize=10)
        if k == 0:
            ax.text(-1.2, -1.2, 1.5, "чёрные оси —\nнаправления чтения $w_m$;\n"
                                     "зелёное — где сигнал ещё спрятан",
                    fontsize=8.5, color=GRAY)
    fig.tight_layout()
    _save(fig, "fig_room.png")


# ── 7. Сцепленность доменного и классового признака ──────────────────────
def fig_entanglement() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, (angle, title) in zip(axes, [
            (90.0, "ортогональны: инвариантность\nне мешает классификации"),
            (21.0, "сцеплены: убирая домен,\nубираешь и класс")]):
        th = np.deg2rad(angle)
        a = np.array([np.cos(th), np.sin(th)])      # доменный сигнал
        b = np.array([1.0, 0.0])                    # классовый сигнал
        ax.annotate("", xy=b * 1.0, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.4))
        ax.annotate("", xy=a * 1.0, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=TGT, lw=2.4))
        ax.text(*(b * 1.06), "класс\n(ν/μ)", color=ACC, fontsize=9.5)
        ax.text(*(a * 1.06), "домен\n(MC/exp)", color=TGT, fontsize=9.5)
        if angle > 45:
            ax.plot([0, 0], [0, 1], color=TGT, ls=":", lw=1.2)
        ax.set_xlim(-0.35, 1.5); ax.set_ylim(-0.35, 1.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_entanglement.png")


# ── 8. Наш прогон: доменный AUC против пробы ─────────────────────────────
def fig_our_run() -> None:
    import csv
    hist = ROOT / ("trained_models/classifier/"
                   "261002_0535_da_nu_classifier_exp_full_E1_lambda0.01_DArepair/"
                   "da_training_history.csv")
    fixed = ROOT / ("inference/nu_classifier/analysis/da_band_probe/results/"
                    "probe_trajectory_FIXED.csv")
    if not hist.exists():
        print("  !! истории прогона нет, рис. 8 пропущен"); return
    rows = list(csv.DictReader(open(hist)))
    ep = [int(r["epoch"]) for r in rows]
    dom = [float(r["val_domain_auc"]) for r in rows]
    pall = [float(r["probe_auc_all"]) for r in rows]
    ptail = [float(r["probe_auc_tail"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.plot(ep, dom, "o-", color="black", lw=1.8, ms=5, label="val$\\_$domain$\\_$auc (жив ли D)")
    ax.plot(ep, pall, "s-", color=ACC, lw=1.8, ms=5, label="probe$\\_$auc$\\_$all (проба)")
    ax.plot(ep, ptail, "^--", color="#8ecf8e", lw=1.4, ms=5, label="probe$\\_$auc$\\_$tail")
    if fixed.exists():
        fr = list(csv.DictReader(open(fixed)))
        fe = [int(r["epoch"]) for r in fr]; fp = [float(r["probe_auc_all"]) for r in fr]
        ax.plot(fe, fp, ":", color="#999999", lw=1.4, label="FIXED: probe$\\_$auc$\\_$all")
    ax.axvline(4.5, color="#c9a227", ls="--", lw=1.4)
    ax.text(4.6, 0.86, "refresh D\n(конец эпохи 4)", fontsize=8.5, color="#7a6000")
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.set_xlabel("эпоха", fontsize=11); ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim(0.45, 0.90); ax.legend(fontsize=8.5, loc="lower left", frameon=False)
    fig.tight_layout()
    _save(fig, "fig_our_run.png")


def main() -> None:
    print(f"пишу в {OUT}")
    for fn in (fig_architecture, fig_failure_chain, fig_optimal_d_1d, fig_readout_2d,
               fig_nullspace, fig_room, fig_entanglement, fig_our_run):
        fn()


if __name__ == "__main__":
    main()
