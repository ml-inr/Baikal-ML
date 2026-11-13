"""Сводка по всем прогонам «чистого протокола»: правила (lambda, D-LR) для DA.

Собирает все прогоны семейства `*_tenth_*` с константными lr и λ-выключателем
(delayed_constant/constant, feature-оптимизатор adamw), читает:
  * da_config.yaml (в каталоге прогона) — λ, lr дискриминатора, оптимизаторы, сид;
  * da_training_history.csv — последняя валидная проба (или проба каждые 2 эпохи);
и делает одну сводную картинку 2x2 + таблицу results.csv.

Панели:
  A. карта (λ, D-LR): цвет = разрыв dom−game (обман), × = мёртвый критик (dom≈0.5);
  B. разрыв против D-LR по λ (AdamW сплошные, SGD пунктир) + контроль;
  C. состояние критика (dom) против D-LR — где проходит смерть;
  D. изъятие (проба game) — насыщение по всем ячейкам.

Палитра: референсная палитра dataviz-скилла, взята как есть (upstream-валидирована);
node для повторного запуска валидатора на машине нет. λ кодируется ordinal-шкалой
синего: ступени 250..650 (задокументированный порог «не светлее 250» соблюдён).
Смерть критика — статусный critical-цвет с меткой × (не цветом единым).

Запуск: python inference/nu_classifier/analysis/da_parameter_rules/analyze.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = Path(__file__).resolve().parent

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
LAM_LEVELS = [0.0003, 0.001, 0.003, 0.01, 0.03]
# λ — это identity (разные сущности), ей категориальные хью в фиксированном порядке
# палитры (слоты 1-5). Синяя шкала ниже — только для ВЕЛИЧИНЫ (панель A).
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LAM2C = dict(zip(LAM_LEVELS, CAT))
CRIT = "#d03b3b"
GAP_CMAP = LinearSegmentedColormap.from_list(
    "gap", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#104281"])


def cell_info(run: str) -> dict | None:
    cfg = yaml.safe_load(open(Path(run) / "da_config.yaml"))
    t = cfg["training"]
    if "scheduler" in t:
        return None                                   # decayed-LR family — не берём
    f, d = t["feature_optimizer"], t["discriminator_optimizer"]
    ls = cfg["domain_adaptation"]["lambda_scheduler"]
    if f["optimizer"] != "adamw" or ls["type"] not in ("delayed_constant", "constant"):
        return None
    lam = float(ls.get("lambda", 0.0))
    return dict(tag=Path(run).name.split("tenth_")[-1], lam=lam,
                d_opt=d["optimizer"], d_lr=float(d["learning_rate"]), seed=cfg["experiment"]["seed"])


def load_rows() -> pd.DataFrame:
    rows = []
    for run in sorted(glob.glob("trained_models/classifier/*_tenth_*")):
        csv = Path(run) / "da_training_history.csv"
        if not csv.exists():
            continue
        info = cell_info(run)
        if info is None:
            continue
        h = pd.read_csv(csv)
        hp = h.dropna(subset=["probe_auc_game_all"])
        if not len(hp):
            continue
        r, last = hp.iloc[-1], h.iloc[-1]
        log = Path(f"trained_models/classifier/logs/train_E1_weights_tenth_{info['tag']}.log")
        fin = log.exists() and "Training complete" in log.read_text()
        info.update(epoch=int(r.epoch), last_epoch=int(last.epoch),
                    val_auc=r.val_auc, dom=r.val_domain_auc, game=r.probe_auc_game_all,
                    mu=r.probe_auc_all, gap=r.val_domain_auc - r.probe_auc_game_all,
                    finished=bool(fin))
        rows.append(info)
    df = pd.DataFrame(rows)
    # усредняем сиды по одинаковым ячейкам
    keys = ["lam", "d_opt", "d_lr"]
    agg = (df.groupby(keys, as_index=False)
             .agg(dom=("dom", "mean"), game=("game", "mean"), gap=("gap", "mean"),
                  mu=("mu", "mean"), val_auc=("val_auc", "mean"),
                  epoch=("epoch", "max"), n=("tag", "count"),
                  finished=("finished", "all")))
    return df, agg


def main() -> None:
    raw, df = load_rows()
    raw.to_csv(HERE / "results_raw.csv", index=False)
    df.to_csv(HERE / "results.csv", index=False)
    print(df.to_string(index=False))

    ctrl = df[(df.lam == 0)]
    g0 = float(ctrl.game.mean())
    ctrl_band = (g0 - 0.007, g0 + 0.007)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "savefig.facecolor": SURF})
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10.5))
    fig.suptitle("DA-правила: λ × lr дискриминатора (чистый протокол, 1/10, z=128)",
                 fontsize=13, y=0.985)

    # ── A. карта ─────────────────────────────────────────────────────────
    a = ax[0, 0]
    norm = Normalize(0, 0.27)
    for _, r in df.iterrows():
        y = r.lam if r.lam > 0 else 0.00012
        mk = "o" if r.d_opt == "adamw" else "s"
        face = "none" if not r.finished else GAP_CMAP(norm(abs(r.gap)))
        a.scatter(r.d_lr, y, s=190, marker=mk, facecolor=face,
                  edgecolor=GAP_CMAP(norm(abs(r.gap))) if not r.finished else INK2,
                  linewidth=1.6, zorder=3)
        if r.dom < 0.52:
            a.scatter(r.d_lr, y, s=150, marker="x", color=CRIT, linewidth=2.4, zorder=4)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(2e-4, 3e-1); a.set_ylim(8e-5, 5e-2)
    a.set_yticks([0.00012] + LAM_LEVELS)
    a.set_yticklabels(["0\n(контроль)"] + [f"{l:g}" for l in LAM_LEVELS])
    a.set_xlabel("lr дискриминатора (постоянный)"); a.set_ylabel("λ (вкл. на 8-й эпохе)")
    a.set_title("A. Карта: цвет = разрыв dom−game (обман), × = критик мёртв", loc="left", fontsize=11)
    a.grid(True, which="both", color="#e8e7e3", lw=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=GAP_CMAP)
    cb = fig.colorbar(sm, ax=a, fraction=0.046, pad=0.02)
    cb.set_label("|разрыв| (0 = нет обмана)")
    a.scatter([], [], s=90, marker="o", facecolor=INK2, label="AdamW-критик")
    a.scatter([], [], s=90, marker="s", facecolor=INK2, label="SGD-критик")
    a.scatter([], [], s=120, marker="x", color=CRIT, linewidth=2.4, label="dom≤0.52")
    a.scatter([], [], s=90, marker="o", facecolor="none", edgecolor=INK2, label="недозрелые (открытые)")
    a.legend(loc="lower left", fontsize=8, framealpha=0.9)

    def lam_lines(panel, col, ylabel, title, ref=None):
        for lam in LAM_LEVELS:
            for opt, ls_ in (("adamw", "-"), ("sgd", "--")):
                sub = df[(df.lam == lam) & (df.d_opt == opt)].sort_values("d_lr")
                if not len(sub):
                    continue
                panel.plot(sub.d_lr, sub[col], ls_, color=LAM2C[lam], lw=2,
                           marker="o" if opt == "adamw" else "s", ms=8,
                           mfc="none" if not sub.finished.all() else LAM2C[lam],
                           mec=LAM2C[lam], mew=1.6, label=f"λ={lam:g} {opt.upper()}", zorder=3)
        panel.set_xscale("log"); panel.set_xlim(2e-4, 3e-1)
        if ref is not None:
            panel.axhspan(*ref, color="#f0efec", zorder=0)
                # подпись полосы — текстом рядом, не цветом
        panel.grid(True, which="both", color="#e8e7e3", lw=0.6)
        panel.set_xlabel("lr дискриминатора"); panel.set_ylabel(ylabel)
        panel.set_title(title, loc="left", fontsize=11)
        for s in ("top", "right"):
            panel.spines[s].set_visible(False)

    lam_lines(ax[0, 1], "gap", "разрыв dom − game",
              "B. Обман растёт с λ; скорость критика его почти не лечит", ref=(-0.02, 0.02))
    ax[0, 1].text(2.6e-4, 0.025, "полоса контроля (λ=0)", fontsize=8, color=INK2)
    ax[0, 1].axhline(0, color=INK2, lw=1)
    ax[0, 1].legend(fontsize=8, ncol=2)

    lam_lines(ax[1, 0], "dom", "val dom AUC (живой критик)",
              "C. Смерть критика: порог по lr зависит от λ (и исчезает у SGD)")
    ax[1, 0].axhline(0.5, color=CRIT, lw=1.5, ls=":")
    ax[1, 0].text(2.6e-4, 0.503, "dom=0.5 — критик мёртв", fontsize=8, color=CRIT)
    ax[1, 0].legend(fontsize=8, ncol=2)

    d = ax[1, 1]
    d.axhspan(*ctrl_band, color="#f0efec", zorder=0)
    for _, r in df.iterrows():
        if r.lam == 0:
            d.scatter(r.d_lr, r.game, s=170, marker="D", facecolor="#b9b8b2",
                      edgecolor=INK2, zorder=3)
            continue
        mk = "o" if r.d_opt == "adamw" else "s"
        d.scatter(r.d_lr, r.game, s=170, marker=mk, facecolor=LAM2C[r.lam],
                  edgecolor=INK2 if not r.finished else "none", linewidth=1.5, zorder=3)
    d.set_xscale("log"); d.set_xlim(2e-4, 3e-1); d.set_ylim(0.68, 0.80)
    d.grid(True, which="both", color="#e8e7e3", lw=0.6)
    d.set_xlabel("lr дискриминатора"); d.set_ylabel("проба game (свежий читатель)")
    d.set_title("D. Изъятие насыщено: все живые ячейки садятся на одну полку", loc="left", fontsize=11)
    d.text(2.6e-4, ctrl_band[1] + 0.002, "контроль λ=0", fontsize=8, color=INK2)
    for lam in LAM_LEVELS:
        d.scatter([], [], s=90, marker="o", facecolor=LAM2C[lam], label=f"λ={lam:g}")
    d.scatter([], [], s=90, marker="D", facecolor="#b9b8b2", label="λ=0")
    d.legend(fontsize=8, ncol=3, loc="lower right", framealpha=0.9)
    for s in ("top", "right"):
        d.spines[s].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / "da_parameter_rules.png"
    fig.savefig(out, dpi=170)
    print(f"\nкартинка: {out}")


if __name__ == "__main__":
    main()
