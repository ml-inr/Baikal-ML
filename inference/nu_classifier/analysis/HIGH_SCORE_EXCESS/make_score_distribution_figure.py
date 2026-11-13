"""Презентационное распределение logit для одного чекпоинта (слайд).

Повторяет презентационную ячейку survival-ноутбука: плотность logit для exp_full
и МК-классов после катов источника (h8s3), палитра, подписи и линия порога.
Читаются только prediction-БД через `excess.survival` — без хитов и без фитов.

Каты не задаются вручную: они берутся из `run_info.json` каталога предсказаний
(поля `min_hits`/`min_strings`), поэтому заголовок рисунка совпадает с тем, с чем
БД действительно собиралась. Если источники в `run_info.json` расходятся или файла
нет, заголовок остаётся без метки катов, а в provenance пишется `"unknown"`.

Запуск (каталог запуска не важен: reader и пути в скрипте абсолютные):
    cd inference/nu_classifier/analysis/HIGH_SCORE_EXCESS
    python make_score_distribution_figure.py \
        --checkpoint <имя каталога в preds> --name <stem> \
        --model-label "h8s3 μ-only DA (e23)"
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from excess import survival  # noqa: E402

FIGURES = HERE / "figures"
PREDS = ROOT / "inference" / "nu_classifier" / "preds"

# Первые три слота проверенной colorblind-safe палитры.
COLOURS = {"muatm": "#2a78d6", "exp": "#eb6834", "nu": "#1baf7a"}
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"


def fmt_n(n: int) -> str:
    """Сокращённая запись числа событий для легенды."""
    if n >= 1e7:
        return f"{n / 1e6:.1f}M"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(n)


def read_run_info(checkpoint: str) -> dict:
    """`run_info.json` каталога предсказаний (пустой словарь, если файла нет)."""
    path = PREDS / checkpoint / "run_info.json"
    return json.loads(path.read_text()) if path.exists() else {}


def source_cuts(info: dict) -> Optional[Dict[str, int]]:
    """Общие для всех источников каты из `run_info.json`, иначе None."""
    pairs = {(run.get("min_hits"), run.get("min_strings"))
             for run in info.get("runs", {}).values()
             if run.get("min_hits") is not None}
    if len(pairs) != 1:
        return None
    min_hits, min_strings = pairs.pop()
    return {"min_hits": int(min_hits), "min_strings": int(min_strings)}


def checkpoint_epoch(path: Optional[Path]) -> Optional[int]:
    """Эпоха из файла чекпоинта; None, если файла нет или он старый."""
    if path is None:
        return None
    try:
        import torch
        return int(torch.load(path, map_location="cpu",
                              weights_only=False).get("epoch"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="имя каталога в inference/nu_classifier/preds/")
    ap.add_argument("--name", required=True, help="stem файла в figures/")
    ap.add_argument("--model-label", required=True,
                    help="подпись модели на слайде, например «h8s3 μ-only DA (e23)»")
    ap.add_argument("--threshold-logit", type=float, default=4.51,
                    help="рабочий порог на рисунке (score = sigmoid(порог))")
    ap.add_argument("--bins", type=int, default=60, help="число бинов гистограммы")
    ap.add_argument("--x-range", type=float, nargs=2, default=(-5.2, 4.95),
                    metavar=("MIN", "MAX"), help="диапазон logit на рисунке")
    ap.add_argument("--experiment-label", default="Baikal-GVD 2020",
                    help="левая часть подзаголовка")
    args = ap.parse_args()

    if not (PREDS / args.checkpoint).is_dir():
        raise SystemExit(f"нет каталога предсказаний: {PREDS / args.checkpoint}")

    x_range = (float(args.x_range[0]), float(args.x_range[1]))
    threshold_logit = float(args.threshold_logit)
    threshold_score = 1.0 / (1.0 + np.exp(-threshold_logit))

    ev, report = survival.load_events(args.checkpoint)
    print(report.to_string())
    exp, muatm, nuatm, nue2 = (ev["exp_full"], ev["muatm"],
                               ev["nuatm"], ev["nue2"])
    frames = {"muatm": muatm, "exp": exp, "nuatm": nuatm, "nue2": nue2}

    info = read_run_info(args.checkpoint)
    cuts = source_cuts(info)
    tag = f"h{cuts['min_hits']}s{cuts['min_strings']}" if cuts else None
    title = "Classifier score distribution"
    if tag:
        title += f" after {tag} cuts"
    subtitle = (f"{args.experiment_label}  ·  exp_full vs MC  ·  "
                f"{args.model_label}")

    bins = np.linspace(x_range[0], x_range[1], args.bins + 1)

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }):
        fig, ax = plt.subplots(figsize=(10, 5.8))
        fig.subplots_adjust(left=0.09, right=0.985, top=0.78, bottom=0.25)

        # Порог — нейтральная линия, не ещё одна серия.
        ax.axvline(threshold_logit, color=INK_2, lw=1.4, ls=(0, (4, 3)),
                   zorder=2)

        # Непрозрачные линии 2 pt вместо alpha=0.5.
        ax.hist(muatm["logit"], bins=bins, density=True, histtype="step",
                color=COLOURS["muatm"], lw=2.0, zorder=3)
        ax.hist(exp["logit"], bins=bins, density=True, histtype="step",
                color=COLOURS["exp"], lw=2.0, zorder=3)
        ax.hist(nue2["logit"], bins=bins, density=True, histtype="step",
                color=COLOURS["nu"], lw=2.0, ls="--", zorder=3)

        # Подпись прямо у кривой: бирюзовый цвет сам по себе низкоконтрастный.
        ax.text(1.7, 4e-3, "ν (nue2, MC)", color=INK_2, fontsize=11,
                ha="left", va="bottom", zorder=5)

        ax.set_yscale("log")
        ax.set_xlim(*x_range)
        ax.set_ylim(1e-6, 5.0)
        ax.set_xlabel("Classifier logit")
        ax.set_ylabel("Probability density")

        # Рецессивная сетка: только основные линии, под данными.
        ax.grid(True, which="major", axis="both", color=GRID, lw=0.8, ls="-",
                zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(which="major", length=0, pad=6, colors=INK_2)

        fig.text(0.09, 0.955, title, fontsize=16, fontweight="semibold",
                 color=INK, ha="left", va="top")
        fig.text(0.09, 0.895, subtitle, fontsize=11, color=INK_2,
                 ha="left", va="top")

        handles = [
            Line2D([], [], color=COLOURS["muatm"], lw=2.0),
            Line2D([], [], color=COLOURS["exp"], lw=2.0),
            Line2D([], [], color=COLOURS["nu"], lw=2.0, ls="--"),
            Line2D([], [], color=INK_2, lw=1.4, ls=(0, (4, 3))),
        ]
        labels = [
            f"Atmospheric μ (MC)   N = {fmt_n(len(muatm))}",
            f"Experimental data   N = {fmt_n(len(exp))}",
            f"ν (nue2, MC)   N = {fmt_n(len(nue2))}",
            f"MC EAS / ν separation (score = {threshold_score:.3f})",
        ]
        ax.legend(handles, labels, loc="upper center",
                  bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
                  handlelength=2.6, columnspacing=2.4, labelspacing=0.8,
                  fontsize=11.5, labelcolor=INK)

        FIGURES.mkdir(parents=True, exist_ok=True)
        png = FIGURES / f"{args.name}.png"
        pdf = FIGURES / f"{args.name}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

    checkpoint_path = Path(info["checkpoint"]) if info.get("checkpoint") else None
    payload = {
        "script": ("inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/"
                   "make_score_distribution_figure.py"),
        "figure": png.name,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_dir": args.checkpoint,
        "prediction_checkpoint": info.get("checkpoint"),
        "prediction_checkpoint_epoch": (checkpoint_epoch(ROOT / checkpoint_path)
                                        if checkpoint_path else None),
        "prediction_db": {src: f"preds/{args.checkpoint}/{src}_thr0p8.duckdb"
                          for src in ("exp_full", "mc_merged")},
        "population": ("scored: все проскоренные test-МК (NOT_TRAINED) и exp "
                       "(NOT_EXCLUDED); без хитов и фитов"),
        "cuts": cuts if cuts else "unknown",
        "model_label": args.model_label,
        "n_events": {name: int(len(frame)) for name, frame in frames.items()},
        "observed_min": {
            name: {"n_sn_hits": int(frame["n_sn_hits"].min()),
                   "n_sn_strings": int(frame["n_sn_strings"].min())}
            if len(frame) else None
            for name, frame in frames.items()},
        "threshold": {"logit": threshold_logit, "score": threshold_score},
        "above_threshold": {
            name: int((frame["logit"] > threshold_logit).sum())
            for name, frame in frames.items()},
        "x_range": list(x_range),
        "bins": {"min": float(bins[0]), "max": float(bins[-1]),
                 "n": len(bins) - 1},
    }
    (FIGURES / f"{args.name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"written {png} and {pdf}")
    print(f"threshold logit={threshold_logit:.2f} score={threshold_score:.4f}; "
          f"above: " + ", ".join(f"{k}={v}" for k, v in
                                 payload["above_threshold"].items()))


if __name__ == "__main__":
    main()
