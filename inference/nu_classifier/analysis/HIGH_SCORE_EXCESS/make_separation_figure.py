"""Гистограмма logit для разделения ШАЛ / нейтрино (для презентации).

Строит распределения logit для exp_full, muatm, nuatm, nue2 после катов h8s3
(n_sn_hits >= 8, n_sn_strings >= 3) и сошедшегося фита. Данные берутся из того же
пайплайна, что и full-ноутбуки: `excess.full.load_events` (фиты + catalog_v2 +
prediction-БД выбранного чекпоинта).

Запуск (из корня репозитория):
    python inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/make_separation_figure.py \
        --checkpoint <имя каталога в preds> --name separation_logit_<tag>
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from excess import full, survival  # noqa: E402

FIGURES = HERE / "figures"
PREDS = ROOT / "inference" / "nu_classifier" / "preds"

COLOURS = {"exp_full": "#d62728", "muatm": "#1f77b4",
           "nuatm": "#2ca02c", "nue2": "#ff7f0e", "neutrino": "#2ca02c"}
BINS = np.linspace(-8.0, 6.0, 141)
MIN_HITS, MIN_STRINGS = 8, 3
ZOOM = (1.0, 5.0)


def selected(frame, with_fit: bool) -> "np.ndarray":
    """logit событий после катов h8s3 (и сошедшегося фита, если with_fit)."""
    mask = ((frame.n_sn_hits >= MIN_HITS)
            & (frame.n_sn_strings >= MIN_STRINGS)
            & np.isfinite(frame.logit))
    if with_fit:
        mask &= frame.status == "ok"
    return frame.loc[mask, "logit"].to_numpy()


def density(values: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=BINS)
    return counts / (len(values) * np.diff(BINS))


def draw_density(ax, values: np.ndarray, *, colour: str, label: str,
                 dashed: bool = False) -> None:
    """Плотность по ненулевым бинам: пустые бины не рисуем, чтобы лог-шкала
    не давала вертикальных спиков до нижней границы."""
    y = density(values)
    x = 0.5 * (BINS[1:] + BINS[:-1])
    keep = y > 0
    ax.plot(x[keep], y[keep], drawstyle="steps-mid", color=colour, lw=1.7,
            ls="--" if dashed else "-", label=label)


def checkpoint_epoch(checkpoint_path: Path) -> int | None:
    try:
        import torch
        return int(torch.load(checkpoint_path, map_location="cpu",
                              weights_only=False).get("epoch"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True, help="stem файла в figures/")
    ap.add_argument("--population", choices=("scored", "fits"), default="scored",
                    help="scored = все проскоренные события (reader, без фитов); "
                         "fits = сошедшийся фит BARS (full-пайплайн)")
    args = ap.parse_args()

    if args.population == "scored":
        EV, _ = survival.load_events(args.checkpoint)
        frames = {"exp": EV["exp_full"], "muatm": EV["muatm"],
                  "nuatm": EV["nuatm"], "nue2": EV["nue2"]}
        with_fit = False
    else:
        events, covered = full.load_events(args.checkpoint)
        frames = full.domain_frames(events)
        with_fit = True

    values = {
        "exp_full": selected(frames["exp"], with_fit),
        "muatm": selected(frames["muatm"], with_fit),
        "nuatm": selected(frames["nuatm"], with_fit),
        "nue2": selected(frames["nue2"], with_fit),
    }
    values["neutrino"] = np.concatenate([values["nuatm"], values["nue2"]])

    # Метрики разделения по logit (= по score, монотонно).
    neutrino = values["neutrino"]
    auc_mu_nu = float(roc_auc_score(
        np.r_[np.zeros(len(values["muatm"])), np.ones(len(neutrino))],
        np.r_[values["muatm"], neutrino]))
    auc_mu_exp = float(roc_auc_score(
        np.r_[np.zeros(len(values["muatm"])), np.ones(len(values["exp_full"]))],
        np.r_[values["muatm"], values["exp_full"]]))
    auc_nu_exp = float(roc_auc_score(
        np.r_[np.zeros(len(neutrino)), np.ones(len(values["exp_full"]))],
        np.r_[neutrino, values["exp_full"]]))
    print(f"N: " + ", ".join(f"{k}={len(v):,}" for k, v in values.items()))
    print(f"AUC mu/ν={auc_mu_nu:.4f}  mu/exp={auc_mu_exp:.4f}  ν/exp={auc_nu_exp:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    labels = {"muatm": "muatm (EAS)", "nuatm": "nuatm", "nue2": "nue2",
              "neutrino": "MC ν (nuatm+nue2)", "exp_full": "exp_full (данные)"}

    # (а) весь диапазон, по классам
    ax = axes[0]
    for key in ("muatm", "nuatm", "nue2", "exp_full"):
        v = values[key]
        if len(v):
            draw_density(ax, v, colour=COLOURS[key],
                         label=f"{labels[key]}, N={len(v):,}",
                         dashed=(key == "exp_full"))
    ax.set_xlim(BINS[0], BINS[-1])
    ax.set_title("по классам")
    ax.set_ylabel("плотность (1 / единицу logit)")

    # (б) зум на хвост μ vs exp: там идёт «тонкое» разделение
    ax = axes[1]
    for key in ("muatm", "exp_full"):
        v = values[key]
        if len(v):
            draw_density(ax, v, colour=COLOURS[key],
                         label=f"{labels[key]}, N={len(v):,}",
                         dashed=(key == "exp_full"))
    ax.set_xlim(*ZOOM)
    ax.set_title("хвост: exp_full против muatm")

    centres = 0.5 * (BINS[1:] + BINS[:-1])
    in_zoom = (centres >= ZOOM[0]) & (centres <= ZOOM[1])

    def max_in_zoom(values: np.ndarray) -> float:
        return float(np.max(density(values)[in_zoom])) if len(values) else 0.0

    left_max = max(np.max(density(values[k]))
                   for k in ("muatm", "nuatm", "nue2", "exp_full")
                   if len(values[k]))
    right_max = max(max_in_zoom(values[k])
                    for k in ("muatm", "exp_full") if len(values[k]))
    axes[0].set_ylim(1e-6, left_max * 1.6)
    axes[1].set_ylim(1e-6, max(right_max * 1.6, 1e-4))
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("logit")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    ratios = []
    for threshold in (2.0, 3.0, 4.0):
        fe = float((values["exp_full"] > threshold).mean())
        fm = float((values["muatm"] > threshold).mean())
        if fm > 0:
            ratios.append(f"logit>{threshold:.0f}: exp/μ = {fe / fm:.2f}")
    axes[1].text(0.02, 0.04, "\n".join(ratios), transform=axes[1].transAxes,
                 fontsize=9, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85,
                           ec="0.75"))

    population_note = ("все проскоренные события" if args.population == "scored"
                       else "сошедшийся фит BARS")
    fig.suptitle(f"Разделение ШАЛ / нейтрино: h8s3, logit ({population_note})",
                 fontsize=13)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{args.name}.png"
    pdf = FIGURES / f"{args.name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    run_info_path = PREDS / args.checkpoint / "run_info.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
    checkpoint_path = Path(run_info.get("checkpoint", "")) if run_info else None
    payload = {
        "script": "inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/make_separation_figure.py",
        "figure": png.name,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_dir": args.checkpoint,
        "prediction_checkpoint": run_info.get("checkpoint"),
        "prediction_checkpoint_epoch": (checkpoint_epoch(ROOT / checkpoint_path)
                                        if checkpoint_path else None),
        "prediction_db": {src: f"preds/{args.checkpoint}/{src}_thr0p8.duckdb"
                          for src in ("exp_full", "mc_merged")},
        "population": args.population,
        "cuts": {"status": "ok" if with_fit else "not required",
                 "n_sn_hits": MIN_HITS, "n_sn_strings": MIN_STRINGS},
        "n_events": {k: int(len(v)) for k, v in values.items()},
        "auc": {"muatmvneutrino": auc_mu_nu, "muatmvexp": auc_mu_exp,
                "neutrinovexp": auc_nu_exp},
        "bins": {"min": float(BINS[0]), "max": float(BINS[-1]), "n": len(BINS) - 1},
    }
    (FIGURES / f"{args.name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"written {png} and {pdf}")


if __name__ == "__main__":
    main()
