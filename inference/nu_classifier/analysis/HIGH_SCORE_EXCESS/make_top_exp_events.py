"""Изображения exp-событий с наибольшим скором (для презентации).

Берёт prediction-БД выбранного чекпоинта, исключает плохие раны (`NOT_EXCLUDED`),
отбирает N событий с максимальным score и рисует их штатным event display.

Выходы (в figures/):
  * <name>.png            — сетка 2×5 (по одному 3D-панелю на событие);
  * <name>.json           — какие именно события и при каких условиях;
  * <name>_top2.png/.json — канонический двухрядный display (1–2 примера).

Запуск (из корня репозитория):
    python inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/make_top_exp_events.py \
        --checkpoint <имя каталога в preds> --name exp_top10_<tag>
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from inference import reader  # noqa: E402
from excess import display  # noqa: E402

FIGURES = HERE / "figures"
PREDS = ROOT / "inference" / "nu_classifier" / "preds"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True, help="stem файлов в figures/")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--label", default="",
                    help="подпись чекпоинта на рисунке (например, 'h8s3 μ-only, e23')")
    ap.add_argument("--min-hits", type=int, default=8)
    ap.add_argument("--min-strings", type=int, default=3)
    args = ap.parse_args()

    r = reader.open("exp_full", checkpoint=args.checkpoint)
    events = r.events(where=[reader.NOT_EXCLUDED],
                      columns=["score", "n_sn_hits", "n_sn_strings"])
    events = events[(events.n_sn_hits >= args.min_hits)
                    & (events.n_sn_strings >= args.min_strings)]
    top = (events.nlargest(args.top, "score")
           [["event_fk", "score", "part_key", "local_idx"]]
           .reset_index(drop=True))
    top["source"] = "exp_full"
    top["data_class"] = "exp_full"
    print(top.to_string(index=False))

    print("читаю хиты и фиты ...", flush=True)
    drawn = display.load(top[["source", "data_class", "part_key", "local_idx", "score"]])
    scores = top["score"].to_numpy()
    print("готово к отрисовке", flush=True)

    # ---- сетка 2×5 ----
    rows, cols = 2, int(np.ceil(len(drawn) / 2))
    fig = plt.figure(figsize=(4.6 * cols, 4.4 * rows))
    for i, event in enumerate(drawn):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        union = display._union(event)
        norm, outside = display._time_norm(
            event.t[union] if union.any() else event.t)
        box = ax.get_position()
        per_inch = 1.0 / fig.get_size_inches()[1]
        display.draw_event(ax, event, norm=norm, marker_scale=180,
                           legend_offset=0.30 * per_inch / box.height,
                           view=display.DEFAULT_VIEW)
        legend = ax.get_legend()
        if legend is not None and i > 0:
            legend.remove()
        status = event.info.get("status", "?") if event.info else "?"
        ax.set_title(f"{i + 1}. score={scores[i]:.4f}  ·  {len(event)} pulses\n"
                     f"{event.part_key} #{event.local_idx}  ·  fit {status}",
                     fontsize=9)
        print(f"  panel {i + 1}/{len(drawn)}", flush=True)

    label = args.label or args.checkpoint
    fig.suptitle(f"exp_full: {len(drawn)} событий с наибольшим скором ({label})",
                 fontsize=13)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    grid_path = FIGURES / f"{args.name}.png"
    fig.savefig(grid_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    run_info_path = PREDS / args.checkpoint / "run_info.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
    payload = {
        "script": "inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/make_top_exp_events.py",
        "figure": grid_path.name,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_dir": args.checkpoint,
        "prediction_checkpoint": run_info.get("checkpoint"),
        "source": "exp_full",
        "selection": f"NOT_EXCLUDED; n_sn_hits>={args.min_hits}, "
                     f"n_sn_strings>={args.min_strings}; top score",
        "events": top.to_dict(orient="records"),
        "note": "цвет — время хита, нормировка на событие; кольцо/крест — фит BARS-подобной реконструкции",
    }
    grid_path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"written {grid_path}")

    # ---- канонический display для 1–2 примеров (с легендой и колорбаром) ----
    n_top2 = min(2, len(drawn))
    fig2 = display.figure(drawn[:n_top2],
                          title=f"exp_full, top-{n_top2} score ({label})",
                          view=display.DEFAULT_VIEW)
    display.save(fig2, f"{args.name}_top2", dpi=250,
                 notes=payload)
    print(f"written {FIGURES / (args.name + '_top2.png')}")


if __name__ == "__main__":
    main()
