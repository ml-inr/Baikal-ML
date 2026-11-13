"""Создать пер-чекпоинтный ноутбук анализа из шаблона.

Шаблоны `template_full_score_range.ipynb`, `template_high_score_range.ipynb`,
`template_score_survival.ipynb` и `template_tracknn.ipynb` содержат всю
обвязку в `excess/`; здесь подставляются CHECKPOINT и параметры конкретного
типа. Полученный ноутбук — обычный, со своими ячейками и выводами: можно
открыть несколько параллельно и сравнивать графики.

Пример:
    python -m inference.nu_classifier.analysis.HIGH_SCORE_EXCESS.make_notebook \
        --checkpoint 261021_2009_..._raking@best_da_model --type full

    python -m inference.nu_classifier.analysis.HIGH_SCORE_EXCESS.make_notebook \
        --checkpoint 261024_1202_..._muOnlyDA_s42@best_da_model --type tracknn \
        --track-db-exp .../exp_full_canonical_p070_track.duckdb \
        --track-db-mc  .../mc_merged_canonical_p070_track.duckdb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRED_ROOT = ROOT / "inference" / "nu_classifier" / "preds"
TEMPLATES = {
    "full": HERE / "template_full_score_range.ipynb",
    "high": HERE / "template_high_score_range.ipynb",
    "survival": HERE / "template_score_survival.ipynb",
    "tracknn": HERE / "template_tracknn.ipynb",
}


def parameter_source(
    kind: str,
    checkpoint: str,
    force_reload: bool,
    *,
    track_db_exp: Path | None = None,
    track_db_mc: Path | None = None,
    score_cuts: list[float] | None = None,
    mc_modulus: int = 10,
) -> str:
    lines = [f'CHECKPOINT = "{checkpoint}"']
    if kind == "high":
        lines.append(f"FORCE_RELOAD = {force_reload}   # True = пересобрать кэш")
    elif kind == "tracknn":
        if track_db_exp is None or track_db_mc is None:
            raise SystemExit("для tracknn нужны --track-db-exp и --track-db-mc")
        lines.extend([
            f"TRACK_DB_EXP = {str(track_db_exp.resolve())!r}",
            f"TRACK_DB_MC = {str(track_db_mc.resolve())!r}",
            f"SCORE_CUTS = {score_cuts or [0.5, 0.7, 0.8, 0.9, 0.95]!r}",
            f"MC_MODULUS = {int(mc_modulus)}",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="имя каталога в inference/nu_classifier/preds/")
    ap.add_argument("--type", choices=sorted(TEMPLATES), required=True)
    ap.add_argument("--out-dir", default=str(HERE / "by_checkpoint"))
    ap.add_argument("--force", action="store_true",
                    help="перезаписать существующий ноутбук")
    ap.add_argument("--force-reload", action="store_true",
                    help="для high: FORCE_RELOAD=True (пересобрать кэш)")
    ap.add_argument("--track-db-exp", type=Path, default=None,
                    help="tracknn: DuckDB с exp-предсказаниями track NN")
    ap.add_argument("--track-db-mc", type=Path, default=None,
                    help="tracknn: DuckDB с MC-предсказаниями track NN")
    ap.add_argument("--score-cuts", default="0.5,0.7,0.8,0.9,0.95",
                    help="tracknn: пороги скора через запятую")
    ap.add_argument("--mc-modulus", type=int, default=10,
                    help="tracknn: прореживание MC по hash(part_key) %% N")
    args = ap.parse_args()

    if not (PRED_ROOT / args.checkpoint).is_dir():
        raise SystemExit(f"нет каталога предсказаний: {PRED_ROOT / args.checkpoint}")
    if args.type == "tracknn":
        for label, path in (("--track-db-exp", args.track_db_exp),
                            ("--track-db-mc", args.track_db_mc)):
            if path is None:
                raise SystemExit(f"для tracknn нужен {label}")
            if not path.is_file():
                raise SystemExit(f"{label}: нет файла {path}")
    try:
        score_cuts = [float(item) for item in args.score_cuts.split(",") if item]
    except ValueError as exc:
        raise SystemExit(f"--score-cuts: {exc}") from exc

    template = TEMPLATES[args.type]
    notebook = json.loads(template.read_text())

    for cell in notebook["cells"]:
        if (cell["cell_type"] == "code"
                and "parameters" in cell.get("metadata", {}).get("tags", [])):
            cell["source"] = parameter_source(
                args.type, args.checkpoint, args.force_reload,
                track_db_exp=args.track_db_exp,
                track_db_mc=args.track_db_mc,
                score_cuts=score_cuts,
                mc_modulus=args.mc_modulus,
            )
            break
    else:
        raise SystemExit(f"в шаблоне {template} нет ячейки с тегом parameters")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.checkpoint}__{args.type}.ipynb"
    if out_path.exists() and not args.force:
        raise SystemExit(f"уже существует: {out_path} (--force, чтобы перезаписать)")

    out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
