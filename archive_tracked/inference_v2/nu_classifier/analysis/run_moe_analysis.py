"""MoE (Mixture-of-Experts) score analysis for nu-classifier checkpoints.

Inner-joins predictions from a configurable list of models, averages scores,
then produces:
  - score_dist.png     : 2×2 panels, score histograms per data_class for each quality cut
  - suppression_vs_eff.png : suppression factor vs neutrino efficiency, one curve per cut

Quality cuts swept (hardcoded — user-specified):
  h5s0  : min_sn_hits=5,  min_sn_strings=0
  h5s2  : min_sn_hits=5,  min_sn_strings=2
  h8s3  : min_sn_hits=8,  min_sn_strings=3
  h10s3 : min_sn_hits=10, min_sn_strings=3

Usage (from project root):
    python inference_v2/nu_classifier/analysis/run_moe_analysis.py \\
        --preds-dir  inference_v2/nu_classifier/preds \\
        --checkpoints \\
            260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model \\
            260507_2121_da_nu_classifier_h5s0_lambda0.01_thr0.8@best_da_model \\
        --source mc_merged \\
        --threshold 0.8 \\
        --signal-classes nuatm_2020 nue2_2020 \\
        --bg-classes muatm_2020
"""

import argparse
import json
import logging
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

CUTS: OrderedDict = OrderedDict([
    ("h5s0",  {"min_sn_hits": 5,  "min_sn_strings": 0}),
    ("h5s2",  {"min_sn_hits": 5,  "min_sn_strings": 2}),
    ("h8s3",  {"min_sn_hits": 8,  "min_sn_strings": 3}),
    ("h10s3", {"min_sn_hits": 10, "min_sn_strings": 3}),
])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preds-dir",      required=True,
                        help="Root predictions directory (contains checkpoint subdirs)")
    parser.add_argument("--checkpoints",    nargs="+", required=True,
                        help="Checkpoint dir names under --preds-dir (space-separated)")
    parser.add_argument("--source",         default="mc_merged",
                        help="Prediction source: mc_merged | mc_reco | exp | exp_reco")
    parser.add_argument("--threshold",      type=float, default=0.8,
                        help="Sig-noise threshold tag used in DB filenames")
    parser.add_argument("--signal-classes", nargs="+", default=["nuatm_2020", "nue2_2020"],
                        help="data_class values treated as signal for suppression curve")
    parser.add_argument("--bg-classes",     nargs="+", default=["muatm_2020"],
                        help="data_class values treated as background")
    parser.add_argument("--overlay-sources", nargs="*", default=["exp"],
                        help="Extra sources to overlay on score dist (metrics excluded). "
                             "Pass empty string or omit to disable. "
                             "Choices: exp | exp_reco  (default: exp)")
    parser.add_argument("--npy-dir",         default=None,
                        help="NPY dataset dir used for training; its parts are excluded from "
                             "mc_merged predictions to remove training events")
    parser.add_argument("--catalog",        default="data_manager/catalog_v2.duckdb")
    parser.add_argument("--output-dir",     default=None,
                        help="Output directory (default: analysis/moe_results/moe_{N}models_{ts}/)")
    args = parser.parse_args()

    preds_root = Path(args.preds_dir)
    checkpoint_dirs = [str(preds_root / ckpt) for ckpt in args.checkpoints]
    n_models = len(checkpoint_dirs)

    _ts = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "moe_results" / f"moe_{n_models}models_{_ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"run_moe_{_ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )

    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Models ({n_models}):")
    for d in checkpoint_dirs:
        logger.info(f"  {d}")
    logger.info(f"Source: {args.source}  threshold: {args.threshold}")
    logger.info(f"Signal classes: {args.signal_classes}")
    logger.info(f"Background classes: {args.bg_classes}")

    from archive_tracked.inference_v2.nu_classifier.analysis.load import load_moe_preds, load_training_parts

    exclude_parts: set = set()
    if args.npy_dir:
        exclude_parts = load_training_parts(args.npy_dir)
        logger.info(f"Excluding {len(exclude_parts):,} training parts from mc_merged (npy_dir={args.npy_dir})")
    from archive_tracked.inference_v2.nu_classifier.analysis.plots import (
        plot_score_dist_by_class,
        plot_suppression_vs_efficiency,
    )

    # ── Load data for each cut ────────────────────────────────────────────────
    dfs_per_cut: OrderedDict = OrderedDict()
    n_events_per_cut: dict = {}

    for cut_label, cut_cfg in CUTS.items():
        logger.info(f"Loading cut {cut_label}: {cut_cfg}")
        t0 = time.perf_counter()
        df = load_moe_preds(
            checkpoint_dirs = checkpoint_dirs,
            source          = args.source,
            thr             = args.threshold,
            catalog_path    = args.catalog,
            cuts            = cut_cfg,
            exclude_parts   = exclude_parts or None,
        )
        logger.info(f"  {cut_label}: {len(df):,} events in {time.perf_counter()-t0:.1f}s")
        if df.empty:
            logger.warning(f"  {cut_label}: no events — skipping")
            continue
        dfs_per_cut[cut_label] = df
        n_events_per_cut[cut_label] = {
            cls: int((df["data_class"] == cls).sum())
            for cls in df["data_class"].unique()
        }

    if not dfs_per_cut:
        logger.error("No events loaded for any cut — aborting")
        sys.exit(1)

    # ── Overlay extra sources on score dist (metrics stay MC-only) ───────────
    thr_tag = str(args.threshold).replace(".", "p")
    for exp_source in (args.overlay_sources or []):
        avail = [d for d in checkpoint_dirs
                 if (Path(d) / f"{exp_source}_thr{thr_tag}.duckdb").exists()]
        if not avail:
            logger.info(f"{exp_source}: no checkpoints have this DB — skipping")
            continue
        logger.info(f"{exp_source}: loading from {len(avail)}/{n_models} checkpoints")
        for cut_label, cut_cfg in CUTS.items():
            if cut_label not in dfs_per_cut:
                continue
            df_exp = load_moe_preds(
                checkpoint_dirs = avail,
                source          = exp_source,
                thr             = args.threshold,
                catalog_path    = args.catalog,
                cuts            = cut_cfg,
            )
            if df_exp.empty:
                continue
            dfs_per_cut[cut_label] = pd.concat(
                [dfs_per_cut[cut_label], df_exp], ignore_index=True
            )
            logger.info(f"  {cut_label} + {exp_source}: +{len(df_exp):,} events")

    # ── Score distributions ───────────────────────────────────────────────────
    dist_path = out_dir / "score_dist.png"
    logger.info("Plotting score distributions...")
    plot_score_dist_by_class(
        dfs_per_cut = dfs_per_cut,
        save_path   = str(dist_path),
    )
    logger.info(f"Saved: {dist_path}")

    # ── Suppression vs efficiency ─────────────────────────────────────────────
    supp_path = out_dir / "suppression_vs_eff.png"
    logger.info("Plotting suppression vs efficiency...")
    plot_suppression_vs_efficiency(
        dfs_per_cut     = dfs_per_cut,
        signal_classes  = args.signal_classes,
        bg_classes      = args.bg_classes,
        save_path       = str(supp_path),
    )
    logger.info(f"Saved: {supp_path}")

    # ── Meta ──────────────────────────────────────────────────────────────────
    meta = {
        "checkpoints":     args.checkpoints,
        "n_models":        n_models,
        "source":          args.source,
        "threshold":       args.threshold,
        "signal_classes":  args.signal_classes,
        "bg_classes":      args.bg_classes,
        "cuts":            {k: v for k, v in CUTS.items()},
        "n_events_per_cut": n_events_per_cut,
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Meta: {meta_path}")
    logger.info("=== MoE analysis complete ===")


if __name__ == "__main__":
    main()
