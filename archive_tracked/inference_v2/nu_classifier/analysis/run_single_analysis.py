"""Single-checkpoint score analysis for nu-classifier.

Loads predictions from one checkpoint, applies quality cuts, then produces:
  - score_dist.png     : 2×2 panels, score histograms per data_class for each quality cut
  - suppression_vs_eff.png : suppression factor vs neutrino efficiency, one curve per cut

Quality cuts swept (hardcoded):
  h5s0  : min_sn_hits=5,  min_sn_strings=0
  h5s2  : min_sn_hits=5,  min_sn_strings=2
  h8s3  : min_sn_hits=8,  min_sn_strings=3
  h10s3 : min_sn_hits=10, min_sn_strings=3

Usage (from project root):
    python inference_v2/nu_classifier/analysis/run_single_analysis.py \\
        --preds-dir  inference_v2/nu_classifier/preds \\
        --checkpoint 260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32@best_da_model \\
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
    parser.add_argument("--preds-dir",      default=None,
                        help="Root predictions directory (contains checkpoint subdirs)")
    parser.add_argument("--checkpoint",     default=None,
                        help="Checkpoint dir name under --preds-dir")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Full path to the checkpoint predictions dir "
                             "(overrides --preds-dir / --checkpoint)")
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
                        help="Output directory (default: analysis/single_results/{checkpoint}_{ts}/)")
    args = parser.parse_args()

    if args.checkpoint_dir:
        checkpoint_dir = str(Path(args.checkpoint_dir))
        ckpt_name = Path(args.checkpoint_dir).name
    elif args.preds_dir and args.checkpoint:
        checkpoint_dir = str(Path(args.preds_dir) / args.checkpoint)
        ckpt_name = args.checkpoint
    else:
        parser.error("Provide either --checkpoint-dir or both --preds-dir and --checkpoint")

    _ts = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "single_results" / f"{ckpt_name}_{_ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"run_single_{_ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )

    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Checkpoint: {checkpoint_dir}")
    logger.info(f"Source: {args.source}  threshold: {args.threshold}")
    logger.info(f"Signal classes: {args.signal_classes}")
    logger.info(f"Background classes: {args.bg_classes}")

    from archive_tracked.inference_v2.nu_classifier.analysis.load import load_preds, load_training_parts

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
        df = load_preds(
            checkpoint_dir = checkpoint_dir,
            source         = args.source,
            thr            = args.threshold,
            catalog_path   = args.catalog,
            cuts           = cut_cfg,
            exclude_parts  = exclude_parts or None,
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
        db = Path(checkpoint_dir) / f"{exp_source}_thr{thr_tag}.duckdb"
        if not db.exists():
            logger.info(f"{exp_source}: DB not found at {db} — skipping")
            continue
        logger.info(f"{exp_source}: loading overlay")
        for cut_label, cut_cfg in CUTS.items():
            if cut_label not in dfs_per_cut:
                continue
            t0 = time.perf_counter()
            df_exp = load_preds(
                checkpoint_dir = checkpoint_dir,
                source         = exp_source,
                thr            = args.threshold,
                catalog_path   = args.catalog,
                cuts           = cut_cfg,
            )
            if df_exp.empty:
                continue
            dfs_per_cut[cut_label] = pd.concat(
                [dfs_per_cut[cut_label], df_exp], ignore_index=True
            )
            logger.info(f"  {cut_label} + {exp_source}: +{len(df_exp):,} events  ({time.perf_counter()-t0:.1f}s)")

    # ── Score distributions ───────────────────────────────────────────────────
    dist_path = out_dir / "score_dist.png"
    logger.info("Plotting score distributions...")
    plot_score_dist_by_class(
        dfs_per_cut = dfs_per_cut,
        save_path   = str(dist_path),
        title       = f"Nu-classifier score distributions — {ckpt_name}",
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
        title           = f"Suppression vs efficiency — {ckpt_name}",
    )
    logger.info(f"Saved: {supp_path}")

    # ── Meta ──────────────────────────────────────────────────────────────────
    meta = {
        "checkpoint":      args.checkpoint,
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
    logger.info("=== Single-model analysis complete ===")


if __name__ == "__main__":
    main()
