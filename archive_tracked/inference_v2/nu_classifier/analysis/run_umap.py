"""UMAP embedding analysis for nu-classifier checkpoints.

Loads out-of-training MC events from baikal_mc_merged.h5 (excluding NPY training parts)
and exp events from exp.h5 split by nu-classifier score (high vs low), computes UMAP 2D
and 3D projections, and saves interactive Three.js WebGL HTML plots.

Usage (from project root):
    python inference_v2/nu_classifier/analysis/run_umap.py \\
        --checkpoint  experiments/numu/260508_1724_.../best_da_model.pth \\
        --npy-dir     data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8 \\
        --mc-h5       data_manager/data/h5datasets/baikal_mc_merged.h5 \\
        --exp-h5      data_manager/data/h5datasets/exp.h5 \\
        --preds-dir   inference_v2/nu_classifier/preds \\
        [--threshold 0.8] [--n-per-class 10000] [--n-exp-high 5000] [--n-exp-low 5000] \\
        [--min-hits 5] [--min-strings 0] [--batch-size 512] [--device cuda:0] \\
        [--umap-neighbors 15] [--umap-min-dist 0.1] [--seed 42] \\
        [--output-dir inference_v2/nu_classifier/preds] \\
        [--reuse-coords]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _checkpoint_name(checkpoint: str) -> str:
    p = Path(checkpoint)
    return f"{p.parent.name}@{p.stem}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--npy-dir",          required=True)
    parser.add_argument("--mc-h5",            required=True)
    parser.add_argument("--exp-h5",           required=True)
    parser.add_argument("--preds-dir",        default="inference_v2/nu_classifier/preds")
    parser.add_argument("--catalog",          default="data_manager/catalog_v2.duckdb")
    parser.add_argument("--threshold",        type=float, default=0.8)
    parser.add_argument("--n-per-class",      type=int,   default=10_000)
    parser.add_argument("--n-exp-high",       type=int,   default=5_000)
    parser.add_argument("--n-exp-low",        type=int,   default=5_000)
    parser.add_argument("--min-hits",         type=int,   default=8)
    parser.add_argument("--min-strings",      type=int,   default=2)
    parser.add_argument("--batch-size",       type=int,   default=512)
    parser.add_argument("--device",           default="auto")
    parser.add_argument("--umap-neighbors",   type=int,   default=15)
    parser.add_argument("--umap-min-dist",    type=float, default=0.1)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output-dir",       default=None,
                        help="Root preds dir; defaults to --preds-dir")
    parser.add_argument("--reuse-coords",     action="store_true",
                        help="Skip UMAP computation if umap_coords.npz already exists")
    args = parser.parse_args()

    output_root = Path(args.output_dir or args.preds_dir)
    ckpt_name   = _checkpoint_name(args.checkpoint)
    out_dir     = output_root / ckpt_name / "analysis" / f"umap_nmc{args.n_per_class}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"run_umap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )
    logger.info(f"Output dir: {out_dir}")

    coords_path = out_dir / "umap_coords.npz"
    meta_path   = out_dir / "meta.json"

    # ── Resolve device ────────────────────────────────────────────────────────
    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    logger.info(f"Device: {dev}")

    # ── Load models ───────────────────────────────────────────────────────────
    from inference_v2.shared.model_utils import load_model, load_sn_model

    logger.info("Loading nu-classifier model...")
    model, norm_config, train_config = load_model(args.checkpoint, device=dev)
    model.eval()

    with_probs = train_config.get("model", {}).get("input_dim", 5) == 6

    logger.info("Loading sig-noise model...")
    sn_model, _, sn_dev = load_sn_model(device=args.device)

    # ── Load embeddings ───────────────────────────────────────────────────────
    from archive_tracked.inference_v2.nu_classifier.analysis.embedding_loader import (
        load_mc_embeddings_from_h5,
        load_exp_embeddings_from_preds,
    )

    thr_tag   = str(args.threshold).replace(".", "p")
    preds_dir = Path(args.preds_dir) / ckpt_name
    exp_db    = preds_dir / f"exp_thr{thr_tag}.duckdb"

    if not exp_db.exists():
        logger.error(f"Exp predictions DB not found: {exp_db}")
        sys.exit(1)

    logger.info("=== Loading MC embeddings (out-of-training) ===")
    t0 = time.perf_counter()
    df_mc = load_mc_embeddings_from_h5(
        mc_h5       = args.mc_h5,
        catalog     = args.catalog,
        npy_dir     = args.npy_dir,
        model       = model,
        norm_config = norm_config,
        sn_model    = sn_model,
        sn_dev      = sn_dev,
        threshold   = args.threshold,
        n_per_class = args.n_per_class,
        min_hits    = args.min_hits,
        min_strings = args.min_strings,
        batch_size  = args.batch_size,
        device      = dev,
        seed        = args.seed,
    )
    logger.info(f"MC done: {len(df_mc):,} events in {time.perf_counter()-t0:.0f}s")

    logger.info("=== Loading exp embeddings from predictions DB ===")
    t0 = time.perf_counter()
    df_exp = load_exp_embeddings_from_preds(
        preds_db    = str(exp_db),
        catalog     = args.catalog,
        exp_h5      = args.exp_h5,
        model       = model,
        norm_config = norm_config,
        sn_model    = sn_model,
        sn_dev      = sn_dev,
        threshold   = args.threshold,
        n_high      = args.n_exp_high,
        n_low       = args.n_exp_low,
        min_hits    = args.min_hits,
        min_strings = args.min_strings,
        batch_size  = args.batch_size,
        device      = dev,
        seed        = args.seed,
        with_probs  = with_probs,
    )
    logger.info(f"Exp done: {len(df_exp):,} events in {time.perf_counter()-t0:.0f}s")

    df_all = pd.concat([df_mc, df_exp], ignore_index=True)
    logger.info(f"Total events for UMAP: {len(df_all):,}")

    if len(df_all) == 0:
        logger.error("No events loaded — aborting")
        sys.exit(1)

    # ── UMAP ──────────────────────────────────────────────────────────────────
    if args.reuse_coords and coords_path.exists():
        logger.info(f"Reusing cached UMAP coords from {coords_path}")
        npz = np.load(coords_path, allow_pickle=True)
        coords_2d = npz["coords_2d"]
        coords_3d = npz["coords_3d"]
    else:
        try:
            import umap
        except ImportError:
            logger.error("umap-learn not installed: pip install umap-learn")
            sys.exit(1)

        all_embs = np.stack(df_all["embedding"].values)  # (N, d_model)
        logger.info(f"Embedding matrix: {all_embs.shape}")

        logger.info(f"UMAP 2D (n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})...")
        t0 = time.perf_counter()
        coords_2d = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.seed,
            verbose=True,
        ).fit_transform(all_embs)
        logger.info(f"  2D done in {time.perf_counter()-t0:.0f}s")

        logger.info(f"UMAP 3D ...")
        t0 = time.perf_counter()
        coords_3d = umap.UMAP(
            n_components=3,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.seed,
            verbose=True,
        ).fit_transform(all_embs)
        logger.info(f"  3D done in {time.perf_counter()-t0:.0f}s")

        np.savez_compressed(coords_path, coords_2d=coords_2d, coords_3d=coords_3d)
        logger.info(f"Coords cached to {coords_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    from archive_tracked.inference_v2.nu_classifier.analysis.umap_plots import (
        plot_umap_2d_html, plot_umap_3d_html,
        plot_umap_2d_png, plot_umap_3d_png,
    )

    ckpt_short = Path(args.checkpoint).parent.name[:40]
    plot_umap_2d_png(
        df_all, coords_2d,
        save_path = out_dir / "umap_2d.png",
        title     = f"UMAP 2D — {ckpt_short}",
    )
    logger.info(f"Saved 2D PNG: {out_dir / 'umap_2d.png'}")

    plot_umap_3d_png(
        df_all, coords_3d,
        save_path = out_dir / "umap_3d.png",
        title     = f"UMAP 3D — {ckpt_short}",
    )
    logger.info(f"Saved 3D PNG: {out_dir / 'umap_3d.png'}")

    plot_umap_2d_html(
        df_all, coords_2d,
        save_path = out_dir / "umap_2d.html",
        title     = f"UMAP 2D — {ckpt_short}",
    )
    logger.info(f"Saved 2D HTML: {out_dir / 'umap_2d.html'}")

    plot_umap_3d_html(
        df_all, coords_3d,
        save_path = out_dir / "umap_3d.html",
        title     = f"UMAP 3D — {ckpt_short}",
    )
    logger.info(f"Saved 3D HTML: {out_dir / 'umap_3d.html'}")

    # ── Meta ──────────────────────────────────────────────────────────────────
    class_counts = df_all["data_class"].value_counts().to_dict()
    meta = {
        "checkpoint":      args.checkpoint,
        "npy_dir":         args.npy_dir,
        "mc_h5":           args.mc_h5,
        "exp_h5":          args.exp_h5,
        "exp_db":          str(exp_db),
        "threshold":       args.threshold,
        "n_per_class":     args.n_per_class,
        "n_exp_high":      args.n_exp_high,
        "n_exp_low":       args.n_exp_low,
        "min_hits":        args.min_hits,
        "min_strings":     args.min_strings,
        "umap_neighbors":  args.umap_neighbors,
        "umap_min_dist":   args.umap_min_dist,
        "seed":            args.seed,
        "n_events_total":  len(df_all),
        "class_counts":    class_counts,
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Meta written to {meta_path}")
    logger.info("=== UMAP analysis complete ===")
    logger.info(f"Class counts: {class_counts}")


if __name__ == "__main__":
    main()
