#!/usr/bin/env python3
"""
make_save_preds_streaming.py

Streaming version of make_save_preds.py. Solves two problems:
  1. Never loads the full dataset into RAM — processes chunk_size parts at a time.
  2. Domains (exp / mc) are fully independent — run one or both via --domain.

Output: single Parquet file per domain (exp.parquet / mc.parquet) written
incrementally via ParquetWriter. Arrays (features, signal_mask, reco_hit_prty)
are stored as nested lists natively supported by Parquet.

Usage:
    python make_save_preds_streaming.py --domain exp
    python make_save_preds_streaming.py --domain mc
    python make_save_preds_streaming.py --domain both --chunk-size 20 --device cuda:0
    python make_save_preds_streaming.py --domain exp --save-dir ./my_output
"""

import argparse
import sys
import os
import logging
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.path.abspath("../../.."))
sys.path.insert(0, str(PROJECT_ROOT))

from archive_tracked.inference.prefilter_model.utils import (
    list_h5_parts, load_model,
    load_exp_parts, load_mc_reco_parts,
    predict_scores,
    EXP_RECO_COL_NAMES, MC_RECO_COL_NAMES,
)

# ── Config ────────────────────────────────────────────────────────────────────
#MODEL_NAME      = "da_prefilter_numu_260429_0155_Qclip_softlabels_PlateauLR_lambda0.05_MC2M_dmodel128_zflipBothMCExp_SoftFocalLoss"
MODEL_NAME      = "da_prefilter_numu_260430_0846_Qclip_hardlabels_PlateauLR_lambda0.05_MC2M_dmodel128_zflipBothMCExp_FocalLoss"
CHECKPOINT_PATH = f"experiments/numu/{MODEL_NAME}/best_da_model.pth"
EXP_RECO_H5_PATH   = "data_manager/data/h5datasets/exp_reco_full_2020.h5"
MC_RECO_H5_PATH    = "data_manager/data/h5datasets/baikal_mc_reco.h5"
MC_PARTICLE_TYPES  = ["muatm", "nuatm_conv"]
BAD_EXP_CLUSTERS   = {"c01"}

DEVICE     = "cuda:2"
BATCH_SIZE = 1024
MAX_HITS   = 500


# ── Arrow helpers ─────────────────────────────────────────────────────────────
def _series_to_arrow(series: pd.Series) -> pa.Array:
    sample = series.iloc[0]
    pa_dtype = pa.from_numpy_dtype(sample.dtype)
    lengths = np.fromiter((len(x) for x in series), dtype=np.int32, count=len(series))
    offsets = np.empty(len(series) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    flat = np.concatenate(series.values)
    values = (
        pa.array(flat, type=pa_dtype) if sample.ndim == 1
        else pa.FixedSizeListArray.from_arrays(
            pa.array(flat.ravel(), type=pa_dtype), sample.shape[1]
        )
    )
    return pa.ListArray.from_arrays(offsets, values)


def _df_to_arrow(df: pd.DataFrame) -> pa.Table:
    col_arrays = {}
    for col in df.columns:
        sample = df[col].iloc[0]
        col_arrays[col] = (
            _series_to_arrow(df[col]) if isinstance(sample, np.ndarray)
            else pa.array(df[col].to_numpy())
        )
    return pa.table(col_arrays)


# ── Streaming Parquet writer ──────────────────────────────────────────────────
class StreamingParquetWriter:
    """Appends Arrow tables to a single Parquet file.
    Schema is inferred from the first chunk and reused for all subsequent ones.
    """

    def __init__(self, path: Path, compression: str = "snappy"):
        self.path = path
        self.compression = compression
        self._writer: pq.ParquetWriter | None = None

    def write(self, table: pa.Table) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema,
                                            compression=self.compression)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Chunk processing ──────────────────────────────────────────────────────────
def _predict_and_write(df: pd.DataFrame, model, norm_config,
                       writer: StreamingParquetWriter) -> int:
    if len(df) == 0:
        return 0
    df["score"] = predict_scores(
        model, df["features"].tolist(), norm_config,
        batch_size=BATCH_SIZE, max_hits=MAX_HITS, device=DEVICE,
    )
    writer.write(_df_to_arrow(df))
    return len(df)


# ── Domain runners ────────────────────────────────────────────────────────────
def run_exp(model, norm_config, save_dir: Path, chunk_size: int) -> None:
    all_parts = list_h5_parts(str(PROJECT_ROOT / EXP_RECO_H5_PATH), "exp_reco")
    parts = [p for p in all_parts
             if not any(bad in p for bad in BAD_EXP_CLUSTERS)]
    n_skip = len(all_parts) - len(parts)
    logger.info(f"EXP: {len(parts)} parts  ({n_skip} bad-cluster parts skipped)")

    total = 0
    out_path = save_dir / "exp.parquet"
    with StreamingParquetWriter(out_path) as writer:
        chunks = range(0, len(parts), chunk_size)
        for i in tqdm(chunks, desc="EXP chunks"):
            chunk = parts[i: i + chunk_size]
            df = load_exp_parts(
                h5_path=str(PROJECT_ROOT / EXP_RECO_H5_PATH),
                group_name="exp_reco",
                load_reco=True,
                parts_to_load=chunk,
            )
            total += _predict_and_write(df, model, norm_config, writer)

    logger.info(f"EXP done: {total:,} events → {out_path}")


def run_mc(model, norm_config, save_dir: Path, chunk_size: int) -> None:
    total = 0
    out_path = save_dir / "mc.parquet"
    with StreamingParquetWriter(out_path) as writer:
        with h5py.File(str(PROJECT_ROOT / MC_RECO_H5_PATH), "r") as h5:
            for ptype in MC_PARTICLE_TYPES:
                if ptype not in h5:
                    logger.warning(f"'{ptype}' not found in HDF5, skipping")
                    continue
                parts = sorted(h5[ptype]["raw"]["data"].keys())
                logger.info(f"MC {ptype}: {len(parts)} parts")

                chunks = range(0, len(parts), chunk_size)
                for i in tqdm(chunks, desc=f"MC {ptype}"):
                    chunk = parts[i: i + chunk_size]
                    df = load_mc_reco_parts(
                        h5_path=str(PROJECT_ROOT / MC_RECO_H5_PATH),
                        particle_types=[ptype],
                        parts_to_load=chunk,
                    )
                    total += _predict_and_write(df, model, norm_config, writer)

    logger.info(f"MC done: {total:,} events → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming score prediction + Parquet save"
    )
    parser.add_argument("--domain", choices=["exp", "mc", "both"], default="both",
                        help="Which domain to process (default: both)")
    parser.add_argument("--chunk-size", type=int, default=25,
                        help="Number of H5 parts per chunk (default: 25)")
    parser.add_argument("--save-dir", default=f"./{MODEL_NAME}",
                        help="Output directory (default: ./<MODEL_NAME>)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Save dir: {save_dir}")

    logger.info(f"Loading model from {CHECKPOINT_PATH}")
    model, norm_config, _ = load_model(
        str(PROJECT_ROOT / CHECKPOINT_PATH), device=DEVICE
    )
    logger.info(f"Model ready. amp_clip={model.amp_clip:.4f}")

    if args.domain in ("exp", "both"):
        run_exp(model, norm_config, save_dir, args.chunk_size)
    if args.domain in ("mc", "both"):
        run_mc(model, norm_config, save_dir, args.chunk_size)

    logger.info("All done.")


if __name__ == "__main__":
    main()