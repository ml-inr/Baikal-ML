import datetime
import sys, os, logging

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Project root
PROJECT_ROOT = Path(os.path.abspath("../../.."))
sys.path.insert(0, str(PROJECT_ROOT))

from archive_tracked.inference.prefilter_model.utils import (
    list_h5_parts, load_model, load_mc_test_parts,
    load_exp_parts, load_mc_reco_parts, predict_scores,
    plot_event_3d, load_blind_reco,
    EXP_RECO_COL_NAMES, MC_RECO_COL_NAMES, COMMON_RECO_COLS,
)


# ── Paths ──

MODEL_NAME = "da_prefilter_numu_260409_1917_Qclip_hardlabels_PlatueLR_lambda0.3_MC2M_dmodel128_zflipBothMCExp_FocalLoss"
CHECKPOINT_PATH = f"experiments/numu/{MODEL_NAME}/best_da_model.pth"

EXP_RECO_H5_PATH = "data_manager/data/h5datasets/exp_reco_full_2020.h5"
MC_RECO_H5_PATH = "data_manager/data/h5datasets/baikal_mc_reco.h5"
MC_PARTICLE_TYPES = ["muatm", "nuatm_conv"]


reco_parts = list_h5_parts(str(PROJECT_ROOT / EXP_RECO_H5_PATH), 'exp_reco')
BAD_RECO_PARTS = [
    p for p in reco_parts if 'c01' in p
]

# ── Inference settings ──
DEVICE = "cuda:4"
BATCH_SIZE = 1024
MAX_HITS = 500
SEED = 42

# ── Limiters (set to None for full dataset) ──
MAX_PARTS_EXP = None        # limit parts read from exp_reco.h5
MAX_PARTS_MC  = None        # limit parts per particle type from baikal_mc_reco.h5


def load_reco_data():
    # ── Load exp_reco ──
    df_exp = load_exp_parts(
        h5_path=str(PROJECT_ROOT / EXP_RECO_H5_PATH),
        group_name="exp_reco",
        max_parts=MAX_PARTS_EXP,
        load_reco=True,
    )
    print(f"exp_reco: {len(df_exp):,} events")
    print(f"  reco columns: {[c for c in df_exp.columns if c in EXP_RECO_COL_NAMES]}")

    # ── Load mc_reco ──
    # NOTE: baikal_mc_reco.h5 must be regenerated with the updated root2h5_config_mc_reco.yaml
    # (25 common reco columns + 6 MC-specific vector columns = 31 total).
    # If the file still has the old 21-column format, reco columns will be named reco_0..reco_20.
    df_mc = load_mc_reco_parts(
        h5_path=str(PROJECT_ROOT / MC_RECO_H5_PATH),
        particle_types=MC_PARTICLE_TYPES,
        max_parts_per_particle=MAX_PARTS_MC,
    )
    print(f"mc_reco: {len(df_mc):,} events")
    print(f"  particle counts:\n{df_mc['particle_type'].value_counts().to_string()}")
    print(f"  reco columns: {[c for c in df_mc.columns if c in MC_RECO_COL_NAMES]}")
    return df_exp, df_mc


def enrich_with_preds(df_exp, df_mc):
    model, norm_config, train_config = load_model(
        str(PROJECT_ROOT / CHECKPOINT_PATH), device=DEVICE
    )
    print(f"Normalization: means={norm_config['means']}, stds={norm_config['stds']}")
    print(f"Model amp_clip: {model.amp_clip:.4f} (normalized) "
        f"= Q≈{model.amp_clip * norm_config['stds'][0] + norm_config['means'][0]:.1f} PE")

    print("EXP Prediction...")
    df_exp["score"] = predict_scores(
        model, df_exp["features"].tolist(), norm_config,
        batch_size=BATCH_SIZE, max_hits=MAX_HITS, device=DEVICE,
    )
    print(f"Exp reco scores: mean={df_exp['score'].mean():.4f}, median={df_exp['score'].median():.4f}")

    print("Prediction...")
    df_mc["score"] = predict_scores(
        model, df_mc["features"].tolist(), norm_config,
        batch_size=BATCH_SIZE, max_hits=MAX_HITS, device=DEVICE,
    )
    print(f"MC reco scores: mean={df_mc['score'].mean():.4f}, median={df_mc['score'].median():.4f}")


def save_data_and_preds(df_exp, df_mc):

    # ── Helpers ──────────────────────────────────────────────────────────────────
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
            else pa.FixedSizeListArray.from_arrays(pa.array(flat.ravel(), type=pa_dtype), sample.shape[1])
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

    # ── Save ──────────────────────────────────────────────────────────────────────
    SAVE_DIR = Path(f"./{MODEL_NAME}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    ARRAY_COLS = {"features", "signal_mask", "reco_hit_prty"}
    JOIN_KEYS  = ["h5_part_str", "local_id"]  # match scalars ↔ arrays

    for name, df in [("exp", df_exp), ("mc", df_mc)]:
        scalar_cols = [c for c in df.columns if c not in ARRAY_COLS]
        array_cols  = [c for c in df.columns if c in ARRAY_COLS]

        # Scalars → parquet (supports predicate pushdown)
        df[scalar_cols].to_parquet(SAVE_DIR / f"{name}_scalars.parquet", index=False)

        # Arrays → feather (fast binary, join keys included)
        feather.write_feather(
            _df_to_arrow(df[JOIN_KEYS + array_cols]),
            SAVE_DIR / f"{name}_arrays.arrow",
        )

    print(f"Saved to {SAVE_DIR}/")
    print(f"  exp_scalars.parquet  +  exp_arrays.arrow")
    print(f"  mc_scalars.parquet   +  mc_arrays.arrow")
    print(f"Join key: {JOIN_KEYS}")


def main():
    df_exp, df_mc = load_reco_data()
    enrich_with_preds(df_exp, df_mc)
    save_data_and_preds(df_exp, df_mc)
    print(f'Done')


if __name__=='__main__':
    main()