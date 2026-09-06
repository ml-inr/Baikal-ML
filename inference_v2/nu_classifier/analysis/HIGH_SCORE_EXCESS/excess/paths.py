"""Where everything this analysis reads and writes lives.

One place, so the notebook and the modules cannot disagree about a path.  Import
the names; do not retype the strings.
"""
from __future__ import annotations

from pathlib import Path

#: Directory holding `main.ipynb`.
HERE = Path(__file__).resolve().parents[1]
ROOT = next(p for p in HERE.parents if (p / "CLAUDE.md").exists())

# ── Inputs ────────────────────────────────────────────────────────────────────
H5DIR = ROOT / "data_manager/data/h5datasets"
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"

#: The sig-noise model whose probabilities selected the hits, and the batch size
#: it ran at.  The batch size is part of the identity of the selection, not a
#: speed setting -- see doc/sig_noise_batch_size.md.
SN_TAG = "k_nsol_labelneq0_da_hs128_k0p0001"
SN_BATCH = 256

#: The one checkpoint all four sources were scored with.
MODEL = ("260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256"
         "@best_da_model")
PREDS_DIR = ROOT / "inference_v2/nu_classifier/preds" / MODEL

# ── Outputs ───────────────────────────────────────────────────────────────────
DATA = HERE / "data"
FIGURES = HERE / "figures"
CACHE = DATA / "samples"


def h5(stem: str) -> Path:
    return H5DIR / f"{stem}.h5"


def probs(stem: str) -> Path:
    """The sig-noise probabilities beside a source file."""
    return H5DIR / f"{stem}_probs_{SN_TAG}.h5"


def preds(source: str) -> Path:
    return PREDS_DIR / f"{source}_thr0p8.duckdb"
