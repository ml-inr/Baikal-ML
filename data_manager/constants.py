from pathlib import Path

import numpy as np
from dataclasses import dataclass

# ── Catalog ────────────────────────────────────────────────────────────────
# Generated artifact — not committed to git.
CATALOG_PATH = Path(__file__).parent / "catalog_v2.duckdb"

# ── Known H5 paths ─────────────────────────────────────────────────────────
_DATA = Path(__file__).parent / "data" / "h5datasets"
H5_EXP_PATH       = _DATA / "exp.h5"
H5_EXP_RECO_PATH  = _DATA / "exp_reco.h5"

# ROOT source directories (local copies only)
_ROOT = Path(__file__).parent / "data"
ROOT_EXP_DIR      = _ROOT / "exp_root"
ROOT_EXP_RECO_DIR = _ROOT / "exp_reco_root"


@dataclass
class Constants:
    # Water properties
    N: float = 1.37
    COS_C: float = 1 / N
    SIN_C: float = np.sqrt(1 - COS_C**2)
    TAN_C: float = SIN_C / COS_C
    C_PART: float = 299792458.0  # Speed of particles in m/s
    C_LIGHT: float = 218826621.0  # Speed of light in water in m/s
    EPSILON: float = 1e-9  # Small value to avoid division by zero

    # Fixed values used in channels and other calculations
    CHANNEL_DIVISOR: int = 288
    STRING_DIVISOR: int = 36
    STRINGS_PER_CLUSTER: int = 8
