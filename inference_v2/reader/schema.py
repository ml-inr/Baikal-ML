"""Every layout constant and convention the reader depends on, in one file.

This is deliberately self-contained: `inference_v2/shared` holds its own copies for
the prediction scripts, and the two are not linked.  The point is that when a field
in the reco production changes, there is exactly one place here to change, and a
function that tells you the change happened.

Nothing in this module reads data.  It describes where things are and what they mean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

# ── Hits ──────────────────────────────────────────────────────────────────────
#: Columns of `raw/data`, in written order: charge (p.e.), time (ns), and the three
#: cluster-centred coordinates (m).  Time is mean-centred per event by the converter
#: (`center_times: true`), so absolute hit time is not recoverable from the file.
HIT_VARS: tuple[str, ...] = ("q", "t", "x", "y", "z")

#: String id of a hit.  Verified against the pipeline, which uses the same divisor.
STRING_DIVISOR = 36

#: `raw/data` chunk length per column, measured.  The cost of a read is the number of
#: chunks it touches, not the number of hits it wants -- see `parts.py`.
HDF5_CHUNK_HITS = 72_461

#: Bytes per hit once in a pandas frame, measured on real samples (72.8 MB for
#: 1,775,688 hits).  Used to turn a megabyte budget into a hit count.
BYTES_PER_HIT = 41

# ── The sig-noise filter ──────────────────────────────────────────────────────
SN_THRESHOLD = 0.8


def signal_mask(probs: np.ndarray, threshold: float = SN_THRESHOLD) -> np.ndarray:
    """Which hits the sig-noise filter calls signal.

    Strictly greater, compared in float32, because that is what every counting site
    in the pipeline does (`data_manager/nu_classifier_ds_builder/io.py`,
    `inference_v2/nu_classifier/predict_*.py`).  The distinction is not academic:
    with `>=`, one event in twenty thousand -- one holding a hit whose probability is
    exactly `float32(0.8)` -- gets a different signal-hit count here than the one
    stored in the prediction database.
    """
    return np.asarray(probs) > np.float32(threshold)


# ── MC truth ──────────────────────────────────────────────────────────────────
#: Columns of `prime_prty`, MC only.
PRIME_PRTY_COLUMNS: tuple[str, ...] = (
    "theta_deg", "phi_deg", "energy_gev", "nucleon_n", "response_muons_n",
    "event_weight")

#: `particle_types.npy` in the training datasets encodes the class as int8.
PARTICLE_TYPE_NAMES = {0: "muatm_2020", 1: "nuatm_2020", 2: "nue2_2020"}

# ── BARS reconstruction ───────────────────────────────────────────────────────
# `reco_prty` is a bare float array, so the names live outside the data.  The
# authority is the converter's own `root_paths.reco_ev` list: the order it reads the
# BRecoMuon branches in is the order the columns are written in.  Each name below is
# therefore paired with its branch, and `check_against_converter_config` asserts the
# pairs still match both YAML files.
#
# Two traps this replaces:
#   * `root2h5_config_exp_reco.yaml` has a key literally named `reco_prty_columns`.
#     Do NOT use it: it lists 13 names for a 25-column array.  Stale documentation.
#   * The names cannot be derived from the branches by a rule -- `fNHits` -> `nHits`
#     and `fLLFit` -> `LLFit` need opposite treatment of their leading capitals.
#
# Measured 2026-08-31: `classBDT` and `classBDTLowE` are **-2.0 everywhere** in
# exp_reco, exp_reco_full_2020 and mc_reco, and also -2.0 in the source ROOT.  The
# BDT was never run for this production; the columns exist but carry no information.
RECO_SCALAR_COLUMNS: tuple[tuple[str, str], ...] = (
    ("thetaRec", "fThetaRec"),
    ("phiRec", "fPhiRec"),
    ("thetaErr", "fThetaErr"),
    ("phiErr", "fPhiErr"),
    ("funcValue", "fFuncValue"),
    ("timeChi2", "fTimeChi2"),
    ("chargeTerm", "fChargeTerm"),
    ("LLFit", "fLLFit"),
    ("nHits", "fNHits"),
    ("nStrings", "fNStrings"),
    ("nOMs", "fNOMs"),
    ("pathLength", "fPathLength"),
    ("timeXYZRec", "fTimeXYZRec"),
    ("covMatrixStatus", "fCovMatrixStatus"),
    ("scfMaxTheta", "fScfMaxTheta"),
    ("scfMinTheta", "fScfMinTheta"),
    ("scfTheta", "fScfTheta"),
    ("scfPhi", "fScfPhi"),
    ("pHit", "fPHit"),
    ("evCenterZ", "fEvCenterZ"),
    ("zDist", "fZDist"),
    ("nTriplets", "fNTriplets"),
    ("nCalls", "fNCalls"),
    ("classBDT", "fClassBDT"),
    ("classBDTLowE", "fClassBDTLowE"),
)

#: MC only: two 3-vectors appended after the scalars, one column per component.
RECO_VECTOR_COLUMNS: tuple[tuple[str, str], ...] = (
    ("xyzRec_x", "fXYZRec"), ("xyzRec_y", "fXYZRec"), ("xyzRec_z", "fXYZRec"),
    ("dirRec_x", "fDirectionRec"), ("dirRec_y", "fDirectionRec"),
    ("dirRec_z", "fDirectionRec"),
)

EXP_RECO_COLUMNS: tuple[str, ...] = tuple(n for n, _ in RECO_SCALAR_COLUMNS)
MC_RECO_COLUMNS: tuple[str, ...] = EXP_RECO_COLUMNS + tuple(
    n for n, _ in RECO_VECTOR_COLUMNS)

#: `reco_prty` width -> column names.  Width is how a reader identifies the file.
RECO_COLUMNS_BY_WIDTH = {len(EXP_RECO_COLUMNS): EXP_RECO_COLUMNS,
                         len(MC_RECO_COLUMNS): MC_RECO_COLUMNS}

CONVERTER_CONFIGS = {
    "exp_reco": "data_manager/root2h5/root2h5_config_exp_reco.yaml",
    "mc_reco": "data_manager/root2h5/root2h5_config_mc_reco.yaml",
}


def reco_columns(width: int) -> tuple[str, ...]:
    """Column names of a `reco_prty` array of this width."""
    try:
        return RECO_COLUMNS_BY_WIDTH[width]
    except KeyError:
        raise ValueError(
            f"reco_prty has {width} columns; this schema knows "
            f"{sorted(RECO_COLUMNS_BY_WIDTH)}") from None


def check_against_converter_config(project_root: Path) -> dict[str, int]:
    """Assert the reco schema still matches both converter configs.  Raises if not.

    Compares the branch, not the name: the config's `reco_ev` entries are full ROOT
    paths ending in the branch, and their order is the column order.  Call this when
    you suspect the production changed; it costs two file reads.
    """
    checked: dict[str, int] = {}
    for source, relative in CONVERTER_CONFIGS.items():
        config = yaml.safe_load((Path(project_root) / relative).read_text())
        branches = [path.rsplit(".", 1)[-1]
                    for path in config["root_paths"]["reco_ev"]]
        expected = [branch for _, branch in RECO_SCALAR_COLUMNS]
        if branches != expected:
            raise AssertionError(
                f"{relative}: reco_ev branches no longer match the schema.\n"
                f"  config: {branches}\n  schema: {expected}")
        vectors = [path.rsplit(".", 1)[-1]
                   for path in config["root_paths"].get("reco_vectors", [])]
        expected_vectors = list(dict.fromkeys(b for _, b in RECO_VECTOR_COLUMNS))
        if source == "mc_reco" and vectors != expected_vectors:
            raise AssertionError(
                f"{relative}: reco_vectors no longer match the schema.\n"
                f"  config: {vectors}\n  schema: {expected_vectors}")
        checked[source] = len(branches) + 3 * len(vectors)
    return checked
