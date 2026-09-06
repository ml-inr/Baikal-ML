"""Column layout of `reco_prty`, anchored to the ROOT->HDF5 converter configs.

`reco_prty` is a bare float array, so the names live outside the data and have to
come from somewhere trustworthy.  The authority is the converter's own
`root_paths.reco_ev` list -- the order it reads the BRecoMuon branches in is the
order the columns are written in.  Each name here is therefore paired with the
branch it came from, and `check_against_converter_config` asserts that the pairs
still match both YAML files.  Call it if you suspect drift; it costs two file
reads.

Two traps this replaces:

* `inference/shared_utils.py` holds the same list in the old tree.  Analyses
  under `inference_v2/` should import from here instead, so the new tree does not
  depend on the old one.  The lists were identical when this file was written --
  the check function is what keeps them from silently diverging.
* `root2h5_config_exp_reco.yaml` also has a key literally named
  `reco_prty_columns`.  **Do not use it**: it lists 13 names for a 25-column
  array.  It is stale documentation, not the schema.

The names cannot be derived from the branch names by a rule.  `fNHits` ->
`nHits` and `fLLFit` -> `LLFit` need opposite treatment of their leading
capitals, so the mapping is written out.
"""
from __future__ import annotations

from pathlib import Path

import yaml

#: (column name, BRecoMuon branch) in written order.  Present in both files.
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

#: `reco_prty` width -> column names, which is how a reader identifies a file.
RECO_COLUMNS_BY_WIDTH = {len(EXP_RECO_COLUMNS): EXP_RECO_COLUMNS,
                         len(MC_RECO_COLUMNS): MC_RECO_COLUMNS}

CONVERTER_CONFIGS = {
    "exp_reco": "data_manager/root2h5/root2h5_config_exp_reco.yaml",
    "mc_reco": "data_manager/root2h5/root2h5_config_mc_reco.yaml",
}


def columns_for(width: int) -> tuple[str, ...]:
    """Column names of a `reco_prty` array of this width."""
    try:
        return RECO_COLUMNS_BY_WIDTH[width]
    except KeyError:
        raise ValueError(
            f"reco_prty has {width} columns; this schema knows "
            f"{sorted(RECO_COLUMNS_BY_WIDTH)}") from None


def check_against_converter_config(project_root: Path) -> dict[str, int]:
    """Assert the schema still matches both converter configs.  Raises if not.

    Compares the branch, not the name: the config's `reco_ev` entries are full
    ROOT paths ending in the branch, and their order is the column order.
    """
    checked = {}
    for source, relative in CONVERTER_CONFIGS.items():
        config = yaml.safe_load((Path(project_root) / relative).read_text())
        branches = [path.rsplit(".", 1)[-1]
                    for path in config["root_paths"]["reco_ev"]]
        expected = [branch for _, branch in RECO_SCALAR_COLUMNS]
        if branches != expected:
            raise AssertionError(
                f"{relative}: reco_ev branches no longer match the schema.\n"
                f"  config:   {branches}\n  schema:   {expected}")
        vectors = config["root_paths"].get("reco_vectors", [])
        vector_branches = [path.rsplit(".", 1)[-1] for path in vectors]
        expected_vectors = list(dict.fromkeys(b for _, b in RECO_VECTOR_COLUMNS))
        if source == "mc_reco" and vector_branches != expected_vectors:
            raise AssertionError(
                f"{relative}: reco_vectors no longer match the schema.\n"
                f"  config:   {vector_branches}\n  schema:   {expected_vectors}")
        checked[source] = len(branches) + 3 * len(vector_branches)
    return checked
