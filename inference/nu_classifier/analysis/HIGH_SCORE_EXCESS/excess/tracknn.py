"""Classifier events joined to the track-NN predictions, without hits or fits.

This is the analysis-side loader for ``inference/reader/track_nn.py``.  It keeps
the same domains as ``excess.high`` (clean exp, test MC thinned by part), but
does not read hits, DQM masks or the BARS fit catalogue.  The only extra data
are the per-event scalars from HDF5 (notably ``prime_theta_deg`` for MC) and the
track-NN row selected by ``event_fk``.

The track-NN database is not a natural sample: a missing row means the event
failed the track selection (canonical sig-noise ``prob >= 0.70``, >= 8 hits on
>= 2 strings, no gate), not that its prediction is unknown.  ``report`` therefore
counts missing rows explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import reader  # noqa: E402

MC_CLASSES = ("muatm_2020", "nuatm_2020", "nue2_2020")
KEYS = ("exp_full", "muatm", "nuatm", "nue2")

#: Domain -> (source, class filter, part-thinning modulus), kept parallel to
#: ``excess.high.DOMAINS`` so the reference samples mean the same thing.
DOMAINS = {
    "exp_full": ("exp_full", None, 1),
    "muatm": ("mc_merged", "muatm_2020", 10),
    "nuatm": ("mc_merged", "nuatm_2020", 10),
    "nue2": ("mc_merged", "nue2_2020", 10),
}

#: Track columns that are useful without the BARS fit.  Calibrated radii are
#: deliberately not attached: for exp/canonical rows they are NULL anyway.
TRACK_COLUMNS = [
    "theta_deg", "phi_deg",
    "direction_x", "direction_y", "direction_z",
    "anchor_x_m", "anchor_y_m", "anchor_z_m",
    "sigma_angle_deg", "sigma_anchor_m",
    "n_raw_hits", "n_hits", "n_strings",
    "selected_threshold", "gate_applied",
    "track_point_scope", "uncertainty_calibration_scope",
]

#: The two exp runs flagged in ``reader.SOURCES["exp_full"].bad_runs``.
BAD_RUNS = tuple(reader.SOURCES["exp_full"].bad_runs)
SCORE_COLUMNS = ["score", "n_sn_hits", "n_sn_strings"]


def _with_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the columns every notebook cell would otherwise repeat."""
    frame = frame.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["logit"] = np.log(frame["score"] / (1.0 - frame["score"]))
    frame["track_cos_theta"] = np.cos(np.deg2rad(frame["track_theta_deg"]))
    frame["track_hit_ratio"] = frame["track_n_hits"] / frame["n_sn_hits"]
    frame["track_string_ratio"] = frame["track_n_strings"] / frame["n_sn_strings"]
    frame["bad_run"] = frame["part_key"].isin(BAD_RUNS)
    return frame


def _report(domain: str, frame: pd.DataFrame) -> dict:
    matched = frame["track_theta_deg"].notna()
    return {
        "домен": domain,
        "событий": len(frame),
        "со_скором": int(frame["score"].notna().sum()),
        "track_matched": int(matched.sum()),
        "track_missing": int((~matched).sum()),
        "n_sn_hits>=8": int((frame["n_sn_hits"] >= 8).sum()),
        "track_n_hits>=8": int((frame["track_n_hits"] >= 8).sum()),
    }


def load_scored_mc(
    checkpoint: str,
    track_db_mc: str | Path,
    data_class: str = "muatm_2020",
    score_floor: float = 0.5,
) -> pd.DataFrame:
    """All ``data_class`` MC events above ``score_floor``, without part thinning.

    The 10% reference in ``load_events`` is too sparse for the high-score tail:
    full ``muatm`` has only ~3.6k events at score >= 0.9.  This loader returns
    the complete score-graded subset, so exp and muatm can be compared in the
    same score groups without a thinning mismatch.
    """
    if score_floor < 0 or score_floor >= 1:
        raise ValueError("score_floor must be in [0, 1)")
    frame = reader.open("mc_merged", checkpoint=checkpoint).events(
        where=[
            reader.NOT_TRAINED,
            f"e.data_class = '{data_class}'",
            f"p.score >= {float(score_floor)}",
        ],
        columns=SCORE_COLUMNS,
    )
    frame = reader.open_track_nn(track_db_mc).attach(
        frame, columns=TRACK_COLUMNS
    )
    return _with_derived(frame)


def load_events(
    checkpoint: str,
    track_db_exp: str | Path,
    track_db_mc: str | Path,
    mc_modulus: int = 10,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load clean exp and part-thinned MC, each left-joined to track-NN.

    Parameters
    ----------
    checkpoint:
        Directory name under ``inference/nu_classifier/preds/``.
    track_db_exp, track_db_mc:
        Track-NN DuckDBs for the same nu-classifier event sets.  They must have
        been written on the same inputs; the reader does not re-derive that.
    mc_modulus:
        Keep every ``mc_modulus``-th MC part (``hash(part_key) % modulus == 0``),
        matching ``excess.high``'s 10% reference sample.  Exp is never thinned.

    Returns
    -------
    ``({exp_full, muatm, nuatm, nue2}, report)``.  The frames contain the
    classifier columns, HDF5 per-event scalars (including ``prime_theta_deg``
    for MC) and ``track_*`` columns.  No ``fit_*`` column is produced.
    """
    if mc_modulus < 1:
        raise ValueError("mc_modulus must be >= 1")
    track_exp = reader.open_track_nn(track_db_exp)
    track_mc = reader.open_track_nn(track_db_mc)

    exp = reader.open("exp_full", checkpoint=checkpoint).events(
        where=[reader.NOT_EXCLUDED],
        columns=SCORE_COLUMNS + ["was_da_target"],
    )
    exp = track_exp.attach(exp, columns=TRACK_COLUMNS)
    exp = _with_derived(exp)

    mc = reader.open("mc_merged", checkpoint=checkpoint).events(
        where=[
            reader.NOT_TRAINED,
            f"hash(l.part_key) % {int(mc_modulus)} = 0",
        ],
        columns=SCORE_COLUMNS,
    )
    mc = track_mc.attach(mc, columns=TRACK_COLUMNS)
    mc = _with_derived(mc)

    out: Dict[str, pd.DataFrame] = {"exp_full": exp}
    for name, data_class in zip(("muatm", "nuatm", "nue2"), MC_CLASSES):
        out[name] = mc[mc["data_class"] == data_class].copy()

    report = pd.DataFrame(
        [_report(name, frame) for name, frame in out.items()]
    )
    return out, report

