"""Per-event features, on the corrected track fit.

The feature definitions are the ones from
``analysis/excess_mechanism/features.py`` (plus its sequence and pairwise
modules), reused unchanged so that numbers stay comparable with the earlier
study.  One thing is deliberately different: the Cherenkov fit underneath.

That module's fitter does not pass the synthetic-track oracle -- it converges to
a local minimum 4.6 deg off an exact track and leaves 8.8 ns of residual where
there should be none.  ``src/tracks.py`` fixes it with multi-start and a
continuous polish, and recovers all 30 test tracks to 0.000 ns.  Here the fixed
fitter is substituted in, with an adapter for the two derived quantities the old
one returned and the new one does not.

Consequence to keep in mind: feature values here are *not* bit-identical to
``excess_mechanism/features.duckdb``.  Everything in this directory is recomputed
with the fixed fitter, so it is internally consistent; mixing the two tables is
not.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracks                                                 # noqa: E402

# Loaded by path, not by name: this module is itself called `features`, and a
# plain import would find *this* file again and half-initialise it.
_LEGACY_DIR = (Path(__file__).resolve().parents[3] / "analysis" / "excess_mechanism")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        f"_legacy_{name}", _LEGACY_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load("features", "features.py")
_seq = _load("seq", "seq_features.py")
_pair = _load("pair", "pair_features.py")


def _fit_adapter(pos: np.ndarray, t: np.ndarray, q: np.ndarray) -> dict:
    """The corrected fit, presented the way the legacy feature code expects."""
    fit = tracks.fit_track(pos, t, q)
    resid = fit["resid"]
    on_track = np.abs(resid) < tracks.ON_TRACK_NS
    q_sum = float(q.sum())
    return {
        "fit_rms": fit["fit_rms"], "fit_contrast": fit["fit_contrast"],
        "fit_zenith": fit["fit_zenith"], "fit_azimuth": fit["fit_azimuth"],
        "frac_on_track": float(on_track.mean()),
        "q_offtrack_frac": float(q[~on_track].sum() / q_sum) if q_sum > 0 else 0.0,
        "q_weighted_residual": (float((q * resid).sum() / q_sum)
                                if q_sum > 0 else 0.0),
        "_u": fit["u"], "_p0": fit["p0"], "_resid": resid,
    }


_legacy.fit_track = _fit_adapter          # the one deliberate substitution

COLUMNS: list[str] = list(_legacy.COLUMNS) + list(_seq.COLS) + list(_pair.COLS)


def event_features(hits: np.ndarray, probs: np.ndarray, channels: np.ndarray,
                   n_raw: int) -> tuple:
    """All features of one event: base, sequence and pairwise, in COLUMNS order."""
    return (tuple(_legacy.event_features(hits, probs, channels, n_raw))
            + tuple(_seq.seq_features(hits))
            + tuple(_pair.pair_features(hits, channels)))
