"""High-score excess analysis: sources, sampling, figures.

    import excess
    S = excess.draw_all()
    excess.save(excess.panel(S, ["raw_n_hits", "sig_n_hits"]), "multiplicity")

Layout:

* `excess.paths`    -- every filesystem location, defined once
* `excess.sources`  -- the four sources, their groups, masks and blacklists,
                       each exclusion carrying the measurement behind it
* `excess.sampling` -- block sampling, HDF5 loading, `Sample`, `verify`, cache
* `excess.figures`  -- shared-bin comparison histograms, written to `figures/`
* `excess.display`  -- event displays: one 3D panel per per-hit selection

Facts shared with the rest of `inference` live in `inference.shared`:
`reco_schema` (the `reco_prty` columns, checked against the converter configs)
and `h5_hits` (hit variables, the signal-mask convention, the chunk-aware block
reader).  Nothing here imports from the old `inference/` tree.
"""
import sys

from . import paths

# This package lives under `analysis/`, where the notebooks put the repository
# root on the path by hand.  Doing it here too keeps the modules runnable on
# their own -- `python -m excess.display` -- without that line.
if str(paths.ROOT) not in sys.path:
    sys.path.insert(0, str(paths.ROOT))

from inference.shared.h5_hits import (
    HIT_VARS, PRIME_PRTY_COLUMNS, SN_THRESHOLD, STRING_DIVISOR, signal_mask,
)
from inference.shared.reco_schema import EXP_RECO_COLUMNS, MC_RECO_COLUMNS

from . import display, figures, sampling, signal, sources
from .figures import COLOURS, overlay, panel, save as save_figure
from .sampling import (
    Sample, cache_mismatch, connect, design_effect, draw, draw_all,
    draw_or_load, load, part_sizes, save, stat_columns, verify,
)
from .signal import compare as sn_compare
from .signal import features as sn_features
from .signal import load_sn
from .signal import predict_sn_probs as sn_predict
from .signal import summarise as sn_summarise
from .sources import MIN_SN_HITS, MIN_SN_STRINGS, QUALITY, SOURCES, describe

__all__ = [
    "paths", "sources", "sampling", "figures", "signal", "display",
    "load_sn", "sn_features", "sn_predict", "sn_summarise", "sn_compare",
    "SOURCES", "QUALITY", "MIN_SN_HITS", "MIN_SN_STRINGS", "describe",
    "Sample", "draw", "draw_or_load", "draw_all", "load", "save", "verify",
    "connect", "part_sizes", "stat_columns", "design_effect",
    "cache_mismatch",
    "overlay", "panel", "save_figure", "COLOURS",
    "HIT_VARS", "PRIME_PRTY_COLUMNS", "STRING_DIVISOR", "SN_THRESHOLD",
    "signal_mask", "EXP_RECO_COLUMNS", "MC_RECO_COLUMNS",
]
