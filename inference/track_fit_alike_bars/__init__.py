"""A transparent port of the BARS v2.3.2 muon-track reconstruction.

    from inference.track_fit_alike_bars import fit_events, fit_hits, fit_event
    from inference.track_fit_alike_bars import masks

    fits = fit_events(ev)                       # ev from reader.events()
    fits = fit_hits(chunk.hits)                 # hits already in memory
    fit = fit_event(q, t, x, y, z, channel)     # one event; fit.fit_hits, ...

`masks` supplies the DQM channel mask BARS applies to data, which the HDF5 files do
not carry and which the fit cannot invent: pass it as `channel_mask` when fitting
experimental events.

`geometry` and `isolation` answer the other question -- how likely the modules
around a fitted track were to fire, and how well the track explains the silence
around it (BARS' `fPnonHit`):

    from inference.track_fit_alike_bars import geometry, isolation

    om = geometry.oms("exp_full", "part_s2020_c04_r0031", local_idx=17)
    field = isolation.phit_field(track, om)              # P(>=1 p.e.) per OM
    iso = isolation.pnonhit(track, om, fired=channels)   # BARS-like summary

See README.md for what is reproduced, what is not, and how well it agrees with
BARS' own output.
"""
from . import geometry, isolation, masks
from .config import FitConfig
from .fit import BARS_NAMES, STATUSES, TrackFit, enabled_channels, fit_event
from .frame import FitProgress, fit_events, fit_hits, read_event_hits, tqdm_progress

__all__ = ["FitConfig", "TrackFit", "BARS_NAMES", "STATUSES", "fit_event",
           "fit_events", "fit_hits", "read_event_hits", "enabled_channels",
           "masks", "geometry", "isolation", "FitProgress", "tqdm_progress"]
