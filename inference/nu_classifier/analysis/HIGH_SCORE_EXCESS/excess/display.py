"""Event displays: one 3D panel per event, every selection in one glyph.

The selections are not alternatives to each other -- each answers a different
question -- and a hit answers several at once, so they are drawn nested in one
panel rather than side by side.  Layers, bottom to top:

| layer | means | drawn as |
|---|---|---|
| `is_truth_track`, `is_truth_shower` | MC truth: muon track, shower | halo |
| everything else | pulses no method took | small grey circle |
| union of the methods | selected by **any** of them | filled sphere, colour = time |
| `is_fit` | this project's BARS-like reconstruction | thick edge on the sphere |
| `is_sig` | the sig-noise model, `prob > 0.8` | green square, just outside it |
| `is_scanfit` | BARS' own scanfit mask, as converted | grey dotted outer ring |

The spheres are translucent and the sizes are nested, not overlapping: a square of
the same `s` as a circle has the same bounding box, so a square that hugged a
circle would be scribed *into* it -- its corners reach 1.41 times further.  A hit
taken by everything therefore reads as one glyph with several edges, and a hit
taken by one method shows exactly which -- the co-occurrence is in the glyph, not
in a comparison between panels.

The camera is placed per event by `best_view`: of the angles on a grid, the one
that hides the least of the spheres, with near-ties going to the ordinary
three-quarter view.  Pass `view=(elevation, azimuth)` to fix it instead.

Colour is the hit time, scaled over the **filled** hits and not over all pulses.
The pulses fill the whole readout window -- microseconds -- while the taken ones
sit within a couple of hundred nanoseconds of each other around the light front,
so a scale covering both paints every taken hit the same shade and loses the order
of the hits.  The bar prints how many hits fall outside the range it shows.

**`raw/labels` does not mean the same thing in every file**, and that is the one
mistake this module refuses to let through.  In the MC files it holds the pulse
origin code (`fMagic`): `0` is noise, negative codes are light from the muon
track, `k*10**6 + j` is light from shower `k` -- so "not noise" is `!= 0`, and the
track/shower split is the sign.  In `exp_reco` the same dataset holds BARS'
**scanfit mask**, and in `exp_full` it is zeros: experimental data has no truth at
all.  Asking for truth on an experimental source raises rather than drawing
something plausible.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

# Agg everywhere except where something already drives a screen.  A notebook's
# backend is named `module://...`, and it is the inline one that actually shows a
# figure -- forcing Agg under it would turn every figure into a file nobody looks
# at, which is exactly what a caller asking for a display does not want.
if not matplotlib.get_backend().lower().startswith("module://"):
    matplotlib.use("Agg")
import matplotlib.cm as mplcm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import paths  # noqa: E402
from inference.reader.spec import h5_group  # noqa: E402
from inference.shared.h5_hits import STRING_DIVISOR  # noqa: E402
from inference.track_fit_alike_bars import fit_events, masks  # noqa: E402

#: The h5 file each source lives in, by stem.
SOURCE_STEM = {"mc_merged": "baikal_mc_merged", "mc_reco": "baikal_mc_reco",
               "exp_reco": "exp_reco", "exp_full": "exp_full"}

#: HDF5 group per source when it has several (the MC files do).
SOURCE_GROUPS = {"mc_merged": ("muatm_2020", "nuatm_2020", "nue2_2020"),
                 "mc_reco": ("muatm", "nuatm_conv", "nuatm_prompt", "nue2")}

#: What `raw/labels` holds per source: "truth" (pulse origin code), "scanfit"
#: (BARS' own selection) or "none" (zeros).  Measured, not assumed -- see the
#: module docstring.
LABELS_MEANING = {"mc_merged": "truth", "mc_reco": "truth",
                  "exp_reco": "scanfit", "exp_full": "none"}

#: Where the scanfit mask lives in sources whose `raw/labels` is truth.
SCANFIT_KEY = {"mc_reco": "raw/reco_mask", "exp_reco": "raw/labels"}

#: Short names for the headline counts, keyed like `LABELS`.
SHORT = {"is_sig": "SN", "is_fit": "reco", "is_scanfit": "BARS",
         "is_truth_track": "track", "is_truth_shower": "shower"}

#: Which masks are algorithms' answers -- the union of these is the filled layer.
#: Truth is not one of them: it is the reference the algorithms are asked about,
#: so it gets a halo behind everything and is never part of the union.
METHODS = ("is_sig", "is_fit", "is_scanfit")

#: Square and outer-ring sizes, as multiples of the hit's own size (whose square
#: root is its width).  The quoted figure is the radius each one spans, in units of
#: the sphere's diameter, and they have to be ordered: a square whose box matches a
#: circle's is *not* inscribed in it -- its corners reach 1.41 times further -- so
#: the square has to clear the sphere by its edge *and* the ring by its corner.
#: Sphere 0.50, square's edge 0.63 and corner 0.89, outer ring 0.92.
GLYPH_SIZE = {"is_sig": 1.60, "is_scanfit": 3.40}

#: The reconstruction's cross, sized to reach just past the sphere it marks, and
#: the truth outline, which is drawn on the sphere's own boundary.
CROSS_SIZE = 1.30
TRUTH_OUTLINE_WIDTH = 1.2

#: The reconstruction's cross is white, which reads on any fill; the dark hairline
#: under it is what keeps it readable where the fill is pale -- a late hit -- and
#: is all there is to see in the legend, where the background is the page.
CROSS_CASING_WIDTH = 1.5
CROSS_WIDTH = 1.2



#: Stroke of the contours drawn around a sphere, as (linewidth, dash pattern).
#: Solid and thin: a dashed square at this size reads as a scatter of ticks.
GLYPH_STROKE = {"is_sig": (0.9, (3, 2)), "is_scanfit": (0.7, (2, 3))}

#: The muon tracks, MC only.  Cyan: it is outside the whole time ramp, and the
#: truth outlines gave up their blue, so nothing else in the panel wears it.
TRACK_COLOUR = "#00a6c0"

#: A bundle can hold dozens of muons; drawing them all turns the panel into a
#: thicket, so the first few are drawn and the count is stated in the title.
TRACK_MAX = 8

#: The track's reach and its arrowhead, as shares of the panel's largest extent --
#: not of the track: sized off the track, a head is a fixed fraction of a line
#: whose length is the event's own spread, so a compact event gets a head the size
#: of its whole cluster and a wide one gets one that cannot be seen.  Both are
#: read off the panel, so they come out the same size on every picture.
TRACK_REACH_SHARE = 0.65
TRACK_HEAD_SHARE = 0.05

#: How translucent the spheres are.  A cluster is a pile of hits in a few modules,
#: and at full opacity the front sphere hides the ones behind it entirely.
SPHERE_ALPHA = 0.75

#: The reconstruction is a thin cross over the sphere rather than an edge of it:
#: the sphere's own outline is taken by MC truth, which a hit can carry at the same
#: time as the fit, and one boundary cannot be painted two colours.

#: Colours.  The network's square is a blue-leaning green and the track outline a
#: yellow-leaning one, so "green" never has to be told apart from "green" by
#: position alone.  Both truth outlines sit on the fills, and the fills cover the
#: whole time ramp, so each of them vanishes somewhere on it -- the track one on
#: the dark early hits, the black one on the same.  That is the price of one colour
#: per meaning; the earlier pair (blue, orange) was worse, the orange landing a
#: third of a hair from a plasma colour it could sit on directly.
COLOUR = {"is_fit": "#ffffff", "is_fit_casing": "#1a1a19", "is_sig": "#63d2a4",
          "is_scanfit": "#9a9a9a", "is_truth_track": "#2f9e44",
          "is_truth_shower": "#000000", "missed": "#8c8c8c"}

#: Legend wording, short: the legend sits under the panel, and a long line there
#: costs the panel the width it needs.  The glyph is drawn beside each line anyway.
LABELS = {
    "is_sig": "sig-noise net (square)",
    "is_fit": "reco, this project (cross)",
    "is_scanfit": "scanfit, BARS (outer ring)",
    "is_truth_track": "truth: muon track (outline)",
    "is_truth_shower": "truth: shower (outline)",
}
UNION_LABEL = "taken by any method: fill, colour = time"
TRACK_LABEL = "muon track, arrow = direction of flight"
CONTEXT_LABEL = "taken by none"


@dataclass
class Event:
    """One event and every mask available for it, all as boolean arrays."""

    source: str
    part_key: str
    local_idx: int
    q: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    channel: np.ndarray
    masks: dict[str, np.ndarray]
    info: dict
    #: MC muons of the event, `(n, 7)`: theta, phi (rad), x, y, z (m, already in
    #: the cluster's frame), time, energy.  None for data.
    tracks: np.ndarray | None = None
    #: Camera chosen for this event, filled in by `draw_event`.
    view: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.q)


def _part_arrays(source: str, group: str, part: str, locals_: list[int]) -> dict:
    """Everything stored per hit for a handful of events of one part.

    One pass over the part: the pulses, their channel, the `raw/labels` column and
    -- where it exists -- the scanfit mask, each sliced to the wanted events.
    The probabilities come from the companion file, the same slice.  For MC, the
    muon tracks come too; see `_muon_tracks` for the frame they arrive in.
    """
    stem = SOURCE_STEM[source]
    # The catalog calls the class `muatm_2020` where the `mc_reco` file calls the
    # group `muatm`; the reader knows the one exception and this is its function.
    group = h5_group(source, group)
    out = {"q": [], "t": [], "x": [], "y": [], "z": [], "channel": [],
           "labels": [], "scanfit": [], "prob": [], "tracks": []}
    with h5py.File(paths.h5(stem), "r") as store, \
         h5py.File(paths.probs(stem), "r") as probabilities:
        if part not in probabilities[group]["probs"]:
            raise KeyError(
                f"{paths.probs(stem).name} has no {group}/{part}: the sig-noise "
                f"probabilities cover 10,100 of the 20,004 mc_merged parts, and a "
                f"display without the network panel would be misleading. Pick "
                f"another part, or events that were scored.")
        raw = store[group]["raw"]
        starts = raw["ev_starts"][part]["data"]
        data = raw["data"][part]["data"]
        channels = raw["channels"][part]["data"]
        labels = raw["labels"][part]["data"]
        scanfit = (raw[SCANFIT_KEY[source].split("/", 1)[1]][part]["data"]
                   if source in SCANFIT_KEY else None)
        probs = probabilities[group]["probs"][part]["data"]
        tracks = _muon_tracks(store[group], part, len(starts) - 1)
        for local in locals_:
            a, b = int(starts[local]), int(starts[local + 1])
            out["q"].append(data[a:b, 0].astype(float))
            out["t"].append(data[a:b, 1].astype(float))
            out["x"].append(data[a:b, 2])
            out["y"].append(data[a:b, 3])
            out["z"].append(data[a:b, 4])
            out["channel"].append(channels[a:b].astype(np.int64))
            out["labels"].append(labels[a:b])
            out["scanfit"].append(None if scanfit is None else scanfit[a:b] != 0)
            out["prob"].append(probs[a:b])
            out["tracks"].append(None if tracks is None else tracks[local])
    return out


def _muon_tracks(block, part: str, n_events: int) -> list | None:
    """The MC muons of each event, positions moved into the hits' frame.

    `muons_prty/individ` holds `(n_muons, 7)`: theta and phi in radians, then
    x, y, z, time and energy.  Two things about the frame, both read off the
    converter rather than guessed (`data_manager/root2h5/root2h5_mc_reco.py`):

    * the bulk hit coordinates are shifted by `-cl_centers` (line 408) while the
      muon rows are written untouched (line 470), so the shift has to be redone
      here, per event, with that event's cluster;
    * the angles point along the flight: `muatm` is downgoing by construction, and
      its stored theta is 118-167 degrees, so the vector `(sin t cos p, sin t
      sin p, cos t)` points down the track rather than back up it.

    The reference point, though, is not the light: it is the track's closest
    approach to the *array* centre, a median 260 m from the cluster that saw the
    event (`doc/mc_provenance.md`).  That is why the drawing slides the line along
    itself to the point nearest the hits, rather than drawing it through them.
    """
    if "muons_prty" not in block:
        return None
    individ = block["muons_prty/individ"][part]["data"].astype(float)
    starts = block["muons_prty/mu_starts"][part]["data"]
    # The two MC files lay the cluster centres out differently: `mc_merged` keeps
    # one table for the whole file, `mc_reco` one per part.
    centres = block["clusters_centers"]
    centres = (centres[part]["data"] if part in centres else centres["data"])[:]
    clusters = block["raw/cluster_ids"][part]["data"]
    out = []
    for event in range(n_events):
        rows = individ[int(starts[event]):int(starts[event + 1])].copy()
        rows[:, 2:5] -= centres[int(clusters[event])]
        out.append(rows)
    return out


def _channel_mask(rows: pd.DataFrame):
    """BARS' DQM channel mask, for the sources where the project can build it.

    Only `exp_full`: its mask is a run-level file.  `exp_reco` carries the mask
    per event inside its ROOT files, keyed by event id, which this frame does not
    have -- those events are displayed without it, and the reconstruction panel
    then differs from BARS by exactly that mask.
    """
    if rows["source"].iloc[0] != "exp_full":
        return None
    return masks.dqm_masks("exp_full", rows["part_key"].iloc[0])


def load(events: pd.DataFrame, *, with_fit: bool = True,
         threshold: float = 0.8) -> list[Event]:
    """Every mask this project has for each event of `events`.

    `events` needs `source`, `part_key`, `local_idx` and, for the MC sources with
    several HDF5 groups, `data_class` -- exactly what the reader and the notebooks
    carry.  The reconstruction runs once for the whole frame, not once per event.

    The arrays are read group by group -- one pass per part -- and held by the
    row's position in `events`, not in a list: a selection made across parts (top
    score, say) has the parts interleaved, and a list would then hand the second
    event's hits to whoever asked for it second.
    """
    events = events.reset_index(drop=True)
    if "data_class" not in events.columns:
        events = events.assign(data_class=[
            SOURCE_GROUPS.get(source, (source,))[0] for source in events["source"]])

    stored: dict[int, tuple[int, dict]] = {}
    for (source, group, part), block in events.groupby(
            ["source", "data_class", "part_key"], sort=False):
        arrays = _part_arrays(source, group, part,
                              [int(v) for v in block["local_idx"]])
        for position, local in enumerate(block["local_idx"]):
            stored[int(block.index[position])] = (int(local), {
                key: value[position] for key, value in arrays.items()})

    fits = None
    if with_fit:
        fits = fit_events(events[["source", "data_class", "part_key", "local_idx"]],
                          channel_mask=_channel_mask, keep_objects=True)

    out = []
    for position, (_, row) in enumerate(events.iterrows()):
        local, arrays = stored[position]
        available = {"is_sig": arrays["prob"] > threshold}
        if arrays["scanfit"] is not None:
            available["is_scanfit"] = arrays["scanfit"]
        if LABELS_MEANING[row["source"]] == "truth":
            available["is_truth"] = arrays["labels"] != 0
            available["is_truth_track"] = arrays["labels"] < 0
            available["is_truth_shower"] = arrays["labels"] > 0

        info = {}
        if fits is not None:
            track = fits["fit"].iloc[position]
            taken = getattr(track, "fit_hits", None)
            in_fit = np.zeros(len(arrays["q"]), dtype=bool)
            if taken is not None and len(taken):
                index = np.asarray(taken, dtype=np.int64)
                if index.max() >= in_fit.size:
                    raise ValueError(
                        f"{row['part_key']}#{local}: fit hit {index.max()} is outside "
                        f"the {in_fit.size} pulses the display read -- the two reads "
                        f"disagree about the event")
                in_fit[index] = True
            available["is_fit"] = in_fit
            info = {"status": track.status, "theta_rec": track.theta_rec,
                    "n_hits": track.n_hits, "time_chi2": track.time_chi2,
                    "scf_max_theta": track.scf_max_theta}
        for column in ("score", "logit", "n_sn_hits"):
            if column in events.columns:
                info[column] = row[column]

        out.append(Event(source=row["source"], part_key=str(row["part_key"]),
                         local_idx=local, q=arrays["q"], t=arrays["t"],
                         x=arrays["x"], y=arrays["y"], z=arrays["z"],
                         channel=arrays["channel"], masks=available, info=info,
                         tracks=arrays["tracks"]))
    return out


#: How much wider than life x and y are drawn.  A Baikal cluster is a needle --
#: hundreds of metres tall, tens across -- and at true scale its events are a
#: thread a few millimetres wide.  Stretching the horizontal axes is a display
#: choice, not a measurement: angles of a slanted track are not to scale.
XY_STRETCH = 2.0


def _base(axis, event: Event, true_aspect: bool) -> None:
    """The detector context every panel shares: string lines and the view limits."""
    for x, y in _strings(event):
        axis.plot([x] * 2, [y] * 2, [event.z.min(), event.z.max()],
                  color="0.45", lw=1.2, linestyle=(0, (5, 3)), zorder=0)
    axis.set(xlabel="x, m", ylabel="y, m", zlabel="z, m")
    # Three ticks per horizontal axis: in a 3D projection the x and y labels land
    # near each other, and five numbers apiece run together into an unreadable band.
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.tick_params(labelsize=7, pad=-1)
    if true_aspect:
        extent = max(np.ptp(event.x), np.ptp(event.y), np.ptp(event.z)) or 1.0
        axis.set_box_aspect((XY_STRETCH * np.ptp(event.x) / extent,
                             XY_STRETCH * np.ptp(event.y) / extent,
                             np.ptp(event.z) / extent))


def _sizes(q: np.ndarray, scale: float) -> np.ndarray:
    return scale * np.sqrt(q / max(q.max(), 1e-9))


def _faces(times: np.ndarray, norm) -> np.ndarray:
    """The spheres' colours, translucent enough to read through.

    Set as explicit RGBA rather than through `c=`/`alpha=`: the edge of the same
    sphere has to stay opaque -- it is the fit's own outline -- and a collection
    alpha would fade it along with the face.
    """
    face = plt.get_cmap("plasma")(norm(times))
    face[:, 3] = SPHERE_ALPHA
    return face


#: Fraction of the axes' short side that the drawn 3D box takes.  mplot3d fits
#: the box into the axes with a margin; the search needs the box's size in points
#: to know what one metre is worth on screen, and this is that margin.
BOX_SHARE = 0.8

#: Views tried by the search: elevation, then azimuth, in degrees.  The elevation
#: stops short of both ends: past 60 a vertical cluster is seen along its axis,
#: which is the one angle that hides it, and at 0 the floor of the box collapses
#: and the x and y ticks pile into one corner of the picture.
VIEW_ELEVATIONS = (10, 20, 30, 40, 50, 60)
VIEW_AZIMUTHS = range(0, 360, 15)

#: How much worse than the best a view may be and still be preferred for looking
#: ordinary.  Overlap is flat near the optimum, and a degree further from the
#: default view costs a reader more than a few per cent of hidden sphere.
VIEW_TOLERANCE = 0.05

#: The view the search falls back to when every candidate ties -- an ordinary
#: three-quarter view, which is what mplot3d itself starts from.
DEFAULT_VIEW = (20, -60)


def _project(x, y, z, elev: float, azim: float):
    """The two screen axes of the orthographic view mplot3d draws for (elev, azim).

    Matches the render because `draw_event` asks for `proj_type("ortho")`: under
    perspective the screen position of a hit would depend on its distance, and the
    search would be scoring a picture no one sees.
    """
    e, a = np.radians(elev), np.radians(azim)
    u = -x * np.sin(a) + y * np.cos(a)
    v = (-x * np.cos(a) * np.sin(e) - y * np.sin(a) * np.sin(e) + z * np.cos(e))
    return u, v


def overlap(x, y, z, radius: np.ndarray, elev: float, azim: float) -> float:
    """Total overlap of the spheres as drawn from (elev, azim), in metres.

    Summed over pairs: how much of the smaller sphere is buried in the larger.
    A view with the spheres spread over the screen scores zero, and one that puts
    the cluster end-on -- where every sphere sits on every other -- scores high.
    """
    u, v = _project(x, y, z, elev, azim)
    du = u[:, None] - u[None, :]
    dv = v[:, None] - v[None, :]
    distance = np.hypot(du, dv)
    buried = np.maximum(0.0, (radius[:, None] + radius[None, :]) - distance)
    np.fill_diagonal(buried, 0.0)
    return float(buried.sum() / 2)


def string_overlap(strings: np.ndarray, width: float, elev: float,
                   azim: float) -> float:
    """How much the detector's strings run into one another from (elev, azim).

    Every string is drawn over the same height, so on screen they are parallel
    segments spanning the same range and two of them merge exactly when their
    horizontal separation drops below the point where a reader can still tell them
    apart.  What this rules out is looking along a row of strings -- the view in
    which the detector reads as two or three vertical lines with everything piled
    on them.  Only x and y matter: the projection's horizontal axis has no z in it.
    """
    u, _ = _project(strings[:, 0], strings[:, 1], np.zeros(len(strings)), elev,
                    azim)
    gap = np.abs(u[:, None] - u[None, :])
    buried = np.maximum(0.0, width - gap)
    np.fill_diagonal(buried, 0.0)
    return float(buried.sum() / 2)


def _distance_to_default(view: tuple[float, float]) -> float:
    return abs(view[0] - DEFAULT_VIEW[0]) + abs((view[1] - DEFAULT_VIEW[1]) % 360)


def _least_overlap(candidates, score):
    """The least-overlapping candidate, ties and near-ties going to the ordinary one.

    Overlap is flat near its minimum -- a perfectly axial cluster has whole ranges
    of angle hiding exactly the same spheres -- so among the views within
    VIEW_TOLERANCE of the best, the one nearest the default view is taken.  That
    keeps the choice stable against a hit moving by a millimetre, and spends the
    last few per cent of separation on a familiar angle instead.
    """
    scored = [(score(elev, azim), (elev, azim)) for elev, azim in candidates]
    floor = min(value for value, _ in scored)
    near = [view for value, view in scored
            if value <= floor + VIEW_TOLERANCE * abs(floor) + 1e-9]
    best = min(near, key=_distance_to_default)
    return best, score(*best)


def best_view(x, y, z, radius: np.ndarray, strings: np.ndarray | None = None,
              string_width: float = 0.0) -> tuple[float, float]:
    """The (elevation, azimuth) that hides the least, of the hits and the strings.

    Hits and strings are priced in the same metres and added, so a view that saves
    a little sphere overlap by stacking four strings costs more than it saves.

    A coarse grid first, then two narrowing passes on its winner: the overlap is a
    smooth function of the angle, and 15 degrees of azimuth is a visible step.
    """
    def score(elev, azim):
        total = overlap(x, y, z, radius, elev, azim)
        if strings is not None and len(strings) > 1:
            total += string_overlap(strings, string_width, elev, azim)
        return total

    best, _ = _least_overlap([(e, a) for e in VIEW_ELEVATIONS
                              for a in VIEW_AZIMUTHS], score)
    for step in (5.0, 1.0):
        centre_e, centre_a = best
        grid = [(round(float(e), 2), round(float(a), 2))
                for e in np.arange(centre_e - 2 * step, centre_e + 2 * step + 1e-9,
                                   step)
                for a in np.arange(centre_a - 2 * step, centre_a + 2 * step + 1e-9,
                                   step)
                if VIEW_ELEVATIONS[0] <= e <= VIEW_ELEVATIONS[-1]]
        best, _ = _least_overlap(grid, score)
    return round(best[0], 1), round(best[1] % 360, 1)


def _strings(event: Event) -> np.ndarray:
    """Each hit string's axis, as (x, y) -- one row per string the event touched."""
    channels = event.channel // STRING_DIVISOR
    return np.array([[event.x[channels == string].mean(),
                      event.y[channels == string].mean()]
                     for string in np.unique(channels)])


def _union(event: Event) -> np.ndarray:
    """Hits taken by at least one method -- the layer that carries the colour."""
    union = np.zeros(len(event), dtype=bool)
    for name in METHODS:
        if name in event.masks:
            union |= event.masks[name]
    return union


def _time_norm(times: np.ndarray) -> tuple[mcolors.Normalize, int]:
    """A colour scale for the hit times that survives the readout window.

    The pulses of an event fill the whole window -- several microseconds -- while
    the ones a method takes sit within a couple of hundred nanoseconds of the light
    front.  Scaling the colour to the pulses therefore paints every taken hit the
    same shade, and the order of the hits is the physics this panel exists to show.
    Scaling to the taken hits' own min and max fixes that but lets one stray hit
    flatten the rest, so the range is median +/- 3 scaled MADs.  Returns the norm
    and how many of `times` fall outside it.
    """
    if len(times) == 0:
        return mcolors.Normalize(0.0, 1.0), 0
    median = float(np.median(times))
    spread = 1.4826 * float(np.median(np.abs(times - median)))
    if not np.isfinite(spread) or spread <= 0:
        # More than half the times identical: fall back to the spread it has, and
        # to a nominal width when they are all the same.
        spread = float(np.ptp(times)) / 6.0 or 50.0
    low, high = median - 3 * spread, median + 3 * spread
    # When the robust range covers most of the spread anyway, take the spread: no
    # hit is then drawn in the end colour for nothing.
    if 0 < float(np.ptp(times)) <= 2.5 * (high - low):
        low, high = float(np.min(times)), float(np.max(times))
    norm = mcolors.Normalize(vmin=low, vmax=high)
    outside = int(((times < norm.vmin) | (times > norm.vmax)).sum())
    return norm, outside


def _headline(event: Event) -> str:
    """The event, and the counts to check the glyphs against."""
    first = (f"{event.part_key} #{event.local_idx} -- {event.source}, "
             f"{len(event)} pulses")
    counts = [f"{SHORT[name]} {int(event.masks[name].sum())}"
              for name in METHODS if name in event.masks]
    for name in ("is_truth_track", "is_truth_shower"):
        if name in event.masks:
            counts.append(f"truth {SHORT[name].lower()} {int(event.masks[name].sum())}")
    if "is_truth" in event.masks:
        taken = np.zeros(len(event), dtype=bool)
        for name in METHODS:
            if name in event.masks:
                taken |= event.masks[name]
        missed = int((event.masks["is_truth"] & ~taken).sum())
        counts.append(f"truth light missed by all methods {missed}")
    lines = [first, "   ".join(counts)]
    if event.tracks is not None and len(event.tracks):
        counted = f"muon tracks {len(event.tracks)}"
        if len(event.tracks) > TRACK_MAX:
            counted += f" (first {TRACK_MAX} drawn)"
        lines.append(counted)
    if event.info:
        lines.append("   ".join(
            f"{key}={value:.3g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in event.info.items()))
    return "\n".join(lines)


def _legend(axis, drawn: dict, fontsize: float = 8, y_offset: float = 0.073) -> None:
    """What each glyph means, listed for the layers this event actually has.

    The handles are the scatters themselves, not copies of their looks.  A proxy
    is what drifts: drawn by hand it had a line through the square, because the
    dash pattern of a proxy belongs to the line and not to the marker's edge.
    """
    order = (("track", TRACK_LABEL), ("union", UNION_LABEL),
             ("is_fit", LABELS["is_fit"]),
             ("is_sig", LABELS["is_sig"]), ("is_scanfit", LABELS["is_scanfit"]),
             ("is_truth_track", LABELS["is_truth_track"]),
             ("is_truth_shower", LABELS["is_truth_shower"]),
             ("rest", CONTEXT_LABEL))
    if "track" in drawn:
        # A quiver in a legend is a blob: one arrowhead on a line says the same.
        drawn = dict(drawn, track=Line2D([], [], color=TRACK_COLOUR, lw=1.4,
                                         marker=">", markersize=5))
    if "is_fit" in drawn:
        # The cross itself is the handle, and at the legend's marker size its dark
        # hairline covers the white line it edges -- the swatch came out black
        # while the panel's crosses are white.  A filled plus carries the two
        # colours in the proportions the panel shows.
        drawn = dict(drawn, is_fit=Line2D([], [], ls="", marker="P", markersize=6,
                                          mfc=COLOUR["is_fit"],
                                          mec=COLOUR["is_fit_casing"], mew=0.6))
    handles = [drawn[key] for key, _ in order if key in drawn]
    labels = [text for key, text in order if key in drawn]
    # Under the panel, in two columns: inside the axes it lands exactly where a
    # vertical cluster puts its hits, and to the right it costs the panel width.
    # The handles are the scatters, and a scatter draws in the legend at its own
    # marker size -- which is the hit's charge, up to a third of an inch across.
    axis.legend(handles=handles, labels=labels, fontsize=fontsize, markerscale=0.45,
                loc="upper center", bbox_to_anchor=(0.5, -y_offset),
                framealpha=0.9, ncol=2, columnspacing=1.2, labelspacing=0.5)


def draw_event(axis, event: Event, *, norm=None, marker_scale: float = 300,
               true_aspect: bool = True, legend_offset: float = 0.073,
               view: tuple[int, int] | None = None):
    """The whole event in one panel: every selection as a nested glyph.

    The layers are drawn back to front by hand -- `computed_zorder` is off --
    because mplot3d otherwise re-sorts the collections by depth, which would slide
    a halo in front of the sphere it belongs behind.

    The camera is placed by `best_view` unless `view` says otherwise, so one event
    is not read from an angle that piles its spheres up.
    """
    _base(axis, event, true_aspect)
    axis.computed_zorder = False
    axis.set_proj_type("ortho")
    size = _sizes(event.q, marker_scale)
    union = _union(event)

    if view is None:
        if union.sum() >= 2:
            marked = np.flatnonzero(union)
            display = np.column_stack([XY_STRETCH * event.x,
                                       XY_STRETCH * event.y, event.z])
            # One metre on screen, from the drawn box: a marker's radius is fixed
            # in points, so its size in metres grows as the event gets smaller --
            # which is exactly why a compact event's spheres cover each other.
            spread = np.ptp(display, axis=0)
            axis_box = axis.get_position()
            size_in = axis.figure.get_size_inches()
            short_pt = 72 * min(axis_box.width * size_in[0],
                                axis_box.height * size_in[1])
            points_per_metre = BOX_SHARE * short_pt / float(spread.max())
            radius = 0.5 * np.sqrt(size[marked]) / points_per_metre
            # A string is as wide as the hits drawn on it: that is what a reader
            # needs to tell one string from the next, not the 1.2 pt of its line.
            strings = XY_STRETCH * _strings(event)
            view = best_view(display[marked, 0], display[marked, 1],
                             display[marked, 2], radius, strings,
                             2.0 * float(np.median(radius)))
        else:
            view = DEFAULT_VIEW
    axis.view_init(elev=view[0], azim=view[1])
    event.view = tuple(view)

    drawn: dict = {}
    if norm is None:
        norm = mcolors.Normalize()

    rest = ~union
    if rest.any():
        drawn["rest"] = axis.scatter(
            event.x[rest], event.y[rest], event.z[rest], s=size[rest] * 0.6,
            facecolors="none", edgecolors=COLOUR["missed"], linewidths=0.6,
            alpha=0.45, depthshade=False, zorder=2)

    if union.any():
        drawn["union"] = axis.scatter(
            event.x[union], event.y[union], event.z[union], s=size[union],
            facecolors=_faces(event.t[union], norm), edgecolors="none",
            depthshade=False, zorder=3)

    # Truth outlines every hit it owns, taken or not: a truth hit no method took is
    # exactly what one looks for, and it would otherwise be a grey circle among
    # hundreds.  Drawn at the sphere's own size, so on a filled hit it lands on the
    # sphere's boundary, and on a bare one it is an empty ring of that size.
    for step, name in enumerate(("is_truth_track", "is_truth_shower")):
        if name in event.masks and event.masks[name].any():
            keep = event.masks[name]
            drawn[name] = axis.scatter(
                event.x[keep], event.y[keep], event.z[keep], s=size[keep],
                facecolors="none", edgecolors=COLOUR[name],
                linewidths=TRUTH_OUTLINE_WIDTH, depthshade=False,
                zorder=4.2 + 0.2 * step)

    in_fit = event.masks.get("is_fit", np.zeros(len(event), dtype=bool))
    if in_fit.any():
        axis.scatter(event.x[in_fit], event.y[in_fit], event.z[in_fit],
                     s=size[in_fit] * CROSS_SIZE, marker="+",
                     c=COLOUR["is_fit_casing"], linewidths=CROSS_CASING_WIDTH,
                     depthshade=False, zorder=6)
        drawn["is_fit"] = axis.scatter(
            event.x[in_fit], event.y[in_fit], event.z[in_fit],
            s=size[in_fit] * CROSS_SIZE, marker="+", c=COLOUR["is_fit"],
            linewidths=CROSS_WIDTH, depthshade=False, zorder=6.1)

    for zorder, (name, marker) in enumerate(
            (("is_sig", "s"), ("is_scanfit", "o")), start=8):
        if name in event.masks and event.masks[name].any():
            keep = event.masks[name]
            width, dashes = GLYPH_STROKE[name]
            drawn[name] = axis.scatter(
                event.x[keep], event.y[keep], event.z[keep],
                s=size[keep] * GLYPH_SIZE[name], marker=marker,
                facecolors="none", edgecolors=COLOUR[name], linewidths=width,
                linestyles="solid" if dashes is None else [(0, dashes)],
                depthshade=False, zorder=zorder)

    if event.tracks is not None and len(event.tracks):
        # Drawn last, over every glyph: the track is the ribbon the event hangs
        # off, and a square drawn on top of it hides where it runs.
        #
        # It is slid along itself until it passes closest to the hits --
        # `fTracks.fX/fY/fZ` is the point of closest approach to the *array*
        # centre, which `doc/mc_provenance.md` measures at a median 260 m from the
        # cluster that saw the event -- and stretched with the hits, or the arrow
        # would not lie along the light it produced.
        stretched = np.column_stack([XY_STRETCH * event.x, XY_STRETCH * event.y,
                                     event.z])
        extent = float(np.ptp(stretched, axis=0).max()) or 1.0
        reach = TRACK_REACH_SHARE * extent
        centroid = np.array([XY_STRETCH * event.x.mean(),
                             XY_STRETCH * event.y.mean(), event.z.mean()])
        for row in event.tracks[:TRACK_MAX]:
            theta, phi = float(row[0]), float(row[1])
            direction = np.array([XY_STRETCH * np.sin(theta) * np.cos(phi),
                                  XY_STRETCH * np.sin(theta) * np.sin(phi),
                                  np.cos(theta)])
            direction /= np.linalg.norm(direction)
            point = np.array([XY_STRETCH * row[2], XY_STRETCH * row[3], row[4]])
            base = point + ((centroid - point) @ direction) * direction
            drawn["track"] = axis.quiver(
                *base - reach * direction, *(2 * reach * direction),
                color=TRACK_COLOUR, linewidth=1.4, zorder=20,
                arrow_length_ratio=TRACK_HEAD_SHARE * extent / (2 * reach))

    _legend(axis, drawn, y_offset=legend_offset)
    axis.set_title(_headline(event), fontsize=9.5)


def figure(events, *, marker_scale: float = 300, true_aspect: bool = True,
           view: tuple[float, float] | None = None, title: str | None = None):
    """One 3D panel per event, one event per row: a single event is a single image.

    Hit times are centred per event, so each row carries its own colourbar --
    a scale shared across events would compare different zeros.  Each row places
    its own camera unless `view` fixes one for all of them.
    """
    rows = len(events)
    # Portrait: a Baikal cluster is a needle -- hundreds of metres tall, tens
    # across -- and a square figure maroons it in whitespace.
    fig = plt.figure(figsize=(5.6, 8.0 * rows))
    # Everything under a panel is placed in inches, not in figure fractions: the
    # height of a row does not change with the number of rows, and fractions of
    # the figure would scale the gaps with `rows` alone.
    per_inch = 1.0 / (8.0 * rows)
    # The gap between rows is in inches too, and has to hold the legend and the
    # colourbar of the row above it -- left to the default hspace, the bar lands
    # on the next panel's title.
    # Only the legend sits under a panel now: the colourbar moved to its right,
    # and the fit's verdict went into the title.
    below, top_margin, bottom_margin = 1.00 * per_inch, 0.88, 0.11
    row_height = (top_margin - bottom_margin - (rows - 1) * below) / rows
    if row_height > 0.05:
        fig.subplots_adjust(top=top_margin, bottom=bottom_margin,
                            hspace=below / row_height)
    scales = []
    for i, event in enumerate(events):
        # Over the hits that carry the colour, not over all pulses: the pulses fill
        # the readout window, the taken ones sit in the light front, and one scale
        # for both leaves every taken hit the same shade.
        union = _union(event)
        norm, outside = _time_norm(event.t[union] if union.any() else event.t)
        axis = fig.add_subplot(rows, 1, i + 1, projection="3d")
        box = axis.get_position()
        draw_event(axis, event, norm=norm, marker_scale=marker_scale,
                   true_aspect=true_aspect, view=view,
                   legend_offset=0.45 * per_inch / box.height)
        scales.append((axis, event, norm, outside))

    fig.canvas.draw()
    for axis, event, norm, outside in scales:
        box = axis.get_position()
        cax = fig.add_axes((box.x1 + 0.005, box.y0 + 0.3 * box.height, 0.012,
                            0.4 * box.height))
        bar = fig.colorbar(mplcm.ScalarMappable(norm=norm, cmap="plasma"),
                           cax=cax)
        label = "hit time, ns" + (f"  ({outside} outside the range)" if outside
                                  else "")
        bar.set_label(label, fontsize=8)
        bar.ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, fontsize=12)
    # Carried on the figure so `save` can write it without the caller repeating
    # what it already passed in: a display is an argument about these events, and
    # the event ids and mask counts are what makes it checkable later.
    fig.display_provenance = {
        "events": [{"source": event.source, "part_key": event.part_key,
                    "local_idx": event.local_idx, "n_pulses": len(event),
                    "view": event.view,
                    "masks": {name: int(mask.sum())
                              for name, mask in event.masks.items()},
                    "info": event.info}
                   for event in events],
    }
    return fig


def save(fig, name: str, *, dpi: int = 200, notes: dict | None = None) -> Path:
    """Write the figure into `figures/`, with a sidecar saying what is in it.

    The sidecar is the point: a display is an argument about one event, and
    without the event id, the masks shown and the fit's verdict it cannot be
    checked later -- which is the failure `doc/AUDIT.md` Q8/Q9 records.
    """
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    path = paths.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    provenance = getattr(fig, "display_provenance", {})
    plt.close(fig)
    payload = {"script": "excess/display.py", "figure": path.name, "dpi": dpi,
               "written": datetime.now().isoformat(timespec="seconds"),
               **provenance, **(notes or {})}
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"written {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default="exp_full", choices=sorted(SOURCE_STEM))
    parser.add_argument("--data-class", default=None,
                        help="HDF5 group; default is the source's first")
    parser.add_argument("--part", default=None,
                        help="part key, e.g. part_s2020_c05_r0196")
    parser.add_argument("--local-idx", default=None,
                        help="comma-separated event positions in the part")
    parser.add_argument("--from-parquet", type=Path, default=None,
                        help="frame with source/part_key/local_idx (and data_class)")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--no-fit", action="store_true",
                        help="skip the reconstruction ring (it costs a fit per event)")
    parser.add_argument("--name", default=None, help="output file stem")
    args = parser.parse_args()

    if args.from_parquet is not None:
        frame = pd.read_parquet(args.from_parquet).head(args.rows)
    else:
        if not (args.part and args.local_idx):
            parser.error("нужен либо --from-parquet, либо --part с --local-idx")
        frame = pd.DataFrame({
            "source": args.source,
            "part_key": args.part,
            "local_idx": [int(v) for v in args.local_idx.split(",")],
            "data_class": [args.data_class
                           or SOURCE_GROUPS.get(args.source, (args.source,))[0]]
            * len(args.local_idx.split(",")),
        })
    events = load(frame, with_fit=not args.no_fit)
    name = args.name or f"display_{events[0].part_key}_{datetime.now():%y%m%d_%H%M}"
    fig = figure(events)
    save(fig, name)


if __name__ == "__main__":
    main()
