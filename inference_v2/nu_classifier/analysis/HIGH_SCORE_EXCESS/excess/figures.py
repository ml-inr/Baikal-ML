"""Comparison histograms, and one defined place for the files they produce.

Figures go to `figures/` and nowhere else -- `save` returns and prints the path,
so a picture is never left somewhere the reader has to guess at.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .sampling import Sample

#: Fixed per source, so a curve is recognised by its colour across figures.
COLOURS = {"mc_merged": "#1f77b4", "exp_full": "#d62728",
           "mc_reco": "#2ca02c", "exp_reco": "#ff7f0e"}

#: Below this many distinct integer values, bin on half-integers.
INTEGER_BIN_LIMIT = 400


def _values(sample: Sample, column: str, level: str) -> np.ndarray | None:
    frame = sample.events if level == "events" else sample.hits
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    return values[np.isfinite(values)]


def overlay(samples: dict[str, Sample], column: str, *, level: str = "events",
            bins: int = 80, span: tuple[float, float] | None = None,
            quantile: float = 0.995, logx: bool = False, logy: bool = True,
            density: bool = True, ax=None):
    """One histogram per source on shared bins, normalised to compare shapes.

    Shared bins are the point: the sources differ in size and in range, and
    histograms drawn on their own bins cannot be read against each other.  The
    range covers the central `quantile` of every source unless `span` is given,
    so a single outlier cannot flatten the figure.

    `level` picks `sample.events` or `sample.hits`.  To restrict a source to one
    HDF5 group, pass `sample.group("muatm_2020")` in the dict.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    series = {}
    for name, sample in samples.items():
        values = _values(sample, column, level)
        if values is None:
            continue
        if logx:
            values = values[values > 0]
        if len(values):
            series[name] = values
    if not series:
        raise KeyError(f"no source has column {column!r} at level {level!r}")

    if span is None:
        lo = min(np.quantile(v, 1 - quantile) for v in series.values())
        hi = max(np.quantile(v, quantile) for v in series.values())
        span = (lo, hi) if hi > lo else (lo - 1, lo + 1)

    # Counts (hits, strings, channels) are integers, and bins that ignore that
    # put two integers in one bin and none in the next, which reads as structure
    # that is not there.  Bin those on half-integers instead.
    integral = all(np.all(np.equal(np.mod(v, 1), 0)) for v in series.values())
    if integral and not logx and span[1] - span[0] <= INTEGER_BIN_LIMIT:
        edges = np.arange(np.floor(span[0]) - 0.5, np.ceil(span[1]) + 1.5)
    elif logx:
        edges = np.geomspace(max(span[0], 1e-12), span[1], bins + 1)
    else:
        edges = np.linspace(span[0], span[1], bins + 1)

    for name, values in series.items():
        # A sub-sample keeps its source's colour, so `mc_reco/muatm` still reads
        # as mc_reco next to the sources it is being compared against.
        colour = COLOURS.get(name) or COLOURS.get(samples[name].source)
        ax.hist(values, bins=edges, density=density, histtype="step", lw=1.6,
                label=f"{name} (n={len(values):,})", color=colour)
    ax.set_xlabel(column)
    ax.set_ylabel("density" if density else "events")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=7)
    return ax


def panel(samples: dict[str, Sample], columns, *, ncols: int = 3,
          level: str = "events", size: tuple[float, float] = (4.2, 2.9), **kw):
    """A grid of `overlay` plots, one per column.  Missing columns are skipped."""
    import matplotlib.pyplot as plt

    # An entry is either a column name or `(name, {overrides})`.  Per-column
    # overrides matter because one setting rarely suits a whole panel: energy
    # wants a log axis and zenith angle must not have one.
    entries = [(c, {}) if isinstance(c, str) else (c[0], dict(c[1]))
               for c in columns]
    entries = [(c, o) for c, o in entries
               if any(_values(s, c, level) is not None for s in samples.values())]
    if not entries:
        raise KeyError("none of the requested columns exist in these samples")
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(size[0] * ncols, size[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (column, overrides) in zip(axes, entries):
        overlay(samples, column, level=level, ax=ax, **{**kw, **overrides})
    for ax in axes[len(entries):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def save(fig, name: str, *, dpi: int = 130, directory: Path | None = None
         ) -> Path:
    """Write a figure into `figures/` and say where it went."""
    out = Path(directory or paths.FIGURES)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (name if name.endswith((".png", ".pdf", ".svg"))
                  else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")
    return path
