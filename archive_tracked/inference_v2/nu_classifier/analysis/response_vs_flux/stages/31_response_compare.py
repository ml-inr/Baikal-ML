"""Stage 31 -- the comparison stage 30's residuals were collected for.

Stage 30 withheld one hit at a time and recorded how far the observed arrival
time fell from the track fitted on the other hits.  This stage turns those
residuals into the two numbers the protocol asks for:

1. **Is the response the same?**  Compared *matched* on geometry -- hits,
   strings, and the fitted distance to the withheld module -- because a raw
   comparison cannot separate a different response from a different population,
   which is the whole difficulty.
2. **Does the mismatch track the excess?**  A response error should bite the hard
   tail hardest, so the gap should grow with the score band if it drives the
   excess.

It also asks whether the extra width is **one-sided**.  The track model describes
light from the bare muon at the Cherenkov angle; stochastic showers along the
track (bremsstrahlung, pair production, photonuclear -- MC does simulate them)
radiate from a point, and their light, like scattered light, can only arrive
*later* than the direct front, never earlier.  So a cascade or scattering
explanation predicts a one-sided late excess, while a timing-resolution or
calibration error predicts a symmetric one.

The caveat is that ``t0`` is fitted on the anchor hits, so extra late light on
*those* drags ``t0`` late and shifts the withheld hit's residual early.  A
one-sided cause can therefore masquerade as a symmetric spread, and the median
shift is reported alongside so the reader can judge how much of that is going on.

It also converts the width difference into the extra timing spread that would
produce it, which is what makes the perturbation in test 3 calibrated rather
than arbitrary: MC has to be smeared by *this much* to look like data.

The per-channel calibration residual measured earlier (9-11 ns, see
`analysis/excess_mechanism/measure_channel_offsets.py`) is carried through the
same arithmetic, so it can be compared with what is needed rather than waved at.

Usage:
    python stages/31_response_compare.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import h5io                                                    # noqa: E402
import provenance                                              # noqa: E402

LOG = logging.getLogger("stage31")
HIT_EDGES = [7, 9, 11, 14, 20, 10_000]
HIT_LABELS = ["8-9", "10-11", "12-14", "15-20", "21+"]
DIST_EDGES = [0, 15, 30, 50, 80, 1e9]
DIST_LABELS = ["<15", "15-30", "30-50", "50-80", "80+"]
MIN_CELL = 150
CHANNEL_CALIBRATION_NS = 10.0     # measured instrumental residual, exp only
IQR_TO_SIGMA = 1.349              # for a Gaussian


def iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def tail(values: pd.Series) -> float:
    return float((values.abs() > 50).mean())


def asymmetry(residuals: pd.DataFrame) -> pd.DataFrame:
    """Early and late sides of the residual, compared separately, per cell.

    ``dt > 0`` means light arrived *later* than the fitted track predicts.
    Cascades and scattering can only add late light; a calibration or resolution
    error is symmetric.  Each side is measured from the cell's own median, so a
    common shift does not leak into the width.
    """
    frame = residuals.assign(
        hit_bin=pd.cut(residuals.n_hits, HIT_EDGES, labels=HIT_LABELS),
        dist_bin=pd.cut(residuals.d, DIST_EDGES, labels=DIST_LABELS),
        string_bin=np.clip(residuals.n_strings, 3, 6))

    def sides(group: pd.DataFrame) -> pd.Series:
        middle = group.dt.median()
        return pd.Series({
            "n": len(group), "median": middle,
            "early_half_iqr": middle - group.dt.quantile(0.25),
            "late_half_iqr": group.dt.quantile(0.75) - middle,
            "early_tail": float((group.dt < -50).mean()),
            "late_tail": float((group.dt > 50).mean())})

    grouped = frame.groupby(["hit_bin", "string_bin", "dist_bin", "source"],
                            observed=True).apply(sides, include_groups=False)
    wide = grouped.unstack("source")
    wide = wide[(wide[("n", "exp")] >= MIN_CELL) & (wide[("n", "mc")] >= MIN_CELL)]
    out = pd.DataFrame({
        "n_exp": wide[("n", "exp")], "n_mc": wide[("n", "mc")],
        "median_exp": wide[("median", "exp")], "median_mc": wide[("median", "mc")],
        "early_ratio": wide[("early_half_iqr", "exp")] / wide[("early_half_iqr", "mc")],
        "late_ratio": wide[("late_half_iqr", "exp")] / wide[("late_half_iqr", "mc")],
        "early_tail_ratio": wide[("early_tail", "exp")] / wide[("early_tail", "mc")],
        "late_tail_ratio": wide[("late_tail", "exp")] / wide[("late_tail", "mc")]})
    out["median_shift_ns"] = out.median_exp - out.median_mc
    return out.reset_index()


def matched(residuals: pd.DataFrame) -> pd.DataFrame:
    """Residual width in cells of geometry, experiment against MC."""
    frame = residuals.assign(
        hit_bin=pd.cut(residuals.n_hits, HIT_EDGES, labels=HIT_LABELS),
        dist_bin=pd.cut(residuals.d, DIST_EDGES, labels=DIST_LABELS),
        string_bin=np.clip(residuals.n_strings, 3, 6))
    grouped = frame.groupby(["hit_bin", "string_bin", "dist_bin", "source"],
                            observed=True).dt.agg(n="size", iqr=iqr, tail=tail)
    wide = grouped.unstack("source")
    wide = wide[(wide[("n", "exp")] >= MIN_CELL) & (wide[("n", "mc")] >= MIN_CELL)]
    out = pd.DataFrame({
        "n_exp": wide[("n", "exp")], "n_mc": wide[("n", "mc")],
        "iqr_exp": wide[("iqr", "exp")], "iqr_mc": wide[("iqr", "mc")],
        "tail_exp": wide[("tail", "exp")], "tail_mc": wide[("tail", "mc")]})
    out["iqr_ratio"] = out.iqr_exp / out.iqr_mc
    out["tail_ratio"] = out.tail_exp / out.tail_mc
    # extra Gaussian spread that, added in quadrature to MC, reproduces exp
    gap = out.iqr_exp ** 2 - out.iqr_mc ** 2
    out["implied_extra_sigma_ns"] = np.sqrt(np.maximum(gap, 0)) / IQR_TO_SIGMA
    return out.reset_index()


def by_band(residuals: pd.DataFrame, excess: dict[float, float]) -> pd.DataFrame:
    """Does the response gap grow with the score band, where the excess does?"""
    grouped = residuals.groupby(["band_lo", "source"]).dt.agg(
        n="size", iqr=iqr, tail=tail).unstack("source")
    out = pd.DataFrame({
        "n_exp": grouped[("n", "exp")], "n_mc": grouped[("n", "mc")],
        "iqr_exp": grouped[("iqr", "exp")], "iqr_mc": grouped[("iqr", "mc")],
        "tail_exp": grouped[("tail", "exp")], "tail_mc": grouped[("tail", "mc")]})
    out["iqr_ratio"] = out.iqr_exp / out.iqr_mc
    out["tail_ratio"] = out.tail_exp / out.tail_mc
    gap = out.iqr_exp ** 2 - out.iqr_mc ** 2
    out["implied_extra_sigma_ns"] = np.sqrt(np.maximum(gap, 0)) / IQR_TO_SIGMA
    out["excess_in_band"] = [excess.get(float(b), np.nan) for b in out.index]
    return out.reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = h5io.load_config(HERE)
    data, config_path = HERE / "data", HERE / "config.yaml"
    source = data / "30_loo_residuals.parquet"
    residuals = pd.read_parquet(source)
    LOG.info("%d residuals (%s)", len(residuals),
             residuals.groupby("source").size().to_dict())

    sided = asymmetry(residuals)
    LOG.info("asymmetry over %d cells: early side x%.2f, late side x%.2f; "
             "tails x%.2f early, x%.2f late; median shift %.1f ns",
             len(sided), sided.early_ratio.median(), sided.late_ratio.median(),
             sided.early_tail_ratio.median(), sided.late_tail_ratio.median(),
             sided.median_shift_ns.median())

    cells = matched(residuals)
    wider = int((cells.iqr_ratio > 1).sum())
    LOG.info("matched cells: %d, exp wider in %d, median IQR ratio %.2f, "
             "median tail ratio %.2f", len(cells), wider,
             cells.iqr_ratio.median(), cells.tail_ratio.median())
    LOG.info("implied extra timing spread: median %.1f ns "
             "(measured per-channel calibration residual is %.0f ns)",
             cells.implied_extra_sigma_ns.median(), CHANNEL_CALIBRATION_NS)

    # excess per band, from the stage 20 score fit so the numbers are not retyped
    fit = pd.read_parquet(data / "20_score_fit.parquet")
    excess = {}
    for lo in residuals.band_lo.unique():
        rows = fit[(fit.score_lo >= lo - 1e-9) & (fit.score_hi <= 1.02)]
        rows = rows[rows.score_lo >= lo - 1e-9]
        window = fit[(fit.score_lo >= lo - 1e-9)]
        if len(window):
            weight = window.mc_share
            excess[float(lo)] = float(
                (window.exp_share.sum()) / (weight.sum()) if weight.sum() else np.nan)
    bands = by_band(residuals, excess)
    for _, row in bands.iterrows():
        LOG.info("band %.2f+: IQR ratio %.3f, tail ratio %.3f, "
                 "implied extra sigma %.1f ns, cumulative excess %.2f",
                 row.band_lo, row.iqr_ratio, row.tail_ratio,
                 row.implied_extra_sigma_ns, row.excess_in_band)

    provenance.write(cells, data / "31_matched_cells.parquet",
                     stage="31_response_compare", config_path=config_path,
                     inputs=[source], started=started,
                     notes={"min_cell": MIN_CELL, "cells_exp_wider": wider})
    provenance.write(sided, data / "31_asymmetry.parquet",
                     stage="31_response_compare", config_path=config_path,
                     inputs=[source], started=started)
    provenance.write(bands, data / "31_by_band.parquet",
                     stage="31_response_compare", config_path=config_path,
                     inputs=[source], started=started)
    LOG.info("stage 31 done in %.1f s", time.time() - started)


if __name__ == "__main__":
    main()
