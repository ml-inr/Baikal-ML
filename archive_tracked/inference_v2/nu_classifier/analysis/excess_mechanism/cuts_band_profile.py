"""Excess vs classifier score, band by band, under each candidate cut.

The cuts found by ``find_cuts.py`` bring the aggregate excess (score > 0.5)
close to 1.  This script asks a sharper question: do they *flatten* the
band profile, or do they merely scale it down while leaving the rise with
score intact?  A cut that removes the mechanism should give ~1 in every
band; a cut that only trims population leaves the shape.

Outputs
-------
figures/cuts_band_profile.png
stdout: the band table and a high-band table with event counts
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from archive_tracked.inference_v2.nu_classifier.analysis.excess_mechanism.find_cuts import load  # noqa: E402

HERE = Path(__file__).parent
MC_GROUPS = [5, 6]          # muatm below / above threshold
EXP_GROUPS = [7, 8]         # exp below / above threshold
MIN_EXP = 150               # per-band floor on weighted exp events
MIN_MC = 1500               # per-band floor on weighted MC events

CUTS: dict[str, callable] = {
    "без ката": lambda d: pd.Series(True, index=d.index),
    "seq_backstep_frac < 0.489": lambda d: d.seq_backstep_frac < 0.489,
    "slowness > 0.788": lambda d: d.slowness > 0.788,
    "string_slope_spread > 1.177": lambda d: d.string_slope_spread > 1.177,
    "оба: sp<0.489 и slow>0.788": lambda d: (
        (d.seq_backstep_frac < 0.489) & (d.slowness > 0.788)
    ),
}

BANDS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3),
         (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
HIGH_BANDS = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.01), (0.5, 1.01),
              (0.8, 1.01), (0.95, 1.01)]

BLUE, ORANGE, TEAL, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#c9268a"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def band_excess(
    mc: pd.DataFrame,
    exp: pd.DataFrame,
    lo: float,
    hi: float,
    *,
    min_exp: int = MIN_EXP,
    min_mc: float = MIN_MC,
) -> tuple[float, int]:
    """Weighted share of exp in [lo, hi) divided by the same share for MC.

    Returns ``(nan, n_exp)`` when either sample is too thin to trust.
    """
    in_mc = ((mc.score >= lo) & (mc.score < hi)).values
    in_exp = ((exp.score >= lo) & (exp.score < hi)).values
    n_exp = int(exp.weight.values[in_exp].sum())
    n_mc = float(mc.weight.values[in_mc].sum())
    if n_exp < min_exp or n_mc < min_mc:
        return float("nan"), n_exp
    share_mc = np.average(in_mc, weights=mc.weight)
    share_exp = np.average(in_exp, weights=exp.weight)
    return share_exp / share_mc, n_exp


def main() -> None:
    feats = load()
    mc_all = feats[feats.group_id.isin(MC_GROUPS)]
    exp_all = feats[feats.group_id.isin(EXP_GROUPS)]

    profiles: dict[str, list[float]] = {}
    for label, cut in CUTS.items():
        mc, exp = mc_all[cut(mc_all)], exp_all[cut(exp_all)]
        profiles[label] = [band_excess(mc, exp, lo, hi)[0] for lo, hi in BANDS]
        cells = " ".join(
            f"{x:6.2f}" if np.isfinite(x) else "     -" for x in profiles[label]
        )
        print(f"{label:30s} {cells}")
    header = " ".join(f"{lo:.2f}".rjust(6) for lo, _ in BANDS)
    print(f"{'':30s} {header}")

    print("\nhigh bands, with weighted exp counts")
    print(f"{'cut':30s}" + "".join(f"{f'{lo}-{hi}':>15s}" for lo, hi in HIGH_BANDS))
    for label, cut in CUTS.items():
        mc, exp = mc_all[cut(mc_all)], exp_all[cut(exp_all)]
        cells = []
        for lo, hi in HIGH_BANDS:
            ratio, n_exp = band_excess(mc, exp, lo, hi, min_exp=40, min_mc=400)
            shown = f"{ratio:6.2f}" if np.isfinite(ratio) else f"{'-':>6s}"
            cells.append(f"{shown}(n={n_exp})")
        print(f"{label:30s}" + "".join(f"{c:>15s}" for c in cells))

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color="#e5e4e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, length=0)

    x = np.arange(len(BANDS))
    styles = [(MUTED, "o--"), (ORANGE, "s-"), (BLUE, "^-"), (PINK, "v-"),
              (TEAL, "D-")]
    for (label, row), (colour, style) in zip(profiles.items(), styles):
        first = label == "без ката"
        ax.plot(x, row, style, color=colour, lw=2.0 if first else 2.6, ms=7,
                alpha=0.7 if first else 1.0, label=label)
    ax.axhline(1.0, color=INK, ls=":", lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lo:.2f}–{hi:.2f}" for lo, hi in BANDS],
                       rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("полоса скора классификатора", color=MUTED)
    ax.set_ylabel("избыток в полосе: доля exp / доля MC", color=MUTED)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=MUTED, loc="upper left")
    ax.set_title("Как каты меняют профиль избытка по полосам скора", loc="left",
                 fontsize=13, fontweight="bold", color=INK, pad=12)
    plt.tight_layout()
    out = HERE / "figures" / "cuts_band_profile.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
