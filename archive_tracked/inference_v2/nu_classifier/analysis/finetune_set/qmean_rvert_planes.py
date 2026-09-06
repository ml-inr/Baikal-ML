#!/usr/bin/env python
"""2D distributions in the (q_mean, r_vert) plane --- mean hit charge instead of d_MC.

Companion to `dmc_rvert_planes.py`, which does all the heavy work (embedding, kNN, cuts)
and writes tables/dmc_rvert_planes.csv with one row per plotted event and the columns
pop, view, d_MC, r_vert, q_mean. This script only re-plots that table with q_mean on the
x axis, so run the other script first.

Same populations (exp, nuatm, nue2, muatm), same two views (all events / classifier score
> 0.8) and the same selection cuts. The cuts are still defined in d_MC and r_vert, so the
box cannot be drawn as a rectangle here: instead the r_vert < 0.7 line is shown, and the
"in box" panels keep only events satisfying d_MC > 2.5 AND r_vert < 0.7.

Outputs figures/qmean_rvert_planes*.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
CSV = HERE / "tables/dmc_rvert_planes.csv"
OOD_CUT, RV_CUT = 2.5, 0.7
QLIM, RLIM = (0.0, 20.0), (0.0, 3.0)
ORDER = ["exp", "nuatm_2020", "nue2_2020", "muatm_2020"]
TITLES = {"exp": "Experimental", "nuatm_2020": r"MC $\nu_\mu^{atm}$",
          "nue2_2020": r"MC $\nu_\mu^{cosm}$", "muatm_2020": "MC EAS"}


def draw(df, fname, view, only_box, suptitle):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ylim = (0.0, RV_CUT) if only_box else RLIM
    xb, yb = np.linspace(*QLIM, 80), np.linspace(*ylim, 80)
    for a, k in zip(ax.ravel(), ORDER):
        d = df[(df["pop"] == k) & (df["view"] == view)]
        if only_box:
            d = d[(d.d_MC > OOD_CUT) & (d.r_vert < RV_CUT)]
        a.set_xlabel(r"$\bar q$  [p.e.]")
        a.set_ylabel(r"$r_\mathrm{vert}$")
        if len(d) == 0:
            a.set_title(f"{TITLES[k]} — no events")
            continue
        h = a.hist2d(np.clip(d.q_mean, *QLIM), np.clip(d.r_vert, *ylim),
                     bins=[xb, yb], norm=LogNorm(), cmap="viridis")
        if not only_box:
            a.axhline(RV_CUT, color="red", lw=1.6, ls="--")
        a.set_title(f"{TITLES[k]} — median $\\bar q$ = {d.q_mean.median():.2f} p.e."
                    f"  (N={len(d):,})", fontsize=12)
        fig.colorbar(h[3], ax=a, label="events")
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(HERE / f"figures/{fname}", dpi=140)
    print(f"saved figures/{fname}", flush=True)


def main() -> None:
    df = pd.read_csv(CSV)
    print(f"loaded {len(df):,} rows from {CSV.name}", flush=True)
    print(df.groupby(["pop", "view"]).q_mean.describe()[["count", "50%"]].to_string(),
          flush=True)

    box = r"$d_\mathrm{MC}>2.5$, $r_\mathrm{vert}<0.7$"
    draw(df, "qmean_rvert_planes.png", "all", False,
         r"$(\bar q, r_\mathrm{vert})$ plane, all events; dashed line: $r_\mathrm{vert}<0.7$")
    draw(df, "qmean_rvert_planes_inbox.png", "all", True,
         rf"All events, restricted to the selection box ({box})")
    draw(df, "qmean_rvert_planes_hiscore.png", "hi", False,
         r"Events with classifier score $\xi>0.8$;"
         r" dashed line: $r_\mathrm{vert}<0.7$")
    draw(df, "qmean_rvert_planes_hiscore_inbox.png", "hi", True,
         rf"Events with $\xi>0.8$, restricted to the selection box ({box})")


if __name__ == "__main__":
    main()
