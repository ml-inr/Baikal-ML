#!/usr/bin/env python
"""Ground-truth check on the MC neutrinos captured by the fine-tune background cuts.

Selection under test (the cuts used to build the fine-tune background sample):
    d_nu > 2.5  AND  r_vert < 0.7

Question (paper section 5.3): are the captured simulated neutrinos either
(1) low-multiplicity / low-energy, or (2) near-horizontal by the TRUE MC zenith angle?
If so, no identifiable neutrino is lost by labelling that population as background.

Reuses the per-event table produced by `nue2_safety_check.py`
(tables/nue2_safety.csv: event_fk, base, part_key, local_idx, rvert, qmean, nfilt,
score, ood), so no GPU and no re-embedding is needed. Ground truth is read from
`prime_prty` in baikal_mc_merged.h5: column 0 = zenith theta [deg], column 2 = energy [GeV].

Angle convention (see experiments/numu/horizon_loss_design.md): theta < 90 deg is
UP-going. MC medians are muatm 143.7 (down-going), nuatm 26.7 and nue2 53.2 (up-going).

Sample caveat: nue2_safety.csv covers `nue2_2020` only -- astrophysical nu_mu with an
E^-2 spectrum. The atmospheric nuatm sample is not included (it needs a GPU pass to
produce embeddings/OOD).

Outputs tables/captured_nu_truth.csv (per-event, with theta and logE) and prints the
summary quoted in the paper.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CSV = HERE / "tables/nue2_safety.csv"
MCH5 = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"
OUT = HERE / "tables/captured_nu_truth.csv"

OOD_CUT, RV_CUT = 2.5, 0.7
STEEP_DEG, HORIZON_BAND = 70.0, 20.0
WELL_RESOLVED_HITS = 20


def read_truth(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Read GT zenith [deg] and log10(E/GeV) for each row of `df` from prime_prty."""
    theta = np.full(len(df), np.nan)
    log_e = np.full(len(df), np.nan)
    with h5py.File(MCH5, "r") as f:
        for (base, pk), idx in df.groupby(["base", "part_key"]).groups.items():
            key = f"{base}/prime_prty/{pk}/data"
            if key not in f:
                continue
            pp = f[key][:]
            idx = np.asarray(idx)
            loc = df.loc[idx, "local_idx"].to_numpy()
            ok = loc < len(pp)
            theta[idx[ok]] = pp[loc[ok], 0]
            log_e[idx[ok]] = np.log10(np.clip(pp[loc[ok], 2], 1e-3, None))
    return theta, log_e


def quartiles(a: np.ndarray) -> tuple[float, float, float]:
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def main() -> None:
    df = pd.read_csv(CSV).reset_index(drop=True)
    print(f"sample: {len(df):,} events, data_class={df.base.unique().tolist()}")

    theta_all, _ = read_truth(df)
    df["theta"] = theta_all

    cap = df[(df.ood > OOD_CUT) & (df.rvert < RV_CUT)].reset_index(drop=True)
    theta, log_e = read_truth(cap)
    cap["theta"], cap["logE"] = theta, log_e
    c = cap[~np.isnan(theta)]
    print(f"captured (d_nu>{OOD_CUT} & r_vert<{RV_CUT}): {len(cap):,} "
          f"({100 * len(cap) / len(df):.2f}% of sample); GT read for {len(c):,}")

    print("\n(1) multiplicity and energy")
    m, lo, hi = quartiles(c.nfilt.to_numpy())
    print(f"    n_hits  median {m:.0f} [p25 {lo:.0f}, p75 {hi:.0f}]  "
          f"frac<=12 {np.mean(c.nfilt <= 12):.2f}  "
          f"frac<{WELL_RESOLVED_HITS} {np.mean(c.nfilt < WELL_RESOLVED_HITS):.2f}")
    m, lo, hi = quartiles(c.logE.to_numpy())
    print(f"    log10E  median {m:.2f} [p25 {lo:.2f}, p75 {hi:.2f}]")

    print("\n(2) ground-truth zenith (90 deg = horizon, <90 = up-going)")
    m, lo, hi = quartiles(c.theta.to_numpy())
    near = np.mean(np.abs(c.theta - 90) < HORIZON_BAND)
    print(f"    theta   median {m:.0f} deg [p25 {lo:.0f}, p75 {hi:.0f}]  "
          f"frac |theta-90|<{HORIZON_BAND:.0f} = {near:.2f}  "
          f"frac theta>90 (down-going) = {np.mean(c.theta > 90):.3f}")

    steep_cap = c.theta < STEEP_DEG
    steep_all = df.theta < STEEP_DEG
    horiz_cap = (c.theta >= STEEP_DEG) & (c.theta <= 90)
    horiz_all = (df.theta >= STEEP_DEG) & (df.theta <= 90)
    r_steep = 100 * steep_cap.sum() / max(steep_all.sum(), 1)
    r_horiz = 100 * horiz_cap.sum() / max(horiz_all.sum(), 1)
    print(f"    capture rate: steep (theta<{STEEP_DEG:.0f}) {r_steep:.2f}%  vs  "
          f"near-horizon ({STEEP_DEG:.0f}-90) {r_horiz:.2f}%  "
          f"-> factor {r_horiz / max(r_steep, 1e-9):.1f}")

    print("\n(3) disjunction: captured events that are NEITHER dim NOR near-horizon")
    bad = (c.nfilt > WELL_RESOLVED_HITS) & (c.theta < STEEP_DEG)
    print(f"    n_hits>{WELL_RESOLVED_HITS} AND theta<{STEEP_DEG:.0f}: "
          f"{int(bad.sum())} events = {100 * bad.mean():.2f}% of captured, "
          f"{100 * bad.sum() / len(df):.3f}% of the whole nu sample")

    print("\ncross-tab (rows n_hits, cols GT theta)")
    rows = pd.cut(c.nfilt, [0, 12, 20, 40, 10 ** 6],
                  labels=["8-12", "13-20", "21-40", ">40"])
    cols = pd.cut(c.theta, [0, STEEP_DEG, 110, 180],
                  labels=["<70 (steep up)", "70-110 (horizon)", ">110 (down)"])
    print(pd.crosstab(rows, cols))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cap.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
