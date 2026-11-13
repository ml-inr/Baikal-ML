#!/usr/bin/env python3
"""Truth multiplicity versus the BARS fit's minimized function on muatm h8s3.

The question: is there a range of ``fFuncValue`` (the function the BARS fit
minimises) where only single-track events survive?  The sample is ``muatm_2020``
after the SN model with the h8s3 hit cuts (``n_sn_hits >= 8``, ``n_sn_strings >= 3``),
with converged fits from ``fits.duckdb``.  Truth multiplicity is
``prime_prty[:, 4]`` (``ResponseMuonsN``), i.e. the number of response muons in the
cluster, not the bundle size in the whole event.

The script prints the single-track fraction in quantile bins of ``func_value`` and
of ``func_value / (n_hits - 5)`` (the normalisation used by the BARS quality cuts),
plus the value ranges for single- and multi-track events.

Usage:
    python -m inference.track_fit_alike_bars.analysis.multiplicity_vs_func_value
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from inference import reader

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256@best_da_model"
)
DEFAULT_H5 = ROOT / "data_manager/data/h5datasets/baikal_mc_merged.h5"


def fit_rows(checkpoint: str) -> pd.DataFrame:
    """Converged muatm h8s3 fits from the catalogue, keyed for the truth join."""
    catalogue = reader.open_fits()
    pred = (ROOT / "inference/nu_classifier/preds" / checkpoint
            / "mc_merged_thr0p8.duckdb")
    with catalogue.connection() as con:
        con.execute(f"ATTACH '{pred}' AS pred (READ_ONLY)")
        return con.execute(f"""
            SELECT f.event_fk, f.part_key, f.local_idx, f.func_value,
                   f.time_chi2, f.charge_term, f.n_hits, f.n_strings
            FROM fitness.fits f
            JOIN {catalogue.manifest_sql()} m USING (event_fk)
            JOIN pred.predictions p USING (event_fk)
            WHERE m.data_class = 'muatm_2020'
              AND f.status = 'ok'
              AND p.n_sn_hits >= 8 AND p.n_sn_strings >= 3
        """).df()


def truth_multiplicity(df: pd.DataFrame, h5_path: Path) -> np.ndarray:
    """ResponseMuonsN per row, read from the main h5 by (part_key, local_idx)."""
    out = np.full(len(df), -1, dtype=np.int64)
    with h5py.File(h5_path, "r") as f:
        for part, rows in df.groupby("part_key").indices.items():
            values = f[f"muatm_2020/prime_prty/{part}/data"][:, 4]
            out[rows] = values[df["local_idx"].to_numpy()[rows]].astype(np.int64)
    if (out < 0).any():
        raise RuntimeError(f"{int((out < 0).sum())} rows have no truth multiplicity")
    return out


def binned(df: pd.DataFrame, column: str, n_bins: int) -> pd.DataFrame:
    edges = np.quantile(df[column], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    b = pd.cut(df[column], edges, include_lowest=True, duplicates="drop")
    table = df.groupby(b, observed=True).agg(
        events=("n_mu", "size"),
        single=("n_mu", lambda x: int((x == 1).sum())),
        multi=("n_mu", lambda x: int((x > 1).sum())),
    )
    table["single_frac"] = table["single"] / table["events"]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    df = fit_rows(args.checkpoint)
    df["n_mu"] = truth_multiplicity(df, args.h5)
    df["func_value_per_dof"] = df["func_value"] / (df["n_hits"] - 5).clip(lower=1)

    print(f"muatm h8s3, converged fits: {len(df):,}")
    print(f"  single track: {int((df.n_mu == 1).sum()):,} "
          f"({100 * (df.n_mu == 1).mean():.2f}%)")
    print(f"  multi track:  {int((df.n_mu > 1).sum()):,} "
          f"({100 * (df.n_mu > 1).mean():.2f}%)  "
          f"median n_mu = {df.n_mu.median():.0f}")

    for column, label in (("func_value", "func_value"),
                          ("func_value_per_dof", "func_value / (n_hits - 5)")):
        print(f"\n{label}:")
        print(binned(df, column, args.bins).to_string())
        single = df.loc[df.n_mu == 1, column]
        multi = df.loc[df.n_mu > 1, column]
        print(f"  single range {single.min():.3g} .. {single.max():.3g} "
              f"(median {single.median():.3g})")
        print(f"  multi  range {multi.min():.3g} .. {multi.max():.3g} "
              f"(median {multi.median():.3g})")


if __name__ == "__main__":
    main()
