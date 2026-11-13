#!/usr/bin/env python
"""Score density of muatm and exp after the h8s3 cut — discriminator-pool model.

Standalone: reads the predictions written on 2026-10-06 for the pool run
(`261002_1101_..._pool10@best_da_model`, checkpoint = epoch 31) and draws the two
populations the high-score excess is made of:

  * MC muons  (`muatm_2020`) — the expected background;
  * real data (`exp`)        — everything the search sees.

Both after the h8s3 topology cut of the sig-noise model (`n_sn_hits >= 8` and
`n_sn_strings >= 3`, the same numbers as ``excess.sources.QUALITY``), both
normalised to unit area so that samples of different sizes are comparable, and on
a logarithmic density scale because the bulk of the score lies four decades below
the tail.

It deliberately does not import the `excess` package: that package joins a `splits`
table (which marks training events and reference parts) that these fresh prediction
databases do not carry yet, and it is its own analysis line. Nothing here modifies
it. The consequence to keep in mind: `mc_merged` predictions include events the
model was fitted on, so the muon curve is slightly optimistic in the tail (see
doc/claims_log.md and inference/reader/training.py for how the maintained reader
handles that).

Usage (from this directory, or anywhere):

    python inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/score_density_pool10.py
    python ... --n-muatm 100000 --n-exp 200000 --bins 100 --out <name>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import duckdb
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CLAUDE.md").exists())

PREDS = (ROOT / "inference/nu_classifier/preds"
         / "261002_1101_da_nu_classifier_exp_full_E1_lambda0.01_pool10@best_da_model")
CATALOG = ROOT / "data_manager/catalog_v2.duckdb"
FIGURES = HERE / "figures"

#: The h8s3 quality cut, identical to excess/sources.py.
MIN_SN_HITS, MIN_SN_STRINGS = 8, 3

#: Fixed, not data-driven: two figures made months apart must share their x axis.
SCORE_RANGE = (0.0, 1.0)
COLOURS = {"muatm": "#1f77b4", "exp": "#d62728"}


def fetch_scores(database: Path, n_events: int, *, data_class: str | None,
                 seed: int = 42) -> np.ndarray:
    """Scores of `n_events` h8s3-passing events, sampled reproducibly.

    Selection is by the hash of the event key, so re-running gives the same sample and
    no random state is involved.  (`USING SAMPLE` would be shorter, but DuckDB applies
    it at the scan, i.e. *before* the h8s3 filter, which silently shrinks the sample.)
    `data_class` selects the particle class in the catalog; None means everything, which
    is what the experimental database holds.
    """
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA disable_progress_bar")
    con.execute(f"ATTACH '{database}' AS preds (READ_ONLY)")
    con.execute(f"ATTACH '{CATALOG}' AS cat (READ_ONLY)")
    where = [f"p.n_sn_hits >= {MIN_SN_HITS}", f"p.n_sn_strings >= {MIN_SN_STRINGS}"]
    if data_class is not None:
        where.append(f"e.data_class = '{data_class}'")
    query = f"""
        SELECT score FROM (
            SELECT p.score, p.event_fk
            FROM preds.predictions p
            JOIN cat.events e ON e.id = p.event_fk
            WHERE {' AND '.join(where)}
        )
        ORDER BY hash(event_fk)
        LIMIT {n_events}
    """
    scores = con.execute(query).fetchnumpy()["score"].astype(np.float64)
    con.close()
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-muatm", type=int, default=100_000)
    ap.add_argument("--n-exp", type=int, default=200_000)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="score_density_muatm_exp_h8s3_pool10.png")
    args = ap.parse_args()

    muatm = fetch_scores(PREDS / "mc_merged_thr0p8.duckdb", args.n_muatm,
                         data_class="muatm_2020", seed=args.seed)
    exp = fetch_scores(PREDS / "exp_full_thr0p8.duckdb", args.n_exp,
                       data_class=None, seed=args.seed)
    print(f"muatm: {len(muatm):,} events, mean score {muatm.mean():.4f}, "
          f"median {np.median(muatm):.5f}")
    print(f"exp  : {len(exp):,} events, mean score {exp.mean():.4f}, "
          f"median {np.median(exp):.5f}")
    for thr in (0.5, 0.9, 0.99):
        print(f"  above {thr}: muatm {int((muatm > thr).sum()):>6,}   "
              f"exp {int((exp > thr).sum()):>6,}")

    bins = np.linspace(SCORE_RANGE[0], SCORE_RANGE[1], args.bins + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for scores, label, colour in ((muatm, "MC muatm", COLOURS["muatm"]),
                                  (exp, "exp", COLOURS["exp"])):
        ax.hist(scores, bins=bins, density=True, histtype="step", lw=1.8, color=colour,
                label=f"{label}  ({len(scores):,} events)")
    ax.set_yscale("log")
    ax.set_xlim(*SCORE_RANGE)
    ax.set_xlabel("classifier score")
    ax.set_ylabel("density per bin  (area under each curve = 1)")
    ax.set_title(f"score after h8s3, {args.bins} bins — pool10 (epoch 31)\n"
                 f"{len(muatm) // 1000}k muatm vs {len(exp) // 1000}k exp")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper center")

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / args.out
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
