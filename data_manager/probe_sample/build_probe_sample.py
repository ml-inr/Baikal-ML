"""Build the fixed held-out sample the trainer's invariance probe runs on.

The probe inside `src/training/da_nu_classifier_trainer.py` (and, since
2026-11, `da_prefilter_numu_trainer.py`) measures, every N epochs and on frozen
features, how separable MC-muons and experimental events still are.  For that
comparison to be comparable between epochs (and between runs), the sample must
be fixed: same events, same hits, defined once here.

Classifier composition (default; all events out of training):
  * muatm, natural   -- 4000 from `reader.NOT_TRAINED`;
  * muatm, tail      -- 1500 with score >= 0.8, drawn from the 100 parts with the
                        most high-score muons (the tail is 0.05% dense, a plain
                        stream would scan hundreds of parts for a handful);
  * exp holdout      -- 4000 natural + 1500 with score >= 0.8 (`NOT_DA_TARGET`,
                        `NOT_EXCLUDED`);
  * nu, natural      -- 4000 (nuatm + nue2), 2026-10-10: the trainer's probe also
                        measures the in-game task, whose source side is the
                        natural nu+mu mixture; without nu the meter cannot
                        reproduce it;
  * nu, high score   -- 1500 with score >= 0.8 (same rationale as muatm tail;
                        for neutrinos high score is the norm, not a tail).

Prefilter mode (`--natural-only --hit-mode all --strictness parts`):
  * the prefilter is trained on **all** raw hits, so `--hit-mode all` keeps the
    hits the sig-noise filter would drop;
  * it has no per-event back-links, so the MC exclusion must be
    `reader.NOT_TRAINED_PARTS`, not `reader.NOT_TRAINED`;
  * its `target_domain` is an HDF5 file, not an NPY directory, so
    `NOT_DA_TARGET` cannot resolve -- pass `--skip-da-target-exclusion` and read
    `exp_full`, which is disjoint from the `exp.h5` used as the DA target;
  * `--natural-only` drops the tail blocks (they are selected with the
    classifier's score) and keeps muatm (0), exp (2), nu (4).

Group schema: 0,1 = muatm (natural, tail); 2,3 = exp (natural, tail);
4,5 = nu (natural, high score).  Consumers that predate 2026-10-10 assume
``exp == group >= 2`` and would mislabel groups 4,5 -- treat >= 4 as MC-nu.
Natural-only uses groups 0, 2, 4.

NB: the reader's stream order is not stable across runs, so re-running this
builder re-draws *different* events for the first four groups.  The canonical
classifier file must be extended in place (old events kept byte-identical, new
blocks appended), never regenerated wholesale -- the trainer's per-epoch probe
values are comparable only while the underlying events stay the same.

Writes an npz with concatenated hits, offsets, group/band labels and the reference
score (from the checkpoint's predictions, used only to fix the bulk/tail split).

Usage (classifier, unchanged):
    python data_manager/probe_sample/build_probe_sample.py

Usage (prefilter):
    python data_manager/probe_sample/build_probe_sample.py \
        --out data_manager/probe_sample/probe_sample_prefilter.npz \
        --checkpoint <prefilter-run>@best_da_model \
        --hit-mode all --natural-only --strictness parts \
        --skip-da-target-exclusion
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from inference import reader                                         # noqa: E402

CHECKPOINT = ("260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256"
              "@best_da_model")
PREDS = ROOT / f"inference/nu_classifier/preds/{CHECKPOINT}"
logger = logging.getLogger("build_probe_sample")


def top_tail_parts(k: int, preds: Path, preds_db: str,
                   splits_db: Path | None = None) -> list[str]:
    """Pick the parts with the most high-score muons, for the tail blocks.

    ``preds`` is the prediction directory carrying the score (the classifier's
    or the prefilter's); ``splits_db`` is an optional database that carries the
    ``splits`` table when the score database does not (the prefilter's does
    not).  Defaults reproduce the classifier query exactly.
    """
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute(f"ATTACH '{preds / preds_db}' AS mc (READ_ONLY)")
    if splits_db is None or splits_db == preds / preds_db:
        splits_table = "mc.splits"
    else:
        con.execute(f"ATTACH '{splits_db}' AS rich (READ_ONLY)")
        splits_table = "rich.splits"
    parts = con.execute(f"""
      SELECT s.part_key FROM mc.predictions p
      JOIN {splits_table} s USING (event_fk)
      WHERE s.data_class='muatm_2020' AND NOT s.used_for_labels AND p.score >= 0.8
      GROUP BY 1 ORDER BY count(*) DESC LIMIT {k}""").df().part_key.tolist()
    con.close()
    return parts


def _mc_exclusion(strictness: str):
    """MC exclusion matching what the checkpoint's dataset supports."""
    if strictness == "parts":
        return reader.NOT_TRAINED_PARTS
    if strictness == "all":
        return reader.NOT_TRAINED
    raise ValueError(
        f"strictness must be 'all' or 'parts', got {strictness!r}")


def collect(source: str, extra_where: list[str], n: int, *,
            checkpoint: str = CHECKPOINT, hit_mode: str = "sig",
            strictness: str = "all", skip_da_target: bool = False):
    r = reader.open(source, checkpoint=checkpoint)
    if source == "exp_full":
        where = ([reader.NOT_EXCLUDED]
                 + ([] if skip_da_target else [reader.NOT_DA_TARGET])
                 + extra_where)
    else:
        where = [_mc_exclusion(strictness)] + extra_where
    hits, offsets, score = [], [0], []
    t0 = time.time()
    for chunk in r.stream(where=where, with_hits=True, budget_mb=128):
        if len(chunk.hits):
            r.verify(chunk)
        if hit_mode == "sig":
            frame = chunk.hits[chunk.hits["is_sig"]]
        elif hit_mode == "all":
            frame = chunk.hits
        else:
            raise ValueError(
                f"hit_mode must be 'sig' or 'all', got {hit_mode!r}")
        grouped = {pos: sub for pos, sub in frame.groupby("event")}
        for pos in range(len(chunk.events)):
            sub = grouped.get(pos)
            if sub is None:
                continue
            hits.append(sub[["q", "t", "x", "y", "z"]].to_numpy(np.float32))
            offsets.append(offsets[-1] + len(sub))
            score.append(float(chunk.events.score.iloc[pos]))
            if len(hits) >= n:
                break
        if len(hits) >= n:
            break
    logger.info("[%s %s] %d events (%.0fs)", source, extra_where, len(hits),
                time.time() - t0)
    return np.concatenate(hits), np.array(offsets, np.int64), np.array(score, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE / "probe_sample.npz"))
    parser.add_argument("--mu-parts", type=int, default=100)
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="checkpoint whose exclusions and scores are used")
    parser.add_argument("--preds", default=str(PREDS),
                        help="prediction directory for tail part selection")
    parser.add_argument("--preds-db", default="mc_merged_thr0p8.duckdb",
                        help="prediction database inside --preds")
    parser.add_argument("--splits-db", default=None,
                        help="database carrying the splits table; "
                             "default = --preds-db")
    parser.add_argument("--hit-mode", choices=("sig", "all"), default="sig",
                        help="'sig' = sig-noise-selected hits (classifier); "
                             "'all' = every raw hit (prefilter)")
    parser.add_argument("--strictness", choices=("all", "parts"),
                        default="all",
                        help="MC exclusion; 'parts' is the only one prefilter "
                             "datasets support")
    parser.add_argument("--skip-da-target-exclusion", action="store_true",
                        help="do not apply NOT_DA_TARGET (prefilter checkpoints "
                             "have no target npy_dir)")
    parser.add_argument("--natural-only", action="store_true",
                        help="natural groups only (muatm, exp, nu); no tails")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.natural_only:
        blocks_spec = [
            ("mc_merged", ["e.data_class = 'muatm_2020'"], 4000, 0),
            ("exp_full", [], 4000, 2),
            ("mc_merged",
             ["e.data_class IN ('nuatm_2020', 'nue2_2020')"], 4000, 4),
        ]
    else:
        splits_db = Path(args.splits_db) if args.splits_db else None
        parts = top_tail_parts(
            args.mu_parts, Path(args.preds), args.preds_db, splits_db)
        listed = ", ".join(f"'{p}'" for p in parts)
        blocks_spec = [
            ("mc_merged", ["e.data_class = 'muatm_2020'"], 4000, 0),
            ("mc_merged", ["e.data_class = 'muatm_2020'",
                           f"l.part_key IN ({listed})", "p.score >= 0.8"],
             1500, 1),
            ("exp_full", [], 4000, 2),
            ("exp_full", ["p.score >= 0.8"], 1500, 3),
            ("mc_merged",
             ["e.data_class IN ('nuatm_2020', 'nue2_2020')"], 4000, 4),
            ("mc_merged", ["e.data_class IN ('nuatm_2020', 'nue2_2020')",
                           "p.score >= 0.8"], 1500, 5),
        ]

    blocks = []
    for source, where, n, group in blocks_spec:
        h, off, sc = collect(
            source, where, n,
            checkpoint=args.checkpoint,
            hit_mode=args.hit_mode,
            strictness=args.strictness,
            skip_da_target=args.skip_da_target_exclusion,
        )
        blocks.append(
            dict(group=group, hits=h, offsets=off, score=sc, n=len(off) - 1))

    hits = np.concatenate([b["hits"] for b in blocks])
    running, offset_parts = 0, []
    for b in blocks:
        offset_parts.append(b["offsets"][:-1] + running)
        running += len(b["hits"])
    offsets = np.concatenate(offset_parts + [np.array([running], np.int64)])
    group = np.concatenate([np.full(b["n"], b["group"], np.uint8) for b in blocks])
    score = np.concatenate([b["score"] for b in blocks])
    # 0 = bulk (score < 0.5), 1 = mid, 2 = tail (score >= 0.8); fixed at build time
    band = np.where(score >= 0.8, 2, np.where(score < 0.5, 0, 1)).astype(np.uint8)

    np.savez_compressed(args.out, hits=hits, offsets=offsets, group=group,
                        band=band, ref_score=score)
    counts = pd.DataFrame({"group": group, "band": band}).value_counts().sort_index()
    logger.info("saved %s: %d events, %d hits\n%s", args.out, len(group), len(hits),
                counts.to_string())


if __name__ == "__main__":
    main()
