"""Assert that a change to the scoring path did not change what it scores.

The prediction pipeline is shared by every source and every model, so an optimisation there
silently rewrites results that are already published. This compares two prediction DuckDBs
built from the same checkpoint, probs file and parts, and fails on any disagreement.

What is compared, and how strictly:

* `event_fk` — the exact same set of events must be scored. Any addition or omission fails.
* `n_sn_hits`, `n_sn_strings` — integers, compared **exactly**. These decide the h8s3
  selection, so a single differing event changes the physics.
* `score` — compared exactly by default. Reordering float operations can legitimately move
  the last bits, so `--score-tol` relaxes it; the report always states how many events cross
  the 0.8 threshold, which is the number that actually matters.
* `embeddings` — compared with a tolerance, and reported as max |delta| per component.

Usage:
    # baseline: current code, some parts, into a scratch dir
    python inference_v2/nu_classifier/predict_mc.py ... --output-dir /tmp/base
    # candidate: after the change, identical arguments
    python inference_v2/nu_classifier/predict_mc.py ... --output-dir /tmp/cand
    python inference_v2/test_scoring_equivalence.py --baseline /tmp/base/... --candidate ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np

THRESHOLD = 0.8


def _load(db: Path) -> tuple[duckdb.DuckDBPyConnection, int]:
    conn = duckdb.connect(str(db), read_only=True)
    n = conn.execute("SELECT count(*) FROM predictions").fetchone()[0]
    return conn, n


def compare(baseline: Path, candidate: Path, score_tol: float, emb_tol: float) -> bool:
    ok = True
    a, na = _load(baseline)
    b, nb = _load(candidate)
    print(f"  baseline : {na:>10,} events  {baseline}")
    print(f"  candidate: {nb:>10,} events  {candidate}")

    a.execute(f"ATTACH '{candidate}' AS cand (READ_ONLY)")

    only_base, only_cand = a.execute("""
        SELECT (SELECT count(*) FROM predictions p LEFT JOIN cand.predictions c
                  USING (event_fk) WHERE c.event_fk IS NULL),
               (SELECT count(*) FROM cand.predictions c LEFT JOIN predictions p
                  USING (event_fk) WHERE p.event_fk IS NULL)
    """).fetchone()
    if only_base or only_cand:
        ok = False
        print(f"  FAIL event sets differ: {only_base:,} only in baseline, "
              f"{only_cand:,} only in candidate")
    else:
        print(f"  event sets identical ({na:,})")

    # integer columns: any difference at all is a failure
    for col in ("n_sn_hits", "n_sn_strings"):
        n_diff = a.execute(f"""
            SELECT count(*) FROM predictions p JOIN cand.predictions c USING (event_fk)
            WHERE p.{col} IS DISTINCT FROM c.{col}
        """).fetchone()[0]
        status = "ok" if n_diff == 0 else "FAIL"
        if n_diff:
            ok = False
        print(f"  {col:14s} {status}  differing events: {n_diff:,}")

    row = a.execute(f"""
        SELECT max(abs(p.score - c.score)),
               count(*) FILTER (WHERE abs(p.score - c.score) > {score_tol}),
               count(*) FILTER (WHERE (p.score > {THRESHOLD}) != (c.score > {THRESHOLD}))
        FROM predictions p JOIN cand.predictions c USING (event_fk)
    """).fetchone()
    max_d, n_over, n_cross = row[0] or 0.0, row[1], row[2]
    if n_over or n_cross:
        ok = False
    print(f"  score          {'ok' if not (n_over or n_cross) else 'FAIL'}  "
          f"max|delta|={max_d:.3e}  over tol: {n_over:,}  crossing {THRESHOLD}: {n_cross:,}")

    has_emb = a.execute("""
        SELECT count(*) FROM duckdb_tables() WHERE table_name = 'embeddings'
    """).fetchone()[0]
    if has_emb:
        rows = a.execute("""
            SELECT p.embedding, c.embedding
            FROM embeddings p JOIN cand.embeddings c USING (event_fk)
        """).fetchall()
        if not rows:
            print("  embeddings     FAIL  no overlapping rows")
            ok = False
        else:
            d = max(float(np.abs(np.asarray(x, dtype=np.float64)
                                 - np.asarray(y, dtype=np.float64)).max()) for x, y in rows)
            if d > emb_tol:
                ok = False
            print(f"  embeddings     {'ok' if d <= emb_tol else 'FAIL'}  "
                  f"{len(rows):,} compared  max|delta|={d:.3e}")
    else:
        print("  embeddings     absent in baseline — not compared")

    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline",  required=True, help="prediction .duckdb built before the change")
    ap.add_argument("--candidate", required=True, help="prediction .duckdb built after it")
    ap.add_argument("--score-tol", type=float, default=0.0,
                    help="allowed |delta| on score (default 0 = bit-exact)")
    ap.add_argument("--emb-tol",   type=float, default=1e-5)
    a = ap.parse_args()

    ok = compare(Path(a.baseline).resolve(), Path(a.candidate).resolve(),
                 a.score_tol, a.emb_tol)
    print("\n" + ("EQUIVALENT" if ok else "DIFFERENCES FOUND"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
