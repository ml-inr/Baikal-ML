"""Quick timing benchmarks for the embedding loader data path.

Tests catalog query strategies and h5 per-event read performance.

Usage:
    python inference_v2/nu_classifier/analysis/profile_loaders.py \
        --catalog data_manager/catalog_v2.duckdb \
        --mc-h5   data_manager/data/h5datasets/baikal_mc_merged.h5 \
        --npy-dir data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8 \
        [--ptype muatm_2020] [--n 50]
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def _t(label: str, t0: float) -> None:
    print(f"  [{time.perf_counter() - t0:6.2f}s]  {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--mc-h5",   required=True)
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--ptype",   default="muatm_2020")
    parser.add_argument("--n",       type=int, default=50,
                        help="Events to sample in benchmark queries")
    args = parser.parse_args()

    import duckdb
    import h5py
    import numpy as np
    import pandas as pd

    print(f"\n=== Catalog: {args.catalog} ===")
    print(f"=== MC h5:   {args.mc_h5} ===")
    print(f"=== ptype:   {args.ptype}, n={args.n} ===\n")

    # 1. Training parts
    t0 = time.perf_counter()
    part_keys_arr = np.load(Path(args.npy_dir) / "h5_part_keys.npy", allow_pickle=True)
    training_parts = set(np.unique(part_keys_arr).tolist())
    _t(f"Load training parts: {len(training_parts)} parts", t0)

    conn = duckdb.connect(args.catalog, read_only=True)

    # 2. COUNT events (index probe)
    t0 = time.perf_counter()
    n_ev = conn.execute(
        "SELECT COUNT(*) FROM events WHERE source='mc_merged' AND data_class=?",
        [args.ptype],
    ).fetchone()[0]
    _t(f"COUNT events ({args.ptype}): {n_ev:,}", t0)

    # 3. DISTINCT part_keys from events.run (no h5_locations join)
    t0 = time.perf_counter()
    df_parts = conn.execute(
        "SELECT DISTINCT run AS part_key FROM events WHERE source='mc_merged' AND data_class=?",
        [args.ptype],
    ).df()
    available_parts = [p for p in df_parts["part_key"].tolist() if p not in training_parts]
    _t(
        f"DISTINCT parts (events.run): {len(df_parts)} total, {len(available_parts)} outside training",
        t0,
    )

    tp_df = pd.DataFrame({"pk": list(training_parts)})
    conn.register("_train_parts", tp_df)

    # 4. SLOW baseline: JOIN with h5_locations
    print("\n--- Query A: JOIN h5_locations (old approach) ---")
    t0 = time.perf_counter()
    df_a = conn.execute(f"""
        SELECT event_fk, part_key, local_idx
        FROM (
            SELECT e.id AS event_fk, h.part_key, h.local_idx
            FROM events e
            JOIN h5_locations h ON h.event_fk = e.id
            WHERE e.source     = 'mc_merged'
              AND e.data_class  = ?
              AND h.part_key   NOT IN (SELECT pk FROM _train_parts)
        )
        USING SAMPLE {args.n} ROWS (reservoir, 42)
        ORDER BY part_key, local_idx
    """, [args.ptype]).df()
    _t(f"Query A: {len(df_a)} rows, {df_a['part_key'].nunique()} parts", t0)

    # 5. FAST: events-only query (events.run = part_key, events.event_id = local_idx)
    print("\n--- Query B: events-only (no h5_locations join) ---")
    t0 = time.perf_counter()
    df_b = conn.execute(f"""
        SELECT event_fk, part_key, local_idx
        FROM (
            SELECT id AS event_fk, run AS part_key, event_id AS local_idx
            FROM events
            WHERE source     = 'mc_merged'
              AND data_class  = ?
              AND run         NOT IN (SELECT pk FROM _train_parts)
        )
        USING SAMPLE {args.n} ROWS (reservoir, 42)
        ORDER BY part_key, local_idx
    """, [args.ptype]).df()
    _t(f"Query B: {len(df_b)} rows, {df_b['part_key'].nunique()} parts", t0)

    conn.execute("DROP VIEW IF EXISTS _train_parts")

    # 7. Two-step: select N_PARTS parts in Python, then query only those events
    print("\n--- Query C: two-step (distinct parts → selected subset → events) ---")
    N_PARTS = 200
    rng_c = np.random.default_rng(99)
    sel = rng_c.choice(available_parts, size=min(N_PARTS, len(available_parts)), replace=False).tolist()
    sel_df = pd.DataFrame({"pk": sel})
    conn.register("_sel_parts", sel_df)
    t0 = time.perf_counter()
    df_c = conn.execute(f"""
        SELECT event_fk, part_key, local_idx
        FROM (
            SELECT id AS event_fk, run AS part_key, event_id AS local_idx
            FROM events
            WHERE source     = 'mc_merged'
              AND data_class  = ?
              AND run         IN (SELECT pk FROM _sel_parts)
        )
        USING SAMPLE {args.n} ROWS (reservoir, 42)
        ORDER BY part_key, local_idx
    """, [args.ptype]).df()
    _t(f"Query C ({N_PARTS} parts): {len(df_c)} rows, {df_c['part_key'].nunique()} parts", t0)
    conn.execute("DROP VIEW IF EXISTS _sel_parts")

    conn.close()

    if df_b.empty:
        print("No events sampled — skipping h5 benchmark")
        return

    # 6. H5 read benchmark: per-event slices vs full-part load
    print("\n--- H5 read benchmark ---")
    part_key   = df_b.iloc[0]["part_key"]
    part_events = df_b[df_b["part_key"] == part_key]
    local_idxs  = part_events["local_idx"].values

    with h5py.File(args.mc_h5, "r") as h5:
        ptype_grp = args.ptype
        grp = h5[ptype_grp]["raw"]

        t0 = time.perf_counter()
        ev_starts = grp[f"ev_starts/{part_key}/data"][:].astype(np.int64)
        n_part_events = len(ev_starts) - 1
        total_hits    = int(ev_starts[-1])
        _t(
            f"ev_starts for {part_key!r}: {n_part_events:,} events, {total_hits:,} total hits",
            t0,
        )

        # Full-array load (old approach)
        t0 = time.perf_counter()
        _ = grp[f"data/{part_key}/data"][:]
        _t(f"Full data array load ({total_hits:,} hits)", t0)

        # Per-event slice reads
        t0 = time.perf_counter()
        data_ds = grp[f"data/{part_key}/data"]
        chan_ds  = grp[f"channels/{part_key}/data"]
        n_hits_read = 0
        for li in local_idxs:
            s, e = int(ev_starts[li]), int(ev_starts[li + 1])
            _ = data_ds[s:e]
            _ = chan_ds[s:e]
            n_hits_read += e - s
        _t(
            f"Per-event slices: {len(local_idxs)} events, {n_hits_read} hits total",
            t0,
        )


if __name__ == "__main__":
    main()
