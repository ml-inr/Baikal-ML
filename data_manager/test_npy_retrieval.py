"""Performance test: retrieve h5 data for events from the nu-classifier NPY dataset
via the DuckDB catalog.

Flow:
  1. Load npy metadata arrays to identify events
     (particle_types, h5_part_keys, h5_local_event_ids)
  2. Batch-query catalog_v2.duckdb → get h5_path + confirmed local_idx per event
  3. Group results by (h5_path, part_key) → one h5 part read per group
  4. Read from baikal_mc_merged.h5 per part:
       - theta, energy  (prime_prty[:, 0/2])
       - channels       (raw/channels)
       - GT signal mask (raw/labels != 0)
  5. Sig-hit probs come from probs.npy (already stored in the npy dataset)

Usage:
    python -m data_manager.test_npy_retrieval [--n-events 100000] [--seed 0]
    python -m data_manager.test_npy_retrieval --n-events -1   # all events
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NPY_DIR      = Path("data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8")
CATALOG_PATH = Path("data_manager/catalog_v2.duckdb")


# ---------------------------------------------------------------------------
# Load npy dataset
# ---------------------------------------------------------------------------

def load_npy_dataset(npy_dir: Path) -> dict:
    t0 = time.perf_counter()
    with open(npy_dir / "dataset_info.json") as f:
        info = json.load(f)

    particle_encode: Dict[str, int] = info["particle_encode"]
    particle_decode  = {v: k for k, v in particle_encode.items()}

    ds = {
        "particle_types": np.load(npy_dir / "particle_types.npy"),
        "h5_part_keys":   np.load(npy_dir / "h5_part_keys.npy", allow_pickle=True),
        "h5_local_ids":   np.load(npy_dir / "h5_local_event_ids.npy"),
        "offsets":        np.load(npy_dir / "offsets.npy"),
        "probs":          np.load(npy_dir / "probs.npy", mmap_mode="r"),
        "labels":         np.load(npy_dir / "labels.npy"),
        "particle_decode": particle_decode,
        "n_events":       0,
    }
    ds["n_events"] = len(ds["particle_types"])

    print(f"Loaded {ds['n_events']:,} events from NPY dataset  "
          f"({time.perf_counter()-t0:.2f}s)")
    return ds


# ---------------------------------------------------------------------------
# Catalog batch lookup
# ---------------------------------------------------------------------------

def catalog_lookup(
    sample_idx: np.ndarray,
    ds: dict,
    catalog_path: Path,
) -> pd.DataFrame:
    """Query catalog for h5 locations of all sampled events.

    Returns DataFrame with columns:
        query_idx  — position in sample_idx
        h5_path    — absolute path to the source HDF5 file
        part_key   — e.g. 'part_10033'
        local_idx  — 0-based event index within that part
    """
    decode = ds["particle_decode"]
    ptypes = ds["particle_types"]
    keys   = ds["h5_part_keys"]
    lids   = ds["h5_local_ids"]

    data_classes = np.array([decode[int(ptypes[i])] for i in sample_idx])
    seasons      = np.array([int(dc.rsplit("_", 1)[-1]) for dc in data_classes],
                            dtype=np.int32)
    runs         = keys[sample_idx]
    event_ids    = lids[sample_idx].astype(np.int64)

    df_q = pd.DataFrame({
        "query_idx":  np.arange(len(sample_idx), dtype=np.int64),
        "data_class": data_classes,
        "season":     seasons,
        "run":        runs,
        "event_id":   event_ids,
    })

    conn = duckdb.connect(str(catalog_path), read_only=True)
    conn.register("_tmp_q", df_q)

    result = conn.execute("""
        SELECT
            q.query_idx,
            h5.h5_path,
            h5.part_key,
            h5.local_idx
        FROM _tmp_q q
        JOIN events e
            ON  e.source     = 'mc_merged'
            AND e.data_class = q.data_class
            AND e.season     = q.season
            AND e.run        = q.run
            AND e.event_id   = q.event_id
        JOIN h5_locations h5
            ON  h5.event_fk  = e.id
    """).df()

    conn.unregister("_tmp_q")
    conn.close()
    return result


# ---------------------------------------------------------------------------
# H5 retrieval
# ---------------------------------------------------------------------------

def retrieve_from_h5(
    catalog_result: pd.DataFrame,
    sample_idx: np.ndarray,
    ds: dict,
) -> dict:
    """Group by (h5_path, part_key) and bulk-read each part once.

    Returns per-event arrays aligned to sample_idx order.
    """
    n = len(sample_idx)
    out = {
        "theta":           np.full(n, np.nan, dtype=np.float32),
        "energy":          np.full(n, np.nan, dtype=np.float32),
        "n_hits":          np.zeros(n, dtype=np.int32),
        "n_sig_hits_gt":   np.zeros(n, dtype=np.int32),
        "n_sig_hits_prob": np.zeros(n, dtype=np.int32),
    }

    offsets = ds["offsets"]

    # Group catalog rows by (h5_path, part_key)
    groups: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    for row in catalog_result.itertuples(index=False):
        groups[(row.h5_path, row.part_key)].append(
            (int(row.query_idx), int(row.local_idx))
        )

    n_parts = len(groups)
    t_h5    = 0.0
    done    = 0

    # Need to know particle type per query_idx to look up the right h5 group
    decode  = ds["particle_decode"]
    ptypes  = ds["particle_types"]
    ptype_for_qi = {
        i: decode[int(ptypes[sample_idx[i]])]
        for i in range(n)
    }

    # Group further by (h5_path, ptype, part_key) — ptype needed for h5 group key
    full_groups: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)
    for row in catalog_result.itertuples(index=False):
        qi    = int(row.query_idx)
        ptype = ptype_for_qi[qi]
        full_groups[(row.h5_path, ptype, row.part_key)].append(
            (qi, int(row.local_idx))
        )

    h5_files: Dict[str, h5py.File] = {}
    try:
        for (h5_path, ptype, pk), events in full_groups.items():
            if h5_path not in h5_files:
                h5_files[h5_path] = h5py.File(h5_path, "r")
            mc_f = h5_files[h5_path]

            t0 = time.perf_counter()
            mc_grp       = mc_f[ptype]
            ev_starts    = mc_grp[f"raw/ev_starts/{pk}/data"][:].astype(np.int64)
            prime_prty   = mc_grp[f"prime_prty/{pk}/data"][:]
            labels_all   = mc_grp[f"raw/labels/{pk}/data"][:]
            t_h5 += time.perf_counter() - t0

            for qi, local_i in events:
                s = int(ev_starts[local_i])
                e = int(ev_starts[local_i + 1])
                gt_mask = labels_all[s:e] != 0

                ps = int(offsets[sample_idx[qi]])
                pe = int(offsets[sample_idx[qi] + 1])

                out["theta"][qi]           = prime_prty[local_i, 0]
                out["energy"][qi]          = prime_prty[local_i, 2]
                out["n_hits"][qi]          = e - s
                out["n_sig_hits_gt"][qi]   = int(gt_mask.sum())
                out["n_sig_hits_prob"][qi] = pe - ps

            done += 1
    finally:
        for f in h5_files.values():
            f.close()

    out["_t_h5"]    = t_h5
    out["_n_parts"] = n_parts
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_events: int, seed: int) -> None:
    ds      = load_npy_dataset(NPY_DIR)
    n_total = ds["n_events"]

    rng = np.random.default_rng(seed)
    if n_events >= n_total:
        sample_idx = np.arange(n_total, dtype=np.int64)
        print(f"Using all {n_total:,} events")
    else:
        sample_idx = rng.choice(n_total, size=n_events, replace=False)
        print(f"Sampled {n_events:,} / {n_total:,} events  (seed={seed})")

    # ── Catalog lookup ───────────────────────────────────────────────────────
    print(f"\nQuerying catalog ({CATALOG_PATH})...")
    t0 = time.perf_counter()
    cat = catalog_lookup(sample_idx, ds, CATALOG_PATH)
    t_cat = time.perf_counter() - t0

    n_found = len(cat)
    n       = len(sample_idx)
    if n_found != n:
        print(f"WARNING: {n - n_found} events not found in catalog!")
    print(f"  {n_found:,} events resolved  ({t_cat:.2f}s, "
          f"{n_found/t_cat:,.0f} lookups/s)")

    # ── H5 retrieval ─────────────────────────────────────────────────────────
    print(f"\nRetrieving from h5...")
    t0 = time.perf_counter()
    out = retrieve_from_h5(cat, sample_idx, ds)
    t_h5_total = time.perf_counter() - t0
    t_h5_io    = out["_t_h5"]
    n_parts    = out["_n_parts"]

    t_total = t_cat + t_h5_total

    print(f"\n{'='*58}")
    print(f"Events retrieved       : {n_found:,}")
    print(f"Parts visited          : {n_parts:,}  "
          f"({n_found/max(n_parts,1):.0f} events/part avg)")
    print(f"")
    print(f"Catalog lookup         : {t_cat:.2f}s  "
          f"({n_found/t_cat:,.0f} lookups/s)")
    print(f"H5 retrieval total     : {t_h5_total:.2f}s  "
          f"({n_found/t_h5_total:,.0f} events/s)")
    print(f"  of which h5 I/O      : {t_h5_io:.2f}s  "
          f"({100*t_h5_io/t_h5_total:.0f}%)")
    print(f"  per-event extraction : {t_h5_total-t_h5_io:.2f}s  "
          f"({100*(t_h5_total-t_h5_io)/t_h5_total:.0f}%)")
    print(f"")
    print(f"End-to-end             : {t_total:.2f}s  "
          f"({n_found/t_total:,.0f} events/s)")

    print(f"\nSample spot-check (first 5 events):")
    print(f"  {'#':>4}  {'theta':>7}  {'log10(E)':>9}  "
          f"{'n_hits':>7}  {'gt_sig':>7}  {'prob_sig':>8}")
    for i in range(min(5, n_found)):
        e_val = out["energy"][i]
        print(f"  {i:>4}  "
              f"{out['theta'][i]:>7.2f}  "
              f"{np.log10(max(e_val, 1e-6)):>9.3f}  "
              f"{out['n_hits'][i]:>7}  "
              f"{out['n_sig_hits_gt'][i]:>7}  "
              f"{out['n_sig_hits_prob'][i]:>8}")

    bad = int(np.isnan(out["theta"]).sum())
    if bad:
        print(f"\nWARNING: {bad} events with missing theta (not found or not read)")
    else:
        print(f"\nSanity check OK: all events retrieved.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events", type=int, default=100_000,
                        help="Events to sample (default 100k; -1 = all)")
    parser.add_argument("--seed",     type=int, default=0)
    args = parser.parse_args()

    n = args.n_events if args.n_events > 0 else int(1e18)
    run(n, args.seed)


if __name__ == "__main__":
    main()
