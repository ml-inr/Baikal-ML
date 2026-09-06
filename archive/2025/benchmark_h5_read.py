"""
Benchmark HDF5 read performance — focused on the real bottleneck.

Key finding from first run:
- 890 GB file, 20004 parts for muatm_2020
- Each part: ~2M hits, ~39 MB gzip-compressed, ~55 MB uncompressed
- Reading one part (decompress): ~350-500 ms
- Slicing from numpy in memory: ~0.5 µs/event
- So the ONLY thing that matters is: how many part loads per batch?

This script answers:
1. How many events per part in a realistic training index?
2. With grouped ordering, how many part loads per batch?
3. How long does a batch take with grouped vs scattered access?
4. What's the real events-per-part distribution from the catalog?

Usage:
    python benchmark_h5_read.py
"""

import time
import sys
import numpy as np
import h5py
import polars as pl
from collections import OrderedDict, Counter
from pathlib import Path

H5_PATH = "/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5"
PARTICLE = "muatm_2020"
CATALOG = "data_manager/h5_catalogs/catalogs_mc_merged/muatm_2020.parquet"


def fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def fmt_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 ** 2:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 ** 3:
        return f"{nbytes / 1024 ** 2:.1f} MB"
    else:
        return f"{nbytes / 1024 ** 3:.2f} GB"


# ---------------------------------------------------------------
# Test 1: Real catalog analysis — how many events per part?
# ---------------------------------------------------------------
def analyze_catalog(catalog_path: str, n_sample: int = 1_200_000):
    """Analyze events-per-part distribution from real catalog."""
    print(f"\n=== Real catalog analysis: {catalog_path} ===")
    t0 = time.perf_counter()

    # Check which column exists
    schema = pl.scan_parquet(catalog_path).collect_schema()
    if "h5_part_num" in schema.names():
        part_col = "h5_part_num"
    else:
        part_col = "h5_part"

    total = pl.scan_parquet(catalog_path).select(pl.len()).collect().item()
    print(f"  Total events in catalog: {total:,}")

    # Sample like the real training does
    rng = np.random.RandomState(42)
    n_to_load = min(n_sample, total)
    if n_to_load < total:
        sample_idx = np.sort(rng.choice(total, size=n_to_load, replace=False))
        df = (
            pl.scan_parquet(catalog_path)
            .select([part_col])
            .with_row_index("_idx")
            .filter(pl.col("_idx").is_in(sample_idx))
            .drop("_idx")
            .collect()
        )
    else:
        df = pl.scan_parquet(catalog_path).select([part_col]).collect()

    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(df):,} events in {fmt_time(elapsed)}")

    parts = df[part_col].to_list()
    part_counts = Counter(parts)
    counts = sorted(part_counts.values())
    n_parts = len(part_counts)

    print(f"\n  Unique parts used: {n_parts:,}")
    print(f"  Events per part:")
    print(f"    min={counts[0]}, max={counts[-1]}")
    print(f"    median={counts[len(counts)//2]}")
    print(f"    mean={np.mean(counts):.1f}")
    print(f"    p10={counts[len(counts)//10]}, p90={counts[9*len(counts)//10]}")

    # Distribution buckets
    buckets = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    print(f"\n  Events-per-part distribution:")
    for b in buckets:
        n = sum(1 for c in counts if c <= b)
        print(f"    <= {b:4d} events: {n:,} parts ({n/n_parts*100:.1f}%)")

    return parts, part_counts


# ---------------------------------------------------------------
# Test 2: Grouped ordering — cache misses per batch (NO HDF5 I/O)
# ---------------------------------------------------------------
def simulate_cache_misses(
    parts: list,
    part_counts: Counter,
    batch_size: int = 512,
    cache_size: int = 8,
):
    """Simulate cache behavior with grouped ordering. Pure computation, no I/O."""
    print(f"\n=== Cache miss simulation (batch_size={batch_size}, cache_size={cache_size}) ===")
    rng = np.random.RandomState(42)

    # Build event list with grouped ordering (like PrefilterDataset)
    from itertools import groupby

    # Create (part_key, event_index) pairs
    events_by_part = {}
    for pk in parts:
        events_by_part.setdefault(pk, [])
        events_by_part[pk].append(pk)

    # Group: all events from same part together, part order shuffled
    unique_parts = list(events_by_part.keys())
    rng.shuffle(unique_parts)
    grouped_events = []
    for pk in unique_parts:
        grouped_events.extend([pk] * len(events_by_part[pk]))

    n_events = len(grouped_events)
    n_batches = (n_events + batch_size - 1) // batch_size
    print(f"  Total events: {n_events:,}, batches: {n_batches}")

    # Simulate batch-by-batch cache behavior
    cache = OrderedDict()
    batch_misses = []
    batch_unique_parts = []
    for bi in range(min(n_batches, 50)):  # first 50 batches
        batch = grouped_events[bi * batch_size:(bi + 1) * batch_size]
        misses = 0
        for pk in batch:
            if pk in cache:
                cache.move_to_end(pk)
            else:
                cache[pk] = True
                misses += 1
                while len(cache) > cache_size:
                    cache.popitem(last=False)
        unique_in_batch = len(set(batch))
        batch_misses.append(misses)
        batch_unique_parts.append(unique_in_batch)

    avg_misses = np.mean(batch_misses)
    avg_unique = np.mean(batch_unique_parts)
    max_misses = np.max(batch_misses)
    est_time_per_batch = avg_misses * 0.45  # ~450ms per part read

    print(f"\n  First 50 batches:")
    print(f"    Avg cache misses/batch: {avg_misses:.1f}")
    print(f"    Max cache misses/batch: {max_misses}")
    print(f"    Avg unique parts/batch: {avg_unique:.1f}")
    print(f"    Estimated I/O time/batch: {fmt_time(est_time_per_batch)}")
    print(f"    Estimated I/O time/epoch: {fmt_time(est_time_per_batch * n_batches)}")

    # Show first 10 batches in detail
    print(f"\n  Batch-by-batch (first 10):")
    for bi in range(min(10, len(batch_misses))):
        print(
            f"    Batch {bi:3d}: misses={batch_misses[bi]:3d}, "
            f"unique_parts={batch_unique_parts[bi]:3d}, "
            f"est_io={fmt_time(batch_misses[bi] * 0.45)}"
        )

    # Now compare with SCATTERED (shuffled event-level, not part-level)
    print(f"\n  --- Comparison: scattered (event-level shuffle) ---")
    scattered = list(grouped_events)
    rng.shuffle(scattered)

    cache = OrderedDict()
    scattered_misses = []
    for bi in range(min(n_batches, 50)):
        batch = scattered[bi * batch_size:(bi + 1) * batch_size]
        misses = 0
        for pk in batch:
            if pk in cache:
                cache.move_to_end(pk)
            else:
                cache[pk] = True
                misses += 1
                while len(cache) > cache_size:
                    cache.popitem(last=False)
        scattered_misses.append(misses)

    avg_scattered = np.mean(scattered_misses)
    est_scattered = avg_scattered * 0.45
    print(f"    Avg cache misses/batch: {avg_scattered:.1f}")
    print(f"    Estimated I/O time/batch: {fmt_time(est_scattered)}")
    print(f"    Speedup from grouping: {avg_scattered / max(avg_misses, 0.01):.1f}x")


# ---------------------------------------------------------------
# Test 3: Actual HDF5 I/O — grouped batch read
# ---------------------------------------------------------------
def benchmark_real_batch(
    path: str, particle: str, parts: list, part_counts: Counter,
    batch_size: int = 512, n_batches: int = 5, cache_size: int = 8,
):
    """Read real batches from HDF5 with grouped ordering."""
    print(f"\n=== Real HDF5 batch read (grouped, cache={cache_size}) ===")
    f = h5py.File(path, "r")
    grp = f[particle]["raw"]
    rng = np.random.RandomState(42)

    # Build grouped event list
    unique_parts = list(part_counts.keys())
    rng.shuffle(unique_parts)
    grouped_events = []
    for pk in unique_parts:
        pk_str = f"part_{pk}" if isinstance(pk, int) else pk
        count = part_counts[pk]
        grouped_events.extend([pk_str] * count)

    # Run N batches
    cache = OrderedDict()
    for bi in range(n_batches):
        batch = grouped_events[bi * batch_size:(bi + 1) * batch_size]
        t0 = time.perf_counter()
        misses = 0
        loaded_bytes = 0
        for pk in batch:
            if pk in cache:
                cache.move_to_end(pk)
                data = cache[pk]
            else:
                data = grp["data"][pk]["data"][:]
                loaded_bytes += data.nbytes
                cache[pk] = data
                misses += 1
                while len(cache) > cache_size:
                    cache.popitem(last=False)
            # Simulate event slice
            if len(data) > 100:
                _ = data[:100]
        elapsed = time.perf_counter() - t0
        print(
            f"  Batch {bi}: {fmt_time(elapsed)}, "
            f"misses={misses}, loaded={fmt_size(loaded_bytes)}"
        )

    f.close()


# ---------------------------------------------------------------
# Test 4: How many events per part does the TRAINING config have?
# ---------------------------------------------------------------
def analyze_training_config():
    """Analyze what the real training config looks like."""
    print(f"\n=== Training config analysis ===")
    # From da_prefilter_numu_baseline.yaml:
    # source: muatm_2020=1200000, nue2_2020=600000, nuatm_2020=600000
    # target: exp=300000
    # batch_size source=512, target=256

    configs = {
        "muatm_2020": {"events": 1_200_000, "catalog": "data_manager/h5_catalogs/catalogs_mc_merged/muatm_2020.parquet"},
        "nue2_2020": {"events": 600_000, "catalog": "data_manager/h5_catalogs/catalogs_mc_merged/nue2_2020.parquet"},
        "nuatm_2020": {"events": 600_000, "catalog": "data_manager/h5_catalogs/catalogs_mc_merged/nuatm_2020.parquet"},
    }

    total_source_events = 0
    total_source_parts = 0

    for pt, cfg in configs.items():
        cat_path = cfg["catalog"]
        if not Path(cat_path).exists():
            print(f"  {pt}: catalog not found at {cat_path}")
            continue

        schema = pl.scan_parquet(cat_path).collect_schema()
        if "h5_part_num" in schema.names():
            part_col = "h5_part_num"
        else:
            part_col = "h5_part"

        total = pl.scan_parquet(cat_path).select(pl.len()).collect().item()
        n_parts_total = pl.scan_parquet(cat_path).select(pl.col(part_col).n_unique()).collect().item()

        n_events = min(cfg["events"], total)
        # Estimate parts used (proportional)
        est_parts = int(n_parts_total * (n_events / total))
        events_per_part = n_events / max(est_parts, 1)

        total_source_events += n_events
        total_source_parts += est_parts

        print(f"  {pt}: {n_events:,} events from ~{est_parts:,} parts "
              f"(~{events_per_part:.0f} events/part)")

    batch_size = 512
    n_batches = total_source_events // batch_size
    events_per_part_avg = total_source_events / max(total_source_parts, 1)
    parts_per_batch = batch_size / events_per_part_avg

    print(f"\n  Total source: {total_source_events:,} events, ~{total_source_parts:,} parts")
    print(f"  Avg events/part: {events_per_part_avg:.1f}")
    print(f"  Batch size: {batch_size}")
    print(f"  Est. unique parts/batch (grouped): {parts_per_batch:.1f}")
    print(f"  Est. I/O time/batch (grouped, ~450ms/part): {fmt_time(parts_per_batch * 0.45)}")
    print(f"  Est. I/O time/epoch: {fmt_time(parts_per_batch * 0.45 * n_batches)}")
    print(f"  Total batches/epoch: {n_batches}")


if __name__ == "__main__":
    print("=" * 60)
    print("HDF5 Read Benchmark — Focused on real training bottleneck")
    print("=" * 60)
    print(f"\nKey fact: reading one gzip-compressed part takes ~350-500ms")
    print(f"Everything else (slicing, tensor creation) is <1ms")
    print(f"So bottleneck = number of part cache misses per batch\n")

    # Step 1: Analyze the real catalog
    if Path(CATALOG).exists():
        parts, part_counts = analyze_catalog(CATALOG)

        # Step 2: Simulate cache behavior (no I/O needed)
        simulate_cache_misses(parts, part_counts, batch_size=512, cache_size=8)

        # Step 3: Real HDF5 I/O test (only 5 batches)
        if Path(H5_PATH).exists():
            benchmark_real_batch(H5_PATH, PARTICLE, parts, part_counts,
                                batch_size=512, n_batches=5, cache_size=8)
    else:
        print(f"Catalog not found: {CATALOG}")

    # Step 4: Full training config analysis
    analyze_training_config()

    print("\n=== Done ===")
