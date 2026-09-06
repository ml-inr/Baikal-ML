"""
Benchmark: direct HDF5 slice reads — the approach now used in PrefilterDataset.

Each event reads a contiguous slice [start:end] from a gzip-compressed part.
With ~1.1 ms/event, a batch of 512 takes ~560ms.
With num_workers=4, batches are prefetched in parallel processes.

Usage:
    python benchmark_h5_direct_slice.py
"""

import time
import numpy as np
import h5py
import polars as pl
from pathlib import Path

H5_PATH = "/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5"
CATALOG = "data_manager/h5_catalogs/catalogs_mc_merged/muatm_2020.parquet"
PARTICLE = "muatm_2020"


def fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def load_real_events(n: int = 3000):
    """Load real event boundaries from catalog."""
    schema = pl.scan_parquet(CATALOG).collect_schema()
    part_col = "h5_part_num" if "h5_part_num" in schema.names() else "h5_part"

    df = (
        pl.scan_parquet(CATALOG)
        .select([part_col, "hit_start_idx", "hit_end_idx", "n_hits"])
        .head(n)
        .collect()
    )

    events = []
    for row in df.iter_rows(named=True):
        pk = row[part_col]
        if isinstance(pk, int):
            pk_str = f"part_{pk}"
        else:
            pk_str = pk
        events.append({
            "part_key": pk_str,
            "start": row["hit_start_idx"],
            "end": row["hit_end_idx"],
            "n_hits": row["n_hits"],
        })
    return events


def benchmark_direct_slice_batches(events, batch_size: int = 512, n_batches: int = 5):
    """Simulate training batches with direct HDF5 slicing (no cache)."""
    print(f"\n=== Direct slice: {n_batches} batches of {batch_size} ===")
    f = h5py.File(H5_PATH, "r")
    grp = f[PARTICLE]["raw"]

    # Shuffle events like real training
    rng = np.random.RandomState(42)
    shuffled = list(events)
    rng.shuffle(shuffled)

    for bi in range(n_batches):
        batch = shuffled[bi * batch_size:(bi + 1) * batch_size]
        n_parts = len(set(e["part_key"] for e in batch))

        t0 = time.perf_counter()
        for ev in batch:
            pk = ev["part_key"]
            s, e_ = ev["start"], ev["end"]
            data = grp["data"][pk]["data"][s:e_]
            labels = grp["labels"][pk]["data"][s:e_]
            channels = grp["channels"][pk]["data"][s:e_]
        elapsed = time.perf_counter() - t0

        print(
            f"  Batch {bi}: {fmt_time(elapsed)}, "
            f"{n_parts} unique parts, "
            f"{fmt_time(elapsed / batch_size)}/event"
        )

    f.close()


def benchmark_per_event_timing(events, n_events: int = 200):
    """Detailed per-event timing to understand variance."""
    print(f"\n=== Per-event timing: {n_events} random events ===")
    f = h5py.File(H5_PATH, "r")
    grp = f[PARTICLE]["raw"]

    rng = np.random.RandomState(42)
    subset = rng.choice(len(events), size=n_events, replace=False)

    times = []
    for idx in subset:
        ev = events[idx]
        pk = ev["part_key"]
        s, e_ = ev["start"], ev["end"]

        t0 = time.perf_counter()
        data = grp["data"][pk]["data"][s:e_]
        labels = grp["labels"][pk]["data"][s:e_]
        channels = grp["channels"][pk]["data"][s:e_]
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times = np.array(times) * 1000  # convert to ms
    print(f"  Per event (ms): mean={times.mean():.2f}, "
          f"median={np.median(times):.2f}, "
          f"p95={np.percentile(times, 95):.2f}, "
          f"max={times.max():.2f}")
    print(f"  Estimated batch of 512: {fmt_time(times.mean() * 512 / 1000)}")
    print(f"  Estimated epoch (2400 batches): {fmt_time(times.mean() * 512 * 2400 / 1000)}")

    f.close()


def benchmark_same_part_vs_different(events, n_events: int = 100):
    """Compare reading events from same part vs different parts."""
    print(f"\n=== Same part vs different parts ({n_events} events) ===")
    f = h5py.File(H5_PATH, "r")
    grp = f[PARTICLE]["raw"]

    # Group events by part
    from collections import defaultdict
    by_part = defaultdict(list)
    for ev in events:
        by_part[ev["part_key"]].append(ev)

    # Find a part with enough events
    big_part = max(by_part.keys(), key=lambda k: len(by_part[k]))
    same_part_events = by_part[big_part][:n_events]

    # Different parts
    diff_part_events = []
    for pk, evts in by_part.items():
        diff_part_events.append(evts[0])
        if len(diff_part_events) >= n_events:
            break

    # A) Same part
    t0 = time.perf_counter()
    for ev in same_part_events:
        s, e_ = ev["start"], ev["end"]
        _ = grp["data"][big_part]["data"][s:e_]
        _ = grp["labels"][big_part]["data"][s:e_]
        _ = grp["channels"][big_part]["data"][s:e_]
    t_same = time.perf_counter() - t0

    # B) Different parts
    t0 = time.perf_counter()
    for ev in diff_part_events:
        pk = ev["part_key"]
        s, e_ = ev["start"], ev["end"]
        _ = grp["data"][pk]["data"][s:e_]
        _ = grp["labels"][pk]["data"][s:e_]
        _ = grp["channels"][pk]["data"][s:e_]
    t_diff = time.perf_counter() - t0

    n_same = len(same_part_events)
    n_diff = len(diff_part_events)
    print(f"  A) Same part ({big_part}, {n_same} events): "
          f"{fmt_time(t_same)} ({fmt_time(t_same / n_same)}/event)")
    print(f"  B) Different parts ({n_diff} events, {n_diff} parts): "
          f"{fmt_time(t_diff)} ({fmt_time(t_diff / n_diff)}/event)")

    f.close()


def estimate_epoch_time():
    """Estimate total epoch I/O time based on measurements."""
    print(f"\n=== Epoch time estimates ===")
    # From benchmark: ~1.1ms per event (3 reads: data+labels+channels)
    ms_per_event = 1.1
    source_events = 1_200_000 + 600_000 + 600_000  # 2.4M
    target_events = 300_000
    source_batch = 512
    target_batch = 256

    source_batches = source_events // source_batch
    target_batches = target_events // target_batch

    # Sequential (num_workers=0)
    source_time = ms_per_event * source_batch * source_batches / 1000
    target_time = ms_per_event * target_batch * target_batches / 1000
    print(f"  num_workers=0:")
    print(f"    Source: {source_batches} batches × {fmt_time(ms_per_event * source_batch / 1000)}/batch "
          f"= {fmt_time(source_time)}")
    print(f"    Target: {target_batches} batches × {fmt_time(ms_per_event * target_batch / 1000)}/batch "
          f"= {fmt_time(target_time)}")
    print(f"    Total I/O per epoch: {fmt_time(source_time + target_time)}")

    # With workers (prefetching hides some latency)
    for nw in [2, 4, 8]:
        effective = (source_time + target_time) / nw
        print(f"  num_workers={nw}: ~{fmt_time(effective)} "
              f"(ideal, assumes perfect overlap)")


if __name__ == "__main__":
    print("HDF5 Direct Slice Benchmark")
    print("=" * 50)
    print(f"File: {H5_PATH}")
    print(f"Particle: {PARTICLE}")

    print("\nLoading event boundaries from catalog...")
    events = load_real_events(n=3000)
    print(f"Loaded {len(events)} events")

    # Show events-per-part for this sample
    from collections import Counter
    part_counts = Counter(e["part_key"] for e in events)
    counts = sorted(part_counts.values())
    print(f"Unique parts: {len(part_counts)}")
    print(f"Events/part: min={counts[0]}, max={counts[-1]}, "
          f"median={counts[len(counts)//2]}, mean={np.mean(counts):.1f}")

    benchmark_same_part_vs_different(events)
    benchmark_per_event_timing(events, n_events=200)
    benchmark_direct_slice_batches(events, batch_size=512, n_batches=5)
    estimate_epoch_time()

    print("\n=== Done ===")
