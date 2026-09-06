# Event Catalog v2 — DuckDB Backend

The catalog maps every physics event across all HDF5 sources to a single row,
enabling cross-source queries, provenance lookup, and duplicate detection.

---

## Schema

```
events
  id           BIGINT PK
  source       VARCHAR   -- 'exp' | 'exp_full' | 'exp_reco' | 'mc_merged' | 'mc_reco'
  data_class   VARCHAR   -- same as source for exp; particle type for MC (e.g. 'muatm_2020')
  season       INTEGER   -- e.g. 2020
  cluster      INTEGER   -- detector cluster index
  run          VARCHAR   -- run number string; MC: part_key as-is (e.g. 'part_1000')
  event_id     BIGINT    -- local index within part;
                          --   exp_reco: physics id_in_run (header_prty[:,3]);
                          --   exp_full: CC timestamp sec*1e9+nsec (header_prty[:,3])
  feature_hash VARCHAR   -- xxh64 of int32 channel sequence (NULL if --no-hash)

h5_locations
  event_fk  → events.id
  h5_path   VARCHAR   -- absolute path to HDF5 file
  part_key  VARCHAR   -- HDF5 group key within raw/ev_starts/
  local_idx INTEGER   -- index into ev_starts array

root_locations   (exp/exp_reco only when --root-dir provided)
  event_fk  → events.id
  root_path VARCHAR
  local_idx INTEGER

npy_locations   (populated separately by NPY dataset builders)
  event_fk  → events.id
  npy_dir   VARCHAR
  npy_tag   VARCHAR
  local_idx INTEGER
```

---

## Populated sources

| source | data_class | ~events |
|--------|-----------|---------|
| exp | exp | 650 K |
| exp_full | exp_full | ~133 M |
| exp_reco | exp_reco | 14.4 M |
| mc_merged | muatm_2020 | ~21 M |
| mc_merged | nuatm_2020 | ~4 M |
| mc_merged | nue2_2020 | ~1 M |

`exp_full` = full-statistics 2020 experimental data (29 clean runs, clusters c02–c07)
from `data_manager/data/exp_root/` → `exp_full.h5`. Five c01 runs were excluded for a
hit-time calibration bug (spurious early hits). Separate `source` namespace, so the
old `exp` (650 K) catalog and its predictions are untouched.

---

## Opening the catalog

```python
import duckdb

# Read-only (safe while no build is running)
conn = duckdb.connect("data_manager/catalog_v2.duckdb", read_only=True)

# Read-only snapshot while a build IS running (copy first)
import shutil, os, time
snap = "/tmp/catalog_snap.duckdb"
for _ in range(60):
    try:
        shutil.copy2("data_manager/catalog_v2.duckdb", snap)
        conn = duckdb.connect(snap, read_only=True)
        break
    except duckdb.IOException:
        time.sleep(0.5)
# ... use conn ...
conn.close()
os.unlink(snap)
```

---

## Common queries

### Event counts by source

```python
conn.execute("""
    SELECT source, data_class, COUNT(*) AS n
    FROM events
    GROUP BY source, data_class
    ORDER BY source, data_class
""").df()
```

### Look up a single event and its HDF5 location

```python
row = conn.execute("""
    SELECT e.*, h.h5_path, h.part_key, h.local_idx
    FROM events e
    JOIN h5_locations h ON h.event_fk = e.id
    WHERE e.source = 'exp_reco'
      AND e.season = 2020 AND e.cluster = 3 AND e.run = '228' AND e.event_id = 42
""").fetchone()
```

### All events for a run

```python
conn.execute("""
    SELECT e.event_id, e.feature_hash, h.local_idx
    FROM events e
    JOIN h5_locations h ON h.event_fk = e.id
    WHERE e.source = 'exp' AND e.season = 2020 AND e.cluster = 7 AND e.run = '202'
    ORDER BY e.event_id
""").df()
```

### Cross-source hash collisions (exp ∩ exp_reco)

```python
conn.execute("""
    SELECT COUNT(*) AS n_collisions
    FROM (
        SELECT feature_hash FROM events WHERE source = 'exp'    AND feature_hash IS NOT NULL
        INTERSECT
        SELECT feature_hash FROM events WHERE source = 'exp_reco' AND feature_hash IS NOT NULL
    )
""").fetchone()
# → (5247,)
```

### Detailed collision rows

```python
conn.execute("""
    SELECT
        e1.season, e1.cluster, e1.run,
        e1.event_id AS exp_eid,
        e2.event_id AS reco_eid,
        e1.feature_hash
    FROM events e1
    JOIN events e2 ON e1.feature_hash = e2.feature_hash
    WHERE e1.source = 'exp' AND e2.source = 'exp_reco'
    ORDER BY e1.season, e1.cluster, e1.run
""").df()
```

### MC events for one particle type

```python
conn.execute("""
    SELECT cluster, COUNT(*) AS n
    FROM events
    WHERE source = 'mc_merged' AND data_class = 'muatm_2020'
    GROUP BY cluster
    ORDER BY cluster
""").df()
```

### Find duplicate channel sequences within a source

```python
conn.execute("""
    SELECT feature_hash, COUNT(*) AS n
    FROM events
    WHERE source = 'exp_reco' AND feature_hash IS NOT NULL
    GROUP BY feature_hash
    HAVING COUNT(*) > 1
    ORDER BY n DESC
    LIMIT 20
""").df()
```

---

## Using the Python retriever

```python
from data_manager.catalog_v2.retriever import EventCatalog

with EventCatalog() as cat:
    # Summary dict: {(source, data_class): count}
    print(cat.summary())

    # Single event lookup
    ev = cat.get(source='exp_reco', season=2020, cluster=3, run='228', event_id=42)
    hits = ev.load_hits()     # (n_hits, 5) float32  [amp, t, x, y, z]
    reco = ev.load_reco()     # dict of reco scalars (exp_reco only)
    print(ev.info)

    # Batch query → list[Event]
    events = cat.query(source='exp', season=2020, cluster=7)
    hits_list = [e.load_hits() for e in events[:10]]

    # Batch query → pandas DataFrame (metadata only)
    df = cat.query_df(source='mc_merged', data_class='nuatm_2020')
```

---

## Building the catalog

Run from the project root with the `baikal25` conda environment active.

### exp, exp_full and exp_reco

```bash
bash data_manager/catalog_v2/run_build_exp.sh exp
bash data_manager/catalog_v2/run_build_exp.sh exp_full
bash data_manager/catalog_v2/run_build_exp.sh exp_reco
```

`build_exp` auto-detects `header_prty` in the HDF5 file and uses column 3 as the
physical `event_id` (exp_full → CC timestamp). Each source rebuild first deletes its
own existing rows, so sources can be (re)built independently.

Logs written to `data_manager/catalog_v2/logs/`.

### MC (no hashing, faster)

```bash
bash data_manager/catalog_v2/run_build_mc.sh --no-hash
# or with hashing (slow):
bash data_manager/catalog_v2/run_build_mc.sh
```

### Monitor a running build

```bash
tail -f data_manager/catalog_v2/logs/<latest>.log
```

### Snapshot query while build is running

```python
import duckdb, shutil, os, time
snap = "/tmp/snap.duckdb"
for _ in range(60):
    try:
        shutil.copy2("data_manager/catalog_v2.duckdb", snap)
        conn = duckdb.connect(snap, read_only=True)
        print(conn.execute("SELECT source, data_class, COUNT(*) FROM events GROUP BY 1,2").df())
        conn.close(); os.unlink(snap); break
    except duckdb.IOException:
        time.sleep(0.5)
```

---

## Notes

- **DuckDB exclusive lock**: only one writer at a time; readers are blocked while a write
  connection is open. Use the snapshot pattern above to query during builds.
- **feature_hash**: xxh64 of the int32 channel-number sequence (order-sensitive, hit-count-sensitive).
  Two events with identical `feature_hash` are physically the same detector response.
- **MC run field**: stores the raw HDF5 part key (e.g. `part_1000`, `part_1000_0`), not a run number.
- **exp_reco event_id**: physics `event_id_in_run` from `header_prty[:, 3]`, not a local index.
