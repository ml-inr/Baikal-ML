# Data Format Documentation

*Covers HDF5 data files, Parquet event catalogs, and the `data_manager/` pipeline.*

## Overview

The data pipeline has two layers:

1. **HDF5 files** — processed detector data (hits, labels, physics truth) converted from ROOT files by `data_manager/root2h5/`
2. **Parquet event catalogs** — per-event metadata indexes built from HDF5 files by `data_manager/build_catalog_*.py`, stored in `data_manager/h5_catalogs/`

Additionally, `data_manager/root2df/` provides a Polars-based ROOT file reader for direct ROOT → DataFrame access (used for exploration and validation, not for training).

### HDF5 Files

| File | Description | Location |
|------|-------------|----------|
| `exp.h5` | Experimental data (2020 season) | `data_manager/data/h5datasets/exp.h5` |
| `exp_reco.h5` | Experimental data with reconstruction parameters | `data_manager/data/h5datasets/exp_reco.h5` |
| `baikal_mc_merged.h5` | All MC particle types merged | `/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5` |
| `baikal_2020_sig-noise_mid-eq_normed.h5` | Normalized MC with train/val/test splits | `/home2/ivkhar/Baikal/data/normed/` |
| `example_mc_5_muatm_files.h5` | Small MC sample for testing | `data_manager/data/h5datasets/` |

### Parquet Catalogs

| Directory | Source HDF5 | Files | Events |
|-----------|-------------|-------|--------|
| `h5_catalogs/catalogs_mc_merged/` | `baikal_mc_merged.h5` | per particle type (e.g. `muatm_2020.parquet`) | ~577M (muatm_2020) |
| `h5_catalogs/catalogs_mc_signoise_normed/` | `baikal_2020_sig-noise_mid-eq_normed.h5` | `train.parquet`, `val.parquet`, `test.parquet` | ~19M (train) |
| `h5_catalogs/catalogs_exp/` | `exp.h5` | `exp.parquet` | ~650K |
| `h5_catalogs/catalogs_exp_reco/` | `exp_reco.h5` | `exp_reco.parquet` | ~334K |

Some catalog directories also contain `signoise_encoder_*_preds/` subdirectories with prediction parquets from an external signal/noise hit-classifier NN.

Each catalog directory includes `source_path.txt` (path to source HDF5), `usage.ipynb` (examples).

## Data source

The HDF5 files are created using ROOT files, obtained with BARS. All ROOT files are stored at cluster62.inr.ac.ru cluster:

- **Experimental runs**: `/home/albert/Baikal/Data/exp_root_files/`
- **MC .root files**: `/home3/ivkhar/Baikal/data/initial_data/MC_2020/<particle_name>/root/all`
- **Experimental reco ROOT files**: `data_manager/data/exp_reco_root/root_files/` (26 files across 7 clusters)
- **MC reco ROOT files**: `data_manager/data/mc_reco_root/` (muatm, nuatm subdirs)

**Experimental** .root files' names indicate the season, cluster and run numbers in format: `s{year}_c{cluster}_r{run}`.  For example: s2020_c07_r0041.
By september 2025, only runs for the 2020 season have been uploaded, being evenly distributed throughout the year, and each .root file contains only the first 25,000 events of the run.

## File Structure

### Base Structure (Both Experimental & Monte Carlo)
```
{particle}/                          # Top-level group (e.g., "exp", "muatm")
├── clusters_centers/data            # Cluster center coordinates [shape: (n_clusters, 3)]
├── coords_are_cluster_centered/data # Boolean flag for coordinate centering
├── ev_ids/                          # Event identifiers
│   └── part_{filename}/data         # Event IDs for each file part [shape: (n_events,)]
└── raw/                             # Raw detector data
    ├── cluster_ids/                 # Cluster assignments per event
    │   └── part_{filename}/data     # [shape: (n_events,)]
    ├── data/                        # Main feature data **[MOST IMPORTANT]**
    │   └── part_{filename}/data     # [shape: (n_hits, 5)] 
    ├── labels/                      # Hit classification labels
    │   └── part_{filename}/data     # [shape: (n_hits,)]
    ├── channels/                    # Detector channel IDs
    │   └── part_{filename}/data     # [shape: (n_hits,)]
    ├── ev_starts/                   # Event boundary indices **[CRITICAL]**
    │   └── part_{filename}/data     # [shape: (n_events+1,)]
    ├── num_un_strings/              # Number of unique detector strings per event
    │   └── part_{filename}/data     # [shape: (n_events,)]
    └── t_res/                       # Time residuals (MC only)
        └── part_{filename}/data     # [shape: (n_hits,)]
```

### Additional Monte Carlo Structure
```
{particle}/                          # MC data has additional physics info
├── prime_prty/                      # Primary particle properties **[MC ONLY]**
│   └── part_{filename}/data         # [shape: (n_events, 6)]
└── muons_prty/                      # Muon information **[MC ONLY]**
    ├── aggregate/                   # Event-level muon summary
    │   └── part_{filename}/data     # [shape: (n_events, 2)]
    ├── individ/                     # Individual muon tracks
    │   └── part_{filename}/data     # [shape: (n_muons, 7)]
    └── mu_starts/                   # Muon boundary indices
        └── part_{filename}/data     # [shape: (n_events+1,)]
```

## Key Data Arrays

### 1. Main Features: `raw/data/{part}/data`
**Shape**: `(n_hits, 5)` | **Type**: `float32`

The primary input features for ML models:
```python
data[:, 0]  # Amplitude - signal strength (normalized)
data[:, 1]  # Time - hit timing (nanoseconds, centered per event)
data[:, 2]  # X coordinate - detector position (meters, cluster-centered)
data[:, 3]  # Y coordinate - detector position (meters, cluster-centered) 
data[:, 4]  # Z coordinate - detector position (meters, cluster-centered)
```

**Example hit**: `[0.695, -2336.3, 145.0, 109.8, -172.6]`

### 2. Event Boundaries: `raw/ev_starts/{part}/data`
**Shape**: `(n_events+1,)` | **Type**: `int32`

Critical for splitting flattened data into individual events:
```python
# Event i spans hits from ev_starts[i] to ev_starts[i+1]
event_0_hits = data[ev_starts[0]:ev_starts[1]]  # First event
event_1_hits = data[ev_starts[1]:ev_starts[2]]  # Second event
```

**Example**: `[0, 49, 106, 154, ...]` → Event 0: 49 hits, Event 1: 57 hits, etc.

### 3. Labels: `raw/labels/{part}/data`
**Shape**: `(n_hits,)` | **Type**: `int32`

Coding integer indicating hit origin (noise, track, caskade). Uses the same magic numbers as the root file.

### 4. Event Identifiers: `ev_ids/{part}/data`
**Shape**: `(n_events,)` | **Type**: `|S25`

Unique string IDs for traceability: `b'exp_s2020_c01_r0027_0'` (exp) or `b'muatm_12586_0'` (MC)

## Monte Carlo Specific Arrays

### 5. Primary Particle Properties: `prime_prty/{part}/data` **[MC ONLY]**
**Shape**: `(n_events, 6)` | **Type**: `float32`

Ground truth information about the initial particle:
```python
prime_prty[:, 0]  # Theta - zenith angle (degrees)
prime_prty[:, 1]  # Phi - azimuth angle (degrees) 
prime_prty[:, 2]  # Energy - primary particle energy (GeV)
prime_prty[:, 3]  # NucleonN - nucleon number
prime_prty[:, 4]  # ResponseMuonsN - number of response muons
prime_prty[:, 5]  # EventWeight - Monte Carlo event weight
```

**Example**: `[145.8, 273.3, 12.0, 14.0, 1.0, 1.0]` → θ=145.8°, φ=273.3°, E=12 GeV

### 6. Aggregate Muon Properties: `muons_prty/aggregate/{part}/data` **[MC ONLY]**
**Shape**: `(n_events, 2)` | **Type**: `float32`

Event-level muon summary:
```python
mu_agg[:, 0]  # FirstMuonTime - time of first muon (ns)
mu_agg[:, 1]  # SumEnergyBundle - total energy of muon bundle (GeV)
```

### 7. Individual Muon Tracks: `muons_prty/individ/{part}/data` **[MC ONLY]**
**Shape**: `(n_muons, 7)` | **Type**: `float32`

Detailed information for each muon track:
```python
mu_individ[:, 0]  # Theta - muon direction zenith (radians)
mu_individ[:, 1]  # Phi - muon direction azimuth (radians)
mu_individ[:, 2]  # X - muon track position (meters)
mu_individ[:, 3]  # Y - muon track position (meters)
mu_individ[:, 4]  # Z - muon track position (meters)
mu_individ[:, 5]  # Time - muon timing (ns)
mu_individ[:, 6]  # Energy - muon energy (GeV)
```

### 8. Muon Boundaries: `muons_prty/mu_starts/{part}/data` **[MC ONLY]**
**Shape**: `(n_events+1,)` | **Type**: `int32`

Similar to `ev_starts` but for splitting muon data by events:
```python
# Muons for event i span from mu_starts[i] to mu_starts[i+1]
event_0_muons = mu_individ[mu_starts[0]:mu_starts[1]]
```

### 9. Time Residuals: `raw/t_res/{part}/data` **[MC ONLY]**
**Shape**: `(n_hits,)` | **Type**: `float32`

Calculated time residuals for each hit based on true muon tracks (physics-based feature).

## Data Processing Applied

Based on `root2h5_exp.py` configuration:

1. **Coordinate Centering**: `shift_coords_to_cl_center: true`
   - All spatial coordinates shifted to cluster center
   - Cluster center stored in `clusters_centers/data`

2. **Time Centering**: `center_times: true` 
   - Hit times centered to have zero mean per event

3. **Quality Filtering**: `exclude_big_ts: true`
   - Hits with time residuals > `1e5` ns removed

4. **Cluster Processing**: `take_single_cluster: true`, `split_multi: true`
   - Single-cluster events extracted
   - Multi-cluster events split into individual clusters

## Variable-Length Sequences

**Key Challenge**: Events have different numbers of hits (variable sequence lengths)

```python
# Example event sizes from exp.h5
Event 0: 49 hits
Event 1: 57 hits  
Event 2: 48 hits
# ... ranges from ~30 to ~200+ hits per event
```

## Usage for PyTorch Dataset

```python
class NuMuDataset(Dataset):
    def __init__(self, h5_path, particle="exp"):
        self.h5_path = h5_path
        self.particle = particle
        # Load all parts and concatenate
        
    def __getitem__(self, idx):
        # Use ev_starts to extract event idx
        start = self.ev_starts[idx]
        end = self.ev_starts[idx + 1]
        
        features = self.data[start:end]  # Shape: (n_hits, 5)
        labels = self.labels[start:end]  # Shape: (n_hits,)
        
        return features, labels
        
    def __len__(self):
        return len(self.ev_starts) - 1
```

## Data Statistics 

### Experimental Data (exp.h5)
- **Total Events**: ~625,000 (25,000 per part × 25 parts)
- **Total Hits**: ~40M hits
- **Average Hits/Event**: ~64 hits

### Monte Carlo Data Example (example_mc_5_muatm_files.h5) 
- **Total Events**: ~152,000 (varies per part: 26K-33K events)
- **Total Hits**: ~11M hits  
- **Average Hits/Event**: ~75 hits
- **Muons/Event**: 1-20+ muons per event (variable)
- **Primary Energy**: 7-1000+ GeV
- **Clusters**: 7 detector clusters (vs 1 in experimental)

### Common Statistics
- **Coordinate Range**: ~[-400, 400] meters (cluster-centered)
- **Time Range**: Variable per event (centered to 0 mean)  
- **Amplitude Range**: [0, ~15] (detector-dependent units)

## Important Notes for Model Development

1. **Variable Sequences**: Must handle different event sizes
2. **Temporal Order**: Hits are sorted by time within each event
3. **Spatial Distribution**: 3D coordinates in detector geometry
4. **Multi-Part Loading**: Data split across multiple file parts
5. **Memory Efficiency**: Consider lazy loading for large datasets

## Experimental vs Monte Carlo Comparison

| Feature | Experimental Data | Monte Carlo Data |
|---------|------------------|------------------|
| **Ground Truth** | No physics truth | Full physics information |
| **Physics Info** | None | Primary particle, muon tracks |
| **Time Residuals** | None | Calculated from true tracks |
| **Use Case** | Real-world validation, Domain Adaptation | Training with ground truth |
| **Complexity** | Simpler structure | Rich physics metadata |

## Parquet Catalog Format

Catalogs are Polars-readable Parquet files containing per-event metadata extracted from HDF5 files. They allow fast filtering and selection without loading full hit arrays.

### Common Columns (all catalogs)

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | Binary | Unique event identifier |
| `n_hits` | Int32 | Hits per event |
| `n_unique_strings` | Int32 | Unique detector strings per event |
| `mean_z` | Float32 | Mean z-coordinate of hits |
| `hit_start_idx` | Int64 | Start index in flat `raw/data` array of the corresponding h5 part |
| `hit_end_idx` | Int64 | End index in flat `raw/data` array |

### MC-only Columns (`mc_merged`, `mc_signoise_normed`)

| Column | Type | Description |
|--------|------|-------------|
| `cluster_id` | Int32 | Detector cluster ID |
| `n_signal_hits` | Int32 | Hits with nonzero label |
| `n_unique_signal_strings` | Int32 | Unique strings with signal hits |
| `energy`, `theta`, `phi`, `weight` | Float32 | MC truth from `prime_prty` |
| `h5_part_num` | Int32 | Numeric part index (mc_merged only) |

### Exp-reco-only Columns (`exp_reco`)

| Column | Type | Description |
|--------|------|-------------|
| `n_signal_hits` | Int32 | Hits flagged by ScanfitMask |
| `n_unique_signal_strings` | Int32 | Unique strings with flagged hits |
| `reco_theta`, `reco_phi` | Float32 | Reconstructed angles |
| `reco_theta_err`, `reco_phi_err` | Float32 | Reconstruction errors |
| `reco_func_value`, `reco_time_chi2`, `reco_charge_term`, `reco_ll_fit` | Float32 | Fit quality |
| `reco_n_hits`, `reco_n_strings`, `reco_n_oms`, `reco_path_length`, `reco_time_xyz` | Float32 | Reco event properties |

### Catalog Usage

```python
import polars as pl

# Lazy scan — reads only requested columns from disk
df = pl.scan_parquet("data_manager/h5_catalogs/catalogs_mc_merged/muatm_2020.parquet")

# Filter + select (predicate pushdown)
big = df.filter(pl.col("n_hits") > 10).select("event_id", "n_hits").collect()
```

### Rebuilding Catalogs

```bash
python data_manager/build_catalog_mc_merged.py [--particles muatm_2020 nue2_2020]
python data_manager/build_catalog_mc_normed.py [--regimes train val test]
python data_manager/build_catalog_exp.py
python data_manager/build_catalog_exp_reco.py
```

**Warning**: `build_catalog_mc_merged.py` for `muatm_2020` takes ~60 GB RAM.

## Signal/Noise Hit-Classifier Predictions

Some catalog directories contain `signoise_encoder_nl5_nh1_dff512_hs512_bs128_preds/` subdirectories with per-event predictions from an external signal/noise hit-classifier NN (Plotnikov's model). These predictions assign signal/noise probabilities to each hit and are used in the signal-hits regime (Iteration 9).

## root2df: ROOT → DataFrame Reader

`data_manager/root2df/` provides a Polars-based ROOT file reader using uproot, as an alternative to the HDF5 pipeline for direct exploration:

- `main.py` — `RootFileReader` context-manager class
- `internal_root_paths.py` — dataclass-based ROOT TTree path definitions for MC, Exp, and Reco variants
- `polars_schema.py` — Polars type schemas for DataFrame columns

```python
from data_manager.root2df.main import RootFileReader
from data_manager.root2df.internal_root_paths import MCRootPaths

with RootFileReader("file.root", MCRootPaths(), prefix="muatm_") as reader:
    events_df = reader.read_events_as_df()
    pulses_df = reader.read_pulses_as_df()
    coords_df = reader.read_OM_coords()
```

## constants.py

`data_manager/constants.py` defines physics and detector constants:

- Water properties: refractive index (N=1.37), Cherenkov angle, speed of light in water
- Detector layout: `CHANNEL_DIVISOR=288`, `STRING_DIVISOR=36`, `STRINGS_PER_CLUSTER=8`

## Binary Classification Tasks

### For Neutrino Detection:
- **Positive Class**: Neutrino-induced events
- **Negative Class**: Background/atmospheric muon events
- **MC Advantages**: Can use ground truth `prime_prty` for supervision
- **Experimental Advantages**: Real-world detector response

### Approaches in Use:
1. **Event-level Classification**: Use entire event sequence → binary label (Iterations 4/4b — prefilter)
2. **Hit-level Classification**: External signal/noise NN classifies individual hits
3. **Domain Adaptation**: DANN training on MC, applied to experimental data
4. **Signal-hits Regime**: Strip noise hits first, then classify events (Iteration 9 — planned)