# HDF5 Data Format Documentation

> **Note**: This file documents the HDF5 internal structure only. For the full data pipeline documentation (HDF5 + Parquet catalogs + root2df + constants), see [data_format.md](data_format.md).

*Generated from analysis of `exp.h5` and `example_mc_5_muatm_files.h5` created by `data_manager/root2h5/` scripts*

## Overview

The HDF5 files contain processed neutrino detector data organized into a hierarchical structure optimized for machine learning. Two types of data are available:

- **Experimental Data** (`exp.h5`): Real detector data from experimental runs. The file is stored at cluster63.inr.ac.ru by path: `/home/albert/Baikal2025/data_manager/h5datasets/exp.h5`
- **Monte Carlo Data** (`*.h5` with MC particle names): Simulated data with ground truth physics information

Each file contains multiple "parts" corresponding to different ROOT files.

## Data source

The HDF5 files are created using ROOT files, obtained with BARS. All ROOT files are stored at cluster62.inr.ac.ru cluster:

- **Experimental runs**: `/home/albert/Baikal/Data/exp_root_files/`
- **MC .root files**: `/home3/ivkhar/Baikal/data/initial_data/MC_2020/<particle_name>/root/all`

**Experimental data**  
Experimental ROOT files named indicating the season, cluster and run numbers in format: `s{year}_c{cluster}_r{run}`.  For example: s2020_c07_r0041.  
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