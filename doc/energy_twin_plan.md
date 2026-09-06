# Rebuild plan: the energy-truth companion

Working document. The user-facing result is [hdf5_energy_truth.md](hdf5_energy_truth.md);
this file is the construction plan and the record of what has to be checked.

## 0. Why rebuild rather than patch

The previous companion stored precomputed targets (`E_ca`, `s_ca`, `b_impact`, …) alongside the
truth. That froze three choices into a 8 GB file: the cylinder radius, the propagation model,
and the sort order of the shower chain. All three changed during this session's investigation.
The new file stores **only what ROOT knows**, plus a small number of derived fields that are
expensive to recompute, and leaves the modelling to `truth.py`.

Consequences for the code: the builder no longer computes targets, so the five modules planned
earlier collapse to three.

```
read_root.py   ROOT (cluster62)  →  npz per file        extraction, no physics
truth.py       pure functions: geometry, propagation     no I/O, shared with analysis
build_h5.py    npz + main h5     →  companion h5         assembly and assertions
```

## 1. Stage 1 — `read_root.py` (runs on cluster62)

Replaces `energy_truth/extract_interactions.py`. Same ROOT access technique (`TTree::Draw`
with `"goff"`, because uproot cannot deserialise the custom `BMCInteraction` class), three
differences:

- **no sorting.** The old extractor ordered each chain along the track and stored an
  `int_block` label to preserve the block structure. The new schema keeps ROOT order, so the
  sort disappears and with it the reason for `int_block`.
- **track kinematics are written out on purpose.** They duplicate `muons_prty/individ`, and
  that is the point: the builder compares them value by value to prove its mapping from ROOT
  entries to HDF5 rows. They are not copied into the companion file — the npz are scratch.
- **the start-index rule stays.** `root2h5.py` reads every array as `[st:]`, so `ev_ids`
  refers to ROOT entry `k + st`. `get_start_index` must be reproduced verbatim; `st` was 0 in
  all 120 files sampled, but the converter allows 1 and a silent off-by-one shifts an entire
  file.

Output per ROOT file, `NNNN.npz`:

| array | shape | meaning |
|---|---|---|
| `n_showers_per_track` | (n_tracks,) int32 | `fTracks.fInteractionN` |
| `n_tracks_per_entry` | (n_entries,) int32 | tracks per ROOT entry, for the row mapping |
| `xyz` | (n_showers, 3) float32 | global metres, ROOT order |
| `energy_gev` | (n_showers,) float32 | converted from TeV on write |
| `trk_xyz` | (n_tracks, 3) float32 | reference point, for the mapping assert |
| `trk_theta_deg`, `trk_phi_deg` | (n_tracks,) float32 | **degrees**, as ROOT holds them |
| `trk_energy_gev` | (n_tracks,) float32 | `fMuonEnergy`, for the mapping assert |

Cost of the track arrays: 24 B per track, so ~57 GB of the muatm npz. They are scratch — the
npz are discarded once the companion is built and validated.

Self-checks, each fatal:

1. `sum(n_showers_per_track) == len(xyz)`;
2. `sum(n_tracks_per_entry) == len(n_showers_per_track)`;
3. collinearity: every shower lies on its own track's line, worst perpendicular deviation
   below 0.1 m (the existing check found 1.3 mm — it catches a mis-sliced flat array);
4. `SetEstimate` large enough for the branch, since a truncated `TTree::Draw` buffer already
   produced a bogus result once in this project.

**Re-extraction is required** — the existing npz (400 nue2, 2000 nuatm locally; 20 004 muatm,
59 GB on cluster62) hold sorted chains. ROOT order is in principle recoverable from them (the
block label was assigned before sorting, and within a block the projection is non-decreasing by
construction, so the original order is a stable sort by `(block, s)`) — but that is a subtle
inversion to rely on. Plan: re-extract everything, and use the inversion on ~20 files purely as
a cross-check that the new extractor agrees with the old data.

Cost: the muatm pass took roughly 8–10 h at 4–6 workers last time. Cluster62 has 12 cores and
no batch system; cap at 6 workers, `nohup`, one output file per input so the run is resumable.

## 2. Stage 2 — `truth.py` (pure, no I/O)

Every function vectorised over muons, `float64` internally. The float32 accumulation error in a
global `cumsum` over ~10⁶ GeV already caused one wrong result here; the promotion is not
optional.

```python
def direction(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:      # (n, 3)
def s_along(points, ref, d) -> np.ndarray                              # (r − ref) · d
def track_start(t_first_muon) -> np.ndarray                            # −c · t
def energy_at(s, e_ref, shower_s, shower_e, starts) -> np.ndarray      # §4 of the doc
def cylinder_crossing(ref, d, centre, radius, z_half) -> (s_in, s_out) # NaN if no crossing
def entry_energy(...) -> (e, status)                                   # the 5 status codes
def to_cluster_frame(xyz, centre) -> np.ndarray
```

`energy_at` is the single implementation of the propagation model, used by the builder and by
downstream analysis alike. `a = 0.24` is a module constant, not a literal.

Order of business inside `entry_energy`, since the status codes are a decision tree:

```
the trajectory never enters the volume          → 3, NaN
  (the line may still cross it upstream of where the muon begins)
the muon begins inside the volume               → 1, or 5 if it is a sentinel
e_ref is the sentinel                           → 4, propagate from e_bundle_reg at the start
otherwise                                       → 0, propagate from the reference point
  ... and if the propagated energy is <= 0      → 2, NaN
```

Two ordering traps. "Died before entry" is decided by the propagation, so it can only be
tested after the model has run, not from `e_ref < 0` alone — a muon dead at the reference
point may well have been alive upstream of it. And "born inside" combines with "sentinel",
which is why the tree has six outcomes rather than five: their energies come from different
sources and one of the two carries a weaker claim.

## 3. Stage 3 — `test_truth.py`

Two kinds of tests. Analytic ones, where the answer is known on paper:

- a track through the cylinder axis: `s_out − s_in = 2·z_half / |cos θ|` for a vertical muon,
  `2·radius` for a horizontal one through the centre;
- a track tangent to the cylinder produces one crossing or none, not a negative length;
- propagation with no showers is linear: `E(s) = E_ref − a·s`;
- propagating forwards then backwards returns the starting energy to float64 precision.

And regression tests against the numbers established in this session, which the new code must
reproduce on `part_1000` of each production:

| check | expected |
|---|---|
| reference point perpendicular to its position vector | 100.00 %, worst ≤ 1.4 mm |
| showers satisfy `s ≥ s_track_start` | 100.000 % |
| shower `time_ns ≥ 0` | 100.000 %, minimum exactly 0 |
| sentinel muons have `s_track_start > 0` | 100.00 % |
| non-sentinel muons have `s_track_start < 0` | 100.00 % |
| rows and muons per part | nue2 and nuatm 1.00 muon/row, muatm 4.09 |

## 4. Stage 4 — `build_h5.py`

Per production, per part, streaming, one part at a time; a part is written or the run aborts.

1. read the part's `ev_ids`, `mu_starts`, `muons_prty/individ`, `raw/cluster_ids`,
   `raw/ev_starts`, `raw/labels` from the main file;
2. map ROOT entries to rows through `ev_ids` (the identifier is `{particle}_{file}_{entry}`).
   This is the step that can go silently wrong: `root2h5.py` drops events, duplicates
   multi-cluster ones, and writes single-cluster rows before split ones, so rows are **not** in
   ROOT entry order. Immediately after building the mapping, assert it against the npz track
   arrays — `trk_xyz` and `trk_energy_gev` bit-identical to `individ[:, 2:5]` and
   `individ[:, 6]`, `radians(trk_theta_deg)` equal to `individ[:, 0]` within float32
   precision. A reference point is a continuous 3-vector; no wrong mapping reproduces it;
3. expand the npz shower chains from per-entry to per-row order with `np.repeat`;
4. compute `s_track_start`, shower times, `e_at_entry` and its status through `truth.py`;
5. assert, and abort the whole build on failure:
   - `len(ev_ids)` matches the main file,
   - `ev_ids` bit-identical to the main file,
   - `mu_starts` identical,
   - `ref_xyz`, `direction`, `e_ref` bit-identical to `muons_prty/individ`,
   - recomputed shower `s` reproduces the stored `time_ns` to float32 precision,
   - `shower_starts[-1] == len(showers)`;
6. write with the same `{ptype}/{name}/part_NNNN/data` layout and gzip on the large arrays.

`n_muons_lighting_cluster` comes from `raw/labels`, defined in full in
[hdf5_energy_truth.md](hdf5_energy_truth.md) §3. Three things to handle rather than assume:

- hits with label `0` are noise — the code differs from the `1` used in the upstream `.dat`,
  so the noise test must be on `0` and not inherited from the binary-format documentation;
- a shower code with `j = 0` appears in the data with no established meaning; count those hits
  separately and log the share instead of attributing them to muon 1;
- that `j` is the 1-based index into the row's own muon range is `consistent`, not `verified`:
  it was checked on a handful of rows and against the maximum (248 on `muatm`). The builder
  must therefore **assert `j ≤ mu_starts[row+1] − mu_starts[row]` on every row** — a violation
  would mean `j` is numbered against the ROOT event rather than the row, which for
  multi-cluster events is not the same thing.

Cost of this step: labels are ~60 hits per row, so ~35·10⁹ int32 for muatm, over 100 GB of
reads. Disk-bound, one pass, and it is the reason the field is stored rather than recomputed.

## 5. Stage 5 — validation of the finished file

A separate script, run on the completed file, not on parts:

- row and muon totals per production against the main file (expected 7.7 M / 66.7 M / 578 M
  rows, 7.7 M / 66.7 M / 2362 M muons);
- distribution of `e_at_entry_status` per production per volume, printed and recorded — the
  share of code `4` on `nuatm` should come out near 24 %, and a large deviation means the
  sentinel logic broke;
- the six regression checks of §3 re-run on a random sample of parts, not just `part_1000`;
- the share of showers whose `s` falls inside volume 0, as a sanity figure for later work.

## 6. Budget

All measured on the finished file.

| production | rows | muons | showers | showers/muon | companion |
|---|---|---|---|---|---|
| nue2_2020 | 7,665,396 | 7,665,396 | 69,416,080 | 9.06 | 1.5 GB |
| nuatm_2020 | 66,695,937 | 66,695,937 | 61,149,470 | 0.92 | 5.3 GB |
| muatm_2020 | 576,947,380 | 2,353,789,797 | 1,705,616,749 | 0.72 | 156 GB |
| **total** | **651,308,713** | **2,428,151,130** | **1,836,182,299** | | **163 GB** |

The ten-fold difference in showers per muon between the two neutrino productions is the E^-2
astrophysical spectrum of nue2: PeV muons radiate hundreds of cascades, atmospheric ones at
tens of GeV radiate almost none. It is also why the float32 index loss of section 4 bites nue2
(22.6% of shower hits) and not nuatm (0.09%).

Extraction cost, measured on cluster62 at 6 workers: nue2 400 files in 1m42s, nuatm 2000 in
9m18s, muatm 20004 in 2h44m. No failures in 22,404 files.

## 7. Risks

- **Re-extraction time.** ~10 h of cluster62 for muatm, resumable per file.
- **Row mapping.** Mitigated as described in §4 step 2: the extractor ships the track
  kinematics so the builder can prove the mapping bit for bit, rather than inferring it from
  collinearity, which tests only the nesting inside an entry, not the entry-to-row map.
- **`j = 0` shower codes.** Unknown meaning, counted separately, must not be silently merged.
- **Status `4` semantics.** Rests on a `consistent`-grade reading of `e_bundle_reg`. If that
  reading is later refuted, only one column of one derived field is affected, and the raw
  inputs to recompute it are all in the file.

## 8. Decisions taken

1. **The existing npz are discarded and everything is re-extracted** with the new extractor, in
   ROOT order. The order-inversion argument of §1 is kept only as a cross-check on a handful of
   files.
2. **Everything is built locally.** Only the ROOT reading has to run on cluster62, because the
   ROOT files and the BARS/ROOT environment live there; it emits npz, those travel, and every
   later stage reads the local `baikal_mc_merged.h5`. Nothing but the npz crosses the network.
3. **The 2019 season is skipped** — `muatm_2019`, `nuatm_2019`, `nue2_2019` are in the main file
   but unused by current work, and part of the 2019 ROOT set is unreadable. The companion will
   contain only the three `*_2020` groups, and its attributes will say so explicitly so that a
   missing group reads as a decision rather than as a failed build.
