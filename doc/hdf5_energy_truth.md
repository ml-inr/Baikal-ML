# `baikal_mc_merged_energy_truth.h5` — the muon energy truth companion

This file answers one question: **how much energy did each simulated muon have at each point
of its path, and where did it lose that energy?** The main file
(`baikal_mc_merged.h5`) stores what the detector saw; this one stores what the Monte Carlo
knows about the muon behind it.

It is a **companion**, not a replacement: it holds only what the main file lacks, and its rows
line up with the main file position by position. Nothing here is a training target — targets
are derived from it, and deliberately not frozen into it.

Read [mc_energy_truth.md](mc_energy_truth.md) for the physics of what these quantities mean and
[mc_provenance.md](mc_provenance.md) for where the simulation comes from.

---

## 1. The vocabulary, in the order you need it

Everything below is defined against one picture. A simulated muon is a straight line; every
quantity in this file is either a point on that line, an energy at such a point, or an
interval between two of them.

```
       muon direction of motion  d = (sinθcosφ, sinθsinφ, cosθ)
       ─────────────────────────────────────────────────────────▶

   ×════════════◇═══════════●═════════════╪═════════════════╪══════▷
   │            │           │             │                 │
   │            │           │             └─ shower         └─ shower
   │            │           │                (stochastic energy loss)
   │            │           │
   │            │           └─ REFERENCE POINT,  s = 0
   │            │              the point of the line closest to the
   │            │              origin of the global frame. Purely
   │            │              geometric — nothing physical happens here.
   │            │
   │            └─ where the cylinder around the cluster is entered
   │
   └─ TRACK START, s = s_track_start
      where the muon's simulated trajectory begins

   s  ── the coordinate along the track, in metres:   s(r) = (r − r_ref) · d
         s = 0 at the reference point, growing in the direction of motion.
         Every "where along the track" quantity in this file is an s.
```

| term | meaning |
|---|---|
| **row** | one event **in one cluster**. An event that lit two clusters occupies two rows, with the same MC truth repeated. All arrays in the main file and here are indexed by row, never by event. |
| **muon** | one simulated track. A row has one muon for `nuatm`/`nue2`, on average 4.1 for `muatm` (an air-shower bundle). |
| **reference point** | the point of the track closest to the origin of the global coordinate frame (the centre of the whole array, *not* of the cluster). BARS stores the track as this point plus a direction. It is a bookkeeping anchor: the muon may not even exist there. Verified: the track is perpendicular to the position vector at this point in 100.00 % of muons, worst deviation 1.4 mm. |
| **s** | signed distance along the track from the reference point, metres, positive downstream. |
| **track start** | `s_track_start = −c · t_first_muon`, `c = 0.299792458 m/ns`. Where the simulated trajectory begins. |
| **shower** | one stochastic energy loss of the muon (pair production, bremsstrahlung, photonuclear) — in ROOT, one `BMCInteraction`. Recorded when its **light yield** exceeds 0.001 P.El., not when its energy exceeds a threshold. Continuous ionisation loss is *not* in this list; it is modelled (§4). |
| **volume** | a cylinder around the centre of the row's own cluster, used to ask "what was the muon's energy when it arrived here". Three nested sizes (§5). |

### What `t_first_muon` measures

The clock's zero is the moment the muon passes the **reference point**. A muon moving at
essentially the speed of light in vacuum (β = 0.9999967 at 41 GeV) therefore covers
`c · t_first_muon` metres between the start of its trajectory and the reference point, and the
start sits at `s = −c · t_first_muon`.

Sign, and why it is a useful check: a positive `t_first_muon` puts the start behind the
reference point (the usual case), a negative one puts it ahead — meaning the muon did not yet
exist at the reference point. That is exactly the population whose energy ROOT writes as the
sentinel `−0.001`, and the two agree in **100.00 %** of muons of all three productions.

The boundary belongs to the non-sentinel side. A muon whose trajectory begins exactly at the
reference point has `t_first_muon = 0` and therefore `s_track_start = −0.0`, and it is alive
there — which is what "not a sentinel" means. Test the two populations with `s > 0` and
`s ≤ 0`, not `s < 0`: in IEEE arithmetic `−0.0 < 0` is false, and the case does occur (one
muon in the 10.5 M checked).

### Is the track start the muon's birth point?

Not always, and the difference matters.

- For **sentinel muons** (`e_ref = −0.001`, 24.1 % of `nuatm`, 14.1 % of `nue2`): yes. The
  muon is born there. `verified`.
- For **`muatm`**: no. Atmospheric muons are born kilometres up in the air, but the track
  start has a median height of +483 m and never lies further than ~795 m from the array
  centre. It is the point where the muon **enters the generation volume** and the simulation
  starts tracking it.
- For the remaining neutrino-induced muons: undetermined. Part of them are genuine vertices,
  part are clipped at the generation boundary (99th percentile of the distance is 672 m
  against a hard ceiling of 703 m). `assumed`.

The safe reading is **"where the simulated trajectory begins"**. Since clipping only ever
happens at the *outer* boundary, a start point found *inside* a cluster is always a real
birth point — which is what makes the status code `1` of §5 trustworthy.

---

## 2. Coordinates, units and clocks — read this before computing anything

Four conventions differ between this file and the main file. Each of them has already caused a
wrong result in this project at least once.

| | this file | main file (`raw/data`) |
|---|---|---|
| **spatial frame** | global array frame | shifted to the row's cluster centre |
| **time origin** | muon birth (§3) | mean hit time of the event, subtracted and discarded |
| **energy unit** | GeV | — |
| **angle unit** | radians (`direction`) | radians in `muons_prty/individ`, **degrees** in `prime_prty` |

To put this file's coordinates into the frame the network sees:

```python
c = clusters_centers[cluster_ids[row]]      # both from the main file
xyz_cluster = xyz_global - c
```

The reverse is not exact for time: the per-event time offset the converter subtracted from hit
times was never stored, so shower times and hit times cannot be put on a common clock. Their
layouts look alike on purpose; their zeros are not the same. Do not overlay them.

Energies of showers are **GeV here and TeV in ROOT** — the factor of 1000 is applied on
writing. Everything in this file is GeV.

---

## 3. Layout

Rows correspond **one to one** with the main file: the same `part_NNNN` groups, the same number
of rows, the same number of muons. `ev_ids` and `mu_starts` are copied from the main file and
exist only so that the builder — and you — can assert the alignment instead of trusting it.

```
{ptype}/                                   nuatm_2020 | nue2_2020 | muatm_2020
├── ev_ids                    (n_rows,)        S25      copy of the main file
├── mu_starts                 (n_rows+1,)      int64    copy of the main file
├── shower_starts             (n_muons+1,)     int64
├── showers                   (n_showers, 5)   float32  energy_gev, time_ns, x, y, z
├── n_muons_lighting_cluster  (n_rows,)        int16
├── ref_xyz                   (n_muons, 3)     float32  copied
├── direction                 (n_muons, 2)     float32  theta, phi — radians, copied
├── e_ref                     (n_muons,)       float32  copied
├── s_track_start             (n_muons,)       float32
├── e_at_entry                (n_muons, 3)     float32
└── e_at_entry_status         (n_muons, 3)     int8
```

Every dataset is `{ptype}/{name}/part_NNNN/data`, matching the main file's convention.

Two nested index levels, the same pattern as `muons_prty/mu_starts`:

```
row ──mu_starts──▶ muon ──shower_starts──▶ showers
```

```python
muons   = slice(mu_starts[row],  mu_starts[row + 1])
showers = slice(shower_starts[mu], shower_starts[mu + 1])
```

### Dataset by dataset

| dataset | provenance | notes |
|---|---|---|
| `ev_ids`, `mu_starts` | copied verbatim | checksums; the builder aborts on any mismatch |
| `shower_starts` | built | boundaries of each muon's shower chain |
| `showers` | ROOT `fTracks.fInteractions` | **in ROOT order, unsorted** (§6). `x, y, z` global metres; `energy_gev` converted from TeV; `time_ns` derived (§4) |
| `n_muons_lighting_cluster` | main file `raw/labels` | how many of the row's muons produced light **in this cluster** — see below |
| `ref_xyz`, `direction`, `e_ref` | copied from `muons_prty/individ` | duplicated so this file stands alone; bit-identical to the main file |
| `s_track_start` | derived | `−c · t_first_muon` |
| `e_at_entry`, `e_at_entry_status` | derived, model-dependent | §5 — **never use the energy without the status** |

### `n_muons_lighting_cluster` in full

A row is one event in one cluster, but a `muatm` event is a bundle of on average 4.1 muons
spread over hundreds of metres. Most of them miss the cluster entirely. This field says **how
many of the row's own muons put at least one recorded photon into this cluster** — that is, how
many of them the row's hits can possibly be about.

It is counted from the main file's `raw/labels`, which stores the origin of every hit as the
code BARS calls `fMagic`:

| label | origin |
|---|---|
| `0` | PMT noise |
| `−(10⁶ − j)` | Cherenkov light of the **track** of muon `j` |
| `k·10⁶ + j` | light of the **k-th shower** of muon `j` |

`j` is the muon's 1-based position in the event's track list — i.e. muon
`mu_starts[row] + j − 1` of this file. `k` indexes the shower chain in ROOT order, which is why
`showers` keeps that order (§6).

**The muon index is lost for `k ≥ 17`.** The generator writes every pulse word as a float32
(doc/mc_binary_formats.md), and a float32 holds integers exactly only up to 2²⁴ = 16 777 216.
A code `k·10⁶ + j` therefore loses its low digits once `k` reaches 17, and comes back rounded to
an even number: `j` decodes as 0, 2, 12, 64 — noise, not a muon. Verified on all three
productions: **every** code below 2²⁴ has `j ≥ 1`, and **every** code above it is even. The damage
happens in the generator's own output format, so nothing downstream can repair it.

Such hits are light whose muon cannot be known. They are not attributed to any muon — writing
`j = 2` because the arithmetic said so would invent one — but they are not discarded either: a
row lit only by them was lit by at least one muon, and is counted as 1. For a row holding a
single muon that is exact, not a bound; only for bundles is it a lower bound, and there it is
rare (0.05 % of `muatm` shower hits, no affected row in the part measured). In `nue2`, where
PeV muons produce hundreds of cascades, 22.6 % of shower hits are affected and 1.5 % of rows
are counted this way.


The count is the number of **distinct `j`** among the row's hits, taking both track light and
shower light: a muon whose light reached the cluster only through one of its cascades still lit
it. Noise hits are excluded, so a row of pure noise gives `0`.

```python
lab  = labels[ev_starts[row]:ev_starts[row + 1]]
sig  = lab != 0
lost = sig & (lab >= 2**24)                     # muon index destroyed by float32
j    = np.where(lab < 0, lab + 1_000_000, lab % 1_000_000)
n    = len(np.unique(j[sig & ~lost]))
n    = max(n, 1) if lost.any() else n
```

Range: `0 … mu_starts[row+1] − mu_starts[row]`. The interesting selection is `== 1` — a bundle
event in which the cluster saw exactly one muon, which is the closest MC analogue of a
single-track experimental event.

One caveat beyond the index loss: counting shower light as lighting is **baked in**.
The field is an aggregate, and a stricter variant (direct track light only) needs
another full pass over `raw/labels` — which is why the field exists at all.

`e_ref` is `fMuonEnergy` exactly as ROOT holds it, including its three cases: positive (the
muon is alive at the reference point), exactly `−0.001` (not yet born), any other negative
value (already dead). Documented in [mc_energy_truth.md](mc_energy_truth.md) §2.

---

## 4. How energy along the track is computed

A muon loses energy two ways: **continuously**, by ionising the water, and **in jumps**, by the
stochastic processes recorded as showers. The textbook formula lumps both together as
`dE/dx = a + b·E`, where `b·E` is the *average* of the stochastic part.

Here the stochastic losses are known individually, so using `b·E` as well would count them
twice. The model is therefore the ionisation term plus the explicit shower sum:

```
E(s) = e_ref − a · s − Σ energy_gev[i]  for every shower with 0 < s_i ≤ s
                                          (signs flip for s < 0)
a = 0.24 GeV/m
```

Both terms are signed, so the same expression propagates forwards and backwards: going
backwards the muon *had* more energy, and both the ionisation term and the shower sum add
rather than subtract.

Shower times follow from the same geometry — the muon reaches `s` at

```
time_ns = t_first_muon + s / c
```

so `time_ns = 0` is the muon's birth. This column is redundant by construction and the builder
asserts it rather than trusting it. Verified on `nuatm`: every shower satisfies
`s ≥ s_track_start` and `time_ns ≥ 0`, both at 100.000 %, with the earliest shower sitting
exactly at the start.

**Where this breaks.** The ionisation term is an approximation with no fluctuations, and
`a = 0.24 GeV/m` is a constant where the real value drifts slowly with energy. Over the tens of
metres inside a cluster the error is small; over the several hundred metres between the
reference point and the track start it is not. [mc_energy_truth.md](mc_energy_truth.md) §8
quantifies this.

---

## 5. `e_at_entry` — energy on arrival, and its status

Three nested cylinders around the centre of the row's own cluster:

| column | Δr | cylinder |
|---|---|---|
| 0 | 0 | `r ≤ 60 m`, `|z| ≤ 265 m` — the cluster itself |
| 1 | 30 | `r ≤ 90 m`, `|z| ≤ 295 m` |
| 2 | 60 | `r ≤ 120 m`, `|z| ≤ 325 m` |

The instrumented volume is not sharp — a muon passing 20 m outside the last string still lights
it — so the two enlarged volumes exist to ask the same question with a margin. The exact sizes
live in the file's attributes; with `ref_xyz`, `direction` and `s_track_start` you can recompute
any other radius in a few lines, so nothing is frozen by this choice.

`e_at_entry[:, k]` is the muon's energy where it crosses into volume `k`. That question has no
answer for a large fraction of muons, which is what the status column is for:

| status | meaning | what the energy column holds |
|---|---|---|
| 0 | propagated from the reference point to the entry point | the propagated energy |
| 1 | the muon was **born inside** this volume | its energy at birth |
| 2 | the muon died before reaching the volume | `NaN` |
| 3 | the track never crosses this volume | `NaN` |
| 4 | sentinel muon: propagated from `e_bundle_reg` at the track start | the propagated energy, **weaker footing** — see below |
| 5 | sentinel muon **born inside** this volume | `e_bundle_reg` itself, at the birth point |

Codes `1` and `5` on column 0 are the answer to "was this muon born inside the detector".
They are kept apart because the energy means different things: `1` is propagated from
`e_ref`, which ROOT states directly, while `5` inherits the weaker footing of `4`.

Status `4` rests on reading `fSumEnergyBundleReg` as the muon's energy at the start of its
track. That reading is `consistent`, not `verified` ([claims_log.md](claims_log.md)), and it
covers about a quarter of `nuatm` muons — which is precisely why it gets its own code instead
of being blended into `0`.

`2` and `3` are the two ways of having no answer, and they are different questions: `3` says
the muon's trajectory never reaches the volume at all, `2` says it would have, but the muon ran
out of energy first. Both store `NaN`.

Comparing the three columns is informative by itself: `3, 3, 0` means the track missed the
cluster and only clipped the outermost volume; `2, 0, 0` means the muon died between the middle
and the inner volume.

---

## 6. Why the showers are in ROOT order

They arrive from ROOT **not** sorted along the track: 2.4 % of tracks contain one large
backward jump, splitting the chain into two blocks each of which is ordered. `BMCInteraction`
carries no type field, so that block structure is the only surviving trace of a category
distinction — and electromagnetic and hadronic cascades of equal energy do not produce equal
light.

Sorting would destroy it, so this file keeps ROOT's order verbatim. Sorting along the track is
one line when you need it, and it is not the file's business to decide that you do:

```python
s = (showers[:, 2:5] - ref_xyz[mu]) @ direction_vector(mu)
chain = showers[np.argsort(s)]
```

Keeping ROOT order also means the shower index `k` encoded in the main file's `raw/labels`
(`k·10⁶ + j`, "light of the k-th shower of muon j") addresses these rows directly.

---

## 7. Worked examples

**Energy of a muon where it passes closest to its cluster.**

```python
d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
s_ca = (clusters_centers[cluster] - ref_xyz[mu]) @ d          # closest approach
sh   = showers[shower_starts[mu]:shower_starts[mu + 1]]
s_sh = (sh[:, 2:5] - ref_xyz[mu]) @ d
lost = sh[(s_sh > 0) & (s_sh <= s_ca), 0].sum() if s_ca > 0 else \
      -sh[(s_sh <= 0) & (s_sh > s_ca), 0].sum()
e_ca = e_ref[mu] - 0.24 * s_ca - lost
```

**Events where exactly one muon of the bundle lit the cluster.**

```python
rows = np.flatnonzero(n_muons_lighting_cluster[:] == 1)
```

**Muons born inside the detector.**

```python
born_inside = np.isin(e_at_entry_status[:, 0], (1, 5))
```

**Total energy a muon deposited inside the cluster** — entry and exit energies of volume 0,
falling back to the birth point when the muon started inside.

---

## 8. What is established and what is not

| statement | status |
|---|---|
| rows correspond 1:1 with the main file | `verified` — asserted at build time on every part |
| the reference point is the closest approach to the array origin | `verified` — 100.00 %, worst 1.4 mm |
| `s_track_start = −c · t_first_muon` marks the beginning of the trajectory | `verified` — 100.00 % of showers lie at `s ≥ s_track_start` |
| the sentinel `−0.001` means "not yet born at the reference point" | `verified` — 100.00 % of sentinels start downstream, 0.00 % of the others do |
| the track start is the muon's birth point | `verified` for sentinels, `assumed` otherwise, **false** for `muatm` |
| `raw/labels` carries the per-hit light origin including the muon index | `verified` — decoded up to muon 248 on `muatm` |
| the muon index of a shower code is destroyed for `k ≥ 17` by the float32 pulse word | `verified` — every code below 2²⁴ has `j ≥ 1`, every code above it is even, on all three productions |
| every shower lies at `s ≥ s_track_start` | `verified` with a measured exception rate of 1 in 7·10⁶: one 6 PeV `nue2` cascade sits 6 m upstream |
| `e_bundle_reg` is the muon's energy at the track start (status `4`) | `consistent` — not independently verified |
| the ionisation constant `a = 0.24 GeV/m` | `assumed` — a standard value, not fitted to this MC |

Every claim above is logged in [claims_log.md](claims_log.md) with its evidence.

---

## 9. Rebuilding

```
ROOT (cluster62) ──read_root.py──▶ npz per file ──build_h5.py──▶ this file
                                                  ▲
                                   baikal_mc_merged.h5
```

`truth.py` holds the geometry and the propagation and is shared by the builder and by
downstream analysis, so the model in §4 has exactly one implementation.
