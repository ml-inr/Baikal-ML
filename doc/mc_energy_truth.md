# MC truth for muon energy

**What ground truth about muon energy exists in the Baikal MC, and what can legitimately be
built from it.** The conclusions constrain any energy target we define.

Companion documents: [mc_provenance.md](mc_provenance.md) — where the MC comes from and how to
match an event across the production chain; [mc_binary_formats.md](mc_binary_formats.md) — how
to read the upstream `.dat` / `.wout`; [claims_log.md](claims_log.md) — every claim below with
its evidence and confidence, including the ones that were refuted along the way.

Code: `data_manager/root2h5/energy_truth/`. Working data:
`data_manager/data/mc_energy_truth_interactions/`, archive `~/mc_energy_truth_extract/` on
cluster62.

---

## 1. The four quantities that exist

| quantity | where | defined for | meaning |
|---|---|---|---|
| `fMuonEnergy` | `BMCTrack` | 48% nuatm, 82% nue2 | muon energy at the **reference point**, GeV |
| `fSumEnergyBundleReg` | `BMCEvent` | **100%** | muon energy at the **detector's depth**, GeV |
| `fFirstMuonTime` | `BMCEvent` | **100%** | time at the reference point, measured **from the muon's birth** |
| `fInteractions` | `BMCTrack` | per track | stochastic losses: (E in TeV, x, y, z) |

The **reference point** is where the track passes closest to the centre of the whole array
(`BMCTrack.fX/fY/fZ`; perpendicularity verified to 1e-6, median 260 m from the centre). It is
the origin for the coordinate `s` along the track used everywhere below. Note it may sit
hundreds of metres from the cluster that actually saw the light.

Everything is measured in GeV except the interaction chain, which is in TeV — the extractor
applies the factor of 1000. Units were established against external anchors, not comments;
see [claims_log.md](claims_log.md) §1.

## 2. `fMuonEnergy`: one field, three cases

The sign encodes **where the muon is** relative to the reference point, not data quality:

| | sentinel `−0.001` | other negative | positive |
|---|---|---|---|
| nue2 | 14.03% | 3.79% | 82.18% |
| nuatm | **23.95%** | **28.17%** | 47.89% |
| muatm | 0.00% | 13.3% | 86.7% |

- **positive** — the muon is alive there; a real energy.
- **exactly −0.001** — a placeholder: the muon is **born downstream**, so it does not exist at
  the reference point. Verified on nue2 (15k) and nuatm (982k): 0.0% of these tracks have any
  interaction before the reference point.
- **other negative** — the mirror case, the muon **died upstream**; 100% have all activity
  before the reference point.

The value −0.001 is not a small negative energy: those tracks carry a median of 971 GeV of
summed interaction losses. It is absent from muatm entirely.

**Consequence:** `fMuonEnergy` is unusable for **17.8% of nue2 and 52.1% of nuatm**. Any
target propagated from it inherits that hole.

## 3. `fSumEnergyBundleReg`: the energy that always exists

Established on full statistics (66.4M nuatm + 7.1M nue2 events; raw arrays in
`~/reg_fullstat/` on cluster62).

`Reg` is the muon energy **at the detector's depth**. For *alive* tracks `Reg/E_ν` collapses
with energy (nue2: 0.486 → 0.041 from 0.1 TeV to 10 PeV) — a 4% share is impossible as CC
inelasticity, so the value is taken after a stretch of travel. The muatm cross-check settles
the geometry: there the birth energy is stored separately as `Surf`, and `Reg` is half of it,
falling with zenith angle (0.554 vertical → 0.337 at 60–80°, `corr` with `1/cos θ` = +0.539) —
i.e. after a slant path through water.

**For sentinel tracks `Reg` is the energy at the neutrino vertex.** Their ratio loses the
energy trend entirely (`corr` = +0.03 on nue2) and climbs to 0.769 at 10–100 PeV, which is the
shape and magnitude of the inelasticity `(1−y)`. These muons are born right at the array, so
nothing is lost in between. This makes `Reg` the only usable energy for exactly the events
where `fMuonEnergy` is absent.

For muatm `Reg` sums the whole bundle and is **not** a per-track quantity.

## 4. The birth point, from the time field

```
s_vertex = − c · fFirstMuonTime          (c = 0.299792458 m/ns)
```

`fFirstMuonTime` is the time of passing the reference point measured from the muon's birth, so
its sign says which side the vertex is on: negative for sentinels (born downstream), positive
otherwise. Verified on 38M nuatm tracks: the vertex lies before the first recorded interaction
in **100.000%** of cases and the sign is right in **100.000%**, across all three classes.

The muon's speed is `c` to within 3·10⁻⁶ at typical energies (β = 0.9999967 at 41 GeV), so the
vacuum `c` is exact for this purpose — but it must be `c`, not `c/n`.

Energy cross-check on live tracks: taking `Reg` at the vertex and subtracting the losses down
to the reference point predicts `fMuonEnergy` with a median ratio of **1.079** (p25 = 1.000),
correlation 0.92 — three independent fields agreeing with no tuning. The residual 8% is the
scale expected from unrecorded soft losses.

**This closes the gap that `fMuonEnergy` leaves:** the birth point is available for every
track, including the 52% of nuatm where the energy field is a placeholder.

## 5. The interaction chain, and what it omits

Showers are recorded **by light yield, not by energy**. From the generator's `readme_sim_gvd`:
*"We stored information about all interactions in which number of P.El 0.001."* The
consequences are measurable:

- the minimum recorded shower energy grows with distance to the nearest OM — 0.10 GeV within
  25 m, 0.57 GeV at 25–50 m, **1.77 GeV** beyond 50 m;
- **no shower is recorded further than 107 m** from any OM;
- so the chain is a thin tube around the strings, not a record of the whole track.

Everything the muon lost outside that tube is missing. Any propagation over hundreds of metres
therefore underestimates the loss, and the error grows with distance.

**The two neutrino productions differ in what they record at the vertex.** At equal `E_ν`:

| | first interaction carries | sits at |
|---|---|---|
| nue2 | **45–74% of `E_ν`** — the hadronic vertex shower | **exactly the vertex** (gap 0.00 m) |
| nuatm | 0.1–2.3% — an ordinary muon loss | 31–70 m downstream |

This holds from 100 GeV to 100 TeV, so it is not a visibility effect — a 10 TeV vertex shower
would be seen from anywhere. **nuatm simply does not write the vertex shower.** Practical
consequence: for nuatm, "energy deposited at the vertex" is not recoverable, and `E_dep` there
systematically lacks the hadronic contribution.

Other properties of the chain: interactions are stored in two blocks rather than propagation
order (backward steps are rare, 2.4%, but large), so the extractor sorts along the track and
keeps the block index as `int_block`. `BMCInteraction` has no type and no time field.

## 6. Extraction, and the traps in it

`fInteractions` is an unsplit `TClonesArray` of a custom class: uproot cannot deserialise it,
so the chain is read on cluster62 with PyROOT's `TTree::Draw`, which returns **flat** arrays.
The event/track/interaction nesting is reconstructed from counts and therefore verified, not
assumed:

| check | what it establishes | result |
|---|---|---|
| `sum(trk_n_inter) == n_inter` | no truncation. Says **nothing** about order | passes |
| collinearity | every interaction lies on *its* track's line | max deviation **1.3 mm** over 28k tracks |
| per-entry re-read | event boundaries, via an independent `Draw` | **300/300** muatm events identical |
| nearest-track assignment | the track-level split inside a bundle | **764/764** correct at 23 m median separation |

**Two converter traps**, either of which would corrupt the truth silently:

1. **Row order.** `cast_to_single` writes all single-cluster events first, then multi-cluster
   ones repeated per cluster, so HDF5 row order ≠ ROOT entry order. Harmless only because
   `ev_ids` carries the ROOT entry number as its **value** and the join is by value.
2. **Start offset.** `root2h5` skips a leading empty event (`st = get_start_index`) and numbers
   from zero, so `ev_id = k` means ROOT entry `k + st`. The extractor reproduces this logic, so
   the index matches by construction.

The join was checked directly: all 16,449 rows of part_3070 against `muons_prty/individ` — x,
y, z, E match **bit-exactly**, angles to 7.6e-6 deg (float32 round-trip).

The target computation exists twice: a readable per-muon reference (`targets.py`) and a
vectorised path (`targets_vec.py`, ×16 faster) that is checked against it — all columns agree
to **1e-14** on 16,449 muons, and end-to-end the two builders give bit-identical indices and
counts over 25 parts.

## 7. Targets

Cluster geometry from the hit envelope: radius **60 m**, half-height **265 m**, matching the
reference macro `energy.C`. Energy at a signed distance `s` from the reference point:

```
E(s) = E_ref − a·s − sign(s) · Σ( e_i between 0 and s ),      a = 0.24 GeV/m
```

Forward the muon loses the interactions it passes plus the continuous term; backward both are
added back. This reproduces `energy.C`, which uses ionisation plus the recorded showers and no
radiative term.

Two targets are kept side by side because they fail in different places:

- **`E_ca`** — energy at the closest approach to the cluster centre. Physical and defined for
  every geometry, but propagates from `fMuonEnergy` and inherits its holes.
- **`E_dep`, `L_path`** over a sensitive cylinder, giving `⟨dE/dx⟩ = E_dep / L_path`. This is
  what the light measures; it uses no `fMuonEnergy` and stays defined for every track.

They are stored **separately, never as their ratio** — the ratio is recoverable, the reverse is
not, and `L_path` is needed to interpret the uncertainty. The volume follows `energy.C`, with
one margin `RLim` applied both radially and in z; five values are scanned,
**RLim ∈ {30, 50, 70, 90, 110} m**, bracketing the author's 70 m.

Sanity: `L_path` grows monotonically with radius (median 188 → 397 m on nue2) while tracks
missing the volume fall from 2.9% to 0.7%.

## 8. Where the propagation breaks

The stochastic part is exact — interactions exist only where the muon lived. The **continuous**
term is charged over the whole path, including stretches where the muon does not exist:

| failure | evidence |
|---|---|
| forward past the death point → negative energy | of tracks dead at the reference point, 0% turn positive when the cluster lies ahead, **75%** when it lies behind |
| backward past the birth point → energy from nowhere | **1.93%** of rows give `E_ca > E_ν`, worst ratio 1.72 |

The error is bounded and computable per row: `0.24 GeV/m × (path outside the muon's confirmed
life)`, stored as `sigma_target_min`. With §4 this bound can now be tightened: the birth point
is known, so "confirmed life" no longer has to fall back on the first interaction.

Magnitudes differ sharply between the targets:

| | `E_ca` | `⟨dE/dx⟩` (RLim = 70) |
|---|---|---|
| rows affected | 33.0% | 85.0% |
| relative error p90 | **1.05** | 0.37 |
| p99 | **28.1** | **0.76** |

`⟨dE/dx⟩` is affected more often but its error is bounded by construction, since the same
ionisation term sits inside `E_dep`. `E_ca` is affected rarely, but its denominator collapses
near the death point and the tail reaches 28×. For regression this favours `⟨dE/dx⟩`: a
heavy-tailed target poisons a fit more than a moderate systematic shift.

**No clipping is applied.** Clipping backward at `E(s) = E_ν` almost never binds and returns a
knowingly inflated value when it does; clipping forward at zero destroys information, since the
negative value encodes how far past death the cluster lies. Raw values are stored together with
`sigma_target_min`, so the decision is made at training time and stays reversible.

## 9. Choice of target: open by construction

`create_energy_h5.py` writes every quantity from which any candidate can be derived, so the
choice is made at training time without recomputing.

What the study settles:

- **`E_ca` cannot be the primary target** — it inherits the `fMuonEnergy` holes (§2) and its
  artefact is unbounded (p99 = 28).
- **`E_dep` and `⟨dE/dx⟩` are equivalent in target quality**; they differ by `L_path`, which
  cancels in the relative error. Neither uses `fMuonEnergy`.
- The choice between them is about **validation on experimental data**. `E_dep` compares
  against total collected charge — available on data with no reconstruction, and with a trivial
  baseline (a plain charge sum) that shows what the network adds. `⟨dE/dx⟩` is
  geometry-normalised and ∝ E above ~1 TeV, but validating it needs a reconstruction that
  supplies a path length.
- **`Reg` is now a third candidate**, defined for 100% of tracks and, for sentinels, equal to
  the vertex energy (§3).

## 10. Open items

- Whether the vertex from §4 should replace `s_first` as the left bound in `sigma_target_min`,
  and how much that tightens the bound.
- Bundle targets: muatm averages 4 muons per event (max 338). Targets are per muon; the
  bundle-level definition is deferred, since the energy task selects single-muon events.
- Sub-threshold losses are absorbed into the constant `a` — negligible over metres, 72 GeV over
  300 m; decisive for a 100 GeV muon, irrelevant for a 100 TeV one.
- The single-track selection must be defined identically for MC and experiment, with purity
  checked against `n_muons`. Note that h8s3-passing muatm is 98.6% multi-muon (median 14).
