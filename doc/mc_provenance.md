# Where the MC comes from: sources, productions and how to match them

The ROOT files this project reads are the last step of a longer chain, and several questions
that look like data questions ("why is the muon energy −0.001?", "is the energy at the point of
birth stored anywhere?") can only be answered by knowing which step dropped what. This
document records the chain, where the upstream files live, and how to match an event across
all of it.

Binary layout of the upstream files is documented separately in
[mc_binary_formats.md](mc_binary_formats.md); the physics of the energy truth is in
[mc_energy_truth.md](mc_energy_truth.md).

## 1. The production chain

```
Fortran generator (RANLUX, data/SECTE.dat, FOTV1.dat, cre10*.dat)
   │   writes  nu_mu_NNNN.dat   + .log .lst .out
   ▼
mcread            BARS/bars/programs/mcread
   │   adds optical noise, adds a time jitter to every pulse, applies the trigger
   │   writes  nu_mu_NNNN.wout  (ZDataReader reads .dat, WOutWriter writes .wout)
   ▼
bexport-mc        BARS/bars/programs/bexport-mc
   │   BMCRead parses the binary, MWriteRootFile writes the trees
   │   writes  NNNN.root        (trees: Events, ArrayConfig)
   ▼
root2h5           data_manager/root2h5/
   │   writes  baikal_mc_merged.h5
```

Two consequences worth remembering:

- **`.dat` is not richer than ROOT.** It holds the same MC truth, minus the noise pulses and
  the time jitter that `mcread` adds. Its only extra content is events that failed the
  trigger — 1,000,000 events in a `.dat` against 39,547 in the corresponding ROOT file (4%).
- **Nothing in the chain drops MC truth.** Every field of `BMCTrack` and `BMCInteraction` is
  present upstream and downstream alike. In particular there is no per-interaction muon energy
  anywhere; see [mc_binary_formats.md](mc_binary_formats.md) §4.

## 2. Where the files live

Upstream files are on EOS at JINR, reachable from `cassiopeia` (`10.220.31.220`) with the
`eos` client (`/eos` itself is only mounted on the `lxui` gateway).

```
/eos/baikalgvd/mc/2020/
├── nu2/wout/v3.0/{10102,20204}/{1..52}/nu_mu_N.{wout,ptl6}
├── nuatm/wout/{v2.0,v3.0,v4.0}/{daemonc,daemonp}/{10102,20204}/…
└── prod1/
    ├── muatm/{cors5,cors7sib}
    ├── nu2.100PeV/{simgvd, wout/v1.1/10102/{a..e}/nu_mu_NNNN.wout, bexmc, bexmc-ntuple}
    ├── nu2.lowEmax/…
    └── nuatm/{simgvd, wout/…}
```

- **`.dat` exists only under `prod1/*/simgvd/`** (500 files of ~169 MB for nuatm), alongside
  `.log`, `.lst` and `.out`. The `nuatm/wout/*` and `nu2/wout/*` branches hold `.wout` only.
- `daemonc` / `daemonp` are the **conventional** and **prompt** atmospheric components.
- `10102` / `20204` are trigger configurations; the number also appears inside the `.ptl6`
  text file (`n_neib=10102`).
- `.lst` is a readable summary (date, array geometry, thresholds) and is the quickest way to
  identify a production without parsing anything.
- `v2.0` / `v3.0` / `v4.0` are **production** versions, not format versions: all of them carry
  `format_version=4` and the same `mc_version=1105302`.

Other seasons exist under `/eos/baikalgvd/mc/{2019,2021,2022,2023,2024,2026}` with the same
layout. The corresponding ROOT files on cluster62 are in
`/home3/ivkhar/Baikal/data/initial_data/MC_2020/{muatm,nuatm,nue2_100pev}/root/{a..e,all}/`.

## 3. Which season a file belongs to

"MC_2020" is the **detector configuration**, not the year the files were produced. The
configuration is unambiguous from the channel count, and the production date is a separate
field:

| production | channels | strings | clusters | `fDate` |
|---|---|---|---|---|
| MC_2019 muatm | 1440 | 40 | 5 | 220617 |
| MC_2019 nue2 | 1440 | 40 | 5 | 210617 |
| MC_2020 nuatm | 2016 | 56 | **7** | 220119 |
| MC_2020 muatm | 2016 | 56 | **7** | 221109 |
| MC_2020 nue2_100pev | 2016 | 56 | **7** | 231018 |
| 2024 nuatm | 4860 | 135 | ~17 | 250514 |

So our 2020 files were produced in 2022–2023, and the 2019 ones in 2021–2022. `baikal_mc_merged.h5`
contains **both** seasons — `*_2019` (5 clusters) and `*_2020` (7 clusters) — and only the 2020
groups are used by the current work. Part of the MC_2019 ROOT set is unreadable
(`MC_2019/nuatm/1000.root` fails with `R__unzip: error -3`, and `MC_2019_v2` contains a file
ROOT does not recognise), which is one reason 2019 is left alone.

## 4. Matching an event across the chain

**`fRunN` / `fEventN` are useless for neutrino productions**: both are −1 in every event of
nuatm and nue2 (they are filled only for muatm, e.g. run 211285). The primary vertex
`fX/fY/fZ` is also identically zero for neutrino generation. There is no intrinsic identifier.

What works is a **content fingerprint** of the primary particle:

```
(theta_primary, phi_primary, E_primary, t_first_muon, E_bundle_reg, event_weight)
```

Measured on `nu_mu_1000.dat` against `nuatm/root/all/1000.root`:

- all **39,547 / 39,547** ROOT events were found in the `.dat`,
- the key is unique for **1,000,000 / 1,000,000** `.dat` events,
- zero collisions.

Three of these values already suffice (`theta, phi, E_primary` was unique on both nuatm and
nue2), but the extra three cost nothing and remove any doubt.

All six are already stored in our HDF5: `prime_prty` is
`[theta, phi, E_primary, A_primary, _, weight]`, so matching can be done from the h5 without
re-reading ROOT. Two cautions: `prime_prty` has one row **per cluster**, so multi-cluster
events are duplicated (39,701 rows against 39,547 events) and the join must go through
`ev_ids`, whose value is the ROOT entry number; and everything is float32, so compare the
float32 representation rather than a converted double.

File names correspond directly: `nu_mu_1000.dat` ↔ `nu_mu_1000.wout` ↔ `1000.root`.

## 5. Field-level notes

Established by comparing the same events across `.dat`, `.wout` and ROOT:

| field | meaning | trap |
|---|---|---|
| `fMuonEnergy` | muon energy at the array plane, GeV | three cases incl. the −0.001 sentinel — see [mc_energy_truth.md](mc_energy_truth.md) §1 |
| `fSumEnergyBundleReg` | bundle energy entering the generation volume (size unestablished), **GeV** | annotated `[TeV]` in the schema — wrong; defined for 100% of muons incl. sentinels, see [mc_energy_truth.md](mc_energy_truth.md) §1.1 |
| `fSumEnergyBundleSurf` | bundle energy at the lake surface, GeV | filled **only for muatm** (median 1737 GeV); identically 0 for nuatm and nue2, where the muon is born under water |
| `fPrimaryParticleEnergy` | primary energy, **GeV** | StreamerInfo says TeV/particle and is wrong: `nu2.100PeV` maxes at 9.943e7, i.e. 1e8 GeV |
| `fX/fY/fZ` (event) | shower axis | identically **0** for neutrino generation; non-zero for muatm |
| `fRunN`, `fEventN` | run / event id | **−1** for neutrino productions |
| `fInteractions` | stochastic energy losses (pair production, bremsstrahlung, photonuclear) | energies in TeV while `fMuonEnergy` is in GeV; recorded **by light yield**, not by energy — see §5.1 |
| `fTracks.fX/fY/fZ` | reference point: closest approach to the array centre | verified perpendicular to 1e-6; median 260 m from the centre — often far from the cluster that saw the event |
| `fTracks.fDelay` | nominally the per-track time offset within a bundle | **identically zero** in every production checked, bundles included |
| the neutrino vertex | — | **not stored**; all 83 leaves of the tree were checked |

### 5.1 `fMagic` — the origin of every pulse

Each pulse carries a third word that BARS calls `fMagic` and `mcread` reads as `n_source`
(`Task_MAIN.cpp`: `int ievent = (int)in[j1-1]; if (ievent == 0) logWarning << "Wrong n_source"`).
It identifies **where the light came from**:

| value | meaning | share of pulses (nuatm) |
|---|---|---|
| `1` | PMT noise (`icode_noise = 1` in `Task_MAIN.cpp`) | 91.2% |
| `-999999` | Cherenkov light of the **muon track** itself | 5.9% |
| `k·10⁶ + 1` | light of the **k-th interaction** (shower) on that track | 2.9% |

The shower index was verified rather than assumed: `k` never exceeded the track's
`fInteractionN` in 3341 events, and the maxima agree (both 21). Per event, muon light is
present in 95.5% and shower light in **55.7%**; only 0.5% of events have shower light without
any muon light. The earliest pulse in an event is noise in 100% of cases — signal always
arrives later.

This is more than bookkeeping: it means the MC labels, hit by hit, which light is track and
which is cascade. Useful for asking how much of an event's light carries energy information,
or for building clean per-hit labels. **Our HDF5 does not currently store it.**

Two cautions when using the index.

**The muon index `j` is unusable once `k` reaches 17.** Every pulse word in the generator's
binary output is a float32 (see [mc_binary_formats.md](mc_binary_formats.md)), and float32
represents integers exactly only up to 2^24 = 16,777,216. `k*10^6 + j` crosses that at k = 17,
after which the value is rounded to an even number and `j` decodes as 0, 2, 12, 64 - rounding
noise. Verified on all three 2020 productions: every code below 2^24 has `j >= 1`, every code
above it is even. The loss happens before ROOT, so no downstream stage can undo it. It matters
most for `nue2`, where PeV muons produce hundreds of cascades and 22.6% of shower hits are
affected; for `nuatm` and `muatm` it is 0.09% and 0.05%.

**Order.** The old extractor (`extract_interactions.py`) sorted interactions along the track,
so `k` did not address its output. `data_manager/energy_truth/read_root.py` keeps ROOT order,
so `k` addresses it directly.

Beware also that reading these branches with `TTree::Draw` requires `SetEstimate` large enough
for the number of pulses — silently truncated buffers produced a bogus "44% of pulses have
`fMagic = 0`" in an earlier pass here.

`fMaxDistance = 25000` was read as centimetres by analogy with the channel table in the same
record, giving a 250 m radius. That is **contradicted** by reconstructing entry points from
`Reg` (notebook §9): 86% of them lie outside 250 m. Either the unit differs or the field means
something other than this volume's radius — see [claims_log.md](claims_log.md).

## 6. Comments in BARS are not a reliable source of units

- `fPrimaryParticleEnergy` is `TeV/particle` in the StreamerInfo of our files and
  `GeV/particle` in the current `BMCEvent.h`. **The header is right and the StreamerInfo is
  wrong**: the `nu2.100PeV` production maxes at 9.943e7, and 100 PeV is 1e8 GeV.
- `fSumEnergyBundleReg` is annotated `[TeV]` and is in GeV.
- `simGVD.yaml` carries `schema_version: 4` while the data carry `format_version: 4` — these
  are unrelated numbering schemes, and the YAML describes a **newer** generator than any
  production on EOS.
- Shower energies are TeV while `fMuonEnergy` is GeV, inside the same record.

When a unit matters, verify it against something external. The unit of
`fSumEnergyBundleReg` was got wrong twice: once from an argument that assumed the answer, once
from the bound `Reg ≤ E_primary`, which proves only that the two agree with each other. What
settled it was a production whose *name* states the energy range (`nu2.100PeV` → 1e8 GeV).
See [mc_energy_truth.md](mc_energy_truth.md) §1.1.
