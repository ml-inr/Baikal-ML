# simGVD production chain: code review and improvement plan

*Review date: 2026-10-21. Scope: the Baikal-GVD track Monte Carlo chain, from simGVD through the second stage (MCREAD) to the BARS import.*
*The cascade MC (`zh_nue_reader/gvd_fortran_mc`) is not covered here and will be reviewed separately.*

The document is written for a human developer or an AI agent who has to act on the findings. Each finding gives:
- a location (`file:line`),
- a verification status,
- the consequence,
- a proposed fix.

Line numbers refer to the snapshot in `/home/ivkhar/Baikal/jinr_repo` (a GitLab dump; each project is a single-commit snapshot).

**Status legend**

| tag | meaning |
|---|---|
| **VERIFIED-CODE** | read and checked in the source by the reviewer |
| **VERIFIED-DATA** | confirmed in MC_2020 output (`nue2_100pev/root/all/1000.root`, season 1) |
| **CONSISTENT** | code and data agree with the claim, but the check cannot distinguish it from a rival explanation; used where the cause is inferred rather than proven |
| **REPORTED** | found by a delegated sub-review, consistent with the code but not independently re-checked |
| **ESTIMATE** | a calculation or judgement, not a measurement |

---

## 0. Summary

simGVD is the **current** production code for track events:
- ν_μ CC (samples `nu2`, `nuatm`) and atmospheric muons (`muatm`, from CORSIKA input);
- seasons 2019–2023, see `Baikal/bars/tools/htcondor/`;
- chain: simGVD → `mcstage2` (Fortran MCREAD) → BARS `bmcread` → ROOT.

The code works and has produced the official samples, but has defects that bias physics or make jobs statistically correlated.

**Top problems, verified in code or data (the causal attribution of C14 is `consistent`; see its row):**
1. **MUM's random generators are never seeded.** Every job replays the same MUM random sequence, whatever seed it is given (§3, C1).
2. **MUM's end-of-track record is discarded by a 2 GeV filter.** A crude fallback model (knock-on electrons only, constant continuous loss) then runs on part of **every** muon's track. Losses between 0.1 and 2 GeV produce **no light** anywhere (C2, P1).
3. **No light beyond 150 m from the muon line**, at any energy (P2).
4. **No scattering delays.** All light arrives at the direct time (P3).
5. **The atmospheric 4w generator never fills `nu_cascade`; the E⁻² 8w variant does** (C3). The default Makefile builds 4w, so it reproduces `nuatm` (no vertex cascade), not `nu2` (which has one).
6. **Output is float32.** Source labels ("magic numbers") lose precision above 2²⁴ (C6).
7. **The second stage clamps every pulse at 2000 p.e.** This is the MC's apparent "saturation" (C7).
8. **The inelasticity truth is an unversioned relative-path table.** The repository table matches `nu2` but not `nuatm`; the nuatm production's table is not preserved and `y` is not stored per event (C14).

**Efficiency (code reading, no profiling possible):** the dominant cost is the light loop over all channels × all losses × up to 20 shower slices. It is computed for every generated event before any trigger check. Spatial pre-selection, extended-shower light tables and early rejection are the main speed-ups (§5).

**CERNLIB** can be removed in about a week:
- HBOOK is diagnostics only;
- the physics-relevant parts are random numbers, Simpson integration and one interpolation routine (§6).

**Recommended order:** fix C1–C7 → remove CERNLIB → write a flat-in-log-E generator with stored generation weights → produce structured output → make the speed-ups → upgrade the physics (small-loss light, scattering, PROPOSAL) (§11). If the immediate goal is the data/MC time residual (the experimental excess) rather than code quality, move the scattering-delay fix (P3) to the top of the physics list; see P3 and §11.

---

## 1. Code locations and how the chain works

| component | location | language | notes |
|---|---|---|---|
| simGVD (detector simulation) | `Baikal/simgvd/` (and fork `zaborov/simgvd/`) | Fortran 77 | ~10.5k lines compiled; README calls it "Legacy", but it is the production code. Imported from SVN `simGVD_x64.2016` (rev 180, 2018), then updated to the "UHE" version (K. Konischev). Last commit 2025-09. |
| MUM 1.3 (muon propagation) | `Baikal/simgvd/third-party/mum/mum1_3.f` | Fortran | 16.4k lines; Sokalski, Bugaev, Klimushin |
| support libraries | `third-party/bailib1/bailib1.f` (RANLUX wrapper `rndm_m`), `third-party/negbi/negbi.f` | Fortran | |
| second stage (MCREAD, production) | `Baikal/mcstage2/` | Fortran + C (zlib reader) | noise, charge smearing, 20 ns summation, masks, trigger → `.wout` |
| second stage (C++ port, **not** production-ready) | `Baikal/bars/programs/mcread/` | C++ | see §9 |
| BARS import | `Baikal/bars/bmcread/`, `Baikal/bars/bextractevent/BExtractMCEvent.cc` | C++/ROOT | `.wout` → `BEvent` ROOT files |
| production catalogue | `Baikal/bars/tools/htcondor/reco-<season>-mc-…-<sample>-<version>…/` | bash | inputs at `/eos/baikalgvd/mc/<season>/<sample>/wout/<version>/<trigger>` |
| older MC (SVN dump, 2017) | `Baikal/gvd-mc/` | Fortran | `simGVD_*` variants, `MC_Read_*`, NT200 simulations |

**Data flow:**

```
steer.dat + stdin dialog ──► simGVD ──► .dat (Fortran records, float32 words, gzip) ──► mcstage2 ──► .wout ──► BARS bmcread + BExtractMCEventSumGump ──► ROOT (BEvent, MCEventMask)
   (seed, mode,                │  generation (ν / single μ / bundles / CORSIKA file)        │ noise, smearing, 20 ns sum,     │ pulse merging (Gumbel shapes)
    trigger, N_triggered)      │  MUM transport + losses, light per OM, Poisson p.e.        │ masks, trigger, 2000 p.e. clamp │
```

**Build** (`Baikal/simgvd/Makefile`):
- Compiler flags: `FC=gfortran`, `FFLAGS=-O2 -mcmodel=medium -fcheck=bounds -std=legacy` (`Makefile:2`).
- The generator file compiled is `munu4w_osc.f` (`Makefile:18`). It is the **atmospheric 4w** variant and never fills `nu_cascade`. The E⁻² variants `munu7w_osc.f` / `munu8w_osc.f` do fill it; the productions used both (variant table in §8).
- CERNLIB is linked through the `cernlib` script (`Makefile:23`).
- The default array size is the 5-cluster setup: `parnpmt.cmn:2` has `nch=1440`. The 2020+ geometry needs `parnpmt_7cl.cmn` (nch=2016) copied over it.

**Run:**
- Steering comes from `steer.dat` (directories, constants, jitter, thresholds).
- The rest is an interactive stdin dialog (`baikal3e3b.f:756-930`): seed (0 means use `time()`), output name, mode, trigger, spectrum, energies, and the number of **triggered** events to produce.
- The example `input` file is for mode 4 (bundles read from a file), not neutrino mode (REPORTED).
- Neutrino mode is `ind_muon = 5`.

**Build environment:**
- cluster81–86.inr.ac.ru: Ubuntu 22.04/24.04, gfortran 11/13, Apptainer, 2× EPYC 7702 (256 threads), 0.5–2 TB RAM.
- None of them has CERNLIB, ROOT or CVMFS.
- All simGVD sources compile with gfortran 13; the only unresolved symbols at link time are CERNLIB's.
- HDF5 with its Fortran wrapper (`h5fc`) is installed on cluster81–85; cluster86 has none.
- No node has an NVIDIA GPU.

---

## 2. Event loop and where the time goes

Per generated event, `simulation` (`sim3h1.f:39-342`) does:

1. **Generation**, depending on mode: `nugen3` (ν), `singlemu`, `grpfast`/`grpmodel` (bundles), or `grpread` (CORSIKA file) (`sim3h1.f:150-190`).
2. **`inout2`** (`sim3h1.f:3119`): flags OMs within `rrmax = 150 m` of each muon line and builds the entry/exit points.
3. **`zerotime1`**, then **`muon_path`** (`muon_path_2.f`): energy-only transport through rock and water to the array (MUM `enewS`/`enew`).
4. **`catastloss`** (`sim3h1.f:1475`):
   - MUM `earray1` produces the losses inside the instrumented segment;
   - simGVD keeps only those with energy transfer > 2 GeV (`:1591`);
   - a fallback model follows (see C2).
5. **`ang_int`** (shower directions; adds the ν vertex shower if `nu_int=1`, `:2143`), then **`rearrange_showers`** (`long_shower.f:15`). Showers above 1 GeV are split into 1-m slices:

   | shower energy | slices |
   |---|---|
   | 10 GeV | ≈4 |
   | 1 TeV | ≈10 |
   | 1 PeV | ≈18 |
   | cap | 20 (`long_shower.f:117`) |

6. **The light loop:** `do k = 1, nch; call channel(k,…)` (`sim3h1.f:240-247`), for **every** channel. `channel` (`:343-793`) does, for each muon in the group:
   - **bare muon:** `res_mu`, kept if mean > 10⁻³ p.e. (`:511`), multiplied by 1.1 (`:515`);
   - **for each loss:**
     - distance test;
     - energy-vs-distance pruning (`e01`, `:882`);
     - then, for each slice, `res_int`, kept if mean > 10⁻³ p.e. (`:589`);
   - **each kept source:** Poisson photo-electron count `fel` (`:911`), stored as a "pulse": time, p.e., source label;
   - **cap:** at most 1000 pulses per channel (`:705`).
7. **`response1`** (trigger, `baikal3e3b.f:2783`): a hit is ≥1 p.e.; the conditions are neighbouring channels on a string, then strings, then clusters. Then **`statgen`** and **`evout`** (output, `newout4_1.f`).

**Cost structure (ESTIMATE):**
- Step 6 dominates. It scales with N_channels × N_losses for the distance tests, plus N_passing × N_slices × (cost of `res_int`). `res_int` costs a few hundred ns: geometry with `acos`, two `log10`, one `10**x`, and a spline.
- Step 6 is executed **before** knowing whether the event can trigger.
- MUM itself is probably ≲10 ms per muon even at 100 PeV (REPORTED), plus about 1 min of start-up per job (MUM rebuilds its tables; REPORTED).

---

## 3. Correctness findings

| ID | severity | status | location | finding | consequence | fix |
|---|---|---|---|---|---|---|
| **C1** | High | VERIFIED-CODE | `baikal3e3b.f:131-147` (only `RLUXGO`); MUM `RNDM`/`RM48` at `mum1_3.f:3308, 3352, 3531, 3545, 3680, 3731, 4012, 4050, …` (27 call sites) | Only RANLUX (`rndm_m`) is seeded from the user seed. MUM's `RNDM`/`RM48` are never seeded: there is no `RDMIN`/`RM48IN` anywhere. `POISSN`/`NORRAN` (photo-electron counts, `sim3h1.f:911-920`) are initialized by `poiset(50.)` at `baikal3e3b.f:126` — a fixed seed, not the user seed (REPORTED for their internals). | All jobs replay the same MUM loss sequence and photo-electron fluctuation sequence. Jobs are statistically correlated, so the effective sample size is unknown. | Route every random call through one seeded generator (`rndm_m`), or seed RNDM/RM48/NORRAN from the user seed. Then add a deterministic per-file seed scheme (§10). |
| **C2** | High | VERIFIED-CODE | filter `sim3h1.f:1591` before the type checks `:1644-1656`; MUM end records carry `etr = 0` (`mum1_3.f:3324, 3342, 3387`); fallback `sim3h1.f:1664-1824` | The `etr ≤ 2 → skip` filter also skips MUM's "reached end" (type 1) and "stopped" (type 2) records. `ee` therefore stays 0 and `istop` 0, so the fallback branch runs **for every muon**. The fallback has a constant continuous loss `emlos ≈ 2.18 MeV/cm` (`:1005`) and knock-on electrons only (`v_delta`). It runs from the last kept MUM loss above 2 GeV, or **from the entry point if MUM produced none** (typical below ~1 TeV), to the exit. MUM's stopping point is ignored. If the muon fell below 11 GeV, losses after that point are kept **and** the stretch is re-simulated (`:1689-1708`, `nnit` not reset). | Energy-loss pattern and stopping point partly wrong. Knock-on showers appear only on the track tail. Double counting is possible near 11 GeV. | Process type-1/2 records before the energy filter. Run the fallback only when the muon really drops below the MUM threshold. Decide explicitly how 0.1–2 GeV losses are treated (P1). |
| **C3** | High | VERIFIED-CODE | `munu4w_osc.f:331` (`e_shower` computed, never used); `nu_cascade.cmn` filled only in `munu7w_osc.f:495-502` and `munu8w_osc.f:436-443`; consumed at `sim3h1.f:2143`; `nu_int` zeroed at `baikal3e3b.f:1172`; `Makefile:18` compiles `munu4w` | The **default 4w** generator never passes the hadronic cascade to the light simulation; the 7w/8w variants do. | The default 4w lineage reproduces the official `nuatm` samples (no vertex cascade; 2022 Bartol, 2025 daemonflux). The `nu2` sample was produced with the 8w variant and **does** have one. "MC_2020 contains vertex cascades" is therefore true for `nue2` only; whether the atmospheric 4w should also carry them is a physics decision, not a pure bug fix. | If the atmospheric sample should have the cascade, port the `nu_cascade` filling from `munu7w/8w` (≈6 lines) and expect the `nuatm` sample to change; confirm with the MC group first. Add a regression check: fraction of events with a vertex shower > 0, separately for the 4w and 8w samples. |
| **C4** | Medium (current code) / Low for the 2019–2023 productions | VERIFIED-CODE | `munu4w_osc.f:144-145` | `call bspl1(cr_nu,21,spl_crnu)`: in the **current** `Baikal/simgvd/munu4w_osc.f` `spl_crnu`/`spl_cranu` are undeclared (implicit scalars), and `BSPL1` writes 23 values into each. The 2017 `gvd-mc/.../simGVD_x64/munu4w_osc.f` declares `common /finte_crosses/ ... spl_crnu(23), spl_cranu(23)` and is safe; the declaration was lost in the current file. | Memory overwrite at initialisation in the current build. The results are unused (the only uses are commented out), so it is harmless only by luck. | Delete the two calls, or restore the `(23)` declaration. |
| **C5** | Medium | VERIFIED-CODE | MUM branch `sim3h1.f:1595` (no check against `intrct=2000`, `mod_pargroup.cmn:2`); fallback cap `:1786-1790` jumps to `900` (`:1943`), skipping `nintr(ig)=nnit` (`:1826`); `nintr` zeroed at `:1527` | More than 2000 kept losses per muon overflows the arrays in the MUM branch: the run aborts with `-fcheck=bounds`, or memory is overwritten without it. In the fallback branch the muon silently gets **zero** showers. | The limit is not reached in the checked MC_2020 subset up to ~10⁷ GeV (VERIFIED-DATA: stored showers extend to the end of the light region). It is near the limit for 100 PeV over 2 km (≈1700 kept losses, ESTIMATE); the sample reaches ~10⁸ GeV, so check the 100 PeV tail explicitly. | Make the arrays dynamic or larger, add the check in the MUM branch, and set `nintr` on every exit path. |
| **C6** | Medium | VERIFIED-CODE + VERIFIED-DATA | `newout4_1.f:48` (`dimension out(noutmx)`, default REAL), `:350-366` (source label `new_mag_num` stored into `out`), `:376` (write) | The whole event record is float32, including the per-pulse source labels `ishow·10⁶ + imu`. | From `ishow = 17` the muon index is lost (exact only below 2²⁴). MC_2020 shows all 50,115 labels ≥ 1.7·10⁷ as exact multiples of 10⁶. Multi-muon events cannot be attributed. | Write integer words for labels and counts (two records, or encode as int32), or move to structured output (§10). |
| **C7** | Medium | VERIFIED-CODE | `mcstage2/steer_f_gvd_2a_22_8sdtw_2.f:797-799`; C++ `bars/programs/mcread/Task_MAIN.cpp:77-79` | Each pulse's charge is clamped at 2000 p.e. | Combined with the 20 ns running sum, this produces the soft "saturation" seen in MC_2020: ⟨Q⟩/μ ≈ 1 up to ~3·10³ p.e., 0.76 at ~7·10³, 0.41 at ~2·10⁴. It is an artefact, not a model of OM saturation. | Keep the true charge, and apply a measured saturation model in reconstruction or in an explicit stage. |
| **C8** | Medium | VERIFIED-CODE | `mcstage2/steer_f_gvd_2a_22_8sdtw_2.f:113` (`int_int = time()`); simGVD `baikal3e3b.f:140` when the seed is 0 | Second-stage noise and smearing are seeded from the clock. | Not reproducible. Jobs started in the same second get identical noise. | Deterministic seeds (§10). |
| **C9** | Low | VERIFIED-CODE | `sim3h1.f:705-711` | More than 1000 pulses in one channel: further sources are dropped silently (the print is commented out). The drop order is loop order, not brightness. | Light loss at the brightest OMs of PeV events. | Merge or sum sources, or grow the arrays; count drops. |
| **C10** | Low | VERIFIED-CODE | `mum1_3.f:3288-3292` (`earray1` returns before `numb_e` is reset at `:3299`) | Path > 2 km in the detector, or E outside 10.1 GeV–1 EeV: MUM returns without resetting the history, and `catastloss` reuses the **previous muon's** losses. | Safe for the current geometries (segment ≤ ~1.5 km). Wrong for large arrays (e.g. the 27-cluster `array_H300R60Z15_*_27cl.dat`). | Reset the history first, return an error flag, check it. |
| **C11** | Low | VERIFIED-CODE | `sim3h1.f:916-920` (`fel`) | For mean > 70 p.e. the count is `ifix(ru*sqrt(anu)+anu)`: Gaussian, truncated toward zero. | Bias ≈ −0.5 p.e. (< 1 % at these means). | Use `nint` or an exact Poisson sampler. |
| **C12** | Low | REPORTED | `munu4w_osc.f` + `newout4_1.f:268-270`; `snu_tot_2.f:78-80` | `id_spec=0` (fixed energy) still samples the spectrum but writes `emuona` as E_ν. `emuona < 10 GeV` makes `sigma_nu` return −1 and corrupts the rejection. | Wrong truth in those modes. | Fix the mode logic; validate input ranges. |
| **C13** | Low | VERIFIED-CODE | `parnpmt.cmn:2` (`nch=1440`) | The default build is the 5-cluster (2019) configuration. | A 2020+ geometry with the default file overflows the arrays: an abort with bounds checking, memory overwrite without it. | Select the geometry by configuration, or check the array size against the geometry file at start. |
| **C14** | High (energy truth) / Medium (generator review) | VERIFIED-CODE + VERIFIED-DATA | 4w: `munu4w_osc.f:320-322` (old) / `:325-329` (current); 8w: `munu8w_osc.f:303-311` (current) / `:371-374` (old); `su_dif_2.f:73-108`; table read in `init_munu` | Inelasticity is drawn from `finte2_y`/`cc_over_y.ascii` (4w; 8w below 10⁵ GeV) or `emu_sim` (8w above; old 8w everywhere). `cc_over_y.ascii` is a **relative-path input**, not stored or versioned. The repository table gives `<y>_ν = 0.479` at 30 GeV, matching `nue2` (0.502) but not `nuatm` (0.301); `nuatm` ν̄ is 0.231 vs the table's 0.351. Both the 2022 (Bartol/4w) and 2025 (daemonflux/4w) nuatm samples show ~0.30. | The inelasticity truth is not reproducible from the output; the nuatm/nue2 difference is invisible in the sample metadata. This is a direct systematic for the energy-truth companion and for any y-based analysis. | Store `y`, `E_μ` at the vertex and `E_had` per event, plus a hash/version of `cc_over_y.ascii`; ask the MC group which table each production used. Until then, do not compare y between samples without checking the table. |
| **C15** | Medium | VERIFIED-CODE | `munu8w_osc.f:371-374` (old 8w); `su_dif_2.f:82-120` (`nu_CC_tab`), `:148-200` (`rnu_CC_y`) | The old 8w calls `emu_sim` for **all** energies. For `E < 10 GeV` `nu_CC_tab` returns without filling `sig` (zero or the previous call's state). For `10 GeV ≤ E < 10⁴` the low-energy branch of `rnu_CC_y` returns the total cross-section `rnu_tab`, independent of y, so `sig` is proportional to the y-bin width and `emu_sim` samples an approximately **uniform y** (mean ~0.5), not the physical y shape. The current 8w uses `finte2_y` below 10⁵ and is safe. | If a production used the old 8w at 20–50 GeV, its inelasticity is wrong (near-uniform), not merely stale. The 2023 `nu2` sample's y (0.50) is equally compatible with the old uniform branch (~0.5) and the current `finte2_y` (0.48), so the sample alone does not identify the 8w snapshot. | Use `finte2_y` below 10⁵; guard `emu_sim` against `E < 10⁴`; abort or warn on an unfilled table. |

---

## 4. Physics-model limitations (deliberate simplifications that matter now)

| ID | status | location | limitation | effect | upgrade |
|---|---|---|---|---|---|
| **P1** | VERIFIED-CODE + VERIFIED-DATA; size ESTIMATE | `sim3h1.f:1591` (keep only `etr > 2 GeV`); `:515` (bare-muon light × 1.1 "for 100 MeV threshold for delta electron"); MUM init `baikal3e3b.f:1193-1200` (`em = emi = 0.1 GeV`, `vm = 1e-3`) | MUM samples discrete losses down to 0.1 GeV. simGVD discards those ≤ 2 GeV, so **losses of 0.1–2 GeV emit no light** (except knock-on electrons on the tail, C2). Data: for muons > 100 GeV only 0.19 % of stored showers are < 2 GeV, with a sharp edge at 2 GeV, at all muon energies up to 100 PeV. | The sub-review estimates 0.3–1.4 MeV/cm of energy in such losses between 100 GeV and 100 TeV (ESTIMATE). That is about **15–70 % of the bare-muon Cherenkov light**, so track light is probably underestimated at ≥ TeV. *Unverified caveat:* the bare-muon spline (`sp_nake`) might already include part of it. If the estimate is confirmed, this is a charge-scale systematic for the current analysis; test it with the filter lowered before the next large production. | Add the mean light of < 2 GeV losses as a continuous, energy-dependent track term (cheap), or keep them as point sources (expensive). Validate with a test run with the filter lowered. |
| **P2** | VERIFIED-CODE + VERIFIED-DATA | `sim3h1.f:3134` (`rrmax = 15000` cm, hard-coded; the commented alternative is the spline range) | OMs farther than 150 m from the muon line are never lit, at any energy. The README says "after 50 TeV 250 m", but the code doesn't do that. Data: maximum perpendicular distance of lit OMs = 149.84 m; for showers > 1 PeV, 2.3 % of contributions already sit at 140–150 m. | Small in practice (VERIFIED-DATA + extrapolation, 3 files). For 0.1–1 PeV showers the lit-OM count per 10 m shell falls with an e-folding length of ~16 m beyond 100 m; extrapolating gives ~0.1 extra lit OMs per shower (0.5 %) and negligible charge. For > 1 PeV: e-folding ~28 m, ~2.7 extra lit OMs per shower (~7 %), mostly single photo-electrons, still negligible charge. The effect grows by ~20–30 m of reach per energy decade, so it matters for ≥ 10 PeV events and for multi-cluster light from a bright shower near a neighbouring cluster. | Make the cut depend on the shower energy (use the `e01` energy-vs-distance table), up to the spline range (300 m). |
| **P3** | VERIFIED-CODE | `res_mu` (`sim3h1.f:1341`, direct-time formula with group velocity); `res_int` (`:1173`, `time = (tm + rn_wat*rast)/cvacu + tmu_delay`) | All photons arrive at the direct (unscattered) time. Scattering only reduces intensity, through the attenuation spline `f_sp` (`ptm6.spl.300`, 0–300 m, 5 m steps). | No late-light tails. Time PDFs and reconstruction trained on this MC are too optimistic compared with water. The project's data/MC study attributes the experimental excess to a ~29 ns deficit of the time spread; if that holds, P3 is the dominant physics issue for the current analysis, not a late-stage upgrade. | Photon tables with time delays (e.g. from `generators/LiTra` or NTsim), or parametrised delay distributions per distance. If the goal is the exp excess, do this before the later items. |
| **P4** | VERIFIED-CODE | `ham10HQlab` (OM acceptance polynomial, 2010); shower angular profile `andis` from 5 energies × 19 angles (`a001`, `:829`); `yearwr=1992` water parameters (`baikal3e3b.f:150`); steer jitter σ = 0.5 ns Gaussian + τ = 3 ns exponential | Old, coarse parametrisations. | Shower angular structure is energy-coarse. Water and acceptance are not tied to current calibrations. | Update from current calibration; finer shower tables. |
| **P5** | VERIFIED-CODE | `sim3h1.f:511, 589` | A source is kept at an OM only if its **mean** light > 10⁻³ p.e. | Muon light reaches ≈ 67 m in MC_2020 (VERIFIED-DATA), showers further. The loss is negligible per source, but the threshold explains the "67 m cut". | Keep; document. |
| **P6** | VERIFIED-CODE | `munu4w_osc.f:493` (`dm2 = 0`), no Earth propagation (REPORTED: up-going events use the surface flux) | No oscillations, no Earth absorption. | Matters for atmospheric ν below ~100 GeV (oscillations) and for up-going ν above 10–50 TeV (absorption). | Apply as event weights if the truth (E_ν, cos θ, ν/ν̄) is stored (§8). |
| **P7** | REPORTED | MUM | No multiple scattering (straight tracks); photonuclear cross section is the oldest component (Bezrukov-Bugaev + ZEUS σγp). | About 1° over a few hundred metres for 10–100 GeV muons; a few % in losses at ≥ PeV. | PROPOSAL (§7). |
| **P8** | VERIFIED-CODE | `newout4_1.f` (~`:150-250`) | Only interactions that contributed to a hit channel are written ("storage selection"). The hit threshold is **0.01 p.e.** (`af(i) > 1.e-2`; `sim3h1.f:255`, `baikal3e3b.f:2823`), not 1 p.e. | The true loss list is lost. Biased per-OM yields below ~100 GeV if stored showers are histogrammed; the bias is smaller than a 1 p.e. threshold would imply. | Write the full loss list (§10). |

---

## 5. Efficiency (major items only)

No profiling was possible: CERNLIB is not installed on the available hosts. **First step: build without CERNLIB (§6) and profile with `perf`/`gprof` on fixed seeds, one run per energy decade.**

| ID | status | where | issue | proposal | expected gain (ESTIMATE) |
|---|---|---|---|---|---|
| **E1** | VERIFIED-CODE | `sim3h1.f:240-247`, `:460-600` | Light computed for all `nch` channels × all losses (distance test each), for every generated event. | **Spatial pre-selection:** sort losses by position s along the track. For each OM compute its closest-approach s₀ and the range R(E). Visit only losses with \|s − s₀\| < √(R² − ρ²) (binary search). Add a cluster-level bounding test first. | large at high energy: O(N_OM·N_loss) → O(N_OM·(log N_loss + k)) |
| **E2** | VERIFIED-CODE | `long_shower.f`, `sim3h1.f:579-600` | Each shower > 1 GeV is evaluated as up to 20 one-metre slices (one `res_int` each). | Precompute an **extended-shower light table** (energy, distance, angle) that folds in the longitudinal profile. That is one lookup per shower; keep slices only for d ≲ 10 m. | ×5–15 fewer `res_int` calls at TeV–PeV |
| **E3** | VERIFIED-CODE + REPORTED | `sim3h1.f:240-280`; generator volume `munu4w_osc.f:76-79, 383-423` | The detector response is computed before the trigger check. The generation volume is fixed transversely (~1.3 × 1.3 km) and 900 m downstream. The sub-review estimates ≥ 95 % of 10–100 GeV events never trigger. | **Early rejection:** cheap upper bound on photo-electrons per string/cluster from the bare muon and the brightest losses, before the full loop. **Energy-dependent generation volume;** reject vertices downstream of the array geometrically. | ×2–10 at low energy |
| **E4** | REPORTED | `mum1_3.f:984-1031` | MUM rebuilds all cross-section tables for 2 media at every start (~1 min). | Cache the tables in a file keyed by the settings. | minutes per job |
| **E5** | VERIFIED-CODE | `Makefile:2` | Production build uses `-fcheck=bounds` (typically 10–50 % slower). The fork uses `-O1`. | `-O2/-O3 -march=native` for production; keep bounds checking in CI/test builds. | 10–50 % |
| **E6** | VERIFIED-CODE | `baikal3e3b.f:1335` (`statgen`: 4 double + 8 int/real arrays of nch×nch) | About 260 MB of static statistics per process. This and `tm1_c(nch,1000)`-type arrays force `-mcmodel=medium`. | Allocate dynamically; drop the unused statistics. | memory, start-up |
| **E7** | REPORTED | `bars/bextractevent/BExtractMCEvent.cc:510-518` | The BARS import builds a new ROOT `TF1` formula for every merged pulse pair (Gumbel merging). | Analytic merging or a precompiled function. | possibly the largest per-event cost of the import |
| **E8** | VERIFIED-CODE | `newout4_1.f:350-366` | Linear search over the label table for every pulse. | Hash/array map. | small |

Parallelism: simGVD is single-threaded but events are independent, so production is embarrassingly parallel once C1/C8 are fixed (one job per core, distinct seeds). The nodes have no batch system; use job arrays via GNU parallel or a simple scheduler script.

---

## 6. CERNLIB: usage map and replacement plan

All call sites were found (REPORTED, cross-checked with a link test: VERIFIED-CODE). "Live" means reachable from the main program.

| routine(s) | live/total calls | where | role | replacement |
|---|---|---|---|---|
| RANLUX, RLUXGO/AT/UT | 4/4 | `bailib1.f:220`; `baikal3e3b.f:147, 2081, 2086` | main generator (physics) | James' standalone `ranlux.f` (same algorithm, bit-identical); later RANLUX++/xoshiro256** |
| RNDM | 18/20 | MUM (`mum1_3.f:3308, 3352, 3731, 6296-6301, 8197-8219, …`) | physics, per muon, **unseeded** (C1) | route to `rndm_m` |
| RM48 | 5/7 | MUM (`mum1_3.f:981, 3680, 11078, 13392, 16332`) | physics, per muon, **unseeded** | double precision from the main generator |
| POISSN/POISET, NORRAN | 1+1, 1/4 | `sim3h1.f:916-919`; `baikal3e3b.f:126` | photo-electron counts | own exact Poisson sampler on the main generator |
| SIMPS | 8/53 | MUM `enew`/`enewS`/`earray1` below 10 GeV; init | physics (per muon, low energy) | 15-line Simpson |
| DSIMPS | 62/68 | MUM table building (init) | physics (init) | same |
| FINT | 2/12 | `munu4w_osc.f:805, 807` (`daFLNM`, daemonflux table) | physics, per event (mode 5) | own interpolation; **check CERNLIB's out-of-range behaviour first** |
| RADAPT, RZERO, GAMDIS, POLINT | 3, 1, 2, 1 | `munu4w_osc.f:172-183`; `sim3h1.f:3358-3397, 873` | init only | QUADPACK DQAGS; Brent; incomplete gamma; 3-point Lagrange |
| GAUSS | 1 | `sim3h1.f:844` | print only | delete |
| VZERO/UZERO, DATIME/TIMEX/TIMEST | 24, 4 | various | utilities, metadata | Fortran intrinsics (`DATE_AND_TIME`, `CPU_TIME`) |
| HBOOK (live: HLIMIT, HIDOPT, HBOOK1/2, HF1/2, HROUT/HREND …) | ~120 | `baikal3e3b.f` (`ihbook`, `ihtotal`, `genhfill`, `sumary`) | **diagnostic only**; nothing reads them back | stubs, or a tiny array-histogram written as text/CSV |
| HBOOK (dead: HBFUN1 ×46, HFILL ×54, …) | — | MUM `test*` routines, `user2a_t.f` (calls commented out) | dead code | delete |

**Plan:**
1. **Phase 1, bit-identical (~2–3 days).**
   - Delete the dead code; stub the HBOOK calls.
   - Vendor or reimplement the numerical routines. Alternatively, copy the minimal CERNLIB (GPL) sources into `third-party/cernlib_min`.
   - **Validation:** old and new builds with the same seed must give byte-identical `.dat` files (mask the date word) and `.lst` listing.
2. **Phase 2, modernise (~2–3 days + validation).**
   - One seeded generator for everything (fixes C1).
   - Exact Poisson sampler; QUADPACK.
   - **Validation:** statistical, ≥ 10⁵ events × several seeds. KS/χ² on hit counts, charge per OM, hit times, muon energy at the array, trigger rate and weight sums. Printed integrals (`angul_integ`, `a001`) agree to ~10⁻⁶.

After Phase 1, simGVD builds on cluster81–86 with gfortran alone. `mcstage2` depends on CERNLIB separately (`flib.Linux`, HBOOK ntuples) and needs the same treatment.

---

## 7. MUM vs PROPOSAL

**MUM settings** (VERIFIED-CODE): `init_mu`/`init_muS` (`baikal3e3b.f:1193-1200`) set:
- pure water (`imed=1`) and "Baikal basis rock" (`imed=8`);
- ZEUS σγp (`ipn=2`) and Kokoulin-Kelner-Petrukhin bremsstrahlung (`ibre=2`);
- `em = emi = 0.1 GeV`, `vm = 1e-3`.

MUM only accepts 0.01 ≤ em ≤ 0.51 GeV and blocks on `read*` otherwise (REPORTED, `mum1_3.f:1744-1748`).

The data confirm the absolute cut: the smallest stored showers have a flat ~2 GeV floor at all muon energies, i.e. simGVD's own filter, so MUM produces losses below 2 GeV.

**Estimated losses per km of water** (ESTIMATE, sub-review):

| E_μ | sampled by MUM (> 0.1 GeV) | kept by simGVD (> 2 GeV) |
|---|---|---|
| 100 GeV | ~105 | ~5 |
| 1 TeV | ~170 | ~20 |
| 100 TeV | ~450 | ~190 |
| 100 PeV | ~1270 | ~830 |

**Physics** (REPORTED):
- KKP bremsstrahlung and pair production: the same family PROPOSAL and Geant4 use, so not superseded.
- Photonuclear: Bezrukov-Bugaev with ZEUS σγp, the oldest component.
- Ionisation with e-diagram bremsstrahlung.
- No LPM (negligible below 100 PeV), no multiple scattering.

**PROPOSAL** (maintained C++ lepton propagator with Python bindings, used by IceCube and KM3NeT): replacing MUM is feasible and beneficial.
- **Gains:** maintenance, multiple scattering, cached tables, no 2 km / 1 EeV limits, and an energy cut that can be set to the 2 GeV actually used, with continuous randomisation.
- **Interface needed:**
  - a C++ shim with `extern "C"` functions `init(seed, ecut, vcut, table_dir)` and `propagate(E, dist, medium)`;
  - the shim returns arrays (type, distance, E before/after, E lost) to fill the `/ehistory/`-style arrays;
  - custom Baikal water and rock media;
  - seed from the simGVD seed;
  - regenerate the MUM-derived table `e_10_500m_cearth` (`emu_minimal.f`).
- **Effort:** a few weeks including validation against MUM.

---

## 8. Neutrino generator, and a flat-in-log-E sample with generation weights

**Current scheme** (`munu4w_osc.f`, `nugen3`; VERIFIED-CODE unless noted):
- **Energy:** proposal E^-1.6 (`:247`) on [`emuona`, 10⁸ GeV] (`:148, :290-292`). Rejection against Σσ_CC·Φ normalised at `emuona` (`:285-317`). This is the **current** 4w; the 2017 4w used Bartol and `esp_maximum = 10⁵` GeV, and the 8w uses a power-2 spectrum with `10⁸` GeV.
- **Flux:** Φ = conventional **daemonflux** ν_μ/ν̄_μ (`daFLNM`, REPORTED `:647-816`) in the current 4w. This matches the "daemonc/daemonp" production names. The 2022 `nuatm` v1.0 used **Bartol**; the `nu2` productions used **power 2**. Prompt (`dapFLNM`), Volkova and Butkevich fluxes exist but are unused.
- **Weights:** events are unweighted, `weights(1)=1` (`:318`). `weights(2)` is an energy-dependent vertex-length ratio (REPORTED).
- **Choices:** ν vs ν̄ in proportion to σΦ in the code, but the samples differ: `nuatm` v1.0 is 70/30 ν/ν̄, `nu2` v1.0 is **100 % ν**, `nu2` v3.0 is ~60/40. Inelasticity comes from `finte2_y`/`cc_over_y.ascii` in 4w and in 8w below 10⁵ GeV, and from `emu_sim` above (old 8w: `emu_sim` always). The table is a **relative-path input**; see C14/C15.
- **Coverage:** ν_μ CC only. No NC, ν_e or ν_τ in simGVD; those are in the separate cascade MC.
- **Stored per event:** E_ν (GeV), zenith/azimuth, ν/ν̄ (`anucl = ±1`), weight product. The vertex is in the header **only in v3.0/v4.0**; v1.0 writes zeros there. N_generated and the generation constants appear only in the `.lst` listing (REPORTED). `y`, `E_μ` at the vertex and `E_had` are not stored.

| variant | flux | `E_max` | y sampler | `nu_cascade` | production |
|---|---|---|---|---|---|
| 4w old (2017 lineage) | Bartol | 10⁵ GeV | `finte2_y` | no | `nuatm` v1.0 (2022) |
| 4w current | daemonflux | 10⁸ GeV | `finte2_y` ≤10⁵, `emu_sim` above | no | `nuatm` v4.0 (2025) |
| 7w | power 2 | 10⁸ GeV | `finte2_y` | **yes** | not identified |
| 8w old (2017) | power 2 | 10⁸ GeV | `emu_sim` everywhere (<10 GeV undefined; 10–10⁴ ≈ uniform y) | **yes** | possibly none |
| 8w current | power 2 | 10⁸ GeV | `finte2_y` ≤10⁵, `emu_sim` above | **yes** | `nu2` v1.0?/v3.0 (2023/2025) |

The exact 2023 `nu2` 8w snapshot is not preserved; its y matches `finte2_y`, so it probably used the current 8w.

**Can `nu2` and `nuatm` be one sample?** In principle yes, but they currently differ in **two** things: the flux and the generator variant (4w vs 8w adds the vertex cascade and changes the y path). One sample generated **flat in log₁₀ E_ν** (10 GeV–100 PeV), isotropic, with a stored generation weight (IceCube "OneWeight"-like), can be reweighted to any ν_μ flux if the cascade and y are handled uniformly:
- conventional/prompt atmospheric (with oscillation weights);
- astrophysical power laws.

**What must be stored per event:**
- true E_ν, cos θ, φ, ν/ν̄, vertex;
- E_μ at the vertex, y, E_had;
- interaction length L_i and interaction probability P_int = σ·N_A·ρ·L_i;
- the generation pdf.

**In the header:** A_gen, Ω_gen, E range, spectral index, N_generated.

**Code change** (ESTIMATE ~60–120 lines in `nugen3`, `init_munu`, `evout`):
- sample log-uniform E (γ = 1 needs a special case: `pgamma = 0`);
- drop the flux rejection; ν/ν̄ with p = ½;
- compute OneWeight;
- fill `nu_cascade.cmn` (fixes C3; code exists in `munu7w_osc.f:495-502` and `munu8w_osc.f:436-443`); store `y`/`E_had` and the `cc_over_y.ascii` hash (C14).

Keep OneWeight outside `weights()`, because `statgen` multiplies all `weights()`. Earth absorption (P6) must be added as a weight (σ_tot is available).

**Atmospheric muons (`muatm`) cannot join a neutrino sample.** They are cosmic-ray bundles from CORSIKA databanks (mode 4): down-going, multi-muon.
- For **detector-response** samples (likelihood tables, reconstruction training), a single-muon sample flat in log E over all directions (mode 1) covers the light physics of both.
- For **background estimates**, keep CORSIKA, generated with a hard spectrum and reweighted (`zaborov/corsika-spectrum-adjuster`).

---

## 9. Second stage (MCREAD) and BARS import

**Production uses the Fortran `mcstage2`** (driven by `bars/programs/helpmcstage2`). The C++ port in BARS is a line-by-line translation (REPORTED) with some differences:
- calibration inputs from DQM;
- a time shift to the first trigger;
- channel re-ordering.

**It is not production-ready:**
- **Broken record framing** (VERIFIED-CODE): `WOutWriter.cpp:20-27` writes the 7128 marker and the length, but never the closing Fortran record marker.
- **No seeding** (VERIFIED-CODE): `gRandom` is never seeded (`main.cc:84` is commented out), so every job uses the same default seed.
- **Channel IDs shifted** (REPORTED, consistent with `init.cc:38-43`): `toggleChannelOrder` (0-based) is applied in `DataCreator.cpp:50` to IDs reported to be 1-based. Channel 36 then lands on another string.
- **Unchecked calibration file** (REPORTED): if the qcalib file is missing or short, the arrays stay uninitialised (`NoiseRatesChannelMaskManager.cpp:64-71`).

**What both versions do** (REPORTED):
- read simGVD's per-source photo-electron counts;
- Gaussian charge smearing N(nc, 0.33√nc);
- 2 ns jitter;
- replace each pulse's amplitude by the **running sum over the previous 20 ns** (documented in `bars-docs/docs/import/mc.md` and `simgvd/readme_sim_gvd`);
- noise: Poisson in 5 µs per channel rate, charge from a table;
- per-channel mask and noise rates (from `array.dat` in Fortran, documented in `mcstage2/README`);
- neighbour trigger;
- **2000 p.e. clamp** (C7).

**Lost along the way:**
- true photo-electron counts and pre-jitter times;
- hits on masked channels;
- events failing the trigger;
- the **number of generated/input events**;
- channels outside the triggered cluster.

**BARS import:**
- `BOriginMC::SetFlag` (`bextractevent/BOrigin.cc:42-53`) stores every merged sub-pulse's label. `GetFlag()` returns the **first** (`bars-docs` wrongly says "last").
- The Gumbel merging builds a `TF1` per merged pair (E7).
- The YAML data model `bars/simgvd/simGVD.yaml` is not used by any reader or writer, and contradicts the real format (REPORTED).
- The waveform converter path (`bmcreadwaveform/BMCConvertersimGVD`) skips all second-stage physics and has its own bugs (REPORTED).

---

## 10. Target production chain (proposal)

1. **One structured simGVD output** (HDF5 or ROOT; fix the BARS YAML model first). Contents:
   - generation constants and N_generated;
   - per event: truth (E_ν, cos θ, φ, ν/ν̄, vertex, y, E_had), weights including OneWeight;
   - the **full** loss list (type, energy, position), including losses that produced no light;
   - per channel and source: true p.e. and pre-jitter time;
   - integer source IDs;
   - seeds and code version.
2. **Detector response as a BARS stage** reading this output directly: noise, smearing, masks per season, trigger, 20 ns logic. Season masks and noise can then be re-applied **without re-simulating**.
3. **Deterministic seeds** derived from (production, file, stage), e.g. a hash; no `time()` anywhere. This follows the existing pattern in `reco-muon` (`preproc_mc.cc:41-46, 95`; REPORTED).
4. **No interactive dialog:** a single configuration file per job, stored in the output header.
5. **Job orchestration** for cluster81–86: one job per core, file-level seeds, per-job logs, and a manifest of N_generated / N_triggered.
6. **Regression tests:** fixed-seed byte comparison for refactors; statistical comparison for physics changes.

---

## 11. Work plan

| step | content | effort (ESTIMATE) | validation |
|---|---|---|---|
| 1 | Correctness fixes C1–C8, C13 | 1–2 weeks | fixed-seed comparisons; vertex-shower fraction per variant (4w vs 8w); seeds give different MUM sequences |
| 1b | Store `y`/`E_had` and the `cc_over_y.ascii` hash; check the production table (C14/C15) | 2–3 days | `y` from the generator vs `1 − Reg/E`; table hash recorded |
| 2 | CERNLIB removal, Phase 1, then Phase 2 (§6) | ~1 week | byte-identical, then statistical |
| 3 | Flat-log-E generator with OneWeight, Earth absorption weight (§8) | 3–5 days | reweight to conventional daemonflux and compare with an existing `nuatm` sample |
| 4 | Structured output and BARS detector-response stage (§10) | 2–3 weeks | compare with the `mcstage2` + BARS chain on the same input |
| 5 | Speed-ups E1–E5 | 1–2 weeks | identical physics output; timing per energy decade |
| 6 | P1 (light of < 2 GeV losses) and P2 (energy-dependent light range) | ~1 week | test run with the filter lowered; light vs distance for PeV showers |
| 7 | PROPOSAL (§7) | a few weeks | energy-loss spectra and ranges vs MUM |
| 8 | P3 scattering delays (photon tables) or adopt NTsim | months | data/MC time residuals |

If the immediate goal is the data/MC time residual rather than code quality, move P3 (scattering delays) to the top of the physics work, before steps 6–8; the current order is for code quality, not for closing the experimental excess.

**Time-cost guess for 100,000 triggered events, flat log E, 10 GeV–100 PeV** (ESTIMATE, before speed-ups): about 0.5–5 s per triggered event, i.e. 15–150 CPU-hours. That is minutes to ~1.5 h of wall time on one cluster8x node with ~100 free threads. A per-decade benchmark run after step 2 will replace this estimate.

---

## 12. How to reproduce the checks

All read-only. Fortran files contain KOI8-R comments, so use `grep -a` or `iconv -f koi8-r -t utf-8`.

```bash
cd /home/ivkhar/Baikal/jinr_repo/Baikal/simgvd
grep -n -a "rluxgo\|rdmin\|rm48in\|norrin\|poiset" *.f third-party/*/*.f  # C1: only RLUXGO; POISET(50)
grep -n -a "rndm(\|rm48(" third-party/mum/mum1_3.f | wc -l               # C1: 27 MUM call sites
sed -n 1585,1600p sim3h1.f; sed -n 3318,3328p third-party/mum/mum1_3.f   # C2: filter before type check; end record etr=0
grep -n -a "nu_cascade\|nu_int" munu4w_osc.f munu7w_osc.f munu8w_osc.f   # C3: absent in 4w; present in 7w/8w
sed -n 140,146p munu4w_osc.f                                             # C4 (current 4w only)
grep -n -a "rrmax = 15000" sim3h1.f                                      # P2
md5sum cc_over_y.ascii                                                   # C14: table identity
grep -n -a "emu_sim\|nu_CC_tab\|rnu_CC_y" munu8w_osc.f su_dif_2.f         # C15
```

The wout header vertex and the y table can be checked directly:

```python
import numpy as np
# copy nu_mu_1226.wout from /eos/baikalgvd/mc/2020/nu2/wout/v3.0/10102/29/
words = np.fromfile("nu_mu_1226.wout", dtype="<u4"); f = words.view("<f4")
n = int(words[0]) // 4
# first event record: [n_bytes][payload][n_bytes]
pos = 2 + n
n2 = int(words[pos]) // 4
ev = f[pos + 2: pos + 1 + n2]          # drop the record's leading word count
print("nhit, theta, phi, A, E, vertex:", ev[:8])   # v1.0: 0; v3.0/v4.0: vertex (matches reconstruction on sentinels)
```

MC_2020 data checks use the reader `/home/ivkhar/Baikal/likelihood/mcread.py` (conda env `torch212`):

```python
import numpy as np, sys; sys.path.insert(0, '/home/ivkhar/Baikal/likelihood'); import mcread as M
X = M.extract('/home3/ivkhar/Baikal/data/initial_data/MC_2020/nue2_100pev/root/all/1000.root')
# P1: 2 GeV floor of stored showers for energetic muons
E, E0 = X["sh_E"], X["tr_E0"][X["sh_tr"]]; m = (E0 > 100) & ~X["sh_is_vtx"]; print(np.mean(E[m] < 2))   # ~0.002
# P2: perpendicular distance of lit OMs from the muon line (max ~149.8 m)
k = X["c_kind"] == 2; s = X["c_src"][k]; t = X["sh_tr"][s]; dv = X["geo"][X["c_om"][k]] - X["tr_r0"][t]; u = X["tr_u"][t]
l = (dv*u).sum(1); print(np.linalg.norm(dv - l[:, None]*u, axis=1).max())
```

---

## 13. Open questions for the MC group

1. **Which generator file and build produced the 2019–2023 productions?** Partial answer from the production logs and data: `nuatm` v1.0 (2022) = old 4w/Bartol/`esp_maximum=10⁵`, no `nu_cascade`; `nu2` v1.0 (2023) = 8w/power 2/`10⁸`, `nu_cascade`; `nuatm` v4.0 (2025) = current 4w/daemonflux, no `nu_cascade`; `nu2` v3.0 (2025) = 8w/power 2, `nu_cascade`. The exact commits/builds are still unknown (the repository is a single-commit snapshot). The default Makefile (4w) reproduces `nuatm`, not `nu2`.
2. **Is the 2 GeV loss filter intentional?** If so, is the light of < 2 GeV losses meant to be inside the bare-muon spline (`sp_nake`)? How was `ptm6.spl.300` produced?
3. **Was the 150 m light range (`rrmax`) a deliberate speed trade-off?** The README mentions 250 m above 50 TeV.
4. **Are any productions known to have been run with distinct `RNDM` seeds** (e.g. by a modified MUM)? If not, how many jobs share MUM sequences? The code has no `RDMIN`/`RM48IN`; the Poisson generator is fixed at `poiset(50.)`.
5. **Which `mcstage2` configuration** (`allwide_par.cmn`, `array.dat` per season) was used? The snapshot is single-cluster (288 channels).
6. **What is the status and validation plan of NTsim?** It would address P3/P4 at the root.
7. **Which `cc_over_y.ascii` (or `finte2_y` implementation) was used for `nuatm` v1.0 and v4.0?** The repository table gives `<y>_ν = 0.479` at 30 GeV and matches `nu2` (0.502), but `nuatm` shows 0.301; the nuatm production must have used a different table or an older implementation, not preserved. Was the table changed between the productions, or is `Reg` not the birth energy?
8. **Why is `nu2` v1.0 100 % ν while `nu2` v3.0 has ~40 % ν̄?** The 8w code samples ν/ν̄ in proportion to σΦ; the v1.0 sample does not. A production setting or a flux-table difference?

---

### Appendix: related facts established in the likelihood study

These are from MC_2020; see `likelihood/MC_2020_format_notes.md`.
- **Sub-pulse amplitudes** are running sums over the preceding 20 ns of the channel. Per-source charge is recovered as `inc_k = amp_k − Σ inc_m` (t_k − t_m ≤ 20 ns). Only 20 ns gives zero negative increments.
- **MC → BARS channel mapping:** `bars = 36·((mc−1)//36) + 35 − (mc−1)%36` (BARS uses a lookup table, `BGeomTelMC::ConvertChannelNumFromSimGVD`).
- **Magic numbers** have scale 10⁶ in MC_2019/2020 (the format notes say 1000), with float32 loss from shower 17 (C6).
- **The ν vertex** is the stored interaction with E = E_ν − E_reg in `nue2` v1.0; `nuatm` has no stored vertex interaction. The 2025 wout (`nu2` v3.0, `nuatm` v4.0) writes the interaction vertex in header words 5–7. Verified on sentinels: the first `nu2` v3.0 event and events 5, 6, 8 of `nuatm` v4.0 (file 1226, batch 29) match `ref − c·fFirstMuonTime·d` to ≤2e-4 m. For alive tracks the formula is not valid (the first `nuatm` v4.0 event differs by ~180 m); use the header field. In v1.0 those words are zero; reconstruct only for sentinels. `fMuonEnergy = −0.001` marks a vertex downstream of r₀.
- **The inelasticity `y` is not stored per event.** In `nue2` it can be derived as `1 − Reg/E_ν` because the vertex cascade is stored; in `nuatm` the vertex cascade is absent, so the same expression is only a proxy whose relation to the generator's `y` depends on the (unversioned) `cc_over_y.ascii` table (C14).
- **Light model:** emission at θ_c (n = 1.33), propagation with group index 1.37, no scattering delay (P3).
- **Seasons** (file series 1xxx–4xxx) differ in switched-off channels (128 / 439 / 703 / 716 OMs) and noise rates. Dead OMs must be excluded when building tables.
