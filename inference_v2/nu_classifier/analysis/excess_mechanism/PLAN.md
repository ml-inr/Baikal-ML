# Are groups 6 and 8 the same physics? — plan

**Status: executed 2026-08-25.** Results in `FINDINGS.md`, notebook `groups.ipynb`, feature
table documented in `FEATURES.md`. The hypotheses below were fixed before any fitting and are
scored against the numbers in `FINDINGS.md`: **H1 supported, H2 and H3 not supported** — for the question of whether the two groups differ
*in kind*. The separate question of *why the excess is twice as large* is **not** answered here;
a claim that it was (brightness) was published and retracted on 2026-08-25, see FINDINGS §2a.

Three things in this plan turned out wrong and are corrected in place below rather than
rewritten away: the per-channel time offsets (§5.1, refuted before the run), the `fit_contrast`
threshold (there is none — see FINDINGS), and the neutrino-box removal, which the plan did not
say to apply symmetrically and which manufactured a difference when it was not.

Mode: **разведка** (exploratory). Leakage-safe splits are kept because without them an AUC
means nothing; bootstrap stability of cut lists, the full 23M-muon pass and event displays are
deferred to a later iteration.

## 0. The question

At ξ = 0.8 the eight groups are, on the h8s3 quality selection with verified exclusions
applied (measured 2026-08-25):

| group | class | score | N |
|---|---|---|---|
| 1 | nuatm | < ξ | 5,162 |
| 2 | nuatm | ≥ ξ | 525,904 |
| 3 | nue2 | < ξ | 12,499 |
| 4 | nue2 | ≥ ξ | 1,360,695 |
| 5 | muatm | < ξ | 23,159,967 |
| 6 | muatm | ≥ ξ | **7,751** |
| 7 | exp | < ξ | 3,177,637 |
| 8 | exp | ≥ ξ | **3,044** |

8/(7+8) = 0.0957% against 6/(5+6) = 0.0335%: a factor 2.9 raw, 1.97 after the shape
correction of the excess study.

**The question is whether 6 and 8 are the same physical population** (and the excess is that
population being commoner in data), **or not.**

The score cut is *not* assumed to select the same region of feature space in both domains —
that is the whole point. Under domain shift a DANN score can land somewhere else in data than
in simulation, and mapping where it lands in each domain is question (а)/(б).

## 1. Sample and cleaning

**Cleaning = the verified blacklist and nothing else**, i.e. exactly what `splits.excluded`
already applies: runs `part_s2020_c02_r0020` and `part_s2020_c02_r0249`, whose channels
222/224/225/227 emit up to 3169.8 p.e. The cluster 1 / cluster 4 entries stay `refuted` and are
not applied. MC additionally excludes `used_for_labels`.

Repeated hits are **not** cleaned out. `hits_per_module` therefore stays in the feature set —
with the caveat in §5.

## 2. Features

Rules of construction:

1. **Nothing derived from the nu-classifier**: no score, no embedding, no `distance_to_nu/mu`.
   Otherwise every answer is circular.
2. **Identical code on MC and data.** No MC truth anywhere.
3. **Run/cluster/time identity is not a feature** — it would let a tree memorise runs — but is
   carried alongside for the post-hoc checks of §5 and §6.
4. **No free parameter tuned on the data.** Every tolerance below is fixed a priori and stated.
5. Preference for **dimensionless quantities with a value physics predicts**, so a
   disagreement reads as "so many times the expectation" rather than as an unanchored shift.

All from the sig-noise-filtered hits (prob > 0.8, batch 256), one pass over the HDF5.

### 2.1 The track fit — the backbone

`track_likeness` in the excess study is the r² of hit time against depth: x and y are ignored
entirely. Replace it with the real thing. For a relativistic track through point `p0` with
direction `u`, the Cherenkov arrival time at a module at `r` is exactly

```
t_pred = t0 + [ s + K·d ] / c ,    s = (r − p0)·u,    d = |(r − p0) − s·u|
K = sqrt(n² − 1) = 0.9364   (n = 1.37, doc/data_format.md)
c = 0.299792458 m/ns
```

**The direction is found from timing, not from the hit cloud.** PCA of the positions is not
usable here: modules sit on vertical strings, so the cloud's principal axis is near-vertical
whatever the track did.

**`p0` must lie on the track, and the hit centroid does not.** The formula above is only valid
for a `p0` on the line: moving `p0` *along* `u` is absorbed by `t0`, but moving it
*perpendicular* changes every `d`. The centroid of the hits sits off the track by roughly the
typical perpendicular distance, so anchoring there biases the fit. (An earlier draft of this
plan did exactly that. Recorded rather than silently fixed.)

A global scan, then **five** rounds of alternating refinement on fixed grids, no tuning:

1. scan `u` over 500 directions on the sphere with `p0` at the charge-weighted centroid;
   `t0` in closed form as the mean residual for each. Keep the best `u`.
2. five rounds of `REFINE_ROUNDS = ((60 m, 10°), (20 m, 3°), (7 m, 1°), (2.5 m, 0.3°),
   (1 m, 0.1°))`. Each round scans the two perpendicular offsets of `p0` on a 7×7 grid at that
   round's half-range, then 50 directions within that round's radius of the current `u`.

**Why five and not one.** Direction and perpendicular offset are coupled, so alternating them
has to iterate. Measured on an exact synthetic Cherenkov track, where the answer is known:

| refinement | worst angle error | worst `fit_rms` |
|---|---|---|
| one 10° local pass | 4.05° | 28.5 ns |
| three rounds | 1.02° | 4.5 ns |
| five rounds | 1.87° at 38 hits, 0.68° at 74 | 2.3 ns |

A single pass makes `fit_rms` measure the fitter's grid resolution rather than the event. The
unit test asserts `fit_rms < 3 ns` and angle error `< 3°` on tracks built from the formula
itself, so this cannot silently regress.

About 1,000 evaluations per event, each a (n_hits × n_grid) matrix — vectorised.

Features out of it:

| feature | meaning |
|---|---|
| `fit_rms` | RMS time residual at the best direction, ns |
| `fit_contrast` | (median RMS over the whole grid − best RMS) / median RMS. **Is there a track at all**: an event with no track structure fits every direction about equally badly. This, not r², is the honest "track-likeness" |
| `fit_zenith`, `fit_azimuth` | direction, project convention (90° = horizon, 180° = straight down) |
| `frac_on_track` | fraction of hits within **25 ns** of the prediction. Fixed a priori and not tuned: the vertical module spacing is 15.0 m (measured), which is 68.6 ns of light travel in water, so 25 ns is a third of one module step |
| `q_offtrack_frac` | charge of the off-track hits over total charge. The classic separator of a single muon from a bundle or a cascade |
| `q_weighted_residual` | charge-weighted mean residual, ns. Direct light is early and bright, scattered light late and dim; the sign says which dominates |

Caveat to keep in view: with ~10–15 hits and a 500-direction grid the best-fit RMS is
optimistically low by construction. That is exactly why `fit_contrast` is defined relative to
the same event's own grid — it is the quantity a fluctuation cannot inflate.

### 2.2 Dimensionless ratios, each with an expected value

| feature | definition | expectation |
|---|---|---|
| `slowness` | `t_span / (extent / v_light)`, `v_light = c/n = 0.2188 m/ns`, `extent` = largest pairwise hit separation | ≈ 1 for a relativistic track. ≫ 1 means the event is too slow for its own size: scattering, several sources, or a mis-associated hit. Partly collinear with `t_span / z_span` since most events are vertically extended — reported, not hidden |
| `q_vs_d_slope` | slope of `log q` against perpendicular distance `d` to the fitted track | a **proxy** for the light attenuation length in water, which DATA_QUALITY lists as "not reachable". Direct candidate to explain §5's light deficit. Not the attenuation length itself: the sig-noise filter keeps bright hits preferentially, so at large `d` the survivors are biased bright and the measured slope is flattened. Read only as a comparison between samples processed identically |
| `q_vs_d_r2` | how well that trend holds | |
| `string_slope_spread` | fit `t` against `z` per string (strings with ≥ 3 hits), take the spread of the slopes | ≈ 0 for one track: every string must agree on the slope. Large for a bundle or a cascade. Information the global fit does not contain |
| `n_strings_fitted` | how many strings that spread rests on | |

### 2.3 The repeated-hit diagnostic

One number that can settle an open question of the project — whether the 21.08%-against-3.45%
discrepancy is PMT afterpulsing:

| feature | meaning |
|---|---|
| `dt_repeat_median` | median time between successive hits on the same module. **The mechanism sets the scale**: ion afterpulses sit at hundreds of ns to µs, scattered light at tens of ns |
| `q_repeat_over_first` | charge of repeat hits over charge of first hits. An afterpulse is fainter than its primary; a second photon need not be |

### 2.4 Marginals, kept as the baseline

**Multiplicity and geometry** — `n_hits`, `n_modules`, `n_strings`,
`hits_per_module`, `hits_per_string_max`, `hits_per_string_mean`, `z_span`, `xy_span`, `z_c`,
`r_cyl`, `z_first`, `z_last`, `dz_signed`, and the hit-cloud PCA (`elongation`, `planarity`).
`r_vert` is retained **only** for comparability with the excess study, flagged as an aspect
ratio and not an angle (excess study §6); `fit_zenith` supersedes it.

**Timing** — `t_span`, `t_span_core`, `t_std`, `dtdz_slope`, `track_likeness` (the old r², kept
for comparability), `causality_violation_frac`, `spearman_tz`.

**Charge** — `q_total`, `q_mean`, `q_std`, `q_median`, `q_iqr`, `q_max`, `q_frac_max`,
`frac_q_below_2`, `q_asymmetry`, `centroid_shift`.

**Sig-noise context, secondary and flagged** — `prob_mean`, `prob_min`, `n_raw_hits`,
`survival_frac`. Another network's output; out of the primary set, used only as a robustness
check.

### 2.5 Deliberately not included

Any feature with a tolerance that invites tuning on the data. A PCA or autoencoder over the
whole set: it mixes the blocks but stops being interpretable, and interpretability is the
entire point. `hits_per_metre`, which depends on a geometry that drifts 2.3 m between runs.

### 2.6 On the time convention

Verified 2026-08-25 by measurement, and stated in doc/hdf5_format.md:35: hit times in both
`baikal_mc_merged.h5` and `exp_full.h5` are already mean-centred per event over the **raw**
hits, in the same way. Network and features share one convention; nothing needs re-centring.

The zero point is set largely by noise — raw hits span ~4,900 ns and the per-event median
wanders by 146 ns in MC and 155 ns in data. Every feature above is a *difference within the
event*, so this cancels exactly. The track fit is likewise invariant: `t0` absorbs any per-event
constant.

### 2.7 Validating the track fit before using it

A feature this load-bearing is not taken on trust. Three checks, all before any group is
compared:

1. **Against MC truth.** The `truth` table carries `zenith_deg` and `azimuth_deg` per MC event.
   Compare `fit_zenith` with it as a function of `fit_contrast` and `n_hits`. This is a genuine
   external anchor — the fit never sees truth. It also *calibrates* `fit_contrast`: it tells us
   the value above which the direction is meaningful, instead of us picking one.
2. **A null for `fit_contrast`.** Shuffle hit times within an event, destroying the track
   structure while keeping positions, charges and multiplicity. Refit. The resulting
   distribution is what `fit_contrast` looks like when no track exists, and it fixes the scale.
3. **A hand-computed event.** A synthetic track with hits placed exactly on the Cherenkov cone
   must return `fit_rms` ≈ 0 and the direction it was built from. Unit test, as
   `test_scalars.py` does.

If check 1 shows the fit does not recover MC truth, the track-fit block is reported as a
shape statistic only and every direction claim is dropped.

### 2.8 Cost

Measured, not estimated: **1.73 ms/event** for the full feature set (500-direction scan plus
five refinement rounds), so ~17 minutes single-threaded for 628k events. Reading dominates
instead: `raw/data` is chunked `(43215, 1)`, so a per-event read decompresses five chunks —
42 ms measured — and whole-part reads are the only sensible pattern. Each part is therefore
read exactly once, with every selected event of every group taken in that pass, across 16
worker processes.

**Stratified sample** (628,456 events): all of groups 1, 3, 6, 8; 100,000 each from 2 and 4;
200,000 each from 5 and 7. With 7,751 events in group 6 and 3,044 in group 8, the large classes
are far past the point of adding anything.

**One restriction, measured rather than assumed.** The quota groups (2, 4, 5, 7) are drawn only
from the parts the exhaustive groups (1, 3, 6, 8) already force us to read. Measured:

- **Experimental side: no restriction at all.** All 27 quality runs contain at least one
  group-8 event, so groups 7 and 8 are drawn from 100% of the 3,180,681 quality events. This
  matters more than it looks — had some runs been excluded, group 7 would have been biased
  toward high-rate runs, contaminating the very cluster-concentration checks of §6 that
  separate H2 from H3.
- **MC side: real but immaterial.** 5,373 of 10,100 muatm parts, holding 12,777,844 of
  23,159,967 group-5 events (55%). Group 5 is indistinguishable between selected and other
  parts: median hits 13.0 vs 13.0, charge per hit 3.57 vs 3.59, `q_total` 48.0 vs 48.2,
  `z_span` 150.4 vs 150.4, `t_span` 590.4 vs 590.0, mean score 0.00447 vs 0.00441. Whatever
  makes a part contain a high-scoring muon does not change what its ordinary muons look like.

## 3. What each question becomes

| sketch | method |
|---|---|
| (а) what selects group 6 | **two comparisons, not one.** "Group 6 against all others" is dominated by groups 2 and 4 (1.9M events against 7,751) and a tree would learn only "not a neutrino". So: **(а1) 6 vs 5** — what makes a muon score high; **(а2) 6 vs 2+4** — what separates a misread muon from a real neutrino. Shallow tree (depth ≤ 4, `min_samples_leaf` ≥ 50, class-balanced), trained on one set of MC parts, precision/recall on held-out parts, plus 20 bootstrap refits to see which splits recur |
| (б) same for group 8 | **(б1) 8 vs 7**, **(б2) 8 vs 2+4**. Split by **run**, not by event — neighbouring events in a run are correlated and an event-level split inflates every score |
| (в) what defines groups 2 and 4; is any of 8 like them | one-vs-rest trees for 2 and 4 → a "ν box". Count how many of group 8 fall inside it, and compare against how many of group 6 do — group 6's rate is the false-positive rate of the box |
| (г) set those aside | flag as subset **8ν**, do not delete. Every later number is reported for 8 and for 8∖8ν |
| (д) are 6 and 8 the same? | **(д1)** apply the (а) rules to exp and the (б) rules to MC; a domain-invariant selection transfers with little precision loss, an OOD one does not. **(д2)** a direct discriminator 6 vs 8∖8ν, AUC on held-out parts/runs against a permutation null; AUC ≈ 0.5 means one population and the excess is representation. **(д3)** density-ratio map in the two features (д2) leans on — the N-dimensional version of the excess study's figure 4 |
| (е) anything in 8 unlike all MC? | isolation forest fitted on **all MC groups 1–6** — the question is "unlike any simulation", and a reference of 2+4+6 alone would flag ordinary muons as novel, which they are not. Null: a held-out MC sample must give the nominal tail rate. Then the §6 checks |

AUC is used deliberately: it compares *shapes* and is blind to the 7,751-against-3,044 count
difference, which is the separate question §0 already answers.

**The headline number of this study is AUC(6 vs 8∖8ν) on held-out parts and runs, with its
permutation null.** The null: pool groups 6 and 8∖8ν, permute the group label, refit the same
tree with the same split, repeat 200 times. That distribution is what AUC looks like when the
two samples are one population, and it accounts for finite-sample optimism that a nominal 0.5
does not.

Everything is repeated at ξ = 0.9 as a sensitivity check; group 8 falls to a few hundred events
there, so it is a check on direction, not a second measurement.

## 4. Hypotheses, and what kills each — fixed before any fitting

- **H1 — 8 is 6, just commoner.** Predicts: (д2) AUC ≈ 0.5 held out; (д1) rules transfer both
  ways; (д3) ratio flat. Any one failing kills it. *Prior: the excess study's figure 4 already
  shows the ratio is not flat in 2D, so H1 is probably dead — which is why it must be stated
  now rather than after.*
- **H2 — 8 holds a population that is not in the simulation.** Predicts: AUC > 0.5; the
  discriminating region is under-populated in *every* MC class; the (е) outliers are spread
  evenly over runs, clusters and time.
- **H3 — the difference is instrumental.** Predicts: AUC > 0.5 *and* the discriminating
  features are ones §5 flags as miscalibrated, and/or the region concentrates in particular
  runs or clusters.

**What separates H2 from H3:** re-run (д2) with the three known instrumental axes removed —
`hits_per_module`, the repeat-hit block, and any feature the tree picked that §6's checks show
concentrated in particular runs or clusters. Under H3 the separation collapses; under H2 it
survives on the physics features alone. (The offset-corrected variant that used to stand here
is gone: the offsets turned out to be geometry — §5.1.)

## 5. Confounders known before starting

These are the reason a positive result cannot be read as physics without §6.

1. ~~Per-channel time offsets~~ — **refuted as a confounder on 2026-08-25.** They are
   geometric (module depth within string), MC reproduces the profile at r = 0.989–0.998, and
   the instrumental residual is 9–11 ns ≈ 2 m of light. Timing features need no correction.
   Kept visible here because the plan was built around it and the entry must not be silently
   dropped.
2. **String geometry drifts up to 2.3 m between runs** (§5); MC has a single geometry. Not
   correctable here.
3. **The high-score fraction varies 1.5× across clusters and is unexplained.** Any result must
   be checked for cluster concentration before it is called physics.
4. **Repeated hits: 21.08% of data against 3.45% of MC**, with none of the brightness
   association MC shows — almost certainly instrumental. Since cleaning is blacklist-only,
   `hits_per_module` is in the feature set and will probably dominate any tree. Reported
   explicitly, and (д2) is run with and without it.
5. **No absolute normalisation** exists. Every statement is about composition, never rate.

## 6. Post-hoc checks applied to any positive finding

Concentration of the selected events in: run, cluster, string, channel, time of day, season,
number of dead channels in the run. Uniform → H2 survives. Concentrated → H3.

## 7. Deliverable

`inference_v2/nu_classifier/analysis/excess_mechanism/` — this plan, `build_features.py`
(+ a hand-computed-event test, as `test_scalars.py` does), `groups.ipynb`, `FINDINGS.md`.

## 8. Out of scope this iteration

Bootstrap stability of the cut lists; the full 23M muon pass; validating the offset
calibration; event displays of anything found; any statement about absolute rates.
