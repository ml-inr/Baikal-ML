# Results

Append-only. Each entry names the stage that produced it, the artefact, and a
confidence label: **verified** (anchored outside the dataset), **consistent**
(agrees with the data but does not exclude rivals), **assumed**. Refutations are
added in place, never by rewriting the original claim.

---

## Stage 00 — index and cross-cluster feasibility (2026-08-26)

Artefacts: `data/00_exp_parts.parquet`, `00_cluster_pairs.parquet`,
`00_coincidence_dt.parquet`, `00_coincidence_summary.parquet`,
`00_clock_offsets.parquet`, `00_mc_multicluster.parquet`.

### MC keeps multi-cluster events — **verified**

The converter ran with `split_multi: true`, so one physical event that lit
several clusters is stored as one row per cluster, groupable by the ROOT entry
number in `ev_ids` (`doc/mc_provenance.md` §4). Census over 12 parts per class:

| class | events | multi-cluster | sizes |
|---|---|---|---|
| muatm_2020 | 385,477 | **1.66%** | 2 clusters: 6,226; 3: 168 |
| nuatm_2020 | 471,699 | 0.44% | 2: 2,041; 3: 14 |
| nue2_2020 | 256,120 | 8.60% | 2: 20,306; 3: 1,705; 4: 23; 5: 1 |

### Experimental runs from different clusters do overlap in time — **verified**

Each experimental part is one *(cluster, run)* pair and no run appears for two
clusters, but wall-clock spans intersect: **20 cross-cluster overlapping pairs**.
On 2021-04-02 all six clusters ran together for 14.4 h (~14M events); 2020-04-26
has c02/c03/c05 for 19–23 h; 2020-04-22 has c06/c07 for 24 h.

### No coincidence signal at any clock offset — **consistent**

Nearest-neighbour `dt` within ±100 µs: flat, background reproduced by theory
(13.5 counts/bin predicted and observed at 46 Hz). Widening to the full overlap
by FFT cross-correlation, with slow trigger-rate drifts removed:

* **0 detections in 20 pairs.** Best fluctuations 4.2–6.2 σ at lags scattered
  from −5,911 s to +12,382 s — mutually inconsistent, i.e. the trials
  distribution, not a signal.
* **Sensitivity demonstrated by injection**, not assumed: 50,000 fake
  coincidences copied A→B at +137 ms were recovered at 44.7 σ, 10,000 at 10.9 σ,
  6,000 at 7.6 σ, each at the correct lag.
* **Upper limit:** coincidence fraction < 0.23% of cluster-A events (median over
  pairs), against **1.66%** multi-cluster events in MC muatm — about 8× below.

**Why this is labelled consistent and not verified.** Two readings are not
separated: the timestamps may be unusable, or the coincidences may genuinely be
absent from the trigger stream. The group does not trust the per-event UTC
timestamps, nor absolute hit times, since BARS time calibration is difficult;
this was stated independently of the measurement. The result is recorded for
later use if the premises ever become verifiable.

**Consequence:** the cross-cluster anchor (PROTOCOL test 1s) is unavailable.
Test 1 runs in the half-detector form, as the protocol pre-registered.

---

## Stage 10 — test 4: the excess against noise load (2026-08-26)

Artefacts: `data/10_mc_reference.parquet`, `10_run_noise.parquet`,
`10_run_noise_correlations.parquet`, `10_run_noise_partial.parquet`.
Notebook: `notebooks/10_run_noise.ipynb`. Figure: `figures/10_excess_vs_noise.png`.

29 runs, 3,353,716 quality experimental events. Excess per disjoint score bin,
median over runs: **1.95** (0.5–0.8), **2.64** (0.8–0.9), **3.13** (0.9–1.0).

### The predicted effect is absent, and what is there points the other way — **consistent**

Every noise proxy correlates **negatively** with the excess level, where
noise-driven (b) predicts positive:

| proxy | raw ρ | control cluster | control quality_frac | control both |
|---|---|---|---|---|
| `raw_hits_per_event` | −0.542 (p=0.002) | −0.446 | −0.454 | −0.179 (p=0.35) |
| `sn_reject_frac` | −0.473 (p=0.009) | −0.393 | −0.356 | −0.056 (p=0.77) |
| `trigger_rate_hz` | −0.360 (p=0.055) | −0.436 | −0.143 | −0.037 (p=0.85) |

The **shape** of the score profile is unrelated to noise in every setting
(|ρ| ≤ 0.36, p > 0.2 for `slope`). A response error should bite the hard tail
harder than the bulk, i.e. change the slope; it does not.

**This is not a measured null.** `trigger_rate_hz` and `quality_frac` are
collinear at ρ = −0.95, so controlling for the latter can remove the variance the
test needs. With 29 runs these axes do not separate. The honest statement is that
the design lacks the power to separate them, while noting that no version of the
analysis shows the predicted sign.

### A per-cluster dependence of the excess — **consistent**, and a lead

The excess varies systematically with cluster: **2.2 (c07) to 3.1 (c04)**, a 40%
spread. It is not noise — cluster 4 is simultaneously the quietest (55 raw hits
per event, 37 Hz) and the most excessive. Clusters differ in geometry, in the
number of working modules and in calibration, while MC simulates one
configuration. This points at a *static* description error rather than a dynamic
noise-driven one, and belongs to test 1.

### Two corrections made while running this stage

* **Cumulative thresholds replaced by disjoint bins.** Nested cuts are one
  measurement repeated with different weights, so the twelve correlation tests of
  the first version could not be counted honestly, and the cumulative number is
  dominated by the lowest bin where the excess is weakest.
* **The noise proxy was mis-specified.** `n_channels` counts *modules among the
  hits that survived the sig-noise filter*
  (`inference_v2/nu_classifier/compute_scalars.py:event_scalars`), so
  `n_channels − n_sn_hits` is minus the number of repeated hits, not rejected
  noise. Nothing in the prediction database can measure noise; the real load is
  read from `raw/ev_starts` in the HDF5, over every triggered event.

---

## Stage 20 — test 2: is a flux reweighting sufficient? (2026-08-26)

Artefacts: `data/20_score_fit.parquet`, `20_weights.parquet`,
`20_smoothness_sweep.parquet`, `20_achievable_bound.parquet`,
`20_held_out.parquet`. Notebook: `notebooks/20_flux_reweight.ipynb`.
Figure: `figures/20_flux_reweight.png`.

Truth axes available: zenith and primary energy (`truth` table, 23,171,597
quality muatm events; `event_weight` is identically 1, so the sample is
unweighted). Accepted MC muons sit near the horizon — mean zenith 107.9° against
136.6° for the rest — while their energies are indistinguishable (log₁₀E 2.73
against 2.74). Zenith carries essentially all the leverage.

### A bounded flux reweighting cannot produce the excess — **verified**

This is an exact upper bound, not a fit result. With `w` a function of truth only,
the reweighted acceptance is `E[w·a(T)]/E[w]`; maximising that ratio over
`w ∈ [1/√R, √R]` has its optimum at a vertex, so sorting truth bins by acceptance
and scanning the threshold gives the exact maximum no fit can beat.

| weight range R | max enhancement | observed excess |
|---|---|---|
| ×2 | 1.76–1.77 | 2.71 / 2.81 / 3.29 / 4.39 — **impossible in every band** |
| ×4 | 3.01–3.14 | **impossible** at ξ>0.95 (3.29) and ξ>0.99 (4.39) |
| ×10 | 5.99–6.59 | reachable |

Labelled verified because it is a bound on the arithmetic, anchored outside any
fitting choice — it holds for the optimal weight, ignoring smoothness and
ignoring every other observable.

### The cost of a plausible weight — **verified**

Sweeping the smoothness penalty (weights fitted as `exp(θ)`, so they cannot be
driven to zero the way the first NNLS version did):

| smoothness | worst band off by | weight range needed |
|---|---|---|
| 1 | 0.09 | ×370 |
| 10 | 0.10 | ×131 |
| 100 | 0.49 | ×16 |
| 1000 | 0.98 | ×4 |

To match every band within 10% the weight needs a dynamic range of ×131. Its
shape is ×12 near the horizon together with suppression of the bulk to 0.41 —
and the bulk of the down-going muon flux is well measured.

### The fitted weight makes the observables worse — **consistent**

Median mismatch removed: **−75.5%** over all quality events, **−55.7%** in the
accepted region (negative = disagreement increased). Not uniform: `q_total`,
`q_mean`, `q_max` improve by 35–45% in the accepted region, while `z_c`,
`prob_mean`, `n_channels`, `xy_span` get substantially worse.

### What this does not establish

Only two truth axes exist in the prediction database. Multiplicity was closed
separately — AUC 0.56 for separating the false positives, report 2026-08-10 §4 —
so it carries little leverage, but **impact parameter is not available at all**.
A flux error living in a variable we cannot see would not be caught: the bound is
computed in the truth space we have.

Stated precisely: **hypothesis (a), in the available truth variables, requires a
flux distortion of at least a factor 4–10 concentrated near the horizon, and even
then degrades the observables.** Whether such an error is plausible for
near-horizontal muons is a question for the generator, not for these data.

### A correction made while running this stage

The first version fitted weights by non-negative least squares. It drove 24
occupied truth bins to exactly zero, holding 65.6% of the MC sample, and produced
a U-shaped zenith weight. A fit that deletes two thirds of the sample cannot
refute anything, so the stage was rewritten as a smoothness sweep in log-weight
space, and the conclusion now rests on the exact bound rather than on any fit.

---

## Stages 30–31 — test 1: is the detector response the same? (2026-08-26)

Artefacts: `data/30_loo_residuals.parquet` (175,779 residuals: 95,256 MC,
80,523 experimental, 41 min on 24 workers), `31_matched_cells.parquet`,
`31_by_band.parquet`.

### The design had to change before it could run — **verified**

The protocol's half-detector split needs six strings so each half has three.
Accepted events have a **median of three strings**: only 216 of 7,752 accepted MC
events and 146 of 3,220 accepted experimental events survive that requirement, so
the split cannot reach the population carrying the excess. Replaced by withholding
one hit at a time: the anchor is looser, but it is equally loose in both samples,
and it reaches the whole population. Recorded here rather than quietly swapped.

### The response is not the same — **consistent, strongly**

Matched on geometry (hits × strings × fitted distance to the withheld module),
**all 66 cells with adequate statistics show experimental residuals wider than
simulated ones**. Median IQR ratio **1.29**, median tail ratio (|Δt| > 50 ns)
**1.47**. Not a single cell goes the other way.

Converted to the extra Gaussian spread that would produce it: **28.7 ns** median
over cells. The per-channel calibration residual measured earlier
(`analysis/excess_mechanism/measure_channel_offsets.py`, 9–11 ns) accounts for a
small part of that — √(28.7² − 10²) = 26.9 ns would remain.

Labelled consistent, not verified, because matching on observed geometry does not
guarantee matched *true* tracks; and the matching variable `d` is the fitted
distance, so it carries a little of the fit quality it is meant to control for.
Sixty-six of sixty-six cells in the same direction makes a residual population
difference an uncomfortable explanation, but not an excluded one.

### The extra width is symmetric, which argues against cascades — **consistent**

Artefact: `data/31_asymmetry.parquet`.

The track model describes light from the bare muon at the Cherenkov angle.
Stochastic showers along the track (bremsstrahlung, pair production,
photonuclear — MC does simulate them) radiate from a point, and their light, like
scattered light, can only arrive **later** than the direct front. A cascade or
scattering explanation therefore predicts a one-sided late excess.

Measured per matched cell, from each cell's own median:

| side | ratio exp / MC |
|---|---|
| early half-IQR | 1.29 |
| late half-IQR | 1.37 |
| tail earlier than −50 ns | 1.67 |
| tail later than +50 ns | 1.39 |
| median shift (exp − MC) | −6.3 ns |

The excess is symmetric, and the *early* tail is enhanced more than the late one.

**This does not exclude cascades.** `t0` is fitted on the anchor hits, so extra
late light on *those* drags `t0` late and pushes the withheld hit's residual
early — a one-sided cause can masquerade as a symmetric spread. The observed
−6.3 ns median shift points that way and shows the mechanism is partly at work,
though it is small next to the width difference. A dedicated check would compare
residuals in bright against dim events, since cascades track energy deposition.

### The mismatch does not track the excess — **consistent**

| band | IQR ratio | tail ratio | implied extra σ | cumulative excess |
|---|---|---|---|---|
| ξ ≥ 0.0 | 1.305 | 1.513 | 29.6 ns | 1.00 |
| ξ ≥ 0.1 | 1.599 | 1.667 | 33.3 ns | 1.59 |
| ξ ≥ 0.5 | 1.417 | 1.597 | 23.4 ns | 2.19 |
| ξ ≥ 0.8 | 1.423 | 1.543 | 23.8 ns | 2.87 |
| ξ ≥ 0.9 | 1.339 | 1.531 | 22.5 ns | 3.10 |

The protocol's decisive form expected the gap to grow with the excess if (b)
drives it. It does not: it is flat, and slightly *lower* where the excess is
largest. The response error is already fully present in the lowest band, where MC
and experiment agree in rate to 2%.

**This does not clear (b).** Acceptance at ξ > 0.8 is 3.3 × 10⁻⁴ in MC; the excess
is a ratio of small numbers in the tail. A uniform degradation that moves a
fraction of a percent of events can multiply a tail rate several-fold while
leaving the bulk untouched, and would then show exactly this flat profile. Stage
40 tests that directly instead of arguing about it.

### The likely nature of the gap, and why it matters for the paper

The residual is built from times within one event, so what matters is relative,
not absolute, calibration. The earlier per-channel measurement compared *median*
offsets, which is blind to per-hit timing scatter — a 28.7 ns per-hit resolution
difference would not have shown up there. The group does not trust BARS absolute
time calibration, and this measurement is consistent with MC's timing model being
optimistic.

If that is what it is, the consequence for the paper is the pessimistic one: real
neutrinos are smeared by the same amount, so MC-derived recall does not transfer
either. This is the question stage 40 was built to answer.

---

## Stage 40 — test 3: the measured response error reproduces the excess (2026-08-27)

Artefacts: `data/40_jitter_parts.parquet`, `40_jitter_summary.parquet`.
300 MC muatm parts, 8,640,212 events, both networks re-run from raw hits at every
jitter level, sig-noise batch size pinned at 256.

> **Numbers superseded once.** The first run of this stage selected its parts with
> Python's `hash()`, whose string hashing is salted per process, so two runs drew
> different parts and disagreed by 9% on the calibrated jitter. Replaced by
> `crc32`; two consecutive runs now produce byte-identical artefacts, verified.
> The figures below are from the reproducible run. The earlier ones (215 accepted
> at σ = 0, induced excess 3.04 at 30 ns) are recorded here so the change is
> visible rather than silent.

Stage 31 measured what MC lacks — about **29 ns** of timing spread — from the
leave-one-out residual. That number was fed in unchanged; it was not tuned, and
the excess it has to reproduce played no part in choosing it.

| jitter σ | quality events | accepted | acceptance | induced excess |
|---|---|---|---|---|
| 0 ns | 693,009 | 239 | 3.45 × 10⁻⁴ | 1.00 |
| 15 ns | 651,787 | 337 | 5.17 × 10⁻⁴ | 1.50 |
| 25 ns | 590,731 | 478 | 8.09 × 10⁻⁴ | 2.35 |
| **30 ns** | 556,375 | 538 | 9.67 × 10⁻⁴ | **2.80** |
| 40 ns | 485,336 | 665 | 1.37 × 10⁻³ | 3.97 |

What the jitter does at 30 ns: accepted events **×2.25**, quality events **×0.80**.
Four fifths of the effect is genuinely more events crossing the classifier
threshold; one fifth is the sig-noise filter discarding smeared hits and shrinking
the denominator. Both follow from the same perturbation, but they are not the same
mechanism and are reported apart.

---

## Stage 41 — does jittered MC also *look* like the data? (2026-08-27)

Artefacts: `data/41_jitter_residuals.parquet`, `41_validation.parquet`,
`41_matched_calibration.parquet`.

The jitter goes onto **raw** hits and the sig-noise filter then removes part of
what it smeared, so the surviving residual width is not the input σ and had to be
measured rather than assumed.

### The marginal comparison misleads, and says so — **verified**

| sample | dt IQR | tail > 50 ns | median hits |
|---|---|---|---|
| MC, no jitter | 47.42 | 0.213 | 13 |
| MC + 30 ns | 61.76 | 0.305 | 12 |
| experiment | 54.07 | 0.282 | 10 |

Taken marginally, 30 ns **overshoots**: experiment sits between the two. But the
samples differ in multiplicity — 13 hits against 10 — and a residual narrows with
more hits to anchor on, so the marginal comparison is not the one to read.

### Matched on geometry, the loop closes — **consistent**

| quantity | value |
|---|---|
| cells with adequate statistics | 45 |
| cells where experiment is wider than unjittered MC | 41 |
| input jitter needed to match experiment | **29.9 ns** (IQR 18.3–37.9) |
| excess that jitter produces (stage 40 scan) | **2.80** (range 1.78–3.73) |
| **observed excess at ξ > 0.8** | **2.87** |

The jitter measured independently in stage 31 (~29 ns, from residual widths) and
the jitter needed here to reproduce the experimental widths (29.9 ns) agree, and
the excess that jitter manufactures matches the observed one.

### What this establishes, and what it does not

**Established:** a timing-resolution error of the measured size is *sufficient* to
produce the entire excess, and it is not a tuned parameter — it was measured from
a different observable in a comparison that never saw the acceptance rate.

**Not established:** that it is the *only* possible cause. Test 3 shows
sufficiency, never uniqueness; another response perturbation could do the same.
The calibration is also wide — the cell IQR 18.3–37.9 ns predicts an excess
anywhere between 1.78 and 3.73, so the agreement of central values is far tighter
than the measurement warrants and must not be quoted as a precision match.

**Left unexplained:** jitter does not reconcile the hit multiplicity. Experiment's
median is 10 hits, jittered MC's 12–13. Timing spread alone does not exhaust the
difference between the samples.

---

## Stages 50–51 — test 5: cuts that remove the false positives (2026-08-27)

Artefacts: `data/50_fp_features.parquet` (805,356 events, 74 features, 111 min),
`51_single_cuts.parquet`, `51_cut_results.parquet`, `51_stability.parquet`.

The cut was fitted on MC only, separating muatm the classifier accepts (ξ > 0.5)
from muatm it confidently rejects (ξ < 0.01) — the same particles, so what it
learns is confusability. Experiment and neutrinos were never seen while fitting.

### Such cuts exist, are single-variable, and are stable — **verified**

All five disjoint replicates pick the same features. At the working point that
removes 90% of the false positives:

| cut | selectivity | clean muons kept |
|---|---|---|
| `seq_depth_corr ≤ −0.232` | 9.92 | 99.2% |
| `dtdz_slope ≤ −1.374` | 9.90 | 99.0% |
| `spearman_tz ≤ −0.224` | 9.83 | 99.0% |
| `dz_signed ≤ −16.13` | 9.78 | 97.8% |
| `fit_zenith ≥ 98.4°` | 9.35 | 93.6% |

Muon statistics survive, as the scale argument predicted: 97.8–99.2% of all
quality muatm is kept while 90% of the false positives go.

**A limitation of the replicate test, stated plainly.** For a single cut the
threshold is a quantile of the *false-positive* distribution alone, so it does
not depend on which clean muons were drawn — the replicates tested which
*features* win, not how stable the thresholds are. Feature choice was identical
in all five (selectivity spread 0.006–0.026); threshold stability was not tested
by this design.

Two rules must be reported as failures rather than quietly dropped: the depth-2
tree is degenerate at the 90% point (both leaves carry the same class, so it does
not separate at all), and the depth-3 tree is unstable there (retention 0.298
with a spread of 0.394 across replicates). Both work at the 99% point.

### Experimental events largely go with the muons — but not entirely — **consistent**

At 90% removal the accepted experimental sample retains 0.117–0.174 against
0.098–0.106 for the MC false positives. Normalised by each sample's own low-score
control — which is what removes the global domain shift — the surviving
experimental excess is:

| cut | at 90% removal | at 99% removal |
|---|---|---|
| `dz_signed` | 1.20 | **2.60** |
| `seq_depth_corr` | 1.33 | 1.34 |
| `spearman_tz` | 1.28 | 1.68 |
| `dtdz_slope` | 1.36 | 2.21 |
| GBM | 1.67 | **2.47** |

One would mean the two are the same population. They are close at the loose
working point and separate at the tight one: the harder the cut bites, the better
the experimental events survive relative to the muons. At 99% the strongest rules
leave experimental events surviving 2.5–2.6× better — near the excess itself
(2.87), though whether that is more than coincidence is not established here.

Reading: the accepted experimental sample is mostly the same confusable-muon
population and dies with it, but a residual is not removed by a cut tuned on
simulated muons.

### The recall cost is catastrophic — **REFUTED, see the correction below**

| | at 90% removal | at 99% removal |
|---|---|---|
| nuatm kept | 0.1–0.2% | ≈0.01% |
| nue2 kept | 0.5–0.9% | ≈0.1% |
| nuatm absolute efficiency | — | **0.00002–0.0002** |
| nue2 absolute efficiency | — | **0.0002–0.0012** |

Neutrino efficiency falls from 99.85% of quality events to between 0.002% and
0.12% — a factor of a thousand to ten thousand. Of 10,896 accepted experimental
events, 144–455 remain.

### Why, and why this was predictable

Every winning feature measures **direction**: correlation of hit order with depth,
slope of time against depth, rank correlation of time and z, signed depth change,
fitted zenith. The muons that fool the classifier fool it *because they look
up-going*, and up-going is precisely the ν_μ signature. There is no cut that
removes one without the other.

This reasoning is sound for *directional* cuts and wrong as a general conclusion.
It is left standing above so the error is visible; the correction follows.

### Correction: the search was too narrow, and the answer is positive — **verified**

Two flaws, both found by the user rather than by me.

**Only one-sided rules were scanned.** `x ≤ t` and `x ≥ t`, never `a ≤ x ≤ b`.
The false positives sit in a *band* near the horizon, and a one-sided rule on a
directional quantity has to remove the whole up-going hemisphere to reach them.

**Candidates were shortlisted by the wrong criterion.** They were ranked by how
well they preserve clean muons, and only the top four went through the neutrino
measurement. `extent_m` ranks seventh by that criterion. Measuring what a cut does
to a held-out sample is evaluation, not design, and contaminates nothing — the
pre-filter had no justification.

With all 224 rules evaluated at both working points, **22 remove ≥98% of the false
positives while keeping ≥60% of muon statistics**:

| cut | FP left | muons left | nuatm | nue2 | exp accepted left | exp events left |
|---|---|---|---|---|---|---|
| `track_likeness ≥ 0.690` | 1.1% | 78.6% | **87.1%** | 46.2% | 2.0% | 219 |
| `extent_m ≥ 144.7` | 1.0% | 80.2% | **81.4%** | 50.3% | 1.7% | 190 |
| `seq_depth_absc ≥ 0.850` | 1.1% | 74.1% | 86.3% | 42.2% | 0.9% | 98 |
| `t_span_core ≥ 468` | 1.0% | 67.1% | 80.7% | 41.4% | 2.3% | 245 |

**Neutrino efficiency falls from 99.85% to about 87%, not to 0.01%** — a relative
loss of roughly an eighth, not a factor of ten thousand.

**The retention is validated out of sample.** Neutrinos were split by MC part; the
two halves agree to within 0.0017 across every rule in the table. So the figure is
not an artefact of having chosen the cut for sparing neutrinos.

**A degenerate family had to be excluded, and the muon constraint is what excludes
it.** `seq_depth_corr ≥ 0.845`, `spearman_tz ≥ 0.857` and `dz_signed ≥ 75.7`
remove ≥99% of false positives and keep 85% of neutrinos — while keeping **zero**
muons. They achieve it by selecting up-going events, i.e. by re-deriving the
classifier. Requiring muon statistics to survive is what rules them out; without
that constraint the search would have returned them as the answer.

### What the surviving features mean

`track_likeness` is the r² of a linear fit of hit time against depth;
`extent_m` is the largest distance between any two hits; `t_span` and `t_std`
are temporal extent; `seq_depth_absc` is the *unsigned* correlation of hit order
with depth. Every one measures **whether the event is a long, clean, well-ordered
track** — and none of them measures its direction.

So the false positives are **short, messy, poorly structured events**. Real muons
and real neutrinos are both long clean tracks, which is why a size-and-quality cut
separates the false positives from both, where a directional cut cannot.

### The residual experimental excess is not determined by this method

Normalised by each sample's own low-score control, accepted experimental events
survive these cuts by a factor of **0.87 to 3.09** relative to the MC false
positives, depending on which cut is used — below one for `seq_depth_absc`, near
1.8 for `track_likeness` and `extent_m`, near 2.9 for `t_span`. The spread is too
wide to claim a residual. Between 90 and 290 experimental events remain after the
cut, and whether they exceed the muon expectation is not settled here.
