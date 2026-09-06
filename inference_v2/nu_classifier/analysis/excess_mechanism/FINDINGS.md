# Are the misread muons and the experimental excess the same events? — findings

Notebook: `groups.ipynb`. Feature table: `features.duckdb`, documented in `FEATURES.md`.
Plan with the hypotheses fixed in advance: `PLAN.md`.

**Answer: yes.** Groups 6 and 8 are the same population. The experimental events the
nu-classifier accepts are its own failure mode reproduced on real data, and the excess is that
population being about twice as common in data as the muon simulation predicts. Nothing in a
48-quantity interpretable feature space distinguishes them beyond the generic
simulation-versus-data gap that affects every muon in the sample.

## The groups

At ξ = 0.8, on the h8s3 quality cut with verified exclusions: group 6 = simulated muons scoring
above the threshold (7,751), group 8 = experimental events scoring above it (3,044).
628,456 events were featurised across all eight groups.

## What was measured — `verified` unless marked

### 1. The score cut lands on the same events in both domains — `verified`

A depth-4 tree fitted on simulation to separate high-scoring muons from low-scoring ones scores
**0.988** applied unchanged to experimental data, against 0.991 in its own domain. The reverse
transfer is 0.990 against 0.987. Both domains lean on the same features in the same order:
`spearman_tz`, `z_c`, `fit_zenith`, `dtdz_slope`.

This is the result that answers the worry the study was built around — that under domain shift
a DANN score might select a different region of feature space in data than in simulation. It
does not.

### 2. The separation between 6 and 8 is not specific to the score cut — `verified`

The direct discriminator reaches **0.704** held out, against a permutation null whose 95th
percentile is 0.524. Separable — but that null is anti-conservative here and cannot carry the
conclusion. Group membership is perfectly confounded with part: every group-6 event is in a
simulated part and every group-8 event in an experimental run. Permuting labels destroys the
confounding, so the null contains no part-level effect while the real configuration is saturated
with one. It can say "not one population" while meaning only "simulation is not data".

**The low-score control is what carries the weight**, because the identical confounding is
present on both sides of it. The identical procedure applied to groups **5 and 7** — the same
two classes below the threshold, size-matched, with the same neutrino box applied — reaches
**0.731**.

| feature set | 6 vs 8 (above the cut) | 5 vs 7 (below the cut) | difference |
|---|---|---|---|
| all 48 primary | 0.714 ± 0.009 | 0.731 ± 0.015 | **−0.017** |
| instrumental axes removed (43) | 0.699 ± 0.011 | 0.723 ± 0.012 | **−0.025** |

The pair above the cut separates **slightly less well** than the pair below it. Whatever makes
experimental events distinguishable from simulated ones is present throughout the sample and is
not concentrated in the excess.

### 2a. Charge is the one place where the cut does matter — and it is a selection effect — `verified`

Single-variable separation is **not** uniform. Charge separates the two groups better above the
score cut than below it (q_mean 0.601 vs 0.566, q_total 0.604 vs 0.564, t_std 0.594 vs 0.548);
track structure separates them better below (track_likeness 0.509 vs 0.571, hits_per_module
0.658 vs 0.699). The equal multivariate AUC of §2 is those two effects cancelling, and saying
only "the AUCs match" would hide it.

The charge gap does not survive a selection made without the network:

| ratio of medians, exp/MC | score-selected (6 vs 8) | physical cut | no cut at all |
|---|---|---|---|
| `q_mean` | 1.364 | **1.154** | 1.128 |
| `q_total` | 1.439 | 1.164 | 1.183 |
| `q_max` | 1.641 | 1.245 | 1.292 |
| `n_hits` | 1.100 | 1.000 | 1.000 |

Experimental events are ~15% brighter *everywhere*. The 1.36 appears only when the selection is
made by the score, because the classifier prefers bright events and so takes a brighter slice of
an already-shifted distribution.

**It does NOT supply the mechanism for the factor of two — a claim this document made and now
retracts.** Inside one fixed, network-free population (horizon-like, no track structure), the
fraction the classifier accepts:

| charge per hit | accepted, simulation | accepted, experiment | ratio |
|---|---|---|---|
| < 3 | 0.764% | 1.540% | 2.02 |
| 3–5 | 0.656% | 1.209% | 1.84 |
| 5–8 | 0.683% | 1.300% | 1.90 |
| 8–15 | 0.693% | 1.827% | **2.64** |
| > 15 | 1.034% | 3.925% | **3.80** |

**Read the floor, not the trend.** If brightness were the mechanism, matching on it would bring
the ratio to 1. It does not: even in the faintest band the experimental acceptance is twice the
simulated one. The rise from 2.0 to 3.8 with brightness is real and adds at the top, but the
2.0 floor is not explained by brightness at all.

The population is present in both domains at nearly the same rate (4.07% of simulated muons
against 5.21% of experimental events, a factor 1.28), so it is not explained by a surplus of
events either. **What produces the factor of two is not established.**

An attempt to bound it — fit the acceptance rule on simulation over all 48 features, apply it to
data, stratify on the predicted probability — leaves a residual of **1.35 to 2.04 depending only
on the model's random seed**. With 7,751 accepted events against 200,000 weighted at 115.8 and
an acceptance rate of 0.03%, the estimator is not stable enough to quote. Settling this needs a
cross-fitted, calibrated estimator and more simulated statistics, and is the obvious next step.

**Note also that "horizon-like" is necessary but nowhere near sufficient**: of the 944,040
simulated muons passing the physical cut, the classifier accepts 0.72%; of the 165,603
experimental ones, 1.60%.

### 3. Group 8 is less neutrino-like than group 6 — `verified`

A neutrino box fitted on groups 2+4 against 1, 3 and 5 — deliberately excluding group 6, so it
is out-of-sample for groups 6, 7 and 8 alike — puts **53.6%** of group 8 inside it against
**58.6%** of group 6. Since group 6 contains no neutrinos by construction, 58.6% is the box's
own false-positive rate and group 8 sits below it.

The box is a blunt instrument and the number should not be read as anything but this one
comparison: it labels 58.6% of known non-neutrinos as neutrinos, which is what happens when a
box is drawn around simulated neutrinos and applied to the muons a classifier cannot tell from
them.

### 4. Nothing in group 8 lies outside the simulation — `verified`

An isolation forest fitted on all simulated groups puts **0.1%** of group 8 and **0.0%** of
group 6 below the cut holding 1% of held-out simulation. Both depleted; neither novel.

Both groups *are* shifted toward the atypical in the bulk — 90.7% of group 8 and 81.0% of
group 6 sit below the reference median where 50% is expected. That is what selecting on a
classifier score does to any sample, and the group 6 control is what shows it.

Against **group 6 alone** as the reference the 1% enrichment is 6.2 (87 events). Those events
are bright, busy, near-horizontal muons — roughly twice the charge per hit of the rest of group
8, three times the total charge, a quarter of their modules hit twice, `fit_zenith` near 95°.
Only 2.3% of them remain outliers when the reference is widened to the whole simulation. The
enrichment measures the narrowness of group 6, not novelty in the data.

### 5. No run or cluster carries more than its share — `verified`

Across 27 runs the ratio of selected share to total share has a spread of 0.09 and a maximum of
1.21; across six clusters it runs 0.93 to 1.06. The H3 (instrumental) prediction of
concentration does not occur.

## Hypotheses, fixed before fitting

| | prediction | outcome |
|---|---|---|
| **H1** group 8 is group 6, only commoner | discriminator no better above the cut than below; rules transfer | **supported** |
| **H2** group 8 holds a population the simulation lacks | separation specific to the cut, surviving ablation; a novel tail | **not supported** |
| **H3** the difference is instrumental | separation collapsing under ablation; concentration in runs/clusters | **not supported** |

## The track fit — the one new capability

A Cherenkov track fit (direction from timing, five refinement rounds, `p0` solved on the line
rather than anchored at the hit centroid) recovers MC truth to a **median 1.0° for the 42% of
events above `fit_contrast` 0.9**, and to a few degrees for the 99.1% above 0.6. The earlier
excess study had no usable direction estimate at all; this one does.

Two corrections recorded rather than quietly fixed:

- **A single local refinement pass was not enough.** On a track built from the Cherenkov formula
  itself it left 4.05° of angle error and 28.5 ns of residual — the fitter's grid resolution,
  not the event. `fit_rms` would have measured the wrong thing. Five shrinking rounds give
  0.68–1.87° and under 2.3 ns.
- **`fit_contrast` has no threshold.** The scrambled-times null suggested 0.25, and an earlier
  draft reported that as the value above which a direction was found. MC truth says accuracy
  improves smoothly and that 99.1% of events sit above 0.6, so the null separated real events
  from scrambled ones without sorting real events into fitted and unfitted.

## Errors caught during the work, kept visible

- **Asymmetric removal manufactured a difference.** Setting the neutrino-like subset aside from
  group 8 only raised the AUC from 0.749 to 0.806. Symmetric removal gives 0.704. Both are in
  the notebook's §4 table, the wrong one labelled.
- **Part names are not unique across MC data classes.** `part_1271`, `part_1216` and others
  exist in two classes at once; keying build tasks on `(source, part_key)` merged them and
  applied one class's `local_idx` to the other's arrays, raising `IndexError: index 24036 is out
  of bounds for axis 0 with size 23554`. The 90,247 partial rows were discarded rather than
  resumed — the affected rows cannot be identified after the fact.
- **The novelty tail alone would have misled.** Reading only the 1% enrichment (0.1) suggested
  group 8 was *more* typical of the simulation than simulation itself. The full distribution
  says it is shifted toward the atypical in the bulk, and the group 6 control says that is a
  property of the selection.

## What this does not settle

**Why there are twice as many.** The study asks whether the excess is *made of* something
different and answers no. It does not explain the factor of two. The simulation being 1.4x too
faint at fixed multiplicity, measured in the earlier study and unchanged here, remains the
leading candidate — a statement about the simulation, not the data.

**The mixture.** Group 8 is roughly half predicted muon background and no individual event can
be assigned to the surplus. Every statement is about the mixture. A distinct population confined
to the surplus and distributed exactly like misread muons would be invisible here — though it
would also be indistinguishable from misread muons by construction.

**Simulation as the reference.** "Nothing outside the simulation" constrains the data only as
far as the simulation is right. A physical population the simulation also produces, in the wrong
quantity, registers as no anomaly at all.

**Sensitivity.** Everything is at ξ = 0.8 on 48 quantities. A population differing in some
quantity not among them is not excluded.

**Only 27 experimental runs.** The part-level split is really a run-level split over 27 runs, so
run-to-run detector variation is only weakly controlled by the split itself. §5's concentration
check is what carries that load.
