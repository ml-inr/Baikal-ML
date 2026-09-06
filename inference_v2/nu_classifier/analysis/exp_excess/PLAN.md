# The experimental excess: what these events are

## The question

Experimental data contains a population of events that the nu-classifier confidently calls
neutrino-like and that the muon simulation does not reproduce. At the h8s3 selection,
scored with `260816_2250_..._FIXED_sn256` (epoch 23):

| score cut | muatm | exp | of muatm h8s3 | of exp h8s3 | ratio |
|---|---|---|---|---|---|
| 0.8 | 7,751 | 3,220 | 0.03346% | 0.09601% | 2.87× |
| 0.9 | — | 1,456 | — | 0.04341% | 3.10× |
| 0.99 | — | 106 | — | 0.00316% | 4.39× |

Muon numbers are **out-of-training**; 23,167,718 muatm and 3,353,716 exp events in h8s3. The
0.9 and 0.99 rows still carry the training events and will move slightly.

These numbers are provisional in a second sense as well: they rest on Stage 0 being applied
consistently, and on the shape-only normalisation discussed in Stage 2. Every table in this
document is to be regenerated from `splits` once that exists.

**Why removing training events barely moved the muon side.** Only **2 of the 432** training
muatm parts were ever scored: probabilities exist for 10,100 of the 20,004 muatm parts, and
the training selection sits almost entirely in the unscored half. So the muon sample is
already effectively out-of-training — 3,879 events of 23.2M, 0.017%.

That is luck, not design, and it is fragile: scoring the remaining 9,904 muatm parts would
make the contamination real and large. The neutrino classes have no such luck — they are
100% covered, and h8s3 loses 9.5% of nuatm and **28.5%** of nue2 to the training set. Any
efficiency quoted on neutrinos is wrong unless `splits` is joined. Hence the rule in Stage 0:
every step joins `splits`, no exceptions, whatever the current contamination happens to be.

**What are these events?** Not "why does the background fake a signal" — that phrasing
answers the question before asking it. The expected number of atmospheric neutrinos is not
prior knowledge; it is a quantity to be computed from our own `nuatm` sample with weights
and exposure (Stage 2). Until it is computed, neither "too many" nor "too few" may be said.

And should the excess turn out to be a hundred times the atmospheric expectation, that does
not make it background. It makes it interesting.

> **Correction, 2026-08-21 — read before the stages below.** Parts of this plan were written
> while `r_vert` was believed to measure inclination. Within the population the classifier
> scores high it does not: every `r_vert` band sits at true zenith 107°, quartiles 102–112°.
> The correlation visible across the whole muon sample is a composition effect. Any stage text
> below that reads a `r_vert` band as an angle is wrong on that point; `FINDINGS.md` §5 carries
> the measurement and the corrected reading. Consequence worth stating once: **no direction
> measurement is available for the experimental excess at all** — `dt/dz` fails on track-less
> events, `r_vert` is decoupled, and `exp_reco` shares only 8 of our 29 runs.

## Ground rules

Taken from `CLAUDE.md`, and they exist because this project has repeatedly documented wrong
claims as established:

* **Name the falsifying observation before collecting evidence.** Every hypothesis below
  carries the observation that kills it. A test that cannot kill anything is not a test.
* **Anchor externally.** Agreement between two of our own quantities is `consistent`, never
  `verified`. Prefer physics, an independent reconstruction, or code that wrote the format.
* **State which rivals a test cannot separate.** Most tests here are not decisive alone.
* **Mark confidence in every recorded number**, and log claims in `doc/claims_log.md`.

Hypotheses stand side by side and are tested symmetrically. The last one cannot be confirmed,
only reached by elimination — and elimination must be by evidence, not by implausibility.

| hypothesis | what kills it |
|---|---|
| instrumental — particular runs, modules or conditions | the excess is distributed over runs and OMs like the h8s3 sample as a whole |
| under-modelled muon background | high-scoring MC muons form no distinct topology to be under-generated |
| domain shift — the network extrapolating off its training manifold | excess events sit inside the muon cloud, at the same distances as high-scoring MC muons |
| known neutrinos | events go downward; the rate does not scale with livetime |
| an unfamiliar population | reproduced by any of the above |

## Stage 0 — definitions and splits

Prerequisite for every number below: the scored MC contains **2,640,384 training events of
94,363,339 (2.80%)**, because scoring ran without `--npy-dir-to-exclude`. The model has seen
them and their scores are biased. Fixed at analysis time, not by rescoring — the training
NPY carries `h5_part_keys.npy` / `h5_local_event_ids.npy`.

`build_splits.py` writes `splits(event_fk, in_training BOOL, role VARCHAR)` into each
predictions DB:

* `used_for_labels` from those back-links, **event by event**;
* `was_da_target` for the experimental events fed to the domain discriminator without labels
  — recorded but not excluded, since no labels were involved;
* `role` ∈ {reference, test} from a stable hash of the **part** name, so events from one part
  never straddle the split and growing the sample cannot silently relabel anything.

The two act on different levels on purpose. Role is a property of the part, because
reference and test must not share one. Training exclusion is a property of the event: a part
that contributed to training still holds events the model never saw, and dropping the whole
part throws them away. In nue2 that is 82% of each touched part — enough to make nue2 the
binding class in the reference for no reason. (In nuatm the two rules coincide: its 187
training parts were consumed whole, all 55,598 of their h8s3 events. Those parts hold 9.48%
of nuatm's h8s3 sample; the other 531,066 events were never seen.)

Measured before adopting this: for nue2, untrained events from touched parts and events from
untouched parts have the same score distribution — median 0.994855 against 0.994851, 99.091%
against 99.086% above 0.8, on 1.05M and 324k events. Part-level exclusion buys nothing here.

A materialised table rather than a filter repeated in each script: every later step joins it
and therefore *cannot* accidentally include a training event or leak the reference into the
test set. Analysis selection: h8s3, out-of-training, excluding the known-bad runs
`c02_r0020` and `c02_r0249`.

Comparisons are quoted at a **fixed working point** (MC-muon survival ε), not a fixed score —
scores are not comparable across models. The ξ ↔ ε correspondence is tabulated explicitly.

*Artifacts*: `splits` table; a cut-flow table with counts per class at every step.

## Stage 1 — experimental data quality  ← runs before everything else

Promoted ahead of the physics stages on 2026-08-20: an instrumental defect does not bias a
measurement, it changes which events exist in the sample. Everything downstream inherits
whatever this stage misses.

### What is already established

**The inherited two-run exclusion is correct, and its mechanism is now measured.** Four
channels on one string of cluster 2 — 222, 224, 225, 227 — emit charges up to 3,169.8 p.e.
against a median hit of 0.97. Between 35.8% and 77.6% of their hits exceed 100 p.e., where
the typical channel sits at 0.0066%: a factor of 5,400 to 11,700. They fire in exactly two
runs, `part_s2020_c02_r0020` and `part_s2020_c02_r0249`, and in no other run of that cluster.

The chain is complete and quantified: the converter applies no charge limit, and neither
does the sig-noise network — a 3,170 p.e. hit reaches it as **863 standard deviations**
after normalisation, and hits above 1,000 p.e. are called signal **100%** of the time. The
event gains signal hits it should not have, passes h8s3, and arrives at the classifier as a
bright compact event — where the amplitude is finally clipped at 100 p.e., erasing the
evidence. Events touching those channels score above 0.8 **8.7 times more often** (0.7164%
against 0.0823% in the same runs) at twice the total charge.

Exclusion is by **run**, not by event: four channels behaving that way is a symptom, and
keeping the rest of a run assumes the remainder is healthy, which is untested.

**The cluster 1 and 4 exclusion is refuted on this data.** Recorded as "channel 71 broken
across cluster 4", it does not reproduce: channel 71 in c04 sits at 0.889 of the cluster
median by occupancy and 1.019 by mean charge, and c04 has no anomalous channel at all — it
is the cleanest cluster by both measures. Most likely the list came from `exp_reco`, or the
channel numbering there is global rather than per-cluster. Kept in the record as refuted;
not applied.

**Cost of all this: small.** The headline ratio moves from 2.87x to 2.86x once the two runs
are dropped. The defect is real, explained, and does not account for the excess.

### What remains unexplained, and is this stage's job

**Cluster-level rate structure.** The high-score fraction varies by 1.5x across clusters —
c04 at 1.25 of the mean, c07 at 0.81, with runs inside each cluster agreeing among
themselves. Dead channels do not explain it (c02 has 24 and sits at the mean; c04 has none
and sits highest), nor do hot channels, nor do the medians of hit count, charge or
sig-noise probability, which agree across clusters to within a few percent.

### Checks done since — results in `DATA_QUALITY.md`

1. **Time bursts (bioluminescence)** — `refuted`. Intervals between consecutive high-score
   events follow a Poisson process: 0.099% shorter than 0.1% of the mean against 0.100%
   expected, median 465 s against 492 s. No clustering, so nothing to explain. This was the
   candidate whose expected signature matched the portrait best.
2. **Trigger thresholds across clusters** — no difference. The 1st percentile of charge spans
   0.122–0.201 and the median 0.967–0.993; correlation with the high-score rate +0.28.
3. **String geometry** — real but unquantified. One channel of c04 moves 2.3 m between runs;
   within a 22-hour run a single position is used, so metre-scale sway sits inside `x, y, z`,
   which the sig-noise network reads as features.

### Still open

* **Cluster rate structure**, 1.5x from c04 to c07, unexplained by any defect measured so far.
  One lead: raw hits per event differ by 37% (50 in c02 and c04, 65–69 elsewhere) and
  anticorrelate with the rate at −0.65. Six clusters cannot establish that; it is a
  hypothesis.
* **Noisy channels by raw rate** — occupancy was measured on *signal* hits only, so a channel
  with a high noise rate whose hits are correctly rejected never appeared.
* **Per-channel time-calibration offsets** — a constant shift distorts `dt/dz` and `t_span`
  while leaving charge and occupancy untouched.
* **Quantifying the geometry sway** and its effect on `r_vert` and the hit selection.

### Output of this stage

An exclusion list in `config.yaml` with a status per entry — `verified` with its mechanism,
`refuted` with the failed check, `inherited` where no mechanism is known — materialised as a
column in `splits` so the filter is applied by a join and cannot be forgotten. Every earlier
number regenerated under it.

## Stage 2 — absolute normalisation

Livetime of the experimental runs and MC generation weights, giving **expected** counts:

* expected atmospheric muons at each working point — the denominator of the excess;
* expected atmospheric neutrinos from `nuatm` — the number that decides whether the excess
  is compatible with the known flux, deficient, or far above it;
* expected astrophysical-spectrum neutrinos from `nue2`, for reference.

Nothing quantitative about the excess may be stated before this stage. The ratios quoted
above are *shape* comparisons — each population normalised to its own h8s3 count — and they
silently assume exp's h8s3 sample is muon-dominated.

## Stage 3 — direction — DONE (2026-08-20)

Built, calibrated, and it answered a different question than the one asked.

**Calibration succeeded.** Against MC truth zenith, the sign of `dt/dz` recovers direction
with 99.2% purity over 25.1M events — negative is downward, positive upward — and with no
error at all among the 12.2M events fitted well (`fit_quality > 0.9`). The convention was
measured, not assumed.

**But the estimator does not work on the population under study.** Fit quality collapses
with score: median 0.90 at score < 0.01, 0.14 at score > 0.8. The proof that this is fatal
comes from the muons themselves, whose direction is known by construction:

| population, score > 0.8 | "upward" by sign | median fit quality | depth spread |
|---|---|---|---|
| MC nuatm — genuine ν | 99.62% | 0.991 | 195 m |
| MC muatm — false positives | **82.97%** | 0.140 | 61 m |
| exp — the excess | **79.29%** | 0.135 | 60 m |
| MC muatm at score < 0.01 | 0.20% | 0.902 | 135 m |

Every muon in the simulation travels downward. Low-scoring muons are called upward 0.2% of
the time; high-scoring ones 83%. So for these events the sign measures something other than
direction, and any up/down figure quoted for the excess would be an artefact. **Direction
for this population has no measurable answer by this method** — stated as a limit, not
worked around.

**What did come out of it, and it is the more useful finding.** The classifier's
high-scoring events have no track structure: hit times carry no linear dependence on depth,
and the events are compact — 60 m of depth against 195 m for genuine neutrinos at the same
score. And the excess is topologically indistinguishable from the muon false positives:
fit quality 0.135 vs 0.140, depth 60 m vs 61 m, 11 hits vs 10, 3 strings vs 3.

Consequences for the hypotheses:

* **known neutrinos** — weakened, but by shape rather than by direction: genuine neutrinos at
  the same score look nothing like this.
* **under-modelled muon background** — strengthened: the excess is the same failure mode as
  the MC false positives, roughly three times more abundant in data.
* Direction, if still wanted, needs a different estimator — timing along the fitted *track*
  rather than along depth, or the standard reconstruction where it exists (see the note on
  exp_reco). Not a prerequisite for the stages below.

**Order changed twice.** The portrait moved ahead of the remaining physics stages, and
then data quality moved ahead of everything — see Stage 1.

## Stage 4 — portrait

All 18 scalars plus direction: high-score exp against high-score muons against MC neutrinos.

**Already measured** (medians, h8s3, out-of-training, before the run exclusion): the excess
is indistinguishable from the muon false positives and unlike genuine neutrinos on every
geometric axis — depth spread 60.3 m against 60.6 m for false-positive muons and 195.2 m for
nuatm; `r_vert` 0.567 against 0.555 and 1.541; time span 289 ns against 318 and 754. The one
axis where the excess stands apart from the false-positive muons is brightness: `q_total`
74.4 against 51.5, `q_max` 31.5 against 18.8.

Note the classifier's high-scoring muons are *more compact* than its low-scoring ones
(60 m against 150 m of depth, 318 ns against 595 ns) with fewer hits and more concentrated
charge. It selects short bright clumps, not neutrino-like geometry.

To redo under the Stage 1 exclusion list, and to extend with the charge profile — whether
light falls off from a centre, as a cascade or a flash would have it, rather than lying
along a line.

## Stage 5 — domain shift

Reference set of 2:1:1 (muatm:nuatm:nue2), at most 5% of events, drawn only from
`role='reference'` parts. Three distances computed separately — to the whole MC manifold, to
the neutrino manifold, to the muon manifold. Conflating the first two has already produced a
wrong statement in the paper once.

Binding constraint is nuatm: 47,118 reference events at the current 10% part fraction, giving
a 2:1:1 reference of 188,472 against a 5% ceiling of 1,253,598. The fraction is to be set
from a measured convergence of the distances with reference size, not chosen by eye.

**Kills it**: excess events lying inside the muon cloud at the same distances as high-scoring
MC muons. The earlier measurement said exactly that (3.36 vs 3.38), but on contaminated data
and far smaller statistics.

## Stage 6 — under-modelled muon background

The handle is truth for the MC muons that *themselves* score high: zenith, energy,
multiplicity. If they form a narrow class and data holds three times more of it, that class
is under-generated.

**Kills it**: high-scoring MC muons forming no distinct class — nothing to under-generate.
Note multiplicity has already been refuted as the handle (see `multimuon_not_the_handle`).

## Stage 7 — what remains

Additional requirements applied (hit-pattern elongation — *not* an angle, see the
correction above — and distance to the neutrino manifold), survivors
counted and compared against the Stage 2 expectation. Individual inspection is feasible here:
106 events above 0.99 is a number a person can look at one by one.

## Code and artifacts

```
analysis/exp_excess/
  PLAN.md              this file
  config.yaml          cuts, exclusions with status, reference fractions, seed
  build_splits.py   →  splits table, including the excluded-run column
  check_data_quality.py → occupancy, charge, raw rate, time bursts, geometry
  build_direction.py→  direction table                                  [done]
  build_reference.py→  d_mc, d_nu, d_mu tables
  normalization.py  →  livetime and weights
  excess.ipynb         analysis; reads only
  figures/  report.md
```

Every stage writes a table keyed by `event_fk` into the same prediction DuckDBs. The notebook
computes nothing itself and cannot bypass `splits`. No parallel data store: hits are read
from the source HDF5 through the catalog at 0.63 ms/event, and the interesting sample is
thousands of events.
