# Experimental data quality — Stage 1 findings

Measurements on `exp_full.h5` (29 runs, clusters 2–7, 133M events) scored with
`260816_2250_..._FIXED_sn256`. Every number here is reproducible from the scripts in this
directory; the status vocabulary is the one in `doc/claims_log.md`.

## 1. Anomalous channels — `verified`, mechanism measured

Four channels on one string of cluster 2 (222, 224, 225, 227 — all `channel // 36 == 6`)
emit charges up to **3,169.8 p.e.** against a median hit of 0.97. Between **35.8% and 77.6%**
of their hits exceed 100 p.e., where a normal channel sits at **0.0066%** — a factor of
5,400 to 11,700.

They fire in exactly two runs, `part_s2020_c02_r0020` and `part_s2020_c02_r0249`, and in no
other run of cluster 2. No other cluster holds an anomalous channel at all.

**Why it reaches the analysis.** No charge limit exists anywhere before the classifier: the
ROOT converter cuts only empty events and hits with `|t| > threshold`, and the sig-noise
network applies mean/std normalisation with no clipping — measured directly, a 3,169.8 p.e.
hit arrives as **863 standard deviations**. Hits above 1,000 p.e. are called signal **100%**
of the time (83.8% for 100–1,000 p.e., against 0.6–3.9% for ordinary hits). The event gains
signal hits, passes h8s3, and reaches the classifier as a bright compact event — where the
amplitude is finally clipped at 100 p.e., erasing the evidence.

**Effect, measured within those two runs:**

| | h8s3 | score > 0.8 | rate | median q_total |
|---|---|---|---|---|
| events touching an anomalous channel | 5,304 | 38 | **0.7164%** | 129.7 |
| events not touching one, same runs | 167,731 | 138 | 0.0823% | 54.1 |
| the other 27 runs | 3,180,681 | 3,044 | 0.0957% | — |

**Excluded by run, not by event.** Four channels behaving that way is a symptom; keeping the
rest of a run assumes the remainder is healthy, and that is untested.

**Cost: small.** The headline excess moves from 2.87x to 2.86x. The defect is real,
explained, and does not account for the excess.

## 2. The cluster 1 and 4 exclusion — `refuted`

Recorded in `temp_plot_pred.py` as "channel 71 broken across entire cluster 4", with no
measurement attached. It does not reproduce on this data:

| | channel 71 in c04 |
|---|---|
| occupancy vs cluster median | 0.889 (other clusters span 0.799–1.449) |
| mean charge vs cluster median | 1.019 |
| anomalous channels in c04 | **none** |

Cluster 4 is the cleanest cluster by both measures. Cluster 1 is absent from `exp_full`
entirely and could not be tested. Most likely the list came from `exp_reco` — that script
reads it — or used global rather than per-cluster channel numbering. **Not applied**, kept in
`config.yaml` with status `refuted` so it is not reinstated from the old note.

## 3. Time bursts — `refuted`

The strongest candidate by expected signature: bioluminescence produces bright, compact,
track-less events in flashes lasting seconds to minutes — the observed portrait exactly. It
would have been invisible to earlier work, which aggregated whole 22-hour runs.

Tested on the interval distribution between consecutive high-score events within each run,
excluding the two bad runs:

| interval shorter than | observed | Poisson | ratio |
|---|---|---|---|
| 0.1% of the mean | 0.099% | 0.100% | 1.0 |
| 1% of the mean | 0.895% | 0.995% | 0.9 |
| 10% of the mean | 10.31% | 9.52% | 1.1 |

Median interval 465 s against the Poisson expectation of 492 s. The high-score events arrive
as a Poisson process; there is no clustering to explain. The same holds for the full h8s3
sample.

## 4. Trigger thresholds across clusters — no difference found

| cluster | q min | q 1% | q 5% | median q | hits/event |
|---|---|---|---|---|---|
| c02 | 0.000 | 0.201 | 0.332 | 0.973 | 50.4 |
| c03 | 0.024 | 0.168 | 0.323 | 0.971 | 64.5 |
| c04 | 0.039 | 0.155 | 0.311 | 0.993 | 50.2 |
| c05 | 0.000 | 0.152 | 0.310 | 0.991 | 66.6 |
| c06 | 0.032 | 0.159 | 0.311 | 0.970 | 66.4 |
| c07 | 0.038 | 0.122 | 0.290 | 0.967 | 68.6 |

Thresholds agree closely — the 1st percentile spans 0.122 to 0.201, the median 0.967 to
0.993. Correlation of the high-score rate with the threshold is +0.28: nothing.

**But raw hits per event differ by 37%** — 50 in c02 and c04 against 65–69 elsewhere — and
that anticorrelates with the high-score rate at **−0.65** on six points. Suggestive, not
established: six clusters cannot separate a real dependence from coincidence, and the
mechanism (fewer hits per event making an event look more compact, hence more like what the
classifier scores high) is a hypothesis, not a measurement.

## 5. String geometry — real, unquantified

Strings drift in the current, and geometry is tracked per run. One channel of cluster 4
moves **2.3 m** in x between runs r0108 and r0130. Within a 22-hour run a single position is
used, so a sway of order metres is baked into `x, y, z` — which the sig-noise network takes
as input features, and which `r_vert` and `dt/dz` are computed from. Not yet quantified.


## 6. Noisy channels by raw rate — nothing found

Occupancy in sections 1–2 was measured on *signal* hits, which would miss a channel whose
noise is correctly rejected. Measured on raw hits instead: no channel anywhere exceeds five
times its cluster median, and only one (c07 channel 271, 3.6x) exceeds three. Noise rates are
uniform.

## 7. Per-channel time offsets — **geometry, not calibration** (`verified`)

> **This section was rewritten on 2026-08-25.** It previously reported per-channel offsets as
> a possible instrumental defect with "cause not separable", and that reading was carried into
> the group-structure plan as "present in data, absent in MC". Both are wrong. The old numbers
> are kept below the line because the argument that replaced them matters more than the table.

**Measurement** (`measure_channel_offsets.py`, identical code on both datasets): for every
signal hit (sig-noise prob > 0.8), time minus the median signal-hit time of its own event;
averaged per channel; channels with at least 200 hits.

| | raw std across channels | std after removing the module-position profile | explained | profile correlation with MC |
|---|---|---|---|---|
| MC muatm | 98.9 ns | 18.3 ns | 97% | — |
| exp c02 | 98.1 | 9.0 | 99% | +0.989 |
| exp c03 | 99.0 | 9.9 | 99% | +0.997 |
| exp c04 | 94.9 | 10.8 | 99% | +0.998 |
| exp c05 | 96.9 | 10.1 | 99% | +0.992 |
| exp c06 | 97.1 | 10.3 | 99% | +0.997 |
| exp c07 | 99.6 | 9.8 | 99% | +0.998 |

The "profile" is the mean offset as a function of `channel % 36`, the module's position in its
string. It runs monotonically from about +247 ns at position 0 to −147 ns at position 33, and
it is the same curve in simulation and in data.

**Two independent reasons this is geometry.** The offset is a monotone function of depth within
the string, which is what down-going muons must produce — one end of a string sees the light
first. And **the simulation reproduces the curve at r = 0.989…0.998**, while having no
calibration errors at all by construction. Anything MC reproduces cannot be a calibration
fault. The second reason is the anchor: it is external to the experimental data.

**What is left for the instrument** is the residual after the profile: **9–11 ns** in every
cluster, about 2 m of light travel. That is small next to the structures any timing feature
resolves, and it is not a reason to distrust `dt/dz`, `t_span` or `track_likeness`.

Three caveats, stated rather than buried. MC's residual (18.3 ns) is larger than any
cluster's, which is most likely statistics — 59,374 events spread over 1,631 channels against
120,000+ over ~270. **MC's raw std is not a stable number**: on four parts instead of six it
reads 72.1 ns rather than 98.9, because fewer channels clear the 200-hit cut and the survivors
are the better-populated ones. The residual (17.7 vs 18.3) and every correlation
(+0.983…+0.995) are stable, and those are what the argument rests on. And channel 224 of
cluster 2 at +676 ns is not covered here: it fires only in the two runs already excluded in §1.

---

**Superseded table, kept for the record.** The original measurement reported mean hit time
relative to the event median per channel as: c02 −14.7 (spread 43.6), c03 −1.7 (21.5), c04
+20.0 (27.9), c05 +19.7 (16.7), c06 +7.4 (29.4), c07 +6.0 (22.0); 61 channels beyond ±50 ns,
grouped by string. Those numbers do not reconcile with the ones above in either magnitude or
sign, and the code that produced them cannot be located. They are marked superseded rather
than explained. The likeliest cause of the smaller spread is that they were computed over raw
hits, whose noise is spread over ~4,900 ns and would dilute any per-channel mean toward zero —
but that is a guess, not a finding.

## Still unexplained

The high-score fraction varies **1.5x across clusters** — c04 at 1.25 of the mean, c07 at
0.81, runs inside each cluster agreeing among themselves. Not explained by dead channels
(c02 has 24 and sits at the mean; c04 has none and sits highest), anomalous channels, trigger
thresholds, or the medians of hit count, charge and sig-noise probability, which agree across
clusters to within a few percent. The hits-per-event anticorrelation is the only lead.

## Not reachable with what we hold

DAQ dead time and buffer overflows, water optical properties, calibration provenance, event
splitting at readout boundaries. These need detector monitoring data.
