# `features.duckdb` — the feature table

One row per event. Produced by `build_features.py`, which calls `features.py:event_features`
on the sig-noise-filtered hits read from HDF5. Checked by `test_features.py`.

Written for a reader who has the table and not the code: what each column means, what value
physics expects where it expects one, when it is `NULL`, and what it must not be used for.

## 1. What the table is

```sql
CREATE TABLE features (
    source     VARCHAR,   -- 'mc' | 'exp'
    event_fk   BIGINT,    -- the catalog's event id; unique within a source, NOT across
    group_id   TINYINT,   -- 1..8, see below
    part_key   VARCHAR,   -- HDF5 part; for exp this is the run
    cluster    VARCHAR,   -- 'c02'..'c07' for exp, NULL for MC
    ...52 DOUBLE feature columns...,
    PRIMARY KEY (source, event_fk)
)
```

**The key is `(source, event_fk)`, not `event_fk`.** MC and experimental ids are drawn from
different catalogs and collide. Any join must carry `source`.

`part_key` is not decoration: every train/test split in the analysis is made on it, because
neighbouring events in one run share a detector state and an event-level split leaks that
across the boundary.

## 2. The eight groups

Selection is the project's h8s3 quality cut — `n_sn_hits >= 8 AND n_sn_strings >= 3` — with the
verified exclusions applied: MC drops events the model trained on (`used_for_labels`),
experimental data drops runs `part_s2020_c02_r0020` and `part_s2020_c02_r0249`. The score
threshold is ξ = 0.8.

| group | class | score | in the full sample | in this table | how |
|---|---|---|---|---|---|
| 1 | nuatm | < ξ | 5,162 | all | exhaustive |
| 2 | nuatm | ≥ ξ | 525,904 | 100,000 | quota |
| 3 | nue2 | < ξ | 12,499 | all | exhaustive |
| 4 | nue2 | ≥ ξ | 1,360,695 | 100,000 | quota |
| 5 | muatm | < ξ | 23,159,967 | 200,000 | quota |
| 6 | muatm | ≥ ξ | 7,751 | all | exhaustive |
| 7 | exp | < ξ | 3,177,637 | 200,000 | quota |
| 8 | exp | ≥ ξ | 3,044 | all | exhaustive |

**Quota rows are not a clean random sample of the whole class**, and this matters enough to
state twice. They are drawn `ORDER BY hash(event_fk) LIMIT n` — deterministic and unbiased
*within the parts read* — but only from the parts the exhaustive groups already force open,
because reading a part costs the same whether one event or a thousand is wanted from it. For
muatm that is 5,373 of 10,100 parts.

Measured, so it need not be assumed: on the experimental side the restriction is **empty** —
all 27 runs contain at least one group-8 event, so groups 7 and 8 come from 100% of the
3,180,681 quality events. On the MC side it covers 55% of quality events and is mildly biased
toward larger parts (median 2,284 quality events per selected part against 2,174), but group 5
itself is indistinguishable between selected and unselected parts: hits 13.0 vs 13.0, charge
per hit 3.57 vs 3.59, `q_total` 48.0 vs 48.2, `z_span` 150.4 vs 150.4, mean score 0.00447 vs
0.00441.

**Counts in this table are not rates.** The muon simulation carries `event_weight` identically
1 and no generated livetime is recorded anywhere, so nothing here normalises to a rate. Ratios
between groups are meaningful only within a class.

## 3. Constants, and where each comes from

| | value | source |
|---|---|---|
| `C_VAC` | 0.299792458 m/ns | definition |
| `N_WATER` | 1.37 | `doc/data_format.md` |
| `K_CHERENKOV` | √(n²−1) = 0.9364 | derived |
| `V_LIGHT` | c/n = 0.21882 m/ns | derived |
| `Q_CLIP` | 100 p.e. | the bound the classifier itself applies |
| `STRING_DIVISOR` | 36 | `nu_classifier_ds_builder/io.py:_count_sig_hits_strings` |
| `ON_TRACK_NS` | 25 ns | a third of the measured 15.0 m module step, which is 68.6 ns of light in water |
| sig-noise threshold | 0.8, batch 256 | pipeline convention |

Every tolerance is **fixed a priori**. None was tuned by looking at the result.

## 4. The track fit

The backbone. For a relativistic track through a point `p0` **on the line** with direction `u`,
Cherenkov light reaches a module at `r` at

```
t_pred = t0 + [ s + K·d ] / c ,   s = (r−p0)·u ,   d = |(r−p0) − s·u|
```

Two things about how it is fitted, both of which change the numbers:

**Direction comes from timing, never from the shape of the hit cloud.** Modules sit on vertical
strings, so a PCA axis of the positions is near-vertical whatever the track did.

**`p0` must lie on the track and the hit centroid does not.** Shifting `p0` along `u` is
absorbed by `t0`; shifting it perpendicular changes every `d`. The fit therefore scans 500
directions over the sphere, then five rounds alternating perpendicular offset and local
direction at shrinking scale — (60 m, 10°), (20 m, 3°), (7 m, 1°), (2.5 m, 0.3°), (1 m, 0.1°).
This is not decoration: on a track built from the formula itself, a single 10° pass left 4.05°
of angle error and 28.5 ns of residual, so `fit_rms` would have measured the fitter's grid
rather than the event. Five rounds give 0.68–1.87° and under 2.3 ns.

| column | meaning | units | expected |
|---|---|---|---|
| `fit_rms` | RMS time residual at the best fit | ns | ≈ 0 for a clean single track |
| `fit_contrast` | (median RMS over the 500-direction grid − best) / median | — | **≥ 0.25 means a direction was found**; below that `fit_zenith` carries no information. Calibrated on scrambled times, not chosen |
| `fit_zenith` | 0 = straight up, 90 = horizon, 180 = straight down | deg | interpret only above the contrast cut |
| `fit_azimuth` | atan2(u_y, u_x) | deg | as above |
| `frac_on_track` | hits within `ON_TRACK_NS` of the prediction | — | 1 for a clean track |
| `q_offtrack_frac` | charge of off-track hits / total charge | — | 0 for a clean track; rises for a bundle or a cascade |
| `q_weighted_residual` | charge-weighted mean residual | ns | sign says whether early bright or late dim light dominates |

## 5. Dimensionless quantities, each with a predicted value

| column | definition | expected |
|---|---|---|
| `slowness` | `t_span / (extent_m / V_LIGHT)` | ≈ 1 for a relativistic track, and **below 1 for a down-going muon, which outruns its own light** (measured 0.73 on a vertical synthetic track). ≫ 1 means too slow for its own size: scattering, several sources, or a mis-associated hit |
| `extent_m` | largest pairwise hit separation | m |
| `q_vs_d_slope` | slope of `log q` against perpendicular distance to the fitted track | a **proxy** for light attenuation in water — *not* the attenuation length: the sig-noise filter keeps bright hits preferentially, so at large `d` the survivors are biased bright and the slope is flattened. Compare only between samples processed identically |
| `q_vs_d_r2` | how well that trend holds | — |
| `string_slope_spread` | spread of the per-string `t`-vs-`z` slopes (strings with ≥ 3 hits) | ≈ 0 for one track: every string must agree. Large for a bundle or a cascade |
| `n_strings_fitted` | strings that spread rests on | doubles as the indicator for the `NULL` above |

`slowness` is partly collinear with `t_span / z_span`, since most events are vertically
extended. Recorded rather than hidden.

## 6. The repeated-hit block

One module can register more than one hit. In experimental data this happens in 21.08% of
events against 3.45% of simulated muons, and unlike in simulation it carries no association
with event brightness — the largest data/simulation discrepancy in the project, and almost
certainly instrumental.

| column | meaning | why |
|---|---|---|
| `dt_repeat_median` | median time between successive hits on one module | **the mechanism sets the scale**: ion afterpulses sit at hundreds of ns to µs, scattered light at tens of ns |
| `q_repeat_over_first` | charge of repeat hits over charge of first hits | an afterpulse is fainter than its primary; a second photon need not be |

These two, together with `hits_per_module`, `hits_per_string_max` and `hits_per_string_mean`,
are the **known instrumental axes**. Any analysis that finds a data/simulation difference must
be repeated with them removed, or it cannot tell an instrument from a physical population.

## 7. Marginals

**Multiplicity and geometry** — `n_hits`, `n_modules`, `n_strings`, `hits_per_module`,
`hits_per_string_max`, `hits_per_string_mean`, `z_span`, `xy_span`, `z_c`, `r_cyl` (centroid
distance from the cluster axis), `z_first`, `z_last`, `dz_signed` (`z_last − z_first`, ordered
by time), `elongation` and `planarity` (normalised hit-cloud PCA eigenvalues), `r_vert`.

> `r_vert` = `σ_z / √(σ_x² + σ_y²)` is kept **only** for comparability with the earlier excess
> study. It is an aspect ratio of the hit cloud and **not an angle**; reading it as one was the
> largest interpretive error of that study. `fit_zenith` supersedes it.

**Timing** — `t_span`, `t_span_core` (5th to 95th percentile), `t_std`, `dtdz_slope`,
`track_likeness` (r² of time against depth — the earlier study's statistic, kept for
comparability; it ignores x and y entirely, which is why the track fit replaced it),
`causality_violation_frac`, `spearman_tz`.

**Charge** — `q_total`, `q_mean`, `q_std`, `q_median`, `q_iqr`, `q_max`, `q_frac_max`,
`frac_q_below_2` (single-photoelectron-like fraction), `q_asymmetry` (early half against late
half by time), `centroid_shift` (charge-weighted centroid minus plain centroid, m). All charges
clipped at `Q_CLIP` first.

**Sig-noise context, secondary and flagged** — `prob_mean`, `prob_min`, `n_raw_hits`,
`survival_frac` (`n_hits / n_raw_hits`). These are another network's output. They are kept out
of the primary feature set for the same reason the nu-classifier's output is excluded, and used
only as a robustness check.

## 8. `NULL` means "undefined", not "missing"

Two columns are undefined for a large share of events, and the reason is structural rather than
a failure:

Measured on 328,260 rows of the table, not estimated:

| column | `NULL` when | all rows | group 6 | group 8 |
|---|---|---|---|---|
| `dt_repeat_median` | no module fired twice | 60.1% | **84.5%** | **55.9%** |
| `string_slope_spread` | fewer than two strings have ≥ 3 hits | 12.4% | 18.7% | 17.5% |

Every other column is defined for every event.

**`q_repeat_over_first` is never `NULL`** — a point worth its own line, because it looks like it
should be. With no repeated hits the numerator is zero and the denominator is not, so the value
is a genuine 0.0 rather than undefined. It and `dt_repeat_median` therefore behave differently
in a model: one is always usable, the other needs the sentinel.

**The group difference in the first row is not noise, it is the phenomenon.** A missing
`dt_repeat_median` means the event had no repeated hits at all: 84.5% of misread simulated
muons against 55.9% of accepted experimental events. Any model handed this column can separate
the two groups on missingness alone, which is why §6's ablation exists.

Do **not** impute these with a median. "This event had no repeated hits" is information. The
analysis fills them with a sentinel (`-999`) so a tree can split on the condition explicitly;
`n_strings_fitted` serves as the companion indicator for the second.

## 9. What is deliberately not here

- **Anything derived from the nu-classifier** — no score, no embedding, no `distance_to_nu`
  or `distance_to_mu`. The classifier defines the groups; letting it define the features too
  would make every answer circular.
- **Any MC truth.** Truth is used once, to *check* `fit_zenith` against `truth.zenith_deg`, and
  never as an input.
- **Run, cluster and time as model inputs.** `part_key` and `cluster` are stored so the analysis
  can test whether a finding concentrates in particular runs — the check that separates an
  instrumental cause from a physical one — but a model given them would memorise runs.
- **`hits_per_metre`** and anything else resting on the geometry, which drifts up to 2.3 m
  between runs while MC uses a single geometry.

## 10. Reproducibility

Sampling is `ORDER BY hash(event_fk) LIMIT n`, never `USING SAMPLE`: a reservoir sample depends
on the order a parallel scan returns rows in and differs between runs even with a seed.

Tasks are keyed on `(source, h5_group, part_key)`. **Part names are not unique across MC data
classes** — `part_1271`, `part_1216` and others exist in two classes at once — and keying on
`(source, part_key)` alone silently merges two different parts and applies one class's
`local_idx` to the other's arrays. That produced `IndexError: index 24036 is out of bounds for
axis 0 with size 23554` and killed a full run; the partial output was discarded rather than
resumed, because the affected rows are not identifiable after the fact.
