# Universal angle, track-line and uncertainty release (photon-v3)

This is the selected **single-model, CPU-capable** release. The signal/noise
classifier is a separate upstream model. The angle/line checkpoint itself
contains the five-layer, eight-head Set Transformer (`hidden=256`), one
direction head, a point-on-track head, a two-output uncertainty head, and a
small frozen photon-time-residual correction. It is not an ensemble.

## Files

| File | Purpose |
| --- | --- |
| `angle_track_uncertainty.pt` | All angle/line/uncertainty model weights, normalization and calibration |
| `signal_noise.ckpt` | Frozen hit signal/noise classifier |
| `classifier_norm.h5` | Five-feature normalization required by the classifier |
| `angle_norm.h5` | Five-feature normalization for the selected-hit HDF5 |
| `run_release_v3.sh` | Raw HDF5 to both prediction CSVs on CPU |
| `independent_2019_f_metrics.json` | Independent nue2 holdout metrics, slices and uncertainty deciles |
| `reco_2020_comparison.json` | ID-paired comparison with standard ROOT reconstruction |

The SHA256 of `angle_track_uncertainty.pt` is
`7cb8f77cfdb2a7cf021d090be5246b83f8a94c7513cc24c793ab019c67686f41`.
The SHA256 of `signal_noise.ckpt` is
`be9cc39b93dd6a57a69233faf6755278e3f202c56de3de7d1cc8d2d4a8b1e93d`.
The bundle was copied from
`/home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/photon_uncertainty_recalibration_v1/universal_angle_track_uncertainty_photon_crossseason_v3.pt`
on cluster63. The `v3` name refers to the frozen photon-corrected release;
the angle/point/uncertainty backbone itself descends from the universal `v4`
checkpoint. No checkpoint weights were changed while making this release.

## Run on CPU

From the repository root, use the Python environment with PyTorch, NumPy,
SciPy, pandas and h5py installed (the tested cluster63 interpreter was
`/home/plotnikovgp/tmp/ENTER/bin/python3.11`):

```bash
PYTHON_BIN=/home/plotnikovgp/tmp/ENTER/bin/python3.11 \
  bash pytorch_training/scripts/angle_track_uncertainty/release_v3/run_release_v3.sh \
  /absolute/path/to/raw_events.h5 /absolute/path/to/new_output_directory
```

The source must use the project's raw event HDF5 schema and contain a `test`
split. The script first selects hits with classifier probability `p>=0.50`,
retaining events with at least eight hits on at least two distinct strings.
It then writes:

- `predictions_calibrated_p070.csv`: one pass with `p>=0.70` and calibrated
  angular/line containment radii;
- `predictions_best_quality_gate200.csv`: start at `p>=0.70`; if its predicted
  zenith is below 90 degrees, add eligible `p>=0.50` hits only when their
  maximum centred photon-time residual is below 200 ns, then rerun the same
  model. This is the best fixed quality setting tested on independent 2019-F.

The two-pass option may be slower but still uses **one model**. It does not
fit on the evaluation event. For rows whose input changes on the second pass,
the CSV leaves calibrated radii blank: its raw `sigma_angle_deg` and
`sigma_line_m` remain available, but the existing p>=0.70 calibration cannot
be claimed for those rows. Unchanged rows have calibrated radii. If calibrated
uncertainties are essential for *every* row, use the p>=0.70 file until a
separate conditional calibration has been validated.

The output is one reconstructed **infinite line** per event, not a finite
track segment: `theta_deg`, `phi_deg` and direction components define its
orientation; `anchor_x_m`, `anchor_y_m`, `anchor_z_m` give a point on that line
in the cluster-local coordinate frame. The point loss is the transverse
distance to the true line; the along-line coordinate is not identifiable.
Angular error means the 3D separation of the predicted and true directions.
Line error means the shortest transverse distance from the predicted anchor
to the true infinite line, *not* endpoint or path-length error. The angular
and line 68%/95% radii are separately marginally calibrated, not a joint
track tube.

The model saw `nuatm`, `nue2`, and only **single-track** `muatm` simulations in
its direction/point training. Multi-track `muatm` truth does not define this
single-line target and was excluded. No experimental ground-truth line exists;
do not interpret experimental uncertainty outputs as validated coverage.

### Already selected hits

For a prepared HDF5 containing both p>=0.50 and p>=0.70 hits, run the model
without the classifier:

```bash
python3 pytorch_training/scripts/angle_track_uncertainty/infer_rethresholded_for_audit.py \
  --bundle pytorch_training/scripts/angle_track_uncertainty/release_v3/angle_track_uncertainty.pt \
  --data /absolute/path/to/selected_p050.h5 --split test --threshold 0.70 \
  --conditional-low-threshold 0.50 --gate-theta-max-deg 90 \
  --max-added-photon-residual-ns 200 \
  --output /absolute/path/to/predictions.csv --device cpu --threads 8
```

The 32-event end-to-end CPU smoke test reproduced the exact selected event
IDs/hits and second-pass decisions of the prior GPU run; 29 events passed
the p>=0.70 cut, and the maximum direction difference was 0.011 degrees.

## Independent evaluation: 2019-F nue2

This is a frozen evaluation after the photon gate was chosen on 2019-A/B.
The 51,580 accepted 2019-F events have disjoint generated primaries from
those tuning parts and the 2020 training source. All rows satisfy the **NN
input cut**: at least 8 predicted-signal hits with p>=0.70 on at least 2
strings. There is no standard-reconstruction cut here. The selected-hit span
below is the largest pairwise 3D separation between the NN-selected hits,
not a fitted track length or the GT-hit span.

| Method | Direction q50/q68/q95 (deg) | Line q50/q68/q95 (m) |
| --- | ---: | ---: |
| p>=0.70 one pass | 4.109 / 7.737 / 40.340 | 3.929 / 6.308 / 19.839 |
| Same model, fixed 200 ns gate | **3.730 / 6.931 / 37.734** | **3.736 / 6.024 / 19.784** |

The second pass changed 19,873 of 51,580 rows. A bootstrap over source
parts gives 95% intervals for gate minus one-pass q50/q68 of
[-0.419,-0.342]/[-0.861,-0.745] degrees and
[-0.212,-0.168]/[-0.331,-0.245] m, respectively. Lower is better.

| p>=0.70 selected hits | N | Direction q68 one pass -> gate (deg) |
| --- | ---: | ---: |
| 8-11 | 16,703 | 12.051 -> 10.473 |
| 12-15 | 10,404 | 9.479 -> 8.427 |
| 16-23 | 10,500 | 7.331 -> 6.690 |
| 24-39 | 7,503 | 4.723 -> 4.490 |
| >=40 | 6,470 | 3.095 -> 3.017 |

| p>=0.70 selected-hit span | N | Direction q68 one pass -> gate (deg) |
| --- | ---: | ---: |
| <100 m | 12,262 | 22.108 -> 19.505 |
| 100-200 m | 28,802 | 6.603 -> 5.987 |
| 200-300 m | 7,138 | 2.730 -> 2.513 |
| 300-450 m | 2,598 | 1.904 -> 1.711 |
| >=450 m | 780 | 1.659 -> 1.583 |

Further slices by true zenith, selected-hit purity, and track-hit fraction,
plus exact definitions and event-level source paths, are in
[`independent_2019_f_metrics.json`](independent_2019_f_metrics.json).
Track-heavy and short-span events have substantially worse angular tails.

## Does the uncertainty predictor work?

On **all** 2019-F p>=0.70 predictions, nominal 68%/95% coverage was
68.23%/95.07% for angle and 67.68%/95.03% for transverse line distance.
The calibrated radius ranks events by difficulty: the lowest predicted
angle-sigma decile had observed direction q68=1.521 degrees, the highest
decile q68=39.185 degrees. Decile coverage is not uniform (angle 68% coverage
65.30% in the easiest decile and 67.66% in the hardest; line 75.38% and
65.82%, respectively), so **marginal** calibration is not a conditional
guarantee. On the 31,707 unchanged gate rows, the p>=0.70 radii still cover
69.02%/95.20% of angular errors and 68.27%/95.58% of line errors. The
changed 19,873 gate rows have no validated conditional radii; they are not
included in those gate coverage percentages. All deciles and sample sizes
are recorded in the JSON report.

## Paired comparison with standard reconstruction: 2020 reco MC

Predictions and standard ROOT reconstruction are matched by event ID, with
the same true particle direction and 3D angular-error definition. The NN
input cut is **p>=0.70, at least 8 selected hits on at least 2 strings**.
The ROOT-derived preliminary cut contains the following **10 observable
conditions** from the original 13-condition analysis selection:

1. `fNHits>=8` and `fNStrings>=2`;
2. `fCovMatrStat==3`;
3. `fScfMaxTheta<1.7`;
4. `fEvCenterZ<220` and `fZDist>70`;
5. `log10(fTimeTerm/(fNHits-5))<1.1`;
6. `fNTriplets/fNHits>0.1`;
7. `fThetaErr<2.5` and `fNCalls<350`.

These are ten boolean conditions despite seven numbered groups. The other
three cuts require BRecoMuonBDTEval methods absent from the available BARS
build (`GetLogPHit_30_min`, `GetTRes_30_med`, etc.). **This 10/13 selection
is a superset, not the exact full analysis cut.** A second row adds the older
`impose_reco_cuts=True` condition:
`fCovMatrixStatus==3`, `fInitRecDelta<20`, and
`fFuncValue/(fNHits-5)<30`. The spelling of `fCovMatrStat` vs
`fCovMatrixStatus` reflects the two data products.

| Paired cut | N | Standard reco q50/q68/q95 (deg) | NN p>=0.70 q50/q68/q95 | NN gate200 q50/q68/q95 |
| --- | ---: | ---: | ---: | ---: |
| ROOT-observable 10/13 | 42,003 | 1.983 / 5.576 / 61.315 | 1.954 / 3.360 / 13.334 | **1.823 / 3.072 / 12.847** |
| Same + older `impose_reco_cuts` | 41,566 | 1.946 / 5.380 / 59.849 | 1.939 / 3.330 / 13.308 | **1.809 / 3.046 / 12.800** |

On 10/13, the paired-bootstrap 95% CI for NN gate minus standard reco is
[-0.201,-0.124] degrees for q50 and [-2.677,-2.348] degrees for q68.
However, on the stricter, already-well-reconstructed subset
`fNHits>=20 && fNTriplets>=12` (also 10/13, N=5,679), standard reco wins
the centre: q50/q68=0.622/0.989 degrees versus NN=0.832/1.223 degrees.
The NN mainly improves the broad tail, not every event. Some 2020 generated
primaries overlap the network's 2020 training source; **these reco figures
are descriptive paired diagnostics, not a primary-independent test**. The
independent 2019-F table above is the generalization check.

The machine-readable reco comparison is
[`reco_2020_comparison.json`](reco_2020_comparison.json). The exact source
cut implementation is
[`analyze_reco_central_gap.py`](../analyze_reco_central_gap.py)
(`root_ten_cut`) and the older cut definition is in
[`evaluate_reco_paired_current.py`](../evaluate_reco_paired_current.py).

## MC versus experiment

The experimental sample has no truth and is predominantly atmospheric muons.
For the 4,924 experimental events satisfying the 2-string/8-hit selection,
the fixed gate changed only 18 predictions. After matching MC `muatm` to
experiment in selected-hit count, string count and z spread, the predicted
theta-distribution Wasserstein-1 distance was 0.375 degrees for one pass and
0.369 degrees for the gate. Without z matching it was about 3.06 degrees.
This checks distributional similarity, **not** reconstruction accuracy on
experimental events. Multi-track `muatm` experimental events are outside the
validated single-line target.

## Limitations and reproducibility

- A new detector season or new noise classifier requires a fresh signal
  selection and uncertainty-calibration audit; the reported coverage is for
  the specified predicted-hit pipeline.
- The gate's changed-input rows need separate, honest conditional calibration
  before publishing 68%/95% radii for them.
- The 2020 reco subset is neither the exact 13/13 historical cut nor
  primary-disjoint from training. Do not use its q50 as the independent test
  score or claim superiority on its high-quality-reco subset.
- The point estimate is one line. No track endpoints, finite length, or
  multiple-particle decomposition are predicted.

The full experiment history, plots and precursor checkpoints are documented
in [`README_universal.md`](../README_universal.md). The release metric JSONs
and the source scripts above provide the exact definitions behind the numbers
here.
