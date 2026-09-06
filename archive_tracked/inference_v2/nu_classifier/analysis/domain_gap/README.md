# domain_gap — MC↔exp domain mismatch analysis

Marimo notebooks probing the gap between MC muons and real experimental data in
the nu-classifier output. Motivation: on raw triggered data the genuine neutrino
fraction is ~10⁻⁶ (atmospheric μ:ν ≈ 10⁶), so the ~0.1% high-score exp fraction
is dominated by **misclassified atmospheric muons** — a domain-shift artefact,
not a neutrino signal. These notebooks quantify and dissect that shift.

## Shared helper

`_common.py` — loaders + curve utilities, reused by all notebooks:
- `load_exp_scores(...)` — scored exp events (score, n_sn_hits, n_sn_strings, cluster).
- `load_mc_muatm_scores(..., exclude_training=True)` — scored MC muatm, with
  nu-classifier **training events excluded** via the NPY dataset back-links
  (`h5_part_keys` / `h5_local_event_ids` / `labels`), so curves reflect
  generalisation not memorisation.
- `survival_curve`, `gap_table`, `add_curve`, `default_thresholds`.

## Notebooks

| file | subtask |
|---|---|
| `suppression_curves.py` | Survival `N(score>ξ)/N` vs ξ, exp vs out-of-training MC muon. Overlay + split by energy proxy (`n_sn_hits`) and by cluster. The vertical gap = domain mismatch vs threshold. |
| `domain_classifier.py` | Train a gradient-boosting classifier (HistGradientBoosting) on per-event tabular features (15 base stats + 8 dimensionless ratios, charge clipped at Q=100 PE as the model does) to separate exp-hi vs MC muon. Reports CV AUC, permutation importance, **single-feature (univariate) AUC**, and top-feature distributions. |

`_features.py` provides the feature extractor: resolves event_fk → HDF5 location,
reads only the needed hit slices (never whole multi-GB parts), runs the sig-noise
model, and computes signal-hit stats + `add_derived_features` ratios.

## Run

```bash
conda activate baikal25
marimo edit inference_v2/nu_classifier/analysis/domain_gap/suppression_curves.py
# headless render:
marimo export html inference_v2/nu_classifier/analysis/domain_gap/suppression_curves.py -o /tmp/out.html
```

## Key result (260508_1724 checkpoint, exp_full vs out-of-train MC muatm)

| ξ | exp survive | MC-μ survive | gap (exp/μ) | MC suppression |
|---|---|---|---|---|
| 0.80 | 1.44e-3 | 3.54e-4 | 4.1× | 2.8k× |
| 0.95 | 2.03e-4 | 2.7e-5 | 7.6× | 38k× |
| 0.99 | 6.0e-5 | 2.0e-6 | 31× | 5.1e5× |

The gap **grows with threshold** (4×→31×): real muons increasingly leak into the
high-score region relative to MC. Note the nu-classifier **alone** reaches
~5×10⁵ muon suppression at ξ=0.99 (not 10⁶ at 0.95 — that likely needs the
prefilter stage). MC-muon statistics floor ≈ 1/3.1M.
