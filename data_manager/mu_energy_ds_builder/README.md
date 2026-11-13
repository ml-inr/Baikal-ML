# mu_energy_ds_builder

Builds the NPY dataset for muon-energy regression: the per-event target is the muon energy
at entry into the sensitive volume of the row's own cluster.

This package is a **deliberate, self-contained copy** of the sig-noise / hit-writing part
of `data_manager/nu_classifier_ds_builder` (2026-10-21). The two datasets are kept
independent on purpose: a change to the classifier builder must not silently change the
energy dataset. If a bug is fixed in one copy, check whether the other needs it too.

## Target and selection

- Target: companion `e_at_entry[:, 1]` — the energy at entry into the cylinder
  `r <= 90 m`, `|z| <= 295 m` (the cluster itself is `r <= 60 m`, `|z| <= 265 m`).
  A muon born inside the volume uses its energy at birth.
- Exactly one muon track in the cluster (`mu_starts` difference == 1).
- Valid status `(0, 1, 4, 5)` and a finite target; statuses `2` (died before) and `3`
  (no crossing from the recorded track start) are dropped.
- Sig-noise threshold `0.8`, batch size `256`; h8s3 event cut
  (`n_sig_hits >= 8`, `n_sig_strings >= 3`).
- `nuatm` rows whose muon was born inside the target volume are excluded: the nuatm
  production does not record the vertex cascade, so their detector response differs.
- The dataset is **not** class-balanced. `muatm` is dropped from the energy task: after
  h8s3 and the single-track cut its yield is tiny (1,769 in the pilot) and strongly
  biased to high energy.

## Sig-noise source

`sig_noise_source: probs_h5` reads the per-hit probabilities and the stored
`n_sig_hits_0.8` / `n_sig_strings_0.8` counts from
`baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5`; no model and no GPU are
needed. `sig_noise_source: model` re-runs the model and remains available for a source
without stored probabilities.

The two paths were compared on the 100-part pilot with `compare_probs_source.py`
(`mu_energy_dataset_h8s3_thr0.8_pilot_probscheck/compare_with_model.json`):

- targets, `event_fk`, sign, theta, weights: identical on all common events;
- raw hit 5-vectors: identical for every event with the same selected-hit count;
- the stored probabilities differ numerically from a fresh model run (max |Δp| 0.038),
  which flips 48 of 254,745 events by 1–2 hits and drops 6 `nuatm` events at the h8s3
  boundary. This is a threshold effect, not a data mismatch. The stored file is the
  canonical SN output (same convention as the nu-classifier dataset).

## Output

Per-event arrays (aligned with each other):

```
features.npy            (n_sig_hits_total, 5) float32  — SN-filtered hits
probs.npy               (n_sig_hits_total,)   float32  — sig-noise probabilities
offsets.npy             (n_events+1,)         int64
energy.npy              (n_events,)           float32  — GeV, target
energy_status.npy       (n_events,)           int8
n_tracks.npy            (n_events,)           int32
particle_types.npy      (n_events,)           int8     — 0 muatm, 1 nuatm, 2 nue2
sign.npy                (n_events,)           float32  — fNucleonN
event_weights.npy       (n_events,)           float32  — fEventWeight
flat_weights.npy        (n_events,)           float32  — flat-selection weight
theta.npy               (n_events,)           float32  — degrees
n_sig_hits.npy          (n_events,)           int32
n_sig_strings.npy       (n_events,)           int32
h5_part_keys.npy        (n_events,)           object
h5_local_event_ids.npy  (n_events,)           int32
event_fk.npy            (n_events,)           int64    — catalog_v2.events.id, -1 if absent
dataset_info.json
selection_info.json
build_config.yaml
```

`event_fk` is looked up through `inference.shared.catalog_query.get_event_fks_mc` using
`(source='mc_merged', data_class, season, run=part_key, event_id=local_idx)`.

## Flat selection and split

The training target must be flat in `log10(E)` globally, not per class. The selection is
integrated into the build (`flat_selection` in the config): equal-width `log10(E)` bins;
abundant bins are subsampled to a common per-bin cap with weight 1; sparse bins keep all
events and get `min(cap/n, 20)`. `target_total` solves for the cap and fills the remainder
exactly, so the written pool is exactly N events.

**The dataset stores only this flat train+val population.** There are no test rows and no
split files. The test complement stays in the full h8s3 source; `selection_info.json`
records the candidate count, the flat weights and the test-complement count. Test
predictions read the full source and exclude the dataset's `event_fk` values (see
`inference/track_energy_reconstruction/`).

Train/val is a **training YAML field** (`data.train_split`, `data.split_seed`), as in the
nu-classifier. `MuEnergyNpyDataset.split()` permutes the dataset rows and slices the
aligned `flat_weights.npy`; the loader never filters.

`make_split.py` and the `select_flat.py` CLI are the pre-refactor standalone path
(full NPY dataset + `flat_indices.npy` + `split_*_indices.npy`). They remain for reading
old datasets; new builds use the integrated selection and do not produce those files.

Current full-2020 build (`mu_energy_dataset_h8s3_thr0.8_train1M`, 2026-10-24): 74.4M
rows scanned over all 2,000 nuatm + 400 nue2 parts; 2,383,377 h8s3 single-track
candidates (nuatm 507,992, nue2 1,875,385). The flat selection writes 1,000,000
train+val events (nuatm 232,896, nue2 767,104; 20.9M selected hits; ESS 33.8%;
weights 1–20). The test complement is 1,383,377 events and is not stored.

Previous full build (2026-10-21, subset of parts; preserved as
`mu_energy_dataset_h8s3_thr0.8`): 1,607,364 events (nuatm 48,137, nue2 1,559,227),
32.1M selected hits. Flat selection 1,000,000 (ESS 35.1%); split 900,119 train /
99,881 val / 607,364 test.

`weights.py` implements an earlier per-class flattening with equal class mass. It is
superseded by the integrated flat selection for this task and is kept only because the
pilot notebook uses it; do not mix the two schemes in one run.

## Running

```bash
# Smoke test (4 parts per class, stored probs, no GPU)
python -m data_manager.mu_energy_ds_builder --config \
    data_manager/mu_energy_ds_builder/test_config.yaml

# Pilot (50 parts per class)
python -m data_manager.mu_energy_ds_builder --config \
    data_manager/mu_energy_ds_builder/pilot_config.yaml

# Full 2020 build (all 2,000 nuatm + 400 nue2 parts; probs source, no GPU)
# Writes data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M
python -m data_manager.mu_energy_ds_builder --config \
    data_manager/mu_energy_ds_builder/default_config.yaml

# Verification of the probs source against a model-built dataset
python -m data_manager.mu_energy_ds_builder.compare_probs_source \
    --reference data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_pilot \
    --candidate data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_pilot_probscheck
```

The pilot dataset contains `dataset_weights.ipynb`, which replays the selection and shows
the weight distribution and the target spectrum before and after weighting.
