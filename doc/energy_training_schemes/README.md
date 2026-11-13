# Energy reconstruction — explanatory figures

Pilot illustrations for the muon-energy training story.  Some are **schematic**
explanatory drawings, not to scale; the `*_real` figures are displays of actual
MC events and are labelled as such.

## Files

- `build_figures.py` — regenerates all SVG/PNG figures below.
- `build_deck.py` — builds the native, fully editable schematic target slide
  (`target_definition_3d_native.pptx`); kept as a fallback, not the main visual.
- `build_real_event_deck.py` — builds the native, fully editable real-event slide
  (`real_event_display_native.pptx`).
- `build_real_event.py` — renders an actual MC event with the true track, target
  cylinder, color = hit time and size = charge (`figures/real_event_display.*`).
- `build_event_types_real.py` — builds the real-data accepted/rejected event
  contact sheet (`figures/event_types_real.*`).
- `build_event_types_real_deck.py` — builds the native, fully editable
  event-types slide (`event_types_real_native.pptx`) from the same six events.
- `build_energy_spectrum.py` — compares the weighted training spectrum with the
  natural MC candidate spectrum before flat selection
  (`figures/energy_spectrum_train_vs_real.*`).
- `build_full_deck.py` — assembles `energy_training_schemes.pptx` from the real
  event slide, the native event-types slide, and data/config-derived native
  diagrams.
- `build_acg_quality_deck.py` — builds `acg_quality_deck.pptx`: the working-model
  quality, DANN/uncertainty and sigma slides, plus backup slides for learning
  curves, the non-neural HGB baseline, training configuration, AdamW/sigma/DANN
  choices, and per-class spectra/statistics.
- `figures/target_definition_3d.svg/.png` — 3D schematic fallback of the target energy:
  cluster, track, target cylinder and entry point; not the main visual.
- `figures/target_definition.svg/.png` — 2D side-view fallback of the same definition:
  muon energy where the track enters the `r ≤ 90 m`, `|z| ≤ 295 m` cylinder
  around its own cluster.
- `figures/event_types_real.svg/.png` — accepted/rejected event topologies from
  real MC events (not schematic).
- `event_types_real_native.pptx` — the same six real events as a single slide
  built entirely from native PowerPoint shapes.
- `figures/energy_spectrum_train_vs_real.svg/.png` — effective training spectrum
  (flat-weighted) vs the natural MC candidate spectrum before flat selection.
- `figures/event_types.svg/.png` — older schematic contact sheet; kept only as a fallback.
- `figures/appendix_event_display.svg/.png` — one real selected MC event from the
  test-parts dataset (hits only; no reconstructed track).  The chosen event lights
  the central string as well as the perimeter.
- `figures/real_event_display.svg/.png` — a real event display with the true MC
  muon track and the target cylinder; color = hit time, size = charge.  This is
  the preferred main visual.
- `real_event_display_native.pptx` — the same real event assembled from native
  PowerPoint shapes (hits, OMs, strings, track, target, colour bar).
- `energy_training_schemes.pptx` — full 10-slide deck: title, real event, real event
  types, data/selection, model tasks, training regimes, evaluation, formulas,
  IDs, and 2D event projections.
- `target_definition_3d_native.pptx` — the same 3D target picture assembled from
  native PowerPoint shapes (ellipses, connectors, text boxes, hit circles), not a
  flattened image.

Run:

```bash
python doc/energy_training_schemes/build_figures.py
python doc/energy_training_schemes/build_deck.py
python doc/energy_training_schemes/build_real_event.py
python doc/energy_training_schemes/build_real_event_deck.py
python doc/energy_training_schemes/build_event_types_real.py
python doc/energy_training_schemes/build_event_types_real_deck.py
python doc/energy_training_schemes/build_energy_spectrum.py
python doc/energy_training_schemes/build_full_deck.py
python doc/energy_training_schemes/build_acg_quality_deck.py
```

The final deck is `energy_training_schemes.pptx`; it is assembled from the
native real-event slide, the native event-types slide, and the other native
slides.  The two PNG/SVG event figures remain available for plotting and
inspection.

## Facts used

- Target: `e_at_entry[:, 1]`, `r ≤ 90 m`, `|z| ≤ 295 m`, unit GeV;
  `e_at_entry` statuses `0, 1, 4, 5` are accepted, `2` (died before) and `3`
  (never crosses) are dropped.  Source: `data_manager/mu_energy_ds_builder/README.md`,
  `doc/hdf5_energy_truth.md` §5.
- Dataset is single-track only; `muatm` is dropped from energy training.
- `nuatm` rows born inside the target volume are excluded (no vertex cascade in
  this production); `nue2` born-inside rows use the birth energy.  Source:
  `data_manager/mu_energy_ds_builder/README.md`, `part_split_config.yaml`.
- Event selection: sig-noise threshold 0.8; `n_sig_hits ≥ 8`, `n_sig_strings ≥ 3`;
  flat selection in `log10 E`; round numbers from
  `data_manager/datasets/mu_energy_dataset_h8s3_thr0.8_train1M_partsplit/dataset_info.json`.
- Spectrum figure: natural MC curve = per-bin candidate counts `n` from
  `selection_info.json` (train parts, before flat selection); training curve =
  `selected × flat weight` from the same report, i.e. the effective weighted
  training spectrum.  Both are normalized to unit area per dex.  This is MC, not
  experimental data.  Source: `build_energy_spectrum.py`, `selection_info.json`.
- Learning curves in the backup slides: `training_history.csv` for the three
  Gaussian/mean-only runs; the plot uses MAE because Huber and Gaussian NLL are
  not on a common scale.  Source: `build_acg_quality_deck.py`.
- HGB comparison in the backup slides: the baseline test DuckDB
  `261026_1725_non_neural_energy_baseline_partsplit_s42@hgb/test.duckdb` and the
  AC-G test predictions, paired on `event_fk`; uncertainty is a block bootstrap
  over acquisition blocks.  The notebook comparison block is set to the working
  AC-G checkpoint.  Source: `build_acg_quality_deck.py`,
  `analysis/non_neural_energy_baseline.ipynb`.
- Per-class spectra/statistics: `energy.npy`, `particle_types.npy`,
  `n_sig_hits.npy`, `n_sig_strings.npy` in the training dataset, plus the AC-G
  test DuckDB for the natural-spectrum test sample.  Source:
  `build_acg_quality_deck.py`.

## Appendix event display

`figures/appendix_event_display.*` is one real selected MC event from the
`testparts` dataset.  It is deliberately kept out of the main story because it
shows actual hits, not the geometry definition, and the energy dataset stores no
reconstructed track.

## Deliberate simplifications for the talk

- Cluster is drawn as 7 perimeter strings + 1 central string; the real cluster has
  288 channels (8 strings × 36 OMs) and a more complex geometry.
- The track is a straight line; energy loss along it is not shown as a gradient.
- The target cylinder is drawn as a rectangle in side view and as nested circles
  in top view; the main-deck 3D schematic is an oblique drawing, not a real projection.
- The older `figures/event_types.*` schematic is illustrative; the current
  `event_types_real.*` contact sheet and `event_types_real_native.pptx` use
  selected real MC events.
