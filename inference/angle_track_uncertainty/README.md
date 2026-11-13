# inference/angle_track_uncertainty

Inference for G. Plotnikov's released angle / track-line / uncertainty model
(`trained_models/angle_track_uncertainty/release_v3/`). The model predicts a
direction, a point on an infinite track line, and marginal angular /
transverse-distance uncertainties. It does **not** classify hits: the input is
the set of hits that passed a signal/noise selection.

The upstream provenance, bundle contents, SHA256s and model-card caveats are in
`trained_models/angle_track_uncertainty/release_v3/README.md`.

## Layout

| File | Role |
|---|---|
| `predict.py` | **Project adapter**: streams our HDF5 + SN probs, writes DuckDB keyed by `event_fk` |
| `common.py` | Shared source reading, selection, batching, gate, catalog helpers |
| `store.py` | DuckDB schema and `INSERT OR IGNORE` writer |
| `prepare_selected_h5.py` | Converts our HDF5 into the upstream prepared-HDF5 schema |
| `predict_one_pass.py` | Vendored upstream one-pass CSV script (reference path) |
| `predict_rethresholded.py` | Vendored upstream two-pass p>=0.50 gate CSV script |
| `run_predict_{mc,exp}.sh` | Wrappers; edit variables at the top |
| `tests/test_parity.py` | Self-check against upstream on synthetic fixtures |

## Running the project adapter

```bash
# MC, canonical SN probabilities, one pass
bash inference/angle_track_uncertainty/run_predict_mc.sh

# Experimental data
bash inference/angle_track_uncertainty/run_predict_exp.sh

# Enable the conditional p>=0.50 + 200 ns photon gate
GATE=true bash inference/angle_track_uncertainty/run_predict_mc.sh
```

The wrappers use `--sn-mode canonical`: the per-hit probabilities
`{stem}_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5`. Set `SN_MODE=release` to use
the frozen `signal_noise.ckpt` shipped with the release instead. The two modes write
separate DBs; a run never mixes their rows.

Small smoke test (one part, a few events):

```bash
PARTS=part_0 LIMIT=100 bash inference/angle_track_uncertainty/run_predict_mc.sh

# Resumable part-level skip (uses data_class + h5_part_key)
python -m inference.angle_track_uncertainty.predict ... --skip-done-parts
```

## Output

```
inference/angle_track_uncertainty/preds/
  prediction_history.csv
  release_v3@angle_track_uncertainty/
    {source}_{sn_mode}_p070_track.duckdb
    {source}_{sn_mode}_p070_gate200_track.duckdb   # when GATE=true
    # e.g. mc_merged_canonical_p070_track.duckdb,
    #      mc_merged_release_p070_gate200_track.duckdb
    run_info.json
```

Schema (`store.py`):

```
predictions(
  event_fk BIGINT PRIMARY KEY,
  event_id, data_class, h5_part_key, h5_local_event_id,
  sn_mode, selection_op, gate_applied,
  selected_threshold, n_raw_hits, n_hits, n_strings,
  theta_deg, phi_deg, direction_x/y/z,
  anchor_x_m/y_m/z_m,
  sigma_angle_deg, angle_r68_deg, angle_r95_deg,
  sigma_anchor_m, anchor_r68_m, anchor_r95_m,
  track_point_scope, uncertainty_calibration_scope
)
```

- `event_fk` — `catalog_v2.events.id`; MC and exp lookups follow the same
  conventions as `inference/nu_classifier/predict_{mc,exp}.py`.
- `n_hits` / `n_strings` — hits and strings selected at `selected_threshold`
  (not at the pipeline's usual 0.8).
- `anchor_*` — a point on the predicted infinite line, in cluster-local metres.
- `track_point_scope` — `validated_single_track_MC` for MC `nuatm`/`nue2`;
  `not_validated_for_possible_multi_track_or_experiment` for `muatm` and exp.
- `uncertainty_calibration_scope` — `p070_crossseason` only for unchanged
  release-SN p>=0.70 MC rows on CPU with the release selection conventions;
  `p050_gate_unvalidated` for gate-replaced rows;
  `canonical_sn_selection_unvalidated` for canonical-SN rows;
  `experimental_unvalidated` for experimental rows;
  `changed_threshold_or_transform_unvalidated` otherwise. Radii are SQL NULL
  where the release does not claim calibration.

`INSERT OR IGNORE` makes a rerun over an overlapping part list resumable.
`--check-existing` warns if an already-stored event gets a different theta.

`--event-set-db PATH [--event-set-query SQL]` restricts inference to the
`event_fk` set in another DuckDB (default query:
`SELECT event_fk FROM predictions`). The catalog resolves each `event_fk` to
its source part and local index; parts without target events are skipped and
already-written event_fk rows are excluded on rerun. This is how the track
model is run on the exact event set of a nu-classifier prediction DB:

```bash
python -m inference.angle_track_uncertainty.predict \
    --source data_manager/data/h5datasets/exp_full.h5 \
    --source-name exp_full \
    --sn-mode canonical \
    --event-set-db inference/nu_classifier/preds/<run>/exp_full_thr0p8.duckdb \
    --device cpu --threads 32 --events-per-chunk 32768
```

Without `--conditional-low-threshold` every row has `gate_applied = false` and
`selected_threshold = 0.70`.

## Selection semantics

- Hits are selected with `prob >= threshold` (release semantics). The shared
  `inference/shared/h5_hits.signal_mask` uses `>`; this task intentionally does
  not change that helper.
- Default cut: `p >= 0.70`, at least 8 selected hits on at least 2 strings.
- The optional gate first runs `p >= 0.70`; for `theta < 90` it reruns the
  **same** model with `p >= 0.50` hits when the low threshold adds hits and
  every added hit passes a 200 ns photon-time residual check. Gate rows expose
  raw sigmas but no calibrated radii.
- In `canonical` mode the hit selection inherits the batch size of the stored
  probs file (recorded in `run_info.json`). In `release` mode batch size is part
  of the selection: the release SN uses a float padding mask (`(~mask).float()`),
  the same mechanism as `doc/sig_noise_batch_size.md`; the default is 256.
- The angle/track backbone itself uses a **boolean** padding mask
  (`src_key_padding_mask=~full_mask`); its per-event output is batch-independent
  up to float reduction order (measured differences <=4e-6).

## Limitations

- The model was trained on `nuatm`, `nue2`, and only **single-track** `muatm`.
  Multi-track `muatm` and experimental events have no validated point or
  uncertainty target.
- No experimental ground-truth line exists. Experimental radii are not validated
  coverage; do not present them as such.
- The point is one point on an infinite line, not a track segment.

## Exact upstream reproduction

For a CSV that matches the upstream scripts field-for-field, first prepare a
selected-hit HDF5 and then run the vendored script:

```bash
python -m inference.angle_track_uncertainty.prepare_selected_h5 \
    --source data_manager/data/h5datasets/baikal_mc_merged.h5 \
    --source-name mc_merged --probs-h5 <probs.h5> \
    --output /tmp/selected_p050.h5 --threshold 0.50 --parts part_0

python -m inference.angle_track_uncertainty.predict_one_pass \
    --checkpoint trained_models/angle_track_uncertainty/release_v3/angle_track_uncertainty.pt \
    --data /tmp/selected_p050.h5 --split test \
    --output /tmp/predictions_p070.csv --device cpu

python -m inference.angle_track_uncertainty.predict_rethresholded \
    --bundle trained_models/angle_track_uncertainty/release_v3/angle_track_uncertainty.pt \
    --data /tmp/selected_p050.h5 --split test \
    --threshold 0.70 --conditional-low-threshold 0.50 \
    --gate-theta-max-deg 90 --max-added-photon-residual-ns 200 \
    --output /tmp/predictions_gate200.csv --device cpu
```

The streaming adapter is preferred for full sources: it writes DuckDB and does
not materialise a selected-hit file.
