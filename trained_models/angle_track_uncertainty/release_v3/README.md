# Angle / track-line / uncertainty model — `release_v3`

External model by G. Plotnikov. It reconstructs, per event, a direction, a point on
an infinite track line, and marginal angular / transverse-distance uncertainties.
It is complementary to `inference/track_energy_reconstruction/` (energy) and
`inference/track_fit_alike_bars/` (the BARS track fit).

## Provenance

| Item | Value |
|---|---|
| Repository | `https://github.com/plotnikovgp/Baikal-ML` |
| Commit | `15392372e4dce61b9c8c7245fb7fffe7c8f0a30f` ("Release universal photon-corrected angle and track model", 2026-10-14) |
| Upstream directory | `pytorch_training/scripts/angle_track_uncertainty/release_v3/` |
| Upstream model card | [`MODEL_CARD.md`](MODEL_CARD.md) |
| Upstream license | none in the repository at this commit; copied with attribution |

The released files were copied byte-for-byte. Their SHA256 values:

| File | SHA256 |
|---|---|
| `angle_track_uncertainty.pt` | `7cb8f77cfdb2a7cf021d090be5246b83f8a94c7513cc24c793ab019c67686f41` |
| `signal_noise.ckpt` | `be9cc39b93dd6a57a69233faf6755278e3f202c56de3de7d1cc8d2d4a8b1e93d` |
| `angle_norm.h5` | `8053d7579d1f0e065a5c4bf1092ce8e513e3985b37c086799b169f5c08207bb7` |
| `classifier_norm.h5` | `28e1dae133db7bacce5a908807b2892c8af675fc20a6ba27fc7fd95b3b38f9ea` |
| `MODEL_CARD.md` | `d61ae86901594a291c123b7c21e3e6d505cff655af9700b7b56acc853a60d8f4` |
| `independent_2019_f_metrics.json` | `6effd8ddcd2bd3429369a59c2bfb72a93ac491c06e4d576d0b29a80fe4d78122` |
| `reco_2020_comparison.json` | `32992aa60677cf485d4fa955c520aff09a9624d440533c9f7d652391f7cbdc02` |

Weights (`*.pt`, `*.ckpt`) are ignored by git, like the SN checkpoint. The small
normalization files and the two metrics JSONs are tracked through explicit
`.gitignore` exceptions; without them the bundle is not reproducible.

## Model

- One CPU-capable model, 3,079,266 parameters; bundle `format_version=2`,
  `model_type=photon_corrected_uncertainty`.
- Backbone: 5-layer, 8-head Set Transformer (`hidden=256`, `ff_mult=2`), engineered
  10-feature input derived from five normalized features `[q, t, x, y, z]`.
- Heads: direction (3), point offset from the selected-hit centroid (3), two log
  uncertainties (angle and transverse line distance), plus a frozen 64-hidden
  photon-time-residual correction.
- Preprocessing: `max_seq_len=128`, `center_time=false`. The bundle normalization is
  identical to `angle_norm.h5`.
- Calibration: angle 68/95 = 2.60175/6.45668; line 68/95 = 1.85289/4.22721.

## Hit selection

The model itself does **not** classify hits. It expects selected hits from a
signal/noise model:

- **release mode**: the frozen `signal_noise.ckpt` in this folder, with
  `classifier_norm.h5`; threshold `p >= 0.50` for the prepared file, `p >= 0.70` for
  the calibrated pass;
- **canonical mode** (default in our adapter): the per-hit probabilities produced
  by `trained_models/sig_noise/k_nsol_labelneq0_da_hs128_k0p0001/`
  (`{stem}_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5`).

`signal_noise.ckpt` is **not** the canonical SN checkpoint used elsewhere in this
repository (different SHA256 and normalization). Release mode reproduces the
published release; canonical mode keeps the hit selection consistent with the rest
of our pipeline and will not reproduce the published metrics exactly. The two
modes write separate DuckDB files.

The angle/track backbone uses a **boolean** padding mask
(`src_key_padding_mask=~full_mask`), so its per-event output is batch-independent
up to float reduction order. The bundled release SN uses a **float** mask
(`(~mask).float()`), the same mechanism as `doc/sig_noise_batch_size.md`; its
selection is batch-composition dependent.

## Scope and limitations

- Direction, point, and uncertainty were trained on `nuatm`, `nue2`, and only
  **single-track** `muatm`. Multi-track `muatm` events were excluded.
- The point is one point on an infinite line, not a track segment; the along-line
  coordinate is not identifiable.
- The 68%/95% radii are marginally calibrated. Calibrated radii are only claimed for
  unchanged release-SN `p >= 0.70` MC rows on CPU with the release selection
  conventions. Gate rows, canonical-SN rows, experimental rows and changed
  thresholds have raw sigmas but NULL radii.
- No experimental ground truth exists; experimental uncertainty outputs are not
  validated coverage. Do not interpret them as such.

## Inference

Our project adapter, the prepared-HDF5 converter, and the vendored upstream CSV
scripts live in `inference/angle_track_uncertainty/`; see its README. The upstream
model card documents the original commands and the independent 2019-F evaluation.
