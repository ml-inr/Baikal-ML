# inference/sig_noise

HDF5-probability generation for the external signal/noise (SN) model.
Unlike the other inference task folders, these scripts do **not** write DuckDB
tables under `preds/`: they write a per-hit probability companion next to the
source HDF5 file.

## Scripts

| Script | Input | Output (next to the input) |
|---|---|---|
| `predict_mc_h5.py` | MC h5 with `raw/{data,ev_starts,channels,labels}` | `{stem}_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5` |
| `predict_exp_h5.py` | exp-family h5 with `raw/{data,ev_starts,channels}` | same naming |

The companion holds per-hit probabilities, hit boundaries, channels, and per-event
signal counts at thresholds 0.5 and 0.8. MC files also carry the truth counts
`n_gt_sig_hits` / `n_gt_sig_strings` (`raw/labels != 0`).

## Commands

```bash
bash inference/sig_noise/run_predict_mc.sh
bash inference/sig_noise/run_predict_exp.sh
```

The wrappers pass extra arguments through, e.g. `--append`, `--parts-json`,
`--group`, `--exclude-parts`.

## Batch size is part of the selection

The model passes a **float** padding mask, so padding stays visible to attention
and a hit's probability depends on how much padding shares its batch. Batch size
therefore changes which hits pass the 0.8 cut; it is not a speed knob. The project
convention is **256** for both MC and experimental data
(`doc/sig_noise_batch_size.md`). The scripts refuse to resume an existing output
file with a different value.

## Model location

Weights and model code live in
`trained_models/sig_noise/k_nsol_labelneq0_da_hs128_k0p0001/`. The old import path
`gplotnikov_sig_noise_models...` still works through a compatibility shim.
