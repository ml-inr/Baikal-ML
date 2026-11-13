# Sig-noise model location

The external signal/noise model (SN, author G. Plotnikov) moved from the
repository root to `trained_models/sig_noise/` on 2026-11-08. The HDF5
probability scripts moved to `inference/sig_noise/`. The old root-level
`gplotnikov_sig_noise_models/` package and its compatibility shim were removed on
2026-11-13.

## Current layout

| Old location (removed) | Current location |
|---|---|
| `gplotnikov_sig_noise_models/` | `trained_models/sig_noise/` (all variants, `archive/`, `logs/`, `test.ipynb`) |
| `gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/predict_mc_h5.py` | `inference/sig_noise/predict_mc_h5.py` |
| same for `predict_exp_h5.py` | `inference/sig_noise/predict_exp_h5.py` |

The canonical model is `k_nsol_labelneq0_da_hs128_k0p0001`. Its directory holds
the checkpoint, config, and model code:

```text
trained_models/sig_noise/k_nsol_labelneq0_da_hs128_k0p0001/
  best_mc_2020.ckpt          # not in git (*.ckpt), placed manually
  train_config_mc_2020.yaml
  sig_noise_model_v3.py      # load_model() + predict_flat()
  model_simplified.py
  encoder.py
  layers.py
```

Non-canonical variants (`nn_aug_highq_*` and others) moved with the model and
kept their `.gitignore` status: `nn_aug_highq_*` is tracked; the remaining
variants, `archive/`, and `logs/` are ignored.

## Import path

Use the canonical path for all new and existing code:

```python
from trained_models.sig_noise.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
    load_model, predict_flat,
)
```

The old import path no longer resolves:

```python
# removed 2026-11-13
from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
    load_model, predict_flat,
)
```

Retired code under `archive_tracked/` still contains the old imports. It is
historical and must be updated manually if it is ever rerun. Old script paths
(`gplotnikov_sig_noise_models/.../predict_mc_h5.py`) are not preserved either;
use `inference/sig_noise/predict_{mc,exp}_h5.py` or the wrappers
`run_predict_{mc,exp}.sh`.

`sig_noise_model_v3.py` computes `MODEL_DIR = Path(__file__).parent`, so
`DEFAULT_CHECKPOINT` and `DEFAULT_CONFIG` resolve to the new location without
changes to the model code.

## What did not change

- weights, `train_config_mc_2020.yaml`, and normalization;
- output names:
  `{stem}_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5` next to the source HDF5;
- HDF5 schema and the `sig_noise_batch_size` / `sig_noise_model` attributes;
- the rule that batch size 256 is part of hit selection, not a speed setting
  (`doc/sig_noise_batch_size.md`).

## Verification (2026-11-08)

These checks were made while the compatibility shim still existed; the
canonical path remains the supported one.

- The old and new import paths loaded the same file, and `DEFAULT_CHECKPOINT`
  existed.
- A deterministic fingerprint on 512 events / 32,071 hits (CPU, batch 64) was
  bit-identical before and after the move: SHA-256
  `003b3713eb3155cbc14ae8e92bc860dcf072e1c15c8cdb237640a7517a37419c`.
- `predict_mc_h5.py` produced bit-identical datasets and attributes before and
  after the move on the same small HDF5 file.
- `load_sn_model()`, the `data_manager/*_ds_builder` imports, `excess.signal`,
  and `inference.nu_classifier.predict_*` worked with the new path.

## See also

- `AGENTS.md` — project map and the place of `trained_models/sig_noise/` in the
  tree;
- `inference/sig_noise/README.md` — commands and output format;
- `doc/sig_noise_batch_size.md` — why batch size controls hit selection;
- `doc/AUDIT.md` — historical pre-move map; it is not maintained.
