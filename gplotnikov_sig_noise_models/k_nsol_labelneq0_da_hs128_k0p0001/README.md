# Signal/Noise Hit Classifier — `k_nsol_labelneq0_da_hs128_k0p0001`

Per-hit signal probability model for Baikal-GVD data.
Classifies each raw detector hit as signal (neutrino/muon origin) or noise.
Trained by G. Plotnikov with domain adaptation on MC 2020 data.

## Architecture

Transformer encoder → per-hit binary classification head.

| Parameter          | Value |
|--------------------|-------|
| Input features     | 5 (Q, t, x, y, z) |
| Hidden size        | 128 |
| Transformer layers | 5 |
| Attention heads    | 1 |
| FFN dimension      | 512 |
| Output             | 2 (noise / signal logits per hit) |
| DA coefficient k   | 0.0001 |

The inference-only class `SigNoiseModel` in `model_simplified.py` omits the
domain-adaptation head. The checkpoint is loaded with `strict=False`, so DA
weights in the checkpoint are silently ignored.

Input normalization is hardcoded in `sig_noise_model_v3.py`:

```
means = [1.295, 0.0,   0.643, 0.248, 30.92]
stds  = [3.671, 1386., 40.10, 39.05, 154.6]
```

## Validation performance (MC 2020)

| Metric    | Value  |
|-----------|--------|
| AUC       | 0.9990 |
| Precision | 0.9821 |
| Recall    | 0.9000 |
| Threshold | 0.6628 |

## Files in this directory

| File | Tracked | Description |
|------|---------|-------------|
| `model_simplified.py` | yes | Inference-only `SigNoiseModel` (no GRL/DA head) |
| `sig_noise_model_v3.py` | yes | **Core API**: `load_model()` + `predict_flat()` |
| `sig_noise_model_v2.py` | yes | Earlier inference wrapper (kept for reference) |
| `encoder.py` | yes | Original full encoder with DA head (not used by v3) |
| `layers.py` | yes | `GradientReversal` layer used by `encoder.py` |
| `predict_mc_h5.py` | yes | Example script for this project's `baikal_mc_merged.h5` layout |
| `train_config_mc_2020.yaml` | yes | Model architecture config, read by `load_model()` |
| `best_mc_2020.ckpt` | **NO** | Trained weights — **must be placed manually** |

## Required file not in git

The model checkpoint is excluded from version control (binary weights).

Place the following file in this directory before running inference:

```
gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/best_mc_2020.ckpt
```

The checkpoint is a standard PyTorch `state_dict` saved with `torch.save`.
Obtain it from the original training output or from shared team storage.

## Core API — `sig_noise_model_v3.py`

The reusable entry point is `load_model` + `predict_flat` from `sig_noise_model_v3.py`.
Use these directly when integrating the model into your own pipeline:

```python
from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
    load_model, predict_flat,
)

model, config, device = load_model(device="cuda:0")

probs = predict_flat(
    model=model,
    data=data_raw,       # (n_hits, 5) float32, raw (unnormalized) features [Q, t, x, y, z]
    ev_starts=ev_starts, # (n_events+1,) int64 CSR-style event boundaries
    batch_size=256,
    device=device,
    normalize=True,      # apply built-in normalization
)
# probs: (n_hits,) float32, values in [0, 1]
# threshold ~0.66 gives recall=0.90, precision=0.98
```

`predict_flat` handles chunked GPU transfer and returns a flat array aligned
with the input hits in the same order as `ev_starts`.

## Example script — `predict_mc_h5.py`

`predict_mc_h5.py` is a project-specific wrapper that reads from
`baikal_mc_merged.h5` (this project's HDF5 layout) and writes a companion
probs HDF5 file. It is **not** a generic inference script — if your HDF5
structure differs, write your own wrapper using `load_model` / `predict_flat`
from `sig_noise_model_v3.py` directly.

For reference, the script is run from the project root:

```bash
python gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/predict_mc_h5.py \
    --input  data_manager/data/h5datasets/baikal_mc_merged.h5 \
    --device cuda:0 \
    --batch-size 256
```

Output written next to the input:
`baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5`
