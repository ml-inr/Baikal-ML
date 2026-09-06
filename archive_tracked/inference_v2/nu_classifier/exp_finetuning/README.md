# Exp Background Fine-Tuning

Supervised fine-tuning of a pretrained nu-classifier checkpoint using pseudo-labeled
experimental background events.  No DANN — the exp events are now treated as labeled data.

## Idea

The current nu-classifier was trained with DANN on MC (labeled) + exp (unlabeled domain target).
After training, the model assigns very low scores to the vast majority of exp events (mean ~0.03).
Events with `score < ξ` (default 0.5) are almost certainly atmospheric muon background and can
be used as hard-labeled background (`label = 0`) to continue supervised training.

This is one-sided pseudo-labeling: only low-confidence exp events are added as background.
High-score exp events (potential neutrino candidates) are never touched.

**Per-epoch MC resampling:** the MC training pool (~4.7M events) is much larger than the exp BG
set (~30k).  Each epoch a fresh random MC subset of size `N = len(exp_bg_train)` is drawn,
giving a 1:1 MC:expBG ratio and avoiding MC memorisation.

**Combined validation:** MC val (10%) + exp BG val (20%) form one combined val set.
If the model overfits to the exp BG training subset, `val_loss` rises immediately.
Early stopping monitors `val_loss` on the combined set.

## Workflow

### Step 1 — Build exp-background dataset

```bash
bash inference_v2/nu_classifier/exp_finetuning/run_build_exp_bg.sh
```

Reads `exp_thr0p8.duckdb` from the checkpoint's preds dir, selects events with
`score < 0.5`, runs the sig-noise model on those events from `exp.h5`, writes:

```
exp_finetuning/exp_bg_datasets/{checkpoint}_lt0p5/
    exp_bg_features.npy        (total_sig_hits, 5) float32
    exp_bg_offsets.npy         (n_events+1,)       int64
    exp_bg_n_sig_hits.npy      (n_events,)         int32
    exp_bg_n_sig_strings.npy   (n_events,)         int32
    exp_bg_event_fks.npy       (n_events,)         int64   — catalog FK for traceability
    exp_bg_scores.npy          (n_events,)         float32 — original model score
    exp_bg_dataset_info.json
```

Quick sanity check:
```python
import numpy as np, json
d = "inference_v2/nu_classifier/exp_finetuning/exp_bg_datasets/.../"
offs  = np.load(d + "exp_bg_offsets.npy")
feats = np.load(d + "exp_bg_features.npy", mmap_mode="r")
info  = json.load(open(d + "exp_bg_dataset_info.json"))
print(f"n_events={info['n_events']:,}  total_hits={len(feats):,}  score_mean={info['score_stats']['mean']:.3f}")
```

### Step 2 — Fine-tune

```bash
bash inference_v2/nu_classifier/exp_finetuning/run_finetune.sh
```

The script writes `finetune_seed32.yaml` and launches training.  Outputs:

```
exp_finetuning/finetuned_models/{checkpoint}_finetuned/
    best_finetuned_model.pth        — best val_loss checkpoint
    latest_finetuned_checkpoint.pth — most recent epoch
    finetuning_history.csv
    finetune_config.yaml
    normalization_config.yaml
    tensorboard/
```

### Step 3 — Evaluate

Run the standard single-model analysis on the finetuned checkpoint to compare
suppression curves and exp score distributions before/after fine-tuning:

```bash
# Edit CHECKPOINT in run_single_analysis.sh to point at the finetuned .pth,
# then run inference on exp and mc_merged first:
bash inference_v2/nu_classifier/run_predict_exp.sh
bash inference_v2/nu_classifier/run_predict_mc.sh
bash inference_v2/nu_classifier/analysis/run_single_analysis.sh
```

Expected: exp score distribution shifts toward 0 (sharper separation),
MC suppression curves unchanged or improved.

## Key files

| File | Purpose |
|---|---|
| `build_exp_bg.py` | Dataset builder (DuckDB query + two-pass sig-noise filtering) |
| `run_build_exp_bg.sh` | Shell config for `build_exp_bg.py` |
| `run_finetune.py` | CLI entry point — loads YAML config, calls `ExpFineTuningTrainer` |
| `run_finetune.sh` | Writes `finetune_seed32.yaml` and launches training |
| `src/training/exp_finetuning_trainer.py` | Trainer class |

## Config reference (`finetune_*.yaml`)

```yaml
pretrained_checkpoint: experiments/numu/.../best_da_model.pth
mc_npy_dir:    data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8
exp_bg_npy_dir: inference_v2/nu_classifier/exp_finetuning/exp_bg_datasets/...
output_dir:    inference_v2/nu_classifier/exp_finetuning/finetuned_models
experiment_name: ...
device: cuda:0

epochs: 50           # early stopping terminates well before this
batch_size: 512      # 256 MC + 256 expBG per step
learning_rate: 5e-5  # much lower than original 3e-4
scheduler: plateau
early_stopping:
  monitor: loss      # combined val loss
  mode: min
  patience: 8
```

## Notes

- **Checkpoint compatibility**: `best_finetuned_model.pth` uses the same keys as the
  original DA checkpoint (`base_model_state_dict`, `normalization_config`, `config`),
  so `inference_v2` predict scripts load it without changes.

- **Per-checkpoint selection**: each model selects its own low-score exp events.
  Fine-tune each checkpoint separately using its own `exp_thr0p8.duckdb`.

- **Incremental fine-tuning**: the finetuned checkpoint can be used as the starting
  point for a second round with a tighter `ξ` or different quality cuts.
