# Enhanced Training Features - TensorBoard Integration & Metrics History

## Overview

The training pipeline includes comprehensive experiment tracking and visualization via TensorBoard:

1. **Metrics history saving at each checkpoint** (not just final)
2. **TensorBoard integration** for training curve visualization
3. **CSV training history** for programmatic analysis

---

## TensorBoard Integration

All trainers (`da_numu_trainer.py`, `da_hcut_numu_trainer.py`, `standard_numu_trainer.py`) write TensorBoard logs automatically.

**Logged per epoch:**
- `Loss/train`, `Loss/val`
- `Metrics/train_auc`, `Metrics/val_auc`
- `Metrics/train_f1`, `Metrics/val_f1`
- `Metrics/train_accuracy`, `Metrics/val_accuracy`
- `LR/feature_optimizer`, `LR/classifier_optimizer`
- DA trainers also log: `Domain/discriminator_accuracy`, `Domain/lambda_factor`

**Usage:**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f"{experiment_dir}/tensorboard")
writer.add_scalar("Loss/train", train_loss, epoch)
writer.add_scalar("Metrics/val_auc", val_auc, epoch)
writer.close()
```

**View logs:**
```bash
tensorboard --logdir experiments/numu/
# or for a specific experiment:
tensorboard --logdir experiments/numu/da_numu_251123_small_moderatelambdmidddnoreg_aug_newlr/tensorboard/
```

---

## Experiment Directory Structure

Each experiment saves:
```
experiments/numu/<experiment_name>/
├── da_config.yaml                  # Full config used
├── model_summary.txt               # Architecture and parameter count
├── normalization_config.yaml       # Normalization stats used
├── da_training_history.csv         # Per-epoch metrics (all epochs)
├── da_training_summary.yaml        # Final summary with best metrics
├── best_da_model.pth               # Best model by val_auc
├── latest_da_checkpoint.pth        # Latest checkpoint
├── da_checkpoint_epoch_XXX.pth     # Periodic checkpoints (every save_every epochs)
└── tensorboard/                    # TensorBoard event files
    └── events.out.tfevents.*
```

---

## CSV Training History

`da_training_history.csv` is updated every epoch and contains all metrics in columnar format for easy analysis in notebooks.

Example usage:
```python
import pandas as pd
df = pd.read_csv("experiments/numu/<name>/da_training_history.csv")
df.plot(x="epoch", y=["val_auc", "train_auc"])
```

---

## Configuration

Checkpoint and logging frequency is controlled in the experiment YAML:

```yaml
training:
  save_every: 5       # Save checkpoint every N epochs
  validate_every: 1   # Validate and log metrics every epoch

logging:
  tensorboard: true   # Enable TensorBoard logging
  log_every: 10       # Console log frequency (batches)
```
