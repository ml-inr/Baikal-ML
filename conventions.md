# Code Development Conventions

*This document defines coding rules for our neural network research project. See [vision.md](vision.md) for technical architecture and development principles.*

## Core Principles

Follow KISS, YAGNI, MVP, and Fail Fast principles from vision.md in all code:
- Start with simplest working implementation
- Add features only when validated need exists
- Prioritize readability over cleverness
- Make code changes in small, testable increments

## Python Coding Standards

**General:**
- Use Python 3.10+ features when appropriate
- Follow PEP 8 naming conventions
- Maximum line length: 88 characters (Black formatter standard)
- Use type hints for function signatures and class attributes
- Prefer `pathlib.Path` over `os.path` for file operations

**Imports:**
```python
# Standard library first
import logging
from pathlib import Path

# Third party
import torch
import numpy as np
import yaml

# Local imports from src/
from src.models.simple_mlp import SimpleMLP
from src.data.h5_dataset import H5Dataset
from src.training.trainer import Trainer
from src.utils.metrics import calculate_binary_metrics

# Other local imports
from data_manager.root2h5.root2h5 import process_file
```

**Error Handling:**
- Use specific exceptions, not bare `except:`
- Log errors with context: `logger.error(f"Failed to load {file_path}: {e}")`
- Fail fast - validate inputs early in functions

## PyTorch Model Implementation

**Model Structure:**
```python
class ModelName(torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # Store config for reproducibility
        self.config = config
        # Build layers from config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always include input/output shapes in docstring
        pass
```

**Requirements:**
- All models inherit from `torch.nn.Module`
- Accept config dict in `__init__` for reproducibility
- Document input/output tensor shapes
- Use `torch.nn.functional` for stateless operations
- Set `model.eval()` before inference, `model.train()` before training

**Device Handling:**
```python
# Move model and data to same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

## Data Handling Conventions

**File Naming:**
- HDF5 files: `train.h5`, `val.h5`, `test.h5`
- Configs: `experiment_name.yaml`
- Models: `best_model.pth`, `epoch_XXX.pth`

**Dataset Classes:**
```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: Path, config: dict):
        # Load metadata, not full data
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return dict with 'features', 'labels', 'lengths', 'event_id'
        return {
            'features': self.events[idx],
            'labels': self.labels[idx], 
            'lengths': self.hit_counts[idx],
            'event_id': self.event_ids[idx]
        }
        
    def __len__(self) -> int:
        pass
```

**Data Loading:**
- Use `torch.utils.data.DataLoader` with appropriate `num_workers`
- Handle variable sequence lengths with custom `collate_fn`
- Validate data shapes and types early in pipeline

## Configuration Management

**YAML Structure:**
- Use flat hierarchy (max 2 levels deep)
- All paths relative to project root
- Include `seed` for reproducibility
- Group related parameters logically

**Loading Configs:**
```python
def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Validate required keys
    return config
```

**Validation:**
- Check required keys exist
- Validate data types and ranges
- Convert string paths to `Path` objects

## Experiment Reproducibility

**Seed Management:**
```python
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**Requirements:**
- Set seed from config at script start
- Save exact config used for each experiment
- Use deterministic algorithms when possible: `torch.backends.cudnn.deterministic = True`
- Log environment info (PyTorch version, CUDA version, etc.)

## Testing Approaches

**Test Structure:**
- Simple unit tests for data processing functions
- Integration tests for full pipelines
- Use pytest fixtures for common test data
- Test with small synthetic datasets

**What to Test:**
- Data loading and preprocessing correctness
- Model forward pass with known inputs
- Config validation
- File I/O operations

**Avoid Over-Testing:**
- Don't unit test PyTorch internals
- Focus on business logic, not framework code
- Test edge cases that affect research validity

## Logging and Monitoring

**Local Logging:**
```python
import logging

logger = logging.getLogger(__name__)
# Log at INFO level for training progress
# Use DEBUG for detailed debugging info
```

**TensorBoard Integration:**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f"{experiment_dir}/tensorboard")
writer.add_scalar("Loss/train", train_loss, epoch)
writer.add_scalar("Metrics/val_auc", val_auc, epoch)
writer.close()
```

**Metrics Logging:**
- Log scalars every epoch: loss, AUC, F1, accuracy
- TensorBoard logs stored under `experiments/<name>/tensorboard/`
- View with: `tensorboard --logdir experiments/`

## Documentation Standards

**Function Documentation:**
```python
def process_data(input_path: Path, config: dict) -> Path:
    """Process ROOT files to HDF5 format.
    
    Args:
        input_path: Path to ROOT file
        config: Processing configuration
        
    Returns:
        Path to output HDF5 file
    """
```

**Requirements:**
- Document all public functions and classes
- Include parameter types and return types
- Focus on what the function does, not how
- Update docs when changing function behavior

**Module Documentation:**
- Include brief module purpose in docstring
- Document key classes and their relationships
- Keep README files minimal (reference vision.md)

## Module-Specific Conventions

**src/models:**
- Attention-based architectures for sequence modeling
- Include model summary generation and parameter counting
- Support loading from config with `create_model(config)` factory
- Transformer encoders with positional encoding for time-series data
```python
# Example: src/models/base_models.py
from src.models.base_models import NuMuClassifierModel, create_model
```

**src/data:**
- PyTorch Dataset classes for HDF5 data with dict-based API
- Advanced collate functions with data augmentation and preprocessing
- Data loading utilities with preset normalization support
```python
# Example: src/data/numu_dataset.py
from torch.utils.data import Dataset
from src.data.numu_dataset import NuMuDataset, create_from_ds_numu_dataloader
```

**src/training:**
- Complete training pipelines with early stopping and checkpointing
- Comprehensive metrics tracking (AUC, F1, precision, recall, confusion matrix)
- Class weight calculation and balanced training for imbalanced datasets
- Device consistency and memory management for variable-length sequences
- TensorBoard integration for experiment tracking and visualization
- Support for preset normalization and data augmentation
```python
# Example: src/training/standard_numu_trainer.py
from src.training.metrics import MetricsTracker, BinaryClassificationMetrics
```

**src/utils:**
- Shared utilities: metrics, logging, reproducibility
- Configuration loading and validation
- Experiment tracking helpers

**data_manager:**
- All functions accept config dict
- Return Path objects for file outputs
- Handle missing files gracefully
- Log processing statistics
- `root2h5/` — multiprocessing ROOT→HDF5 converters (MC, Exp, Reco variants), YAML-driven
- `root2df/` — Polars-based ROOT reader using uproot, dataclass path configs
- `build_catalog_*.py` — Parquet catalog builders (MC merged, MC normed, Exp, Exp reco)
- `h5_catalogs/` — output Parquet catalogs with per-event metadata
- `constants.py` — physics/detector constants (refractive index, channel/string divisors)
- `stats_dict/` — normalization statistics YAML files for training

**inference:**
- Load models in eval mode
- Batch predictions efficiently
- Save predictions in standard format (.npz, .csv)
- Include confidence scores when available

## Performance Guidelines

**Memory Management:**
- Use generators for large datasets
- Clear GPU cache when needed: `torch.cuda.empty_cache()`
- Monitor memory usage during long training runs

**Computation:**
- Profile code before optimizing
- Use appropriate PyTorch data types (float32 vs float64)
- Leverage GPU when available, fallback to CPU gracefully

## What NOT to Do

- Don't implement custom optimizers initially (use PyTorch built-ins)
- Don't write complex inheritance hierarchies
- Don't optimize prematurely
- Don't create abstract base classes unless needed
- Don't use global variables for configuration
- Don't hardcode file paths or hyperparameters
- Don't ignore warnings or errors silently