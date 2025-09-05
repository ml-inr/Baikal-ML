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

# Local imports
from models.base import BaseModel
from data_manager.loader import DataLoader
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
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Return (features, label)
        pass
        
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

**ClearML Integration:**
```python
from clearml import Task

# Initialize once per experiment
task = Task.init(project_name="neural_research", task_name=experiment_name)
task.connect(config)  # Log hyperparameters
```

**Metrics Logging:**
- Log scalars every epoch: loss, accuracy, etc.
- Log sample predictions periodically
- Save model artifacts automatically

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

**data_manager:**
- All functions accept config dict
- Return Path objects for file outputs
- Handle missing files gracefully
- Log processing statistics

**models:**
- One model class per file
- Include model summary generation
- Support loading from config
- Implement `get_model(config)` factory function

**training:**
- Separate training loop from model definition
- Support resuming from checkpoints
- Save best model based on validation metric
- Log metrics every epoch

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