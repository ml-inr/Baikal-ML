# Neural Network Research Project - Technical Vision

## Technologies (KISS Approach)

**Environment Management:**
- conda (primary package manager)
- pip (only when package unavailable in conda)

**Core ML Stack:**
- Python 3.10+
- PyTorch (deep learning)
- NumPy (numerical computing)
- scikit-learn (metrics and utilities)
- SciPy (scientific computing)

**Data Processing:**
- PyROOT (ROOT file handling)
- h5py (HDF5 files)
- pandas (data manipulation)
- awkward (complex array structures)

**Jupyter Ecosystem:**
- jupyter (notebook server)
- ipykernel (Python kernel)
- matplotlib (plotting)
- ipywidgets (interactive widgets)

**Configuration & Utils:**
- PyYAML (YAML config files)
- pathlib (file path handling - built-in)

**Development:**
- pytest (testing when needed)

**Installation Priority:**
1. Try conda first: `conda install package_name`
2. Use conda-forge if needed: `conda install -c conda-forge package_name`
3. Only use pip if unavailable: `pip install package_name`

## Development Principles (KISS Approach)

**Core Principles:**
- **KISS** (Keep It Simple, Stupid) - Choose the simplest solution that works
- **YAGNI** (You Aren't Gonna Need It) - Don't build features until actually needed
- **MVP** (Minimum Viable Product) - Start with bare minimum to validate ideas
- **Fail Fast** - Quick experiments to identify what doesn't work early
- **Iterative Development** - Build incrementally, improve based on results

**Research-Specific:**
- **Reproducible Experiments** - Same inputs → same outputs
- **Modular Design** - Independent components that can be swapped/tested
- **Validation First** - Prove concepts work before scaling up

**Implementation:**
- Start with simplest working version
- Add complexity only when validated need exists
- Document what works and what doesn't
- Keep experiments small and focused

## Project Architecture (KISS Approach)

**Simple Flat Structure:**
```
/data_manager     # ROOT → HDF5 conversion & preprocessing
/models          # PyTorch model definitions & architectures
/training        # Training scripts, optimizers, schedulers
/inference       # Model loading, prediction, evaluation
/experiments     # Configs, logs, saved models
/notebooks       # Jupyter exploration & prototyping
```

**Key Components:**
- **data_manager** - Data pipeline (ROOT files → HDF5 datasets)
- **models** - Neural network architectures
- **training** - Training loops and optimization
- **inference** - Model inference, predictions, metrics evaluation
- **experiments** - YAML configs, model weights, results
- **notebooks** - Interactive research and validation

**Data Flow:**
ROOT files → data_manager → HDF5 → training → saved models → inference → predictions/metrics

## Data Pipeline (KISS Approach)

**Pipeline Flow:**
```
ROOT files → data_manager → separate HDF5 files → PyTorch datasets
```

**File Organization:**
- `train.h5` - Training dataset
- `val.h5` - Validation dataset  
- `test.h5` - Test dataset
- `config.yaml` - Data processing configuration

**Data Manager Components:**
- ROOT file reader
- Configurable transformations (extensive codebase planned)
- HDF5 writer with dataset splitting
- YAML-driven cuts and filters

**Configuration Approach:**
```yaml
data:
  input_files: ["file1.root", "file2.root"]
  cuts:
    pt_min: 10.0
    eta_max: 2.5
  splits:
    train: 0.7
    val: 0.15
    test: 0.15
  transformations:
    # TBD - will grow as needed
```

**Key Benefits:**
- Separate files for clean train/val/test workflow
- YAML configs for reproducible data processing
- Flexible transformation system to be developed iteratively

## Configuration Approach (KISS Approach)

**Master Configuration:**
- One YAML config file per experiment
- Simple hierarchical structure
- No inheritance/overrides initially (YAGNI)

**Config Structure:**
```yaml
experiment:
  name: "my_experiment_001"
  seed: 42

data:
  input_files: ["data1.root", "data2.root"]
  output_dir: "processed_data/"
  cuts:
    pt_min: 10.0
    eta_max: 2.5
  splits:
    train: 0.7
    val: 0.15
    test: 0.15

model:
  architecture: "simple_mlp"
  hidden_dims: [128, 64, 32]
  dropout: 0.2

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  optimizer: "adam"

inference:
  model_path: "experiments/my_experiment_001/best_model.pth"
  output_dir: "results/"
```

**Usage:**
- Pass config path to scripts: `python train.py --config experiments/config_001.yaml`
- All components read from same config for consistency
- Experiment results saved in config-named directory

## Experiment Management (KISS Approach)

**Directory Structure Per Experiment:**
```
experiments/
└── my_experiment_001/
    ├── config.yaml           # Full experiment config
    ├── checkpoints/          # Model checkpoints during training
    │   ├── epoch_010.pth
    │   ├── epoch_020.pth
    │   └── latest.pth
    ├── best_model.pth        # Best model by early stopping
    ├── training_log.csv      # Losses and metrics per epoch
    └── model_summary.txt     # Model architecture info
```

**Tracking Components:**
- **Config Saving** - Copy original config to experiment folder
- **Checkpointing** - Save model state every N epochs
- **Best Model** - Save best model based on validation metric
- **Early Stopping** - Monitor validation loss/metric for stopping
- **Metrics Logging** - CSV with epoch, train_loss, val_loss, metrics
- **Model Info** - Architecture summary and parameter count

**Simple Implementation:**
```python
# Save during training loop
torch.save(model.state_dict(), f"checkpoints/epoch_{epoch:03d}.pth")
if val_metric > best_metric:
    torch.save(model.state_dict(), "best_model.pth")
    best_metric = val_metric
```

**CSV Format:**
```csv
epoch,train_loss,val_loss,accuracy,f1_score
1,0.856,0.743,0.672,0.658
```

## Logging Approach (KISS Approach)

**Dual Logging from Day 1:**
- Python `logging` for local debugging
- **ClearML** for experiment tracking and visualization

**Local Logging:**
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Training started")
logger.debug("Batch processing details")
logger.error("Training failed")
```

**ClearML Tracking:**
```python
from clearml import Task

# Initialize experiment
task = Task.init(project_name="neural_research", 
                task_name=config['experiment']['name'])

# Log hyperparameters
task.connect(config)

# Log metrics (task-specific)
task.logger.report_scalar("Loss", "train", loss, iteration)
task.logger.report_scalar("Accuracy", "validation", acc, iteration)
```

**Configurable Metrics:**
```yaml
training:
  metrics_to_track:
    - "accuracy"
    - "f1_score" 
    - "precision"
    - "recall"
    # Add task-specific metrics as needed
```

**What Gets Tracked:**
- Hyperparameters and configs
- Training/validation losses
- Task-specific metrics (configurable)
- Model artifacts and checkpoints
- Training plots and visualizations

## Usage Scenarios (KISS Approach)

**Primary Task:**
Binary classification on time-series data with variable sample counts

**Core Workflows:**

**1. Data Preparation:**
```bash
python data_manager/process_data.py --config experiments/config.yaml
# ROOT → HDF5 with time-series preprocessing
# Handle variable sequence lengths
```

**2. Model Training:**
```bash
python training/train.py --config experiments/config.yaml
# Train binary classifier (RNN/LSTM/Transformer)
# ClearML tracking from start
# Early stopping on validation AUC/F1
```

**3. Model Inference:**
```bash
python inference/predict.py --config experiments/config.yaml
# Load best model
# Generate predictions on test set
# Calculate binary classification metrics
```

**4. Interactive Research:**
```bash
# Jupyter notebooks in VSCode
notebooks/explore_data.ipynb      # Data exploration
notebooks/model_analysis.ipynb    # Model behavior analysis
notebooks/results_viz.ipynb       # Results visualization
```

**Key Challenges Addressed:**
- Variable-length time series (padding/truncation strategies)
- Binary classification metrics (AUC, precision, recall, F1)
- Time-series specific data splits (temporal validation)

## Model Deployment (KISS Approach)

**Simple Inference Pipeline:**
Load trained model → Make predictions → Save results

**Basic Implementation:**
```python
# Load best model
model = load_model("experiments/exp_001/best_model.pth")
model.eval()

# Make predictions
predictions = model(test_data)
probabilities = torch.sigmoid(predictions)

# Save results
results = {
    'predictions': predictions.numpy(),
    'probabilities': probabilities.numpy(),
    'true_labels': test_labels.numpy()
}
np.savez("results/predictions.npz", **results)
```

**Output Formats:**
- NumPy arrays (.npz files)
- CSV files for easy analysis
- Metrics summary (JSON/YAML)

**Deployment Strategy:**
- Start with script-based inference
- Add batch processing when needed
- No web APIs or complex deployment initially (YAGNI)

## Next Steps

1. Set up conda environment with core dependencies
2. Create basic project structure following the architecture
3. Implement minimal data_manager for ROOT → HDF5 conversion
4. Build simple PyTorch model for binary classification
5. Set up ClearML tracking and basic training loop
6. Validate end-to-end pipeline with small dataset