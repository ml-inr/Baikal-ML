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

**Organized Source Structure:**
```
/src                # Core source code
  ├── models/       # PyTorch model architectures
  ├── data/         # Dataset classes and data loaders
  ├── training/     # Training loops, optimizers, schedulers
  └── utils/        # Shared utilities, metrics, logging
/data_manager       # Data pipeline & catalogs
  ├── root2h5/      # ROOT → HDF5 conversion (multiprocessing, YAML configs)
  ├── root2df/      # ROOT → Polars DataFrame reader (for exploration)
  ├── h5_catalogs/  # Parquet event catalogs (per-event metadata indexes)
  ├── data/         # Local HDF5 files, ROOT files, observation notebooks
  ├── stats_dict/   # Normalization statistics for training
  ├── ManualTests/  # Validation notebooks (ROOT↔HDF5 comparison)
  ├── constants.py  # Physics & detector constants
  └── build_catalog_*.py  # Catalog builder scripts (MC, Exp, Reco, Normed)
/inference          # Inference notebooks and results
/experiments        # Configs, logs, saved models
/notebooks          # Jupyter exploration & prototyping
```

**Key Components:**
- **src/** - Core source code with clean module organization
  - **src/models** - Neural network architectures (Transformer + domain discriminator)
  - **src/data** - PyTorch datasets and data loading (numu, hcut variants)
  - **src/training** - Training loops (standard, DA, hcut DA trainers)
  - **src/utils** - Shared utilities, metrics, reproducibility
- **data_manager** - Full data pipeline: ROOT → HDF5 → Parquet catalogs
  - **root2h5/** - Multiprocessing ROOT→HDF5 converters (MC, exp, reco variants)
  - **root2df/** - Polars-based ROOT reader for exploration and validation
  - **h5_catalogs/** - Parquet event catalogs built from HDF5 (see `doc/data_format.md`)
  - **build_catalog_*.py** - Scripts to rebuild catalogs from HDF5 files
  - **constants.py** - Physics constants (Cherenkov angle, detector geometry divisors)
- **inference** - Jupyter notebooks for model evaluation and comparison
- **experiments** - YAML configs, model weights, training logs
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
- **ROOT → HDF5 converters** (`root2h5/`) - Multiprocessing conversion for MC, Exp, and Reco data
  - `root2h5.py` (MC), `root2h5_exp.py` (Exp), `root2h5_exp_reco.py` (Exp Reco)
  - YAML configs: `root2h5_config.yaml`, `root2h5_config_exp.yaml`, `root2h5_config_exp_reco.yaml`, `root2h5_config_mc_reco.yaml`
- **Physics processor** (`eval_tres.py`, `eval_tres_reco.py`) - Time residual calculations
- **ROOT → DataFrame reader** (`root2df/`) - Polars-based uproot reader for direct ROOT exploration
  - Supports MC, Exp, Exp-Reco, MC-Reco path variants via dataclass configs
- **Parquet catalog builders** (`build_catalog_*.py`) - Extract per-event metadata from HDF5 into Parquet
  - `build_catalog_mc_merged.py` - MC merged data (per particle type)
  - `build_catalog_mc_normed.py` - Normalized MC (train/val/test splits)
  - `build_catalog_exp.py` - Experimental data
  - `build_catalog_exp_reco.py` - Experimental reco data
- **Event catalogs** (`h5_catalogs/`) - Parquet files with event-level metadata for fast filtering
- **Constants** (`constants.py`) - Physics and detector geometry constants
- **Normalization stats** (`stats_dict/`) - Preset normalization parameters for training
- **Validation** (`ManualTests/`) - Notebooks comparing ROOT↔HDF5 data integrity
- **Quality control** - Big time residual filtering, coordinate transformations
- **Cluster analysis** - Single/multi-cluster event handling with splitting

**Configuration Approach:**
```yaml
general:
  take_single_cluster: true
  split_multi: true
  shift_coords_to_cl_center: true
  center_times: true
  exclude_big_ts: true
  t_threshold: 1e5

multiprocessing:
  MAX_QUEUE_SIZE: 4
  NUM_WORKERS: 4

input:
  particle: "muatm"
  root_dir_path: '/path/to/root/files/'

output:
  h5_name: "processed_data.h5"
  h5_prefix: "/path/to/output/"
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
- **TensorBoard** for experiment tracking and visualization

**Local Logging:**
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Training started")
logger.debug("Batch processing details")
logger.error("Training failed")
```

**TensorBoard Tracking:**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f"{experiment_dir}/tensorboard")
writer.add_scalar("Loss/train", loss, epoch)
writer.add_scalar("Metrics/val_auc", val_auc, epoch)
writer.close()
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
- Training/validation losses per epoch
- Task-specific metrics (AUC, F1, precision, recall)
- Learning rate schedule
- Domain discriminator accuracy (DA experiments)
- Logs stored in `experiments/<name>/tensorboard/`

## Usage Scenarios (KISS Approach)

**Primary Task:**
Binary classification on time-series data with variable sample counts

**Core Workflows:**

**1. Data Preparation:**
```bash
python data_manager/root2h5/root2h5_mc.py --config data_manager/root2h5/root2h5_config.yaml
# ROOT → HDF5 with physics-specific preprocessing
# Multiprocessing conversion with time residual calculations
# Handle variable sequence lengths and cluster analysis
```

**2. Model Training:**
```bash
python src/training/standard_trainer.py --config experiments/standard_neutrino_baseline.yaml
# Train attention-based Transformer models (800k+ parameters)
# Comprehensive metrics tracking (AUC, F1, precision, recall)
# Early stopping on validation metrics with model checkpointing
```

**3. Model Inference:**
```bash
python inference/predict.py --config experiments/config.yaml
# Load best attention-based model
# Generate predictions on test set
# Calculate comprehensive binary classification metrics
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
5. Set up TensorBoard tracking and basic training loop
6. Validate end-to-end pipeline with small dataset