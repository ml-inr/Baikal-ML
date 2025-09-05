# Development Task List

## Progress Report

| Iteration | Task | Status | Progress |
|-----------|------|--------|----------|
| 1 | Project Setup | ✅ Complete | ✅ 100% |
| 2 | Basic Data Manager | ⏳ Pending | ⬜ 0% |
| 3 | Simple Model + Training | ⏳ Pending | ⬜ 0% |
| 4 | Experiment Management | ⏳ Pending | ⬜ 0% |
| 5 | Inference Pipeline | ⏳ Pending | ⬜ 0% |
| 6 | ClearML Integration | ⏳ Pending | ⬜ 0% |
| 7 | Time-Series Features | ⏳ Pending | ⬜ 0% |
| 8 | Production Pipeline | ⏳ Pending | ⬜ 0% |

**Status Icons:**
- ⏳ Pending
- 🔄 In Progress  
- ✅ Complete
- ❌ Failed/Blocked

---

## Iteration 1: Project Setup
**Goal:** Create environment and basic structure

### Tasks:
- [x] Create conda environment with core dependencies
- [x] Set up project folder structure (`/data_manager`, `/models`, `/training`, `/inference`, `/experiments`, `/notebooks`)
- [x] Create basic `requirements.yml` for conda
- [x] Add `.gitignore` for Python/ML projects
- [x] Create simple `hello_world.py` to test environment

**Test:** Run `python hello_world.py` and import key libraries (torch, numpy, h5py)

---

## Iteration 2: Basic Data Manager
**Goal:** Convert ROOT files to HDF5 format

### Tasks:
- [ ] Create `data_manager/root_reader.py` - read ROOT files with PyROOT
- [ ] Create `data_manager/h5_writer.py` - write data to HDF5 format
- [ ] Create `data_manager/config.py` - load YAML configs
- [ ] Create basic YAML config for data processing
- [ ] Create `data_manager/process_data.py` - main processing script

**Test:** Process one small ROOT file → generate `train.h5`, verify data integrity

---

## Iteration 3: Simple Model + Training
**Goal:** Train basic binary classifier

### Tasks:
- [ ] Create `models/simple_mlp.py` - basic MLP model
- [ ] Create `training/dataset.py` - PyTorch dataset for HDF5 files
- [ ] Create `training/train_simple.py` - basic training loop
- [ ] Create experiment config YAML
- [ ] Add model checkpointing and best model saving

**Test:** Train model on synthetic/small data, save checkpoints, verify convergence

---

## Iteration 4: Experiment Management
**Goal:** Organize experiments and track results

### Tasks:
- [ ] Create experiment directory structure (`experiments/exp_name/`)
- [ ] Add config copying to experiment folder
- [ ] Create `training/metrics.py` - calculate and log metrics
- [ ] Save training logs to CSV format
- [ ] Add model summary generation

**Test:** Run full experiment, verify all files saved correctly, reproduce results

---

## Iteration 5: Inference Pipeline
**Goal:** Load models and make predictions

### Tasks:
- [ ] Create `inference/model_loader.py` - load saved models
- [ ] Create `inference/predict.py` - make predictions on test data
- [ ] Add metrics evaluation for test set
- [ ] Save predictions in standard format (.npz, .csv)
- [ ] Create simple inference script with config support

**Test:** Load trained model, make predictions on test set, calculate accuracy/F1

---

## Iteration 6: ClearML Integration
**Goal:** Add experiment tracking and visualization

### Tasks:
- [ ] Install and configure ClearML
- [ ] Add ClearML task initialization to training scripts
- [ ] Log hyperparameters and metrics to ClearML
- [ ] Add model artifact logging
- [ ] Create ClearML visualization for training curves

**Test:** Run experiment with ClearML tracking, view results in ClearML web UI

---

## Iteration 7: Time-Series Features
**Goal:** Handle variable-length sequences

### Tasks:
- [ ] Add padding/truncation for variable sequences
- [ ] Create time-series specific dataset class
- [ ] Implement RNN/LSTM model for sequences
- [ ] Add time-series data augmentation
- [ ] Update configs for sequence processing

**Test:** Train RNN on time-series data, handle different sequence lengths correctly

---

## Iteration 8: Production Pipeline
**Goal:** End-to-end automated workflow

### Tasks:
- [ ] Create master script for full pipeline
- [ ] Add data validation and quality checks
- [ ] Implement batch prediction for large datasets
- [ ] Add model comparison and selection
- [ ] Create deployment-ready inference script

**Test:** Run complete pipeline from ROOT files to final predictions, validate entire workflow

---

## Testing Strategy

Each iteration includes:
- **Unit Tests:** Test individual functions work correctly
- **Integration Tests:** Test modules work together
- **Validation:** Verify outputs make scientific sense
- **Reproducibility:** Same config → same results

## Notes

- Start each iteration only after previous one is fully tested
- Keep MVP mindset - add minimal functionality to achieve iteration goal
- Document any issues or lessons learned after each iteration
- Update progress report table after completing each iteration