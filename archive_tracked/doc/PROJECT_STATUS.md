# Baikal Neutrino Detection - Project Status

*Last updated: 2026-03-15*

## 🎯 Current Status: Iteration 4 Complete — Validating on Real Data

### ✅ Completed Iterations

**Iteration 1: Project Setup** ✅ 100%
- Conda environment with core dependencies
- Project folder structure
- Basic requirements and configuration

**Iteration 2: Basic Data Manager** ✅ 100%
- ROOT → HDF5 conversion pipeline
- Configuration management
- Data processing validation

**Iteration 2.5: Adapt Data Manager** ✅ 100%
- Customized for Baikal detector data
- Physics-specific processing (time residuals, clustering)
- Production-ready multiprocessing system
- Quality control and coordinate transformations

**Project Restructuring** ✅ 100%
- Organized `src/` directory structure
- Clean separation of source code vs artifacts
- Standard Python project layout

**Iteration 2.6: PyTorch Dataset** ✅ 100%
- Complete neutrino/muon binary classification dataset
- Variable-length sequence handling
- Memory-efficient loading
- Comprehensive test suite (4/4 tests passing)

**Iteration 3: Simple Model + Training** ✅ 100%
- Transformer encoder + MLP head (`NuMuClassifierModel`)
- Full training pipeline with early stopping, checkpointing, TensorBoard
- Comprehensive metrics: AUC, F1, precision, recall, confusion matrix

**Iteration 4: Domain Adaptation Trainer** ✅ 100%
- DANN adversarial training (MC → Exp domain adaptation)
- `src/training/da_numu_trainer.py` + `src/data/numu_dataset.py`
- GRL-based domain discriminator (`src/models/domain_discriminator.py`)
- Extensive lambda sweep experiments conducted (Oct–Nov 2025)
- **Best model**: `experiments/numu/da_numu_251123_small_moderatelambdmidddnoreg_aug_newlr`
  - Val AUC: **0.9777**, domain confusion achieved (~54%)

**Iteration 4b: Hit-Cut (hcut) Task** ✅ 100%
- Side branch: events with <5 signal hits relabeled as class 0 (indistinguishable from muons)
- `src/data/hcut_numu_dataset.py` + `src/training/da_hcut_numu_trainer.py`
- `data_manager/stats_dict/NuCut5hits_mc.yaml` normalization stats
- Lambda sweep experiments (Jan–Feb 2026, hcut3 and hcut5 variants)
- **Best model**: `experiments/numu/da_hcut5_numu_260206_smallnn_bigds_middleddnoreg_0.2lambda_aug_correct`
  - Val AUC: **0.9440**, domain confusion achieved (~52%)

### 🚀 Next Steps

**Iteration 8:** Validate quality and reliability of both best prefilter models on real experimental data.

**Iteration 9:** Signal-hits regime — train a high-purity neutrino classifier (>10⁶× background suppression) operating on signal-only hits, using an external hit-classifier NN for noise stripping on real data.

See `doc/tasklist.md` for the detailed roadmap.

## 📁 Key Files

### Core Implementation
- `src/data/numu_dataset.py` - Standard nu-mu dataset
- `src/data/hcut_numu_dataset.py` - Hit-cut variant dataset
- `src/models/base_models.py` - NuMuClassifierModel (Transformer + MLP)
- `src/models/domain_discriminator.py` - GRL-based domain discriminator
- `src/training/da_numu_trainer.py` - DANN domain adaptation trainer
- `src/training/da_hcut_numu_trainer.py` - hcut variant trainer
- `src/training/standard_numu_trainer.py` - Standard (no DA) trainer
- `data_manager/root2h5/` - ROOT→HDF5 conversion pipeline

### Configuration
- `experiments/da_numu_baseline.yaml` - Main DA config
- `experiments/da_hcut_numu_baseline.yaml` - hcut DA config
- `data_manager/stats_dict/default_mc.yaml` - Standard normalization stats
- `data_manager/stats_dict/NuCut5hits_mc.yaml` - hcut normalization stats

### Documentation
- `CLAUDE.md` - Project guidelines and rules
- `doc/tasklist.md` - Development roadmap and progress
- `doc/workflow.md` - Development process
- `doc/hdf5_format.md` - HDF5 data structure documentation

## 🔧 Environment Setup

```bash
conda activate baikal25
cd /home/albert/Baikal2025
```

## 📊 Data Overview

**MC Data:** `/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5`
**Exp Data:** `/home/albert/Baikal2025/data_manager/h5datasets/exp.h5`

**Particle types:** `muatm_2020` (background), `nue2_2020`, `nuatm_2020` (signal)

**Data characteristics:**
- Variable-length time series (35-200+ hits per event)
- 5D hit features: [amplitude, time, x, y, z]
- 1M GPU memory: ~2500 MiB per 1,000,000 events

---

*Project follows KISS principles with iterative development. See `doc/tasklist.md` for detailed roadmap.*