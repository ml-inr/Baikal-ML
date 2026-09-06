# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

*This file serves as the central hub for all development guidelines for the Baikal Neutrino Detection Neural Network Research Project.*

## Project Guidelines Overview

This project follows KISS (Keep It Simple, Stupid) principles for neural network research focused on binary classification of time-series data. All code generation and development decisions should reference these core documents:

### 📋 Core Documentation

**[vision.md](vision.md)** - Technical Architecture & Principles
- Technologies and development stack
- Project architecture and data pipeline
- Configuration management approach
- Experiment management strategy
- Primary reference for all technical decisions

**[conventions.md](conventions.md)** - Coding Standards
- Python and PyTorch coding patterns
- Data handling and configuration conventions
- Testing approaches for ML components
- Documentation and logging standards
- Follow these rules for all generated code

**[doc/workflow.md](doc/workflow.md)** - Development Process
- Iteration execution workflow
- Code review and validation checkpoints
- Version control and reproducibility practices
- Research experiment cycles
- Must follow this process for all development

**[doc/tasklist.md](doc/tasklist.md)** - Development Plan
- 8-iteration development roadmap
- Progress tracking and task management
- Testing requirements for each iteration
- Current project status and next steps

### 📦 Data Reference

**[doc/hdf5_format.md](doc/hdf5_format.md)** - layout of every HDF5 file (sources, inference
outputs, energy-truth companion)

**[doc/mc_provenance.md](doc/mc_provenance.md)** - where the MC comes from: production chain
(generator → `.dat` → `mcread` → `.wout` → `bexport-mc` → ROOT → h5), upstream file locations
on EOS, which production maps to which detector season, and how to match an event across the
chain. Read this before trusting any field's units — BARS comments disagree with the data.

**[doc/mc_binary_formats.md](doc/mc_binary_formats.md)** - exact word layout of the upstream
`.dat` / `.wout`, verified against ROOT; reader is
`data_manager/root2h5/energy_truth/read_wout.py`

**[doc/mc_energy_truth.md](doc/mc_energy_truth.md)** - what energy ground truth actually
exists: the three cases of `fMuonEnergy`, the targets built from the interaction chain, and
the limits of the propagation

**[doc/reco_quality_cuts.md](doc/reco_quality_cuts.md)** - the recommended BARS quality cuts
for reco events: where each quantity lives, mixed units, unfilled fields, measured pass rates
in both samples, and the known faults (cluster 1, bad charges, multi-cluster fragments)

**[doc/claims_log.md](doc/claims_log.md)** - every factual claim the project relies on, with
its evidence and status (`verified` / `consistent` / `assumed` / `refuted`). Check here before
relying on a statement, and update it when a claim's status changes.

## Verifying Claims

Research claims here have repeatedly been documented as established and turned out wrong. Every
one of those failures shared a cause: the check tested **internal consistency** — one field
against another from the same file, or a pattern against an already-formed hypothesis — instead
of anchoring to something outside the data. More checking does not fix this; different checking
does.

**Before writing a factual claim into `doc/`:**

- **Anchor externally.** A claim about units or semantics is not established until it is checked
  against something outside the dataset: physics that holds independently, a production whose
  name states its range, an independent implementation, or the code that writes/reads the
  format. Comparing two fields of the same file is `consistent`, never `verified`.
- **Read the specification before decoding data.** If a parser, writer or schema exists, read it
  first. Reverse-engineering a format by pattern-matching is a last resort, and it produced a
  day of wrong conclusions when `Task_MAIN.cpp` was two steps away.
- **Separate "consistent with" from "proves".** State explicitly which rival hypotheses a test
  can and cannot distinguish. `Reg ≤ E_primary` proves the two share a unit, not which unit.
- **Name the falsifying observation before collecting evidence.** "If this is true, X; if I see
  Y, it is dead." Skipping this is what turns a search into a search for confirmation.
- **Isolate before editing code.** Reproduce the failure in the smallest setting that shows it
  (one element vs a batch, one file vs a run) rather than guessing at a fix; guessing has twice
  made things worse here.
- **Mark confidence in the text.** Write `verified` / `consistent` / `assumed` where a reader
  would otherwise have to trust the tone, and log the claim in
  [doc/claims_log.md](doc/claims_log.md).
- **Record refutations, don't overwrite them.** When a claim flips, say so in place and add the
  bad argument to the log — the failure modes repeat.

**Confidence is part of the answer.** Reporting a result without saying how well it is
established is an incomplete report, not a concise one.

## Code Generation Rules

**ALWAYS follow these guidelines when generating code:**

### 🏗️ Technical Requirements (from conventions.md)
- Use Python 3.10+ with PyTorch
- Follow PEP 8 with 88-character line limit
- Type hints for all function signatures
- Config-driven everything (YAML files)
- Reproducible experiments (seed management)
- Modular design with clear separation

### 🔄 Development Process (from doc/workflow.md)
1. **Plan Phase**: Propose solution with code snippets
2. **Wait for Agreement**: Never implement without approval
3. **Implementation**: Follow conventions.md standards
4. **Validation**: Test according to iteration requirements
5. **Wait for Confirmation**: Get user approval before proceeding
6. **Progress Update**: Mark tasks complete in tasklist.md
7. **Commit**: Save working state with descriptive message

### 🧪 Research Focus
- **Primary Task**: Binary classification on time-series data
- **Data Flow**: ROOT files → HDF5 → PyTorch → trained models → predictions
- **Tracking**: TensorBoard experiment tracking from day 1
- **Reproducibility**: Same config → same results

## Key Reminders

**Never do without user approval:**
- Implement code solutions
- Move to next development iteration
- Make architectural decisions
- Add complexity or features

**Always include:**
- Type hints and docstrings
- Error handling and logging
- Configuration loading from YAML
- Device handling (CPU/GPU)
- Reproducibility features (seeds, deterministic)

**Research-specific requirements:**
- Handle variable-length time series
- Binary classification metrics (AUC, F1, precision, recall)
- Model checkpointing and best model saving
- TensorBoard experiment tracking
- Separate train/val/test HDF5 files

## Common Development Commands

### Environment Setup
```bash
# Activate conda environment
conda activate baikal25

# Verify environment works
python -c "import torch, numpy, h5py; print('Environment OK')"
```

### Running Tests  
```bash
# Test the PyTorch dataset
cd src/data && python test_numu_dataset.py

# Run basic dataset verification
python -c "from src.data.numu_dataset import NuMuDataset; print('Dataset import OK')"

# Test training pipeline
python src/training/standard_numu_trainer.py --config experiments/standard_neutrino_baseline.yaml --debug

# Test domain adaptation pipeline
python test_da_training.py

# Train domain adaptation model
python src/training/da_numu_trainer.py --config experiments/da_neutrino_baseline.yaml --debug
```

## Key Data Information

### Data Location
- **Main MC Data**: `data_manager/data/h5datasets/baikal_mc_merged.h5`
- **Experimental Data**: `data_manager/data/h5datasets/exp.h5` (top-level group: `"exp"`, legacy 25k-capped, ~650K events)
- **Full-Stat Exp Data**: `data_manager/data/h5datasets/exp_full.h5` (top-level group: `"exp_full"`, ~133M events, 29 clean runs c02–c07, has `header_prty` with physical event_id = CC timestamp)
- **Exp Reco Data**: `data_manager/data/h5datasets/exp_reco.h5` (top-level group: `"exp_reco"`)
- **Precomputed Probs**: `data_manager/data/h5datasets/baikal_mc_merged_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5`
- **NPY Datasets**: `data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8/`
- **Parquet Catalogs**: `data_manager/h5_catalogs/`
- **Data Format**: HDF5 files with variable-length time series
- **Features**: 5D hit data [amplitude, time, x, y, z coordinates]
- **Task**: Binary classification (neutrino=True vs muon=False events)

### Data Characteristics
- Variable sequence lengths: 35-200+ hits per event
- Time-series data with temporal ordering
- 3D spatial coordinates (cluster-centered)
- Multiple particle types: `muatm_2020`, `nue2_2020`, `nuatm_2020`

## Architecture Guidelines

### Two-Stage Detection Pipeline

1. **Prefilter** (`inference/prefilter_model/`) — input: all raw hits (5D). Reduces EAS background ×10–20.
   Trainer: `src/training/da_prefilter_numu_trainer.py`
   Canonical model: `experiments/numu/da_prefilter_numu_260429_0155_..._SoftFocalLoss/`

2. **Nu-classifier** (`inference/nu_classifier_model/`) — input: sig-noise-filtered hits (thr=0.8). High-purity neutrino selection.
   Trainer: `src/training/da_nu_classifier_trainer.py`
   Canonical model: `experiments/numu/260508_1724_da_nu_classifier_h5s0_lambda0.01_thr0.8_seed32/`

**External dependency**: `gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/sig_noise_model_v3.py`
**NPY dataset builder**: `data_manager/nu_classifier_ds_builder/` (HDF5 → memory-mapped NPY arrays for training)

### Current Implementation Status
- ✅ **Data Pipeline**: ROOT → HDF5 → NPY datasets → PyTorch training
- ✅ **Variable Sequences**: Proper handling with padding/masking
- ✅ **Binary Classification**: Neutrino detection ready
- ✅ **Domain Adaptation**: DANN trainer with GRL (prefilter + nu-classifier)
- ✅ **inference_v2/**: Clean predict scripts (npy/mc/exp), incremental DuckDB output, batch runner, UMAP analysis
- ✅ **Batch Predictions**: 8 nu-classifier checkpoints being scored across mc_merged/mc_reco/exp sources
- ⏳ **Next**: UMAP analysis, further representation-space studies, model comparison on exp data

### Key Design Patterns
- **Config-driven everything**: All behavior controlled by YAML files
- **Variable-length sequences**: Custom collate functions for batching
- **Memory efficiency**: Lazy loading with `max_events` limits
- **Physics-informed**: Leverage 3D spatial and temporal structure

### Testing Strategy
- Integration tests for data loading pipeline
- Unit tests for individual components  
- Validation against known physics properties
- Reproducibility tests with fixed seeds

## Current Status

Check [doc/tasklist.md](doc/tasklist.md) for current iteration progress and next steps in the development plan.

**Current**: Iterations 1–7 complete. `inference_v2/` pipeline fully operational — 8-model batch predictions running across mc_merged/mc_reco/exp/npy sources, UMAP analysis tooling built. Next: representation-space analysis and model comparison on experimental data.

---

*Follow these guidelines consistently to maintain code quality, research reproducibility, and project coherence.*