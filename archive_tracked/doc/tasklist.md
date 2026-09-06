# Development Task List

## Progress Report

| Iteration | Task | Status | Progress |
|-----------|------|--------|----------|
| 1 | Project Setup | ✅ Complete | ✅ 100% |
| 2 | Basic Data Manager | ✅ Complete | ✅ 100% |
| 2.5 | Adapt Data Manager | ✅ Complete | ✅ 100% |
| 2.6 | PyTorch Dataset | ✅ Complete | ✅ 100% |
| 3 | Model + Training Pipeline | ✅ Complete | ✅ 100% |
| 4 | Domain Adaptation Trainer (prefilter) | ✅ Complete | ✅ 100% |
| 4b | Hit-Cut (hcut) Task | ✅ Complete | ✅ 100% |
| 5 | Experiment Management | ✅ Complete | ✅ 100% |
| 6 | Inference & Reporting Notebooks | ✅ Complete | ✅ 100% |
| 7 | TensorBoard Integration | ✅ Complete | ✅ 100% |
| 8 | Model Validation on Experimental Data | ⏳ Pending | ⬜ 0% |
| 9 | Signal-Hits Regime (precision selection) | ⏳ Pending | ⬜ 0% |

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

---

## Iteration 2: Basic Data Manager
**Goal:** Convert ROOT files to HDF5 format

### Tasks:
- [x] ROOT file reader with PyROOT
- [x] HDF5 writer with structured output
- [x] YAML config loading
- [x] Main processing script

---

## Iteration 2.5: Adapt Data Manager
**Goal:** Customize for Baikal detector data and physics analysis

### Tasks:
- [x] Complete `root2h5` module with multiprocessing support
- [x] Production-ready scripts for MC, experimental, and reconstructed data
- [x] Physics-specific time residual calculations (`eval_tres.py`)
- [x] Advanced data handling for single/multi-cluster events
- [x] Quality control and coordinate transformations

**Implementation:** `data_manager/root2h5/` — 3 variants (MC/exp/reco), YAML configs, multiprocessing

---

## Iteration 2.6: PyTorch Dataset
**Goal:** PyTorch dataset for neutrino/muon binary classification

### Tasks:
- [x] `NuMuDataset` in `src/data/numu_dataset.py`
- [x] Variable-length sequence handling with custom collate function
- [x] Dict-based API: `{'features', 'labels', 'lengths', 'event_id'}`
- [x] Memory-efficient loading, normalization, augmentation support
- [x] Test suite: 4/4 tests passing

---

## Iteration 3: Model + Training Pipeline
**Goal:** Train binary classifier for neutrino detection

### Tasks:
- [x] `NuMuClassifierModel` in `src/models/base_models.py` — Transformer encoder + MLP head
- [x] Comprehensive metrics in `src/training/metrics.py` — AUC, F1, precision, recall, confusion matrix
- [x] Full training pipeline in `src/training/standard_numu_trainer.py`
- [x] Early stopping, learning rate scheduling, model checkpointing
- [x] YAML-driven config system (`experiments/standard_neutrino_baseline.yaml`)

---

## Iteration 4: Domain Adaptation Trainer (prefilter regime)
**Goal:** DANN adversarial domain adaptation for MC→Exp transfer learning

**Status:** ✅ **COMPLETED**

**Context:** Models in this iteration use **all hits** in each event (signal + noise). They are designed for a **prefilter** role: reducing atmospheric muon background by ×10–20, without precise neutrino selection.

### Tasks:
- [x] `src/training/da_numu_trainer.py` — DANN trainer with GRL
- [x] `src/models/domain_discriminator.py` — gradient reversal layer domain discriminator
- [x] Dual data loading: MC (source, labelled) + Exp (target, unlabelled)
- [x] Loss: L_class + λ * L_domain with progressive lambda scheduling
- [x] `experiments/da_numu_baseline.yaml` — DA config
- [x] Extensive lambda sweep experiments (Oct–Nov 2025)

**Results:**
- Best model: `experiments/numu/da_numu_251123_small_moderatelambdmidddnoreg_aug_newlr`
- Val AUC: **0.9777**, domain discriminator accuracy: ~54% (confusion achieved)
- Validation on real experimental data: **pending (Iteration 8)**

---

## Iteration 4b: Hit-Cut (hcut) Task
**Goal:** Modified nu-mu classification with physically motivated label redefinition

**Status:** ✅ **COMPLETED**

**Context:** Events with fewer than 5 signal hits are physically indistinguishable from atmospheric muon background. These events are relabeled as class 0 (background), making the classification task more physically meaningful. Still uses all hits as input (prefilter regime).

### Tasks:
- [x] `src/data/hcut_numu_dataset.py` — dataset with hit-cut label logic
- [x] `src/training/da_hcut_numu_trainer.py` — DA trainer for hcut task
- [x] `data_manager/stats_dict/NuCut5hits_mc.yaml` — normalization stats
- [x] `experiments/da_hcut_numu_baseline.yaml` — hcut DA config
- [x] Lambda sweep experiments (hcut3 and hcut5 variants, Jan–Feb 2026)

**Results:**
- Best model: `experiments/numu/da_hcut5_numu_260206_smallnn_bigds_middleddnoreg_0.2lambda_aug_correct`
- Val AUC: **0.9440**, domain discriminator accuracy: ~52% (confusion achieved)
- Validation on real experimental data: **pending (Iteration 8)**

---

## Iteration 5: Experiment Management
**Goal:** Organize experiments and track results

**Status:** ✅ **COMPLETED** — implemented organically as part of the training pipelines

### Completed:
- [x] Automatic experiment directory creation under `experiments/numu/<name>/`
- [x] Config saved to experiment folder (`da_config.yaml`)
- [x] Per-epoch CSV training history (`da_training_history.csv`)
- [x] Final summary YAML (`da_training_summary.yaml`)
- [x] Model summary file (`model_summary.txt`)
- [x] Best model and periodic checkpoint saving

---

## Iteration 6: Inference & Reporting Notebooks
**Goal:** Load models, make predictions, evaluate and visualize results

**Status:** ✅ **COMPLETED** — implemented as Jupyter notebooks in `inference/`

**Note:** A script-based inference pipeline was replaced with per-task notebooks, which allows flexible, task-specific analysis pipelines. Each notebook loads a model and produces plots and metrics for a specific experiment or comparison.

### Completed:
- [x] `inference/da_numu_report.ipynb` — full report for standard DA models
- [x] `inference/da_numu_report_hcut_2026.ipynb` — report for hcut models
- [x] `inference/compare2reco.ipynb` / `compare2reco_BothDAandNoDA.ipynb` — comparison with reco-based selection
- [x] `inference/pr_curves_da_numu_251123.ipynb` — PR curves for best prefilter model
- [x] `inference/mc_exp_compare.ipynb` — MC vs Exp distribution comparison
- [x] `inference/dd_weights_analysis.ipynb` — domain discriminator analysis

---

## Iteration 7: TensorBoard Integration
**Goal:** Experiment tracking and visualization

**Status:** ✅ **COMPLETED** — integrated in all trainers

### Completed:
- [x] TensorBoard SummaryWriter in all training pipelines
- [x] Per-epoch logging: loss, AUC, F1, accuracy, domain accuracy, lambda
- [x] Logs stored in `experiments/numu/<name>/tensorboard/`

**Usage:** `tensorboard --logdir experiments/numu/`

---

## Iteration 8: Model Validation on Experimental Data
**Goal:** Verify quality and reliability of the best prefilter models on real Baikal-GVD data

**Status:** ⏳ **Pending**

### Tasks:
- [ ] Run best prefilter model (`da_numu_251123_*`) on full experimental dataset
- [ ] Run best hcut model (`da_hcut5_numu_260206_*`) on full experimental dataset
- [ ] Compare MC and Exp score distributions (domain alignment check)
- [ ] Compare with existing reconstruction-based selection (reco baseline)
- [ ] Evaluate at physics-relevant operating points (high recall, high background rejection)
- [ ] Document findings and decide which model(s) to carry forward

**Key question:** Does DA actually improve MC→Exp generalization vs a no-DA baseline at the same architecture?

---

## Iteration 9: Signal-Hits Regime (precision selection)
**Goal:** Train a high-purity neutrino classifier operating on signal hits only, achieving >10⁶× background suppression with maximum neutrino recall

**Status:** ⏳ **Pending**

**Context and motivation:**
- Prefilter models (Iterations 4/4b) use all hits including noise → limited by noise contamination
- This iteration uses **only signal hits** per event, stripping noise before classification
- On MC: noise stripping via ground truth hit labels (training) **or** hit-classifier NN outputs (testing both regimes)
- On Exp: noise stripping via an external hit-classifier NN (treated as a ready external dependency)
- Hard quality cut applied: **≥8 signal hits on ≥2 strings** (events not meeting this are discarded, not classified)
- DA technique still applied: MC (source) → Exp (target)
- Architecture: same Transformer backbone as prefilter models

**This is a new full research branch**, analogous to but independent from Iterations 4/4b.

### Tasks:
- [ ] Create `src/data/signal_hits_dataset.py` — dataset supporting both GT and hit-classifier NN input modes
- [ ] Implement quality cut (≥8 signal hits on ≥2 strings) in dataset
- [ ] Create `src/training/da_signal_hits_trainer.py` — DA trainer for signal-hits regime
- [ ] Define normalization stats for signal-hits regime (`data_manager/stats_dict/`)
- [ ] Create baseline experiment config (`experiments/da_signal_hits_baseline.yaml`)
- [ ] Train and evaluate with GT signal hits (upper-bound performance reference)
- [ ] Train and evaluate with hit-classifier NN outputs (realistic inference regime)
- [ ] Compare GT vs NN-stripped results to quantify hit-classifier quality impact
- [ ] Validate on experimental data with DA

**Test:** Background suppression >10⁶× at neutrino recall comparable to or better than prefilter regime threshold

---

## Testing Strategy

Each iteration includes:
- **Unit Tests:** Test individual functions work correctly
- **Integration Tests:** Test modules work together
- **Validation:** Verify outputs make scientific sense
- **Reproducibility:** Same config → same results

## Notes

- Start each iteration only after previous one is fully tested
- Keep MVP mindset — add minimal functionality to achieve iteration goal
- Document issues and lessons learned after each iteration
- Update progress table after completing each iteration
