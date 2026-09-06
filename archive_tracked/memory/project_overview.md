---
name: Project Overview
description: High-level summary of the Baikal2025 neutrino detection ML project — goals, architecture, and current status
type: project
---

Baikal2025 is a deep learning research project for neutrino detection using the Baikal-GVD underwater neutrino telescope.

**Core goal:** Binary classification of detector events — neutrino signal vs. atmospheric muon background — using variable-length time-series of detector hits (5D features: amplitude, time, x, y, z).

**Architecture:** Transformer encoder + MLP head (NuMuClassifierModel, ~541k–800k+ params). Domain Adaptation via DANN (Gradient Reversal Layer) to transfer from Monte Carlo (source) → experimental data (target).

**Data pipeline:** ROOT files → HDF5 (data_manager/root2h5/) → Parquet event catalogs (h5_catalogs/) → PyTorch Dataset → training.

**Completed iterations (1–7):** Project setup, data pipeline, PyTorch datasets, Transformer model + standard training, DANN domain adaptation trainer (Iter 4), hit-cut (hcut) task variant (Iter 4b), experiment management, inference notebooks, TensorBoard integration.

**Best models so far:**
- Prefilter DA: `experiments/numu/da_numu_251123_small_moderatelambdmidddnoreg_aug_newlr` — Val AUC 0.9777
- hcut5 DA: `experiments/numu/da_hcut5_numu_260206_smallnn_bigds_middleddnoreg_0.2lambda_aug_correct` — Val AUC 0.9440

**Next steps:**
- Iteration 8 (pending): Validate best prefilter models on real experimental data, compare with reco-based selection baseline
- Iteration 9 (pending): Signal-hits regime — high-purity classifier (>10⁶× background suppression) operating only on signal hits, using external hit-classifier NN for noise stripping on exp data

**Active research notes (as of ~Apr 2026):**
- Returned to hard labels (soft labels caused overconfident wrong predictions on exp data)
- DA presence seems marginally helpful on Upgoing Reco events
- Discrepancy with reco on which hits are signal — training on full MC sample may help
- TODO: train without DA for comparison; test Z-inversion augmentation in DA

**Why:** Physics goal is to identify upward-going neutrino events from within a vast muon background in a deep underwater Cherenkov detector.
