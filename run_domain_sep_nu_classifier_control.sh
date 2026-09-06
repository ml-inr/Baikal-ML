#!/usr/bin/env bash
# Control check: split MC muatm 50/50 and try to separate the two halves.
# Expected: val_domain_auc ≈ 0.5 (no real domain difference).
# Compare against run_domain_sep_nu_classifier.sh (MC vs Exp).
# Run from project root.

set -euo pipefail

python src/training/domain_sep_trainer.py \
    --config experiments/domain_sep_nu_classifier_control.yaml
