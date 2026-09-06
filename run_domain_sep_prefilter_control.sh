#!/usr/bin/env bash
# Control check for prefilter: split MC muatm 50/50 and try to separate the two halves.
# Expected: val_domain_auc ≈ 0.5.  Compare against run_domain_sep_prefilter.sh.
# Run from project root.

set -euo pipefail

python src/training/domain_sep_trainer.py \
    --config experiments/domain_sep_prefilter_control.yaml
