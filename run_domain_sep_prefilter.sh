#!/usr/bin/env bash
# Domain separability test for the prefilter feature space: MC muatm vs Exp.
# Compare with run_domain_sep_nu_classifier.sh to see which stage is harder to adapt.
# Run from project root.

set -euo pipefail

python src/training/domain_sep_trainer.py \
    --config experiments/domain_sep_prefilter.yaml
