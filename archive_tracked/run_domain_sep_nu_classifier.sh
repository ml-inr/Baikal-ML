#!/usr/bin/env bash
# Domain separability test: MC muatm vs Exp.
# Trains encoder + discriminator jointly (no GRL) to maximise domain separability.
# Run from project root.
#
# Edit experiments/domain_sep_nu_classifier.yaml to change device / paths before running.

set -euo pipefail

python src/training/domain_sep_trainer.py \
    --config experiments/domain_sep_nu_classifier.yaml
