# OUTDATED (2026-09-06) — kept to reproduce historical runs, not for new work.
#
# The trainer this wrapper calls lives in an archive directory; the path below
# had been left pointing at the old location, so the wrapper could not run at
# all (doc/AUDIT.md §1.6). The path is now corrected.
#
# Status: superseded by the two-stage pipeline (prefilter -> nu-classifier).
# For new work use scripts/train_da_prefilter_hard_in_bg.sh or
# scripts/train_da_nuclassifier_in_bg.sh.

nohup python src/training/archive/da_numu_trainer.py --config experiments/da_numu_baseline.yaml > experiments/logs/DA_train_$(date +'%Y%m%d_%H%M%S').log &