# OUTDATED (2026-09-06) — kept to reproduce historical runs, not for new work.
#
# The trainer this wrapper calls lives in an archive directory; the path below
# had been left pointing at the old location, so the wrapper could not run at
# all (doc/AUDIT.md §1.6). The path is now corrected.
#
# Status: the hcut branch is closed (doc/AUDIT.md Q14) — it was one of the
# prefilter variants and gave no clear gain. There is no current replacement;
# this wrapper exists only to reproduce the historical runs.

nohup python src/training/archive/da_hcut_numu_trainer.py --config experiments/da_hcut_numu_baseline.yaml > experiments/logs/DA_HCUT_train_$(date +'%Y%m%d_%H%M%S').log &