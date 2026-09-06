# OUTDATED (2026-09-06) — kept to reproduce historical runs, not for new work.
#
# The trainer this wrapper calls lives in an archive directory; the path below
# had been left pointing at the old location, so the wrapper could not run at
# all (doc/AUDIT.md §1.6). The path is now corrected.
#
# Status: superseded by src/training/da_prefilter_numu_trainer.py.
# Differs from the non-_big wrapper only in the config
# (da_prefilter_numu_hardlabels_big.yaml).

nohup python archive/260524/da_prefilter_numu_trainer_upsample.py --config experiments/da_prefilter_numu_hardlabels_big.yaml > experiments/logs/DA_PREFILTER_train_$(date +'%Y%m%d_%H%M%S').log &
