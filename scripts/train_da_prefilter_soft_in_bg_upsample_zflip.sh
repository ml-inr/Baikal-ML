# OUTDATED (2026-09-06) — kept to reproduce historical runs, not for new work.
#
# The trainer this wrapper calls lives in an archive directory; the path below
# had been left pointing at the old location, so the wrapper could not run at
# all (doc/AUDIT.md §1.6). The path is now corrected.
#
# Status: this variant was PROMOTED to src/training/da_prefilter_numu_trainer.py
# (the two files agree to 0.99, doc/AUDIT.md §3.7). The archived copy is kept
# because 0.99 is not 1.00 and old runs should be reproduced with the exact
# file that produced them. For new work use
# scripts/train_da_prefilter_soft_in_bg.sh.

nohup python archive/260524/da_prefilter_numu_trainer_upsample_zflip.py --config experiments/da_prefilter_numu_softlabels_zflip.yaml > experiments/logs/DA_PREFILTER_train_$(date +'%Y%m%d_%H%M%S').log &
