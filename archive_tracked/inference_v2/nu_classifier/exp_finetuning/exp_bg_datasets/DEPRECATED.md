# SUPERSEDED — rebuild after the excess is re-checked (2026-08-16)

The fine-tuning background selection ran on experimental events whose hits were selected
from probabilities computed at **batch size 512**, inconsistent with the MC side at 256.

Deliberately not rebuilt yet: fine-tuning is the *mitigation* for the experimental excess,
so the excess is being re-measured with the base model on 256-selected data first. Whether a
fine-tuning set is needed at all, and on what selection, follows from that.

Context: `doc/sig_noise_batch_size.md`.
