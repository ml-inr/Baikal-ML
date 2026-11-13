# SUPERSEDED — rebuild at batch size 256 (2026-08-16)

This dataset's hit selection comes from `exp_full_probs_*.h5` written at **batch size 512**,
while the MC source dataset (`nu_classifier_dataset_h5s0_thr0.8`) uses **256**. Because the
sig-noise model passes a float padding mask, batch size is part of the hit-selection
definition, so source and target domains were not selected the same way.

Consequence for training: the E-series runs
(`260705_*_E1/E2/E3b`, `260707_*_E5`) used this as their DA target against a 256-selected MC
source. E1 is being retrained on the rebuilt dataset as a control.

Context: `doc/sig_noise_batch_size.md`.
