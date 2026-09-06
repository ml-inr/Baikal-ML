# How the predictions here were produced — read before comparing MC with data (2026-08-16)

**Corrected note.** An earlier version of this file said the `exp*.duckdb` files were stale
because they came from experimental probabilities computed at batch size 512 while MC used
256. That is wrong for most of them, and the truth matters more.

`predict_mc.py` and `predict_exp.py` compute sig-noise **on the fly** unless `--probs-h5` is
passed, and until today they used the nu-classifier's batch size for it — which the runner
scripts set to **1024**. Only 6 of 69 MC logs and 24 of 40 exp logs read precomputed
probabilities.

For the canonical checkpoint `260508_1724_..._seed32`, every run — MC and exp — took the
on-the-fly route at 1024. Its MC and exp predictions were therefore produced identically and
**are mutually consistent**. The 256-versus-512 problem never reached them; it reached the
artifacts built from the probs files (the `exp_full` NPY dataset, and through it the
E-series training runs).

## What this means in practice

* **Do not rescore exp alone.** Exp at 256 against MC predictions at 1024 would create the
  very mismatch this cleanup is about. The rescore is all-or-nothing across both sources,
  and it is a pending decision, not an oversight.
* **Check provenance before mixing checkpoints.** Runs differ in how their probabilities
  were obtained; a comparison across checkpoints can silently compare two hit selections.
  `grep "precomputed SN probs" <dir>/predict_*.log` tells you which route a run took.
* Newer runs record this properly: `run_info.json` now carries `sn_probs_source` and
  `sn_batch_size`, and both scripts take a separate `--sn-batch-size` (default 256).

The experimental excess is being re-examined through
`analysis/exp_excess_investigation`, which reads stored probabilities for both MC and exp —
so with the exp probs rebuilt at 256, both sides come from files written at the same batch
size, independently of anything in this directory.

Background: `doc/sig_noise_batch_size.md`.
