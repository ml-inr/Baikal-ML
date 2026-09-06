# Batch size is part of the sig-noise preprocessing definition (2026-08-16)

## The property

`gplotnikov_sig_noise_models/.../model_simplified.py:45` and `encoder.py:79` pass the
transformer padding mask to `nn.TransformerEncoder` as a **float** tensor:

```python
self.enc(x, src_key_padding_mask=(~mask).float())
```

PyTorch reads a float `src_key_padding_mask` as an **additive attention bias**, not as a
boolean "ignore these positions". Padding therefore stays visible to attention, and a hit's
probability depends on how much padding sits in its batch — i.e. on the longest event
batched with it.

Consequence: **batch size is part of the definition of the hit selection, not a performance
knob.** Change it and `n_sig_hits`, `n_sig_strings` and the h8s3 event selection move with
it. Measured on 4,000 muatm events (249,189 hits), same weights: bs 128 vs 512 gives
max|delta| 0.197 with 212 hits crossing the 0.8 cut; against a padding-free bs=1 reference,
bs=256 differs by up to 0.777 and moves 0.49% of all hits across the threshold.

## This is how the model was trained — `verified`

Not our wrapper's mistake. Grigory's training code on the cluster,
`/net/63/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/models/encoder.py:79`, carries
the identical line, and `train_config_mc_2020.yaml`'s `type: encoder_domain_adaptation`
reaches it through `EncoderDomainAdaptation.forward -> self.encoder(x, mask)`. Our
`encoder.py` is a trimmed copy of his file. Training used `batch_size: 128`.

Anchor: read directly from the code that produced the weights, not inferred from our data.

Do **not** "fix" the mask to boolean in isolation. It would put inference out of step with
training and with every probability already stored, and it would silently invalidate the
comparability of new results with old ones. If it is ever changed, everything downstream has
to be recomputed together.

## The convention: bs = 256

The stored MC probabilities were written at **batch size 256**, so 256 is the project
convention and MC stays as it is. Training used 128; the mismatch is accepted, because
internal consistency between MC and data matters more here than matching training exactly.

**The one real defect:** the experimental probabilities were written at **512**, so MC and
data did not receive the same hit selection — precisely the thing an MC-versus-data excess
measurement depends on. Only the exp side needs redoing.

| file | batch size | action |
|---|---|---|
| `baikal_mc_merged_probs_k_nsol_*.h5` | 256 | keep |
| `exp_full_probs_k_nsol_*.h5` | 512 | **recompute at 256** |
| `exp_probs_k_nsol_*.h5` | 512 | **recompute at 256** |

How large is the 256-vs-512 difference specifically? 82 of 249,189 hits change side of the
0.8 cut, i.e. **0.03%**. (The larger 0.49% figure above is 256 against a padding-free `bs=1`
reference, which is a different comparison.)

### What follows from the exp side

* `data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8` — built from the 512 probs;
  rebuild at 256.
* The **E-series training runs** (`260705_*_E1/E2/E3b`, `260707_*_E5`) used that dataset as
  their DA **target** while their MC **source** came from a 256-selected dataset, so the two
  domains were not selected the same way. Decision: retrain **E1** on the rebuilt dataset as
  a control; accept E2/E3b/E5 as they are unless E1 moves.
* The **May runs are clean**, including the canonical
  `260508_1724_..._seed32`: their DA target is the legacy `exp.h5` population inside
  `nu_classifier_dataset_h5s0_thr0.8`, built on the fly at 256 (`default_config.yaml`).
* `inference_v2/nu_classifier/preds/` — **do not rescore exp alone.** See the section below;
  the stored predictions do not come from the probs files at all.
* The fine-tuning background selection and the fine-tuned model — deferred until the excess
  has been re-measured with the base model, since fine-tuning is the mitigation for it.

MC datasets, all nu-classifier checkpoints other than E1, and all MC predictions are
unaffected and stay in use.

### The stored predictions were not made from the probs files at all

Found while preparing the rescore, and it corrects the section above. `predict_mc.py` and
`predict_exp.py` fall back to running sig-noise **on the fly** when `--probs-h5` is not
given, and — worse — they used the *nu-classifier's* batch size for it, which the runner
scripts set to **1024**. Only 6 of 69 MC logs and 24 of 40 exp logs used precomputed probs.

For the canonical checkpoint `260508_1724_..._seed32`, **every** run, MC and exp alike, went
the on-the-fly route at 1024. So its MC and exp predictions were produced the same way and
**are mutually consistent** — the "MC at 256 versus data at 512" mismatch never reached the
main analysis. It reached the artifacts built from the probs files: the `exp_full` NPY
dataset and, through it, the E-series training runs.

Consequence: rescoring exp at 256 while the MC predictions sit at 1024 would *create* the
inconsistency we are trying to remove. The rescore is all-or-nothing across MC and exp, so
it is deferred as a decision rather than done silently.

The excess can be re-checked without touching `preds/`: the
`analysis/exp_excess_investigation` builder reads stored probabilities for both MC and exp,
so with the exp file rebuilt at 256 both sides come from files written at the same batch
size.

Fixed in the code so the trap cannot be re-entered: `predict_mc.py` and `predict_exp.py` now
take a separate `--sn-batch-size` (default 256), decoupled from the classifier batch size,
and `run_info.json` records `sn_probs_source` and `sn_batch_size`.

### A limit worth knowing

Equal batch size does not strictly mean equal treatment. With a float mask, how much padding
an event sees depends on the *lengths of the other events in its batch*, and MC and
experimental data have different event-length distributions by nature. Matching the batch
size removes the procedural inconsistency, not this intrinsic one. Only `bs=1` — no padding
anywhere — is entirely neutral, and there the mask type stops mattering at all.

Note the prefilter is independent of all of this — it classifies raw hits and never calls
the sig-noise model.
