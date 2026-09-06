# Report 2026-08-10 — two branches closed: SNGP does not fix the exp excess, and muon multiplicity is not the handle

Closes the last open thread from [../2026-07-08/REPORT.md](../2026-07-08/REPORT.md) §5, which
built SNGP (spectral-norm encoder + RFF-GP head + Laplace covariance + mean-field logit) as
the *principled* fix for the exp neutrino-like excess, and predicted only **partial** help
because the false-ν sit at the **edge** of the MC-ν manifold (kNN 3.0) rather than in a void.
The model was trained to convergence on 2026-07-09 (best epoch 55, val AUC 0.9999,
gp_var 0.0048) but never evaluated at a working point. It is now evaluated.

**Verdict: SNGP does not reduce the experimental excess.** It tracks the base DA model,
slightly worse. The fine-tune (report 07-08 §6) remains the only intervention that works.
The prediction of §8 is confirmed — and sharpened: the variance is correlated with
*ν-likeness*, not purely with OOD-ness, which is why it cannot separate the false-ν.

---

## 1. Inference tooling — `predict_sngp.py`

The standard `predict_mc.py` / `predict_exp.py` persist only `score`. For SNGP the
predictive variance `var = φᵀΣφ` is the whole point, and it is computed inside
`RandomFeatureGPHead.forward` and then discarded. New script
[`inference_v2/nu_classifier/predict_sngp.py`](../../predict_sngp.py) persists both:

    predictions(event_fk BIGINT PK, score FLOAT, gp_var FLOAT, n_sn_hits INT, n_sn_strings INT)

- mean-field score reproduced explicitly (`sigmoid(logit / sqrt(1 + λ·var))`), bit-identical
  to `model.classifier(emb)` in eval mode with a valid covariance;
- pre-processing (padding, mask, normalisation, amp clip) mirrors
  `model_utils.predict_scores` exactly → scores directly comparable with the DA/FT DBs;
- both sources: `mc_merged` (per-ptype groups) and `exp_full` (physical event_id from
  `header_prty`); per-part cap applied **before** reading hits (contiguous prefix of a part =
  one physical run, unbiased) so cost scales with the cap, not the part size.

**Out-of-training is automatic with `--probs-h5`:** the precomputed SN-probs file covers
*only* non-training parts (overlap with the training NPY part list is exactly 0 for all three
ptypes: muatm 10000 parts, nuatm 100, nue2 67). Verified explicitly.

Scored (moderate statistics, cuda:0, ~5 min total): mc_merged 516,840 events
(muatm 271k / nuatm 95k / nue2 150k), exp_full 114,832 (100k events/part × 29 parts).
At h8s3: muatm 193,003, nuatm 29,546, nue2 109,277, exp 84,563.
DBs: `preds/sngp_nu_classifier_baseline@best_sngp_model/{mc_merged,exp_full}_thr0p8.duckdb`.

*Sanity check on a possible bug:* SNGP's nuatm yield looked 13× below E1's. It is not a bug —
E1's DB scored nuatm on-the-fly over **2000** parts while the probs file covers **100**. On the
*same* 11 parts SNGP has 4,519 vs E1's 4,805 at h8s3 (difference = last part truncated by the
event budget).

## 2. Working-point comparison — the decisive test (`sngp_working_point.py`)

SNGP's mean-field shrinks *all* logits, so any fixed-threshold statement about it is
confounded by score compression — the trap documented for spectral norm (07-06 §1b). Every
model is therefore compared at a **fixed MC-muon survival μ**, the threshold re-derived per
model from its own out-of-training muatm scores. All three models are restricted to the
**same events** (intersection of scored `event_fk`): muatm 190,265, nuatm 29,021, nue2
106,980, exp 84,563 (FT background excluded; unlabelled DA-target exp kept, as always).

**Excess = (exp fraction above cut)/μ** (1.0 = exp behaves exactly like the MC muon background):

| μ | SNGP | base (E1) | fine-tuned |
|---|---|---|---|
| 1e-2 | 1.31 | 1.21 | **0.93** |
| 3e-3 | 1.48 | 1.36 | **0.91** |
| 1e-3 | **1.98** | **1.79** | **1.11** |
| 3e-4 | 2.25 | 1.89 | 1.10 |
| 1e-4 | 2.60 | 4.02 | 1.66 |

ν efficiency (equal-weighted nuatm+nue2) is the same for all three within ~1% at every point
(e.g. at 1e-3: SNGP 0.996, base 0.996, FT 0.994).

**SNGP ≈ base, consistently on the wrong side of it.** Only μ ≥ 1e-3 is statistically solid
here (193 muons at 1e-3); the 1e-4 and below points are Poisson noise (19 and fewer muons) and
the apparent SNGP<base at 1e-4 should not be read as a win.

*Cross-validation of the reference values.* The base/FT columns above were measured on the
small event set common to all three models. `ft_benefit_audit.py`, run independently on the
full statistics (23,879 muons at μ=1e-3 against 193 here), gives base **1.74×** → FT **1.09×**
at 1e-3 and 1.23× → 0.95× at 1e-2, against 1.79 → 1.11 and 1.21 → 0.93 here — agreement within
~3% across a 100× difference in muon statistics. The SNGP comparison is therefore anchored
correctly, even though its own statistics only support μ ≥ 1e-3.

**Confounder, stated honestly:** SNGP is MC-supervised with **no domain adaptation** (the DANN
was removed from the trainer), while the base model has DA λ=0.01. The λ=0 ablation (07-07 §2)
costs ~1.2× in excess; SNGP is ~1.1× worse than base. So SNGP performs like a no-DA model, and
distance-awareness buys nothing measurable on top. A fully matched test would need SNGP+DA.

## 3. Variance as an explicit OOD veto (`sngp_variance_veto.py`)

Even if the mean-field shrinkage is too weak, the per-event variance is still measured, and
the pilot (07-10) found it ~3× higher for exp false-ν than for genuine MC-ν. Tested as a
*separate cut*: fix the working point on SNGP's muatm scores, then additionally require
`gp_var < v`, scanning v over quantiles of the MC-ν variance, and recompute the muon survival
after the same cut so the working point stays matched.

At μ=1e-3 (score cut 0.632):

| variance cut (MC-ν quantile) | n_mu | excess | ν eff |
|---|---|---|---|
| none | 193 | 1.98 | 0.996 |
| 0.99 | 158 | 1.75 | 0.989 |
| 0.95 | 96 | 1.38 | 0.960 |
| 0.90 | 57 | 1.24 | 0.922 |

(rows below q=0.8 have ≤17 muons — noise, not reported.)

**The veto works, but it is dominated by the fine-tune.** It buys 1.98 → 1.38 at a 4% ν loss,
and 1.98 → 1.24 at an 8% ν loss; the fine-tune reaches **1.11 at a 0.2% ν loss**.

**Mechanism — why it costs so much signal.** Median `gp_var`: **MC-ν 0.0055**, exp 0.0026,
MC-μ 0.0022. The variance is *highest for the neutrino class itself*, not for the OOD events:
the ν region of RFF space is the sparsely covered one. So cutting on low variance is partly an
anti-ν cut, which is exactly why ν efficiency falls as fast as the excess does. The variance
measures "ν-like and rare", not "not-in-training-distribution".

## 4. Muon multiplicity — correcting report 07-09 §4 (`multimuon_baserate.py`)

Report [../2026-07-09/REPORT.md](../2026-07-09/REPORT.md) §4 reported "**79% of high-score
muatm are multi-muon (n_muons≥2, truth)**" and recommended, on that basis, an anti-multi-muon
cut on the candidate sample and a multi-task `n_muons` auxiliary head. That figure was quoted
without a base rate, and the sample it came from (`corner_cases.py`) was itself drawn with
`score>0.5`. Measured properly, in the population the classifier actually sees, the conclusion
**reverses**.

MC truth `n_muons = prime_prty[:,4]`, muatm:

| population | N | n_mu≥2 | n_mu≥3 | median n_mu |
|---|---|---|---|---|
| (a) unconditional (whole random parts) | 347,910 | 0.448 | 0.309 | **1** |
| (b) h8s3, any score — what the net sees | 40,000 | **0.984** | 0.968 | **14** |
| (c) h8s3, score>0.8 — the false positives | 7,573 | 0.685 | 0.509 | **3** |

**Enrichment of the false positives vs (b): 0.70× for n_mu≥2, 0.53× for n_mu≥3.** The false
positives are *depleted* in multiplicity, not enriched — they are the **low-multiplicity tail**
of the h8s3 population (median 3 muons against 14 for a typical h8s3 muatm event).

The physics is straightforward and, in hindsight, obvious: the h8s3 cut (≥8 signal hits on ≥3
strings) by itself selects bright, high-multiplicity bundles, so almost the entire training EAS
class is multi-muon. A 14-muon bundle looks like a broad shower and is confidently rejected;
what fools the classifier is the sparse, clean, track-like signature of a *single or few* muons
— which is exactly what a ν_μ CC event is.

Comparing (c) against (a) instead gives a spurious 1.64× "enrichment"; that comparison is
confounded, because the h8s3 cut is applied to (c) but not to (a).

**Consequences for the two proposed interventions.**
- *Reweighting EAS events with `n_muons>2`*: the condition covers **98.4%** of the h8s3 EAS
  training class, so it is effectively a uniform reweighting — a no-op — and to the extent it
  acts at all it weights *away* from the failure mode. If a multiplicity weight is used at all
  it must go the other way: **low**-multiplicity EAS (n_mu ≲ 3) are the genuine hard negatives
  (~15–30% of the class).
- *Multi-task `n_muons` head*: forces topological representation where multiplicity is high,
  whereas the failures live in the low-multiplicity regime where there is little topology to
  resolve. It targets the wrong regime.
- *Anti-multi-muon cut on the candidate sample* (07-09 §4): would preferentially remove the
  bundles the network already rejects, not the fakes. On the multiplicity axis the ordering is
  ν (1) < false-ν (3) < typical h8s3 muatm (14), so a "low multiplicity" selection keeps the
  fakes together with the signal.

Consistent with this, no truth-level feature in `tables/corner_cases_muatm.csv` separates the
false positives usefully (AUC vs the rest of that score>0.5 sample): `corr(t,z)` 0.656,
`n_mu` 0.563, `n_filt` 0.562, `two_clu` 0.558, `rvert` 0.539, `lo_drop` 0.539,
`corr_flip` 0.502. The strongest is the direction proxy — i.e. the very ambiguity that makes
these events hard, not a handle for removing them.

**Verdict: the muon-multiplicity direction is closed in both forms.** The residual failure mode
is a near-horizon single-muon topology, which differs from a ν_μ CC event only by direction —
the same axis the horizon θ-loss (E5, 07-08 §7) already attacked and lost signal on.

## Conclusions

1. **SNGP does not reduce the exp excess at any solid working point** (1.98 vs base 1.79 at
   μ=1e-3), with ν efficiency unchanged. Negative result, decisive within the statistics.
2. The 07-08 §8 prediction is **confirmed**: the false-ν sit at the edge of the MC-ν
   distribution, so the mean-field shrinkage is far too weak to move them below the cut.
3. **New mechanistic finding:** the GP variance is *anti-correlated with training density in
   the ν region*, i.e. it is highest for MC-ν themselves (0.0055 vs exp 0.0026, μ 0.0022).
   Distance-awareness in this embedding does not equal OOD-awareness for this failure mode.
4. As an explicit veto the variance is a real but weak signal (1.98→1.38 at −4% ν), strictly
   worse than the fine-tune (1.98→1.11 at −0.2% ν).
5. **Muon multiplicity is not a handle on the false positives (§4)** — corrects 07-09 §4.
   Against the population the classifier sees (h8s3, 98.4% multi-muon, median 14 muons), the
   false positives are *depleted* in multiplicity (0.70× for n_mu≥2, median 3). Reweighting
   `n_muons>2` would touch 98% of the EAS class; a multi-task `n_muons` head targets the
   high-multiplicity regime while the failures live in the low-multiplicity one.
6. **The fine-tune remains the operational fix.** SNGP joins the horizon θ-loss (E5, 07-08 §7),
   spectral norm / afterpulse augmentation (07-06) and now the muon-multiplicity direction as
   interventions that do not close the domain gap.

## Next steps (open)

- If SNGP is to be given a fair chance: retrain **SNGP + DA** (add the DANN branch back) so it
  is matched to the base model; only then is "distance-aware output" tested independently of
  the missing domain adaptation.
- Statistics here support μ ≥ 1e-3 firmly. Scaling muatm (10000 probs-covered parts available,
  2M events ≈ 70 s) and exp would extend to 1e-5, but the qualitative answer is already set.
- The multi-task `n_muons` thread is closed by §4. What remains of the corner cases is **case 1**
  (a single distant noise hit inflating the verticality) — an artefact, not a physical ambiguity,
  addressable by a leave-one-out consistency feature or outlier augmentation.
- Untouched threads: SNGP × fine-tune combination; ν-candidate event displays.

## Artifacts

| Path | What |
|---|---|
| `inference_v2/nu_classifier/predict_sngp.py` | SNGP inference persisting score + gp_var (mc_merged, exp_full) |
| `preds/sngp_nu_classifier_baseline@best_sngp_model/` | SNGP prediction DBs (mc_merged, exp_full) |
| `analysis/model_comparison/sngp_working_point.py` | §2 — matched working points, common events |
| `analysis/model_comparison/sngp_variance_veto.py` | §3 — variance as explicit OOD veto |
| `analysis/model_comparison/tables/sngp_working_point.csv` | excess / ν-eff per model per μ |
| `analysis/model_comparison/tables/sngp_variance_veto.csv` | veto trade-off scan |
| `analysis/model_comparison/figures/sngp_working_point.png` | excess and ν-eff vs μ, 3 models |
| `analysis/model_comparison/figures/sngp_variance_veto.png` | veto trade-off curves |
| `analysis/model_comparison/sngp_exp_eval.py` | earlier pilot (07-10): 3× variance separation |
| `analysis/model_comparison/multimuon_baserate.py` | §4 — multiplicity base rates (a)/(b)/(c) and the enrichment |
| `analysis/model_comparison/tables/multimuon_baserate.csv` | §4 — the three-population table |
