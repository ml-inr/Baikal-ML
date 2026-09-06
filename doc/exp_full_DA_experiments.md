# DA nu-classifier — exp_full experiment matrix

Domain-adaptation retraining of the nu-classifier to fix two problems found on
real exp data (see `inference_v2/nu_classifier/analysis/domain_gap/`):
**afterpulses** (hardware hits absent from MC, mistaken for signal) and
**feature collapse / OOD** (exp configurations unseen in MC → over-confident
false "neutrino" predictions; domain classifier separates exp-hi vs MC-μ at
AUC≈0.85).

Three options, tested alone and combined:
- **Opt 1** — larger real target for DA: `exp_full.h5` (~5M events, bad runs
  `c02_r0020/r0249` excluded) instead of `exp.h5` (~650k).
- **Opt 2** — spectral norm on the encoder (`model.spectral_norm.enabled`).
- **Opt 3** — synthetic afterpulse aug into MC source only
  (`dataloader.augmentation.afterpulse`): random-OM, random-time hit(s) with
  Q~U(5,100). **Per-event sequential draw of 1..3** afterpulses (roll the next
  only if the previous fired → P(1)>P(2)>P(3)). Sub-options by base probability:
  **3a** p=0.02 (elevated-physical), **3b** p=0.25 (robust).

All runs share `src/training/da_nu_classifier_trainer.py` and the base config
`experiments/da_nu_classifier_exp_full.yaml`. Between runs, edit only the marked
fields (below) and `experiment.name`. lambda is swept ∈ {0.01, 0.1} via
`domain_adaptation.lambda_scheduler.max_lambda`.

## Matrix (12 runs = 6 configs × 2 lambda)

| # | run name | Opt | lambda | `spectral_norm.enabled` | `afterpulse` (enabled/prob) | purpose |
|---|---|---|---|---|---|---|
| 1 | `exp_full_E1_lambda0.01` | 1 | 0.01 | false | off | baseline: does more target data help DA |
| 2 | `exp_full_E1_lambda0.1`  | 1 | 0.10 | false | off | stronger DA with big target |
| 3 | `exp_full_E2_lambda0.01` | 1+2 | 0.01 | **true** | off | + spectral norm vs collapse |
| 4 | `exp_full_E2_lambda0.1`  | 1+2 | 0.10 | **true** | off | |
| 5 | `exp_full_E3a_lambda0.01`| 1+3a | 0.01 | false | **true / 0.02** | + afterpulse (physical rate) |
| 6 | `exp_full_E3a_lambda0.1` | 1+3a | 0.10 | false | **true / 0.02** | |
| 7 | `exp_full_E3b_lambda0.01`| 1+3b | 0.01 | false | **true / 0.25** | + afterpulse (robust rate) |
| 8 | `exp_full_E3b_lambda0.1` | 1+3b | 0.10 | false | **true / 0.25** | |
| 9 | `exp_full_E4a_lambda0.01`| 1+2+3a | 0.01 | **true** | **true / 0.02** | all options (physical) |
| 10| `exp_full_E4a_lambda0.1` | 1+2+3a | 0.10 | **true** | **true / 0.02** | |
| 11| `exp_full_E4b_lambda0.01`| 1+2+3b | 0.01 | **true** | **true / 0.25** | all options (robust) |
| 12| `exp_full_E4b_lambda0.1` | 1+2+3b | 0.10 | **true** | **true / 0.25** | |

Common: target `data_manager/datasets/nu_classifier_dataset_exp_full_thr0.8`,
source `..._h5s0_thr0.8` (5M), seed 42, focal loss, 100 epochs w/ early stop.

## Run procedure (per row)

1. In `experiments/da_nu_classifier_exp_full.yaml` set: `experiment.name`,
   `model.spectral_norm.enabled`, `dataloader.augmentation.afterpulse.enabled` +
   `.prob`, and `domain_adaptation.lambda_scheduler.max_lambda`.
2. `nohup python src/training/da_nu_classifier_trainer.py --config experiments/da_nu_classifier_exp_full.yaml > experiments/numu/<run_name>.log 2>&1 &`
3. Output → `experiments/numu/<run_name>/`.

## Evaluation (out-of-training)

Per checkpoint: `predict_exp.py` on exp_full (held-out test set, exclude training
back-links) + `predict_mc.py` on out-of-training MC → run the `domain_gap/`
notebooks. Success = domain-classifier AUC ↓ toward 0.5 and/or exp high-score
fraction ↓ toward the physical ~10⁻⁶ (from ~10⁻³), MC signal/bg separation intact.

## Results

### Round 1 — E1/E2/E3b `best_da_model` @ lambda 0.01 (2026-07-06, out-of-training test)

⚠️ **Not a matched working point:** each run's `best_da_model` sits at a very
different epoch (E1=23, E2=43, E3b=37), so this compares models at different
points on their training curves. See Round 2 for the matched-epoch comparison.

Test inference: MC ~700k/ptype (mc_merged, training excluded via back-links) +
exp_full held-out (bad runs c02_r20/r249 dropped, training excluded).
Analysis: `inference_v2/nu_classifier/analysis/model_comparison/compare_da_models.py --preset best`
(→ `comparison_table_best.csv`, `mc_survival_curves_best.png`, `exp_score_tail_best.png`).

| run | options | AUC | ν-eff@0.8 | μ-suppr@0.8 | ν-eff@μ=1e-3 | exp N | exp >0.8 | **exp frac>0.8** |
|---|---|---|---|---|---|---|---|---|
| E1 baseline      | 1     | 0.99991 | 96.71% | 2351× | 98.11% | 1,180,477 | 1,795 | **0.152%** |
| E2 spectral-norm | 1+2   | 0.99990 | 96.42% | 2618× | 98.00% | 1,180,477 | 1,566 | **0.133%** ✓ best |
| E3b afterpulse   | 1+3b  | 0.99992 | 96.84% | 2523× | 98.29% | 1,180,477 | 2,040 | **0.173%** ✗ worse |

**Findings:**
- **MC separation intact for all three** — AUC≈0.9999, ν-eff≈96–97%, μ-suppression
  ~2.4–2.6k×. No modification degraded MC discrimination.
- **Spectral norm (E2) wins on exp:** exp neutrino-like fraction 0.133% vs 0.152%
  baseline (**−13%**). `exp_score_tail.png` shows E2 pulls exp events out of the
  over-confident tail (>0.9, E2 lowest) into the mid-uncertain band (0.05–0.5, E2
  highest) — the anti-collapse / OOD-calibration signature.
- **Afterpulse aug (E3b) backfired:** exp fraction rose to 0.173% (**+14%**). p=0.25
  synthetic afterpulses did not help exp generalisation (possibly wrong Q/topology
  model, or the net learned afterpulses as a *signal* cue).
- **Domain gap not closed:** all still ~1.3–1.7×10⁻³, ~1000× the physical ~10⁻⁶.
  Spectral norm is directionally right but insufficient alone.

Next candidates: E2 at lambda 0.1 (stronger DA push), spectral_norm.coeff<1 +
pre-norm (norm_first), drop/retune afterpulse.

### Round 2 — matched working point (anchor E1 @ epoch 10)

Motivation: Round 1 compared `best_da_model`s at different epochs (23/43/37).
To isolate the *modification* effect from the *training-point* effect, we anchor
at E1 @ epoch 10 and match E2/E3b on **val_loss (primary) + F1 (control)**.
AUC is unusable as a matching axis — it is saturated (≈0.9999, run-range ~1e-4,
argmin dominated by noise). Matched checkpoints:

| model | matched epoch | criterion |
|---|---|---|
| E1  | 10 | anchor |
| E2  | 22 | val_loss / AUC match |
| E2  | 28 | F1 match (bracket — check ordering robust to metric) |
| E3b | 10 | val_loss / AUC (F1 → 8, ≈same) |

Inference: `run_matched_epoch_infer.sh` → analysis `compare_da_models.py --preset matched`
(→ `comparison_table_matched.csv`, `mc_survival_curves_matched.png`,
`exp_score_tail_matched.png`, `score_hist_muatm_vs_exp_matched.png`,
`score_hist_per_model_matched.png`).

Score histograms make the artifact visual: `score_hist_per_model_matched.png`
(exp black vs MC-muatm red per model, with the fixed-0.8 line and each model's
μ=1e-3 working-point cut, dashed) shows E2's whole muatm distribution shifted
left (μ-WP cut 0.70/0.71 vs 0.75 for E1/E3b), while exp sits ~the same factor
above muatm in every panel → gap unchanged. `score_hist_muatm_vs_exp_matched.png`
overlays all models' muatm (left) and exp (right): E2 shifts both by the same
amount.

**Results (out-of-training, exp held-out 1.18M, MC sig=565k/bg=207k):**

| model (matched) | AUC | sig_eff@μ=1e-3 | exp>0.8 (fixed thr) | **exp@μ=1e-3 (fixed WP)** |
|---|---|---|---|---|
| E1 @ep10  | 0.99989 | 97.69% | 0.148% | **0.252%** |
| E2 @ep22  | 0.99988 | 97.66% | 0.093% | **0.261%** |
| E2 @ep28  | 0.99989 | 97.78% | 0.108% | **0.255%** |
| E3b @ep10 | 0.99988 | 97.57% | 0.155% | **0.259%** |

**Conclusion — no modification closed the OOD gap (at lambda 0.01).**
The Round-1 / fixed-0.8 signal "E2 (spectral norm) has fewer exp neutrino-like
events" is an **artifact**: spectral norm's Lipschitz bound compresses *all*
scores downward (signal, muon, exp alike), so at a fixed 0.8 threshold fewer of
everything passes (E2: μ-suppr 3336× vs 1915×, sig_eff 95% vs 96.5% — uniformly
more conservative). Normalising to a **fixed MC muon-suppression working point**
(cut giving μ-survival = 1e-3 per model) removes this global rescaling, and the
exp OOD false-neutrino fraction is then **equal across E1/E2/E3b (~0.25%, spread
~1-2σ over ~2970 events)**. The exp↔MC-muon domain gap (exp survives ~2.5× more
than MC muons at equal WP) is unchanged by spectral norm or afterpulse.

Methodology note: matching epochs (not comparing `best_da_model`s at 23/43/37)
AND comparing at a fixed working point (not a fixed score threshold) were both
necessary — either shortcut gave a spurious "E2 wins".

Next: stronger DA (lambda 0.1), spectral_norm.coeff<1 (tighter Lipschitz) +
pre-norm, or a distance-aware output (SNGP/DUQ-style) — a fixed-threshold score
compression is not enough; the gap must be closed at the working point.
