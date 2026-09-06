# Design — horizon-aware (θ) loss for the nu-classifier (E1 analog)

**Motivation** (see `inference_v2/nu_classifier/reports/2026-07-06/REPORT.md`): the exp
neutrino-like excess is a **horizon (θ≈90°) up/down confusion**. In MC the class
label is almost perfectly θ-separated — **muons θ>90 (down), neutrinos θ<90 (up)** —
so the classifier learned a θ≈90 decision boundary. Right at the boundary the
up/down cue degenerates; the exp domain shift then tips near-horizontal events into
confident "neutrino". Fix: **teach the net to abstain (p≈0.5) for MC events near
θ=90°**, so it is not confident where direction is unknowable.

## Data (DONE)
`theta.npy` (per-event GT zenith, deg) added to
`data_manager/datasets/nu_classifier_dataset_h5s0_thr0.8/` via
`data_manager/nu_classifier_ds_builder/add_theta.py` (reads `prime_prty[:,0]` through
the dataset back-links; 5,256,496 events, 0 NaN). Per-class θ: muatm med 143.7 [94,180];
nuatm 26.7 [0,90]; nue2 53.2 [0,90]. Target (exp) has **no** θ — the loss is
source-only, so this is fine.

## Loss formulation
Horizon weight per source event (1 at horizon, →0 away):
```
w(θ) = max_soften · exp(-(θ-90)² / (2 σ²))            # σ in degrees
```
**Primary: soft-label.** Pull the BCE/soft-focal target toward 0.5 near the horizon:
```
t_soft = y·(1-w) + 0.5·w        # muon(y=0)@horizon→0.5 ; ν(y=1)@horizon→0.5
L_cls  = BCE_or_soft_focal(logits, t_soft)            # replaces the hard-label target
```
(`soft_focal_loss_with_logits` already exists in the trainer imports and takes a soft
target — reuse it.)

**Alternative: confidence penalty** (keeps hard labels, adds a regularizer):
```
L = L_cls(logits, y) + β · mean( w(θ) · (sigmoid(logits) - 0.5)² )
```
Trade-off: soft-label is a single clean objective directly encoding "predict 0.5 at
horizon"; the penalty leaves the main objective intact and decouples strength (β) but
adds a hyperparameter. **Recommend soft-label** to start.

## Code touchpoints (minimal)
1. `src/data/nu_classifier_dataset/dataset.py` — load `theta.npy` if present in
   `__init__`; return `"theta"` in `__getitem__` (NaN if absent).
2. `src/data/nu_classifier_dataset/collate.py` — stack `theta` into the batch dict
   only when items carry it (source has it, exp target does not).
3. `src/training/da_nu_classifier_trainer.py` `_calculate_classification_loss`
   (line ~424) + call site (line ~585): if `horizon_loss.enabled`, compute `w(θ)` and
   the soft target from `source_batch["theta"]`, use it in the source classification
   loss. Domain/target path unchanged.
4. Config block (new), on top of the E1 config:
```yaml
training:
  horizon_loss:
    enabled: true
    sigma_deg: 10.5      # DATA-DERIVED (see below); Gaussian fit to muatm FP falloff
    max_soften: 1.0      # 1.0 = full pull to 0.5 at the horizon
    # (or: mode: soft_label | confidence_penalty ; beta: <float>)
```
Everything else = E1 (spectral_norm off, afterpulse off, λ=0.01). Naming per the
`{YYMMDD}_{HHMM}_da_nu_classifier_exp_full_...` convention.

## σ derived from data (DONE)
Fine θ-profile of MC muatm frac(score>0.8) on the **boosted** E1@ep10 preds (6.0M
muatm, 3.2k score>0.8; `theta_sigma.py`, `tables/mc_muatm_fp_vs_zenith_fine.csv`,
`figures/mc_muatm_fp_vs_zenith_fine.png`): the false-neutrino rate falls off smoothly
from **7.6% at θ≈95–98°** to ~0 by 130°. Gaussian fit `frac ~ A·exp(-(θ-90)²/2σ²)`
gives **σ = 10.5°** (A=0.043). ⇒ set `sigma_deg = 10.5`.

## Open decisions (need user)
- **soft-label vs confidence-penalty** (+ β if penalty). `soft_focal_loss_with_logits`
  already exists (metrics.py:282, built for continuous targets in [0,1]) and equals
  E1's focal when horizon is off — soft-label is a minimal swap, not a re-implement.
- **max_soften** (1.0 = force full 0.5 at horizon; <1.0 = partial).
- **λ** (keep 0.01, or pair with the λ-sweep discussion).
- σ — resolved from data (10.5°).

## Validation plan
Re-run the 2026-07-06 diagnostics on the new checkpoint: MC muatm frac>0.8 vs θ
(should flatten at the horizon), exp excess & exp>0.8 fraction (should drop toward
physical), z-proxy of exp high-score (should lose the horizontal peak). Compare at a
fixed working point (matched-epoch protocol).
