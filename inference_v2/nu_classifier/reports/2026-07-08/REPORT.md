# Report 2026-07-08 — Fine-tune fix: OOD-background selection, ν-safety, and the SNGP build

Follows [../2026-07-07/REPORT.md](../2026-07-07/REPORT.md), which concluded (a) neither
cheap intervention (horizon θ-loss, DA off) removes the working-point exp excess, and
(b) the principled fix is a **distance-aware / OOD-aware output (SNGP)**. This report
covers the design and construction of a **pragmatic interim fix** — fine-tuning E1 on
real exp OOD events as hard background (label 0) — plus the ν-safety checks that gate it,
and the **full SNGP implementation** (the principled fix) now written.

All populations h8s3 (`n_sn_hits≥8, n_sn_strings≥3`); exp = exp_full out-of-training,
bad runs c02 r20/r249 dropped. Reference model **E1 @ epoch 10**
(`260705_0702_da_nu_classifier_exp_full_E1_lambda0.01/da_checkpoint_epoch_010`).

---

## 1. Strategy: fine-tune on exp OOD background ("poor man's SNGP")

Instead of rebuilding the head immediately, teach the current classifier to reject the
OOD population directly: **fine-tune E1 with a labelled exp-background stream (label 0)**,
MC providing the anchor (MC already correctly rejects horizontal muons — 0/457 test-MC
bright-horizontal muons at >0.8, 2026-07-07 §4). Selection = `encoder-OOD ∧ rvert<0.7`;
encoder-OOD is the primary selector (the exact failure mode: over-confident extrapolation
in encoder space), rvert<0.7 a soft safety filter to avoid labelling any rare steep
(potential-ν) event as background. Uses the existing `ExpFineTuningTrainer` (separate from
the DANN trainer; MC anchor 1:1:2 exp:mc_μ:mc_ν, label 0 on bg) — **no change to the main
DA trainer**.

### 1a. Why E1 @ epoch 10 and not "best" (epoch 23)?
Continuity, not a principled "ep10 is better". The entire diagnostic chain (reports
07-06/07-07, OOD characterization) and the prediction DBs the builder queries
(`exp_full_thr0p8.duckdb`, `mc_merged_thr0p8.duckdb`) exist for **ep10**. Crucially the
encoder of the "original" defines the embedding space in which we compute OOD, so original
model and OOD-embedding model must match. "best" = ep23 is selected by **MC val loss**
(0.00275 vs ep10 0.00311); both have MC val AUC ≈ 0.99995 (saturated — MC val does not see
the exp excess at all). The excess is present from epoch 1 (structural), so ep10 ≈ ep23 on
it. Plan: fix ep10 first (fully characterized); replicate on best/ep23 as the deployable
version if it works.

## 2. Why kNN-OOD, not Mahalanobis?

- **Numerical stability.** The MC encoder embedding is effectively ~1.7-dimensional
  (2 PCs = 90% var, 2026-07-06 §4e), so the 128×128 covariance is near-singular;
  Mahalanobis needs its pseudo-inverse, which amplifies noise in the ~126 near-zero-var
  directions. That is exactly why exp_hi got maha ~499 (tiny components in unused tail
  dims blown up) — the absolute scale is noise-soft (we already flagged "use kNN, robust").
- **kNN is non-parametric** — distance to actual nearest MC points, no covariance inverse,
  answers "is this region supported by training data?".
- **ν-efficiency safety.** kNN>threshold = far from *all* MC points (incl. MC-ν at kNN
  ≈0.68) → labelling those exp events 0 does not collide with MC-ν supervision. Mahalanobis
  would select excess events that sit *among* MC-ν in real distance (that's why they score
  high) → relabelling them 0 while MC-ν stay 1 is fragile and risks ν-eff.
- **Caveat / tension:** Mahalanobis *does* flag the score>0.8 excess more directly (it
  differs from MC-ν precisely in the tail direction). kNN under-selects it (§3). The
  principled resolution is SNGP — a *learned* Mahalanobis in the RFF-GP space with a proper
  Laplace covariance that avoids the degeneracy by construction (§5).

## 3. Fine-tune set assembly — the non-monotonic OOD⋄score finding

`analysis/finetune_set/assemble.py` (120k exp sample; MC reference manifold, kNN k=20).
**kNN-OOD and score are non-monotonic** — the strongest OOD is at *mid* score, and the
score>0.8 excess is *not* strongly kNN-OOD:

| score band | N (of 120k) | OOD median | rvert median |
|---|---|---|---|
| 0.0–0.2 (bulk, in-dist μ) | 118 795 | 0.73 | 1.01 |
| 0.2–0.5 | 778 | 4.29 | 0.53 |
| 0.5–0.8 | 343 | **4.47** | 0.53 |
| **0.8–1.0 (false-ν, the target)** | 84 | **3.65** | 0.60 |

MC reference: OOD median 0.68, **p99 = 4.10**, p99.9 = 5.01. The excess (score>0.8) sits at
OOD ≈ 3.65 — **below MC-p99**. Mechanism (consistent with 2026-07-06): score>0.8 events sit
at the **MC-ν blob edge** (that's why they score high → they have MC-ν neighbours → moderate
kNN); the genuinely far-from-everything events are the mid-score ambiguous ones. So the
naïve `OOD>p99` cut captures only **20%** of the false-ν (recall 0.20). Threshold sweep
(all with rvert<0.7):

| OOD cut | pool events | recall(score>0.8) | median score of selected |
|---|---|---|---|
| **2.5** | ~13 500 | **0.73** | 0.11 |
| 3.0 | ~9 350 | 0.63 | 0.17 |
| 4.10 (p99) | ~3 370 | 0.20 | 0.36 |
| 5.01 (p99.9) | ~790 | 0.07 | 0.45 |

Recall plateaus at 0.73 because 27% of score>0.8 have rvert≥0.7 (vertical — kept as
potential ν by the safety filter). **Chosen: OOD>2.5 ∧ rvert<0.7** (recall 0.73, far from
MC → safe to label 0). Built from the full 480k pool:
`build_exp_bg_ood.py` → **13,989 events**, 159,292 filtered hits, score median 0.110
(frac>0.8 = 0.02), rvert median 0.48. Dataset:
`exp_bg_datasets/260705_0702_..._E1_lambda0.01@da_checkpoint_epoch_010_ood2p5_rv0p7/`.

## 4. ν-safety of the selection

Concern (physics): a very-high-energy (VHE) ν lighting up a large detector volume could be
(1) kNN-OOD (bright, rare) and (2) low-rvert (horizontal, or secondary bright cascades
along the track) → swept into background → we would teach the net to reject real high-value
ν. Note rvert is |verticality|, **blind to up/down sign**; the true ν discriminator is
direction (up-going through the Earth = almost certainly ν). The residual danger is
specifically **near-horizontal, bright, up-going** ν.

### 4a. MC nue2 test (`nue2_safety_check.py`)
Do our cuts sweep up real ν? Test on MC nue2 (bright EM cascades), test sample **disjoint**
from the reference (else self-match understates OOD). N=43k, score median 0.990 (98.8%
correctly ν). Fraction that *would* be selected as background (OOD>2.5 ∧ rvert<0.7):

| Q_mean | N | %OOD>2.5 | %rvert<0.7 | **% → BG** |
|---|---|---|---|---|
| 0–3 | 3795 | 0.14 | 0.37 | **11.4%** |
| 3–5 | 7729 | 0.14 | 0.41 | 11.6% |
| 5–8 | 7347 | 0.12 | 0.40 | 9.5% |
| 8–15 | 11908 | 0.08 | 0.37 | 6.4% |
| 15–30 | 11430 | 0.05 | 0.31 | **4.2%** |
| ≥30 | 791 | 0.07 | 0.39 | 5.7% |

**Key result: brighter → SAFER** (the feared "bright→OOD" mechanism is reversed): brighter
nue2 have more hits → embeddings land more solidly in the MC-ν region → lower OOD. Overall
7.7% → BG, but only ~4% for the bright tail. **Caveat:** MC's atmospheric nue2 spectrum is
soft — only 12 events at Q≥50 — so **true VHE (Q≫50) is untestable here**; the reassuring
trend is an extrapolation. Also rvert only weakly protects cascades (37% have rvert<0.7).
The ~4–11% residual is real ν-loss risk; the MC-ν anchor (same nue2 labelled 1 in the
fine-tune) contests that region rather than letting the net blanket-reject.

### 4b. Directional defense — tested and REJECTED (`directional_check.py`)
Proxy (no reco): `corr(t,z)` on filtered hits. Sign calibrated on muatm (all down-going,
θ>90): corr median **−0.945** (strongly negative = down). nue2 cascades: corr median
**+0.801** (positive = "up-like", low fit quality). Filter "background only if clearly
down-going" (corr<−0.3):

| population | N | →BG base | →BG +filter |
|---|---|---|---|
| nue2 (protect) | 32k | 8.34% | **0.33%** |
| muatm (keep as BG) candidates | — | — | 42.7% retained |

On its face the filter protects nue2 25×. **But it is counterproductive here.** Computing
corr(t,z) on the 13,989 selected background events: median −0.302, 50% clearly down-going —
so the filter would *halve* the set. Decisively, **among the actual false-ν (score>0.8) in
the set (N=289), median corr = +0.357 (up-going-like), 59% clearly up (corr>+0.3), only 3.8%
clearly down.** The score>0.8 exp false-ν are up-going/cascade-like — **indistinguishable
from real nue2 by corr(t,z)** (that is exactly why they fool the classifier). Both directions
of the filter destroy the target:

| corr filter | keeps target (score>0.8) | keeps full bg |
|---|---|---|
| require down (corr<−0.3) | **4%** | 50% |
| drop up (corr>+0.3) | **41%** (removes 59%) | 93% |

The "drop clearly-up" variant (the natural VHE-safety insurance, −7.4% of the set) still
removes 59% of the target because the target *is* up-like. corr_t_z (like rvert) is the
**horizon axis** — the strongest exp_hi-vs-all-MC discriminator (KS 0.79, vs rvert 0.64;
2026-07-07 raw_ood_summary) but **not exp-specific** (exp_hi vs muatm_hi KS 0.072, near the
bottom — muon false positives share it). ν-safety cannot be engineered into a geometric
filter because the false-ν and real ν are co-located on every geometric axis. →
**Directional / corr filter dropped entirely.**

**Consequence for ν-safety:** no cheap proxy (direction, verticality, brightness)
separates exp-false-ν from real ν — that separation *is* the OOD/domain problem. So ν-safety
cannot be engineered into the selection; it rests on (i) the MC-ν anchor contesting the
region during fine-tuning, (ii) the net learning the subtle exp-domain feature (the AUC 0.70
charge/topology residual, 2026-07-07 §4), and (iii) the **decisive post-fine-tune ν-efficiency
check** (MC nue2/numu at the working point) — if it drops materially, back off (lower lr,
fewer bg events) or defer to SNGP.

## 5. SNGP — the principled distance-aware fix (implemented)

The fine-tune is an interim patch; the root cause is that the classifier's output is not
distance-aware. SNGP (Liu et al. 2020) makes it so: **spectral-normalized encoder** (bounds
the Lipschitz constant so embedding distance tracks input distance) + **RFF-GP output head**
(Random Fourier Features approximation of a GP) with a **Laplace covariance** accumulated
over training and a **mean-field logit** `logit/√(1+λ·var)` at inference — inputs far from
the training manifold get high variance → shrunk toward 0.5 → the maha-499 exp events are
flagged uncertain rather than confidently ν. Implementation: `RandomFeatureGPHead` +
`NuMuSNGPModel` in `src/models/base_models.py` (encoder shared with `NuMuClassifierModel`,
spectral-norm ON by default) and `src/training/sngp_nu_classifier_trainer.py` (copied from
the DA trainer, DANN removed, per-epoch Laplace precision accumulation + covariance inversion
+ mean-field validation added) — **the main DA trainer is untouched**.

**Verified (smoke tests):** the distance-aware mechanism works — in-sample predictive
variance ≈1.0 vs ~34× that on OOD-ish inputs → mean-field shrinks OOD logits toward 0 (random
inputs score ≈0.49). The full trainer runs end-to-end (SN on 17 encoder weights, 400k params,
covariance rebuilt each epoch, checkpoints saved), and the checkpoint loads through the
standard `load_model` / `predict_scores_and_embeddings` path unchanged. Config
`experiments/sngp_nu_classifier_baseline.yaml` (MC-supervised, num_rff 1024, optional
`warm_start_checkpoint` to reuse the E1 encoder).

**Caveat (see §8):** the false-ν sit at the *edge* of the MC-ν manifold (kNN 3.0 vs
nu-core 0.8), not in an empty void, so SNGP's variance on them is only *moderate* — it will
shrink them partially, not fully. SNGP is the principled fix for generic over-confidence;
for *this* edge-of-nu excess the fine-tune (which plants real background labels exactly in
that corner) is more directly targeted.

---

## 6. Fine-tune result — the excess is roughly halved with ν-efficiency preserved

Fine-tuned E1 on the 13,989-event OOD background (`ExpFineTuningTrainer`, MC anchor 1:1:2,
lr 5e-5) — early-stopped at epoch 8, best epoch 5, MC val AUC 0.9998 (unchanged, so MC
separation is intact). **Final measurement, strictly out-of-training** (training removed via
back-links for both models; E1 muatm 1.89M, FT muatm 22.9M, exp: E1 480k / FT 584k clean),
down to muon survival 1e-5 with Poisson errors (`paper_suppression_curve.py`). The excess is
the exp survival relative to the MC-muon survival, i.e. **exp/μ** (=1 means exp behaves like
the muon background):

| μ (muon survival) | E1 exp/μ | FT exp/μ | FT/E1 excess | E1 ν-eff | FT ν-eff | E1 #μ@cut |
|---|---|---|---|---|---|---|
| 1e-3 | 1.90 | **1.17** | 0.62 | 0.996 | 0.994 | 1889 |
| 3e-4 | 2.35 | **1.16** | 0.49 | 0.985 | 0.981 | 567 |
| 1e-4 | 2.48 | **0.94** | 0.38 | 0.966 | 0.958 | 189 |
| 3e-5 | 2.08 | **1.14** | 0.55 | 0.931 | 0.924 | 57 |
| 1e-5 | 2.50 | **1.20** | 0.48 | 0.898 | 0.884 | 19 |

**Headline result.** The fine-tune **roughly halves the exp neutrino-like excess** (FT/E1 ≈
0.4–0.6, uniform from 1e-3 to 1e-5) **while ν-efficiency is preserved** (≤1.5% vs E1 at every
point). E1's excess is large and *grows* with strictness (1.9× → 2.5×); the fine-tuned model
brings exp down to only **~1.1–1.2× the pure-muon rate** — i.e. exp nearly behaves like the
muon background, with a small honest residual (partly real ν, partly background removable by
the verticality cut below). This is the opposite of the horizon θ-loss (§7), which does not
reduce the excess and *loses* signal. Checkpoint
`experiments/numu/…@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned/best_finetuned_model.pth`;
figure `figures/paper_suppression_curve.png`. (E1's 1e-5 point is 19 muons — Poisson-limited
but consistent; the horizon of MC-muon statistics.)

**Residual removability (`residual_cuts.py`; residual = clean out-of-training exp above the
FT working-point cut ≈0.40 — the FT model compresses scores so fixed-0.8 is unrepresentative;
584k clean out-of-training exp pool).** The 623 residual false-ν are more horizontal (rvert
med 0.60) and dimmer (Q 5.2) than the kept MC-ν (rvert 0.81, Q 9.5), so a verticality cut
preferentially removes them — but at a real signal cost:

| rvert>t | residual excess removed | nue2 kept |
|---|---|---|
| 0.55 | 39% | 82% |
| 0.70 | 72% | 63% |
| 0.90 | 94% | 41% |

So the residual is manageable by the standard directional (up-going) selection at the usual
signal/background tradeoff, not for free. Brightness helps in the same direction (residual
dimmer than signal) but costs low-energy ν. The ~6% surviving rvert>0.9 (vertical, high-score)
are the genuine ν candidates (→ visualization task, Next steps).

## 7. Horizon θ-loss (E5), converged — does NOT reduce the excess

E5 (horizon soft-label, 2026-07-07 §1) trained to convergence (best epoch 8). Working-point
sweep E1@ep10 vs E5@ep8 (matched MC-muon survival, from the persisted DBs, **down to 1e-5**):
the exp-excess ratio E5/E1 stays **≈ 1 across the whole range** (0.91–1.12, no trend), while
E5's **signal efficiency collapses at strict cuts** — 0.83 vs E1 0.97 at μ=1e-4, **0.70 vs
0.90 at μ=1e-5**. So the horizon θ-loss does not remove the excess at any working point and
is strictly worse than the base model on signal. Decisive negative result; the fine-tune (§6)
is the opposite — it lowers the excess with preserved efficiency.
→ `working_point_sweep.py`, `db_working_point_sweep.py`. (NB E5 and FT MC DBs were built via
`--probs-h5` over the 10000 precomputed-covered muatm parts at min-hits5 → ~22.9M muatm h8s3,
verified duplicate-free; E1's canonical DB scored fewer muatm (1.9M), which limits E1's curve
to ~1e-5 at ~20 muons — fine with Poisson errors.)

## 8. Where the false-ν live — the corner is occupied by ν, not empty (`mc_nu_corner.py`)

A physics correction to the "empty corner / OOD extrapolation" framing: the bright∧horizontal
region is **not** empty in MC — it is occupied by ν, which is exactly why the net scores exp
events there as ν. Among MC events with rvert<0.7 AND Q_mean>8:

| class | N in corner | score median | frac>0.8 |
|---|---|---|---|
| muatm | 1026 | **0.018** | 0.002 |
| nuatm | 301 | **0.981** | 0.963 |
| nue2 | 8596 | **0.989** | 0.989 |

The net learned "bright∧horizontal → ν" from MC ν (energetic horizontal numu, bright
cascades), and correctly rejects the sparse MC muons there (0.018). exp bright-horizontal
events (which MC muon simulation lacks) inherit the ν label. kNN distance to the MC-ν
manifold: MC-ν core 0.79, MC-ν bright-horizontal edge 1.30, **exp false-ν 3.02**, exp_lo /
muatm 11.2. So the false-ν sit **near (but just beyond) the ν edge** — ~2.3× the edge, far
closer to ν (3.0) than to muons (11); 80% of them have a horizontal nearest-ν neighbour.

**Refined mechanism:** not extrapolation into a void, but a **charge decision boundary
learned on sparse MC data in the bright-horizontal corner**, which misfires on exp events
whose charge distribution MC does not reproduce (exp brighter, §4/2026-07-07). This explains
(a) why the excess is edge-of-nu (moderate kNN, §3 non-monotonicity), (b) why SNGP only
partially helps (§5 caveat), and (c) why the fine-tune — planting real exp-background labels
in that exact corner — is the more targeted fix (§6).

---

## Conclusions
1. **The OOD-background fine-tune roughly halves the exp excess (2–3× at μ=1e-3…3e-4) with
   ν-efficiency preserved** (§6) — the primary, publishable result.
2. Selection: OOD>2.5 ∧ rvert<0.7 (kNN, recall 0.73 of the false-ν), 13,989 background
   events. The excess is edge-of-MC-ν, not strongly kNN-OOD, so a naïve p99 cut misses 80%.
3. The bright∧horizontal corner is **occupied by MC-ν** (score ~0.98), muons rejected there;
   the false-ν sit just beyond the ν edge (kNN 3.0 vs 0.8 core) — a charge boundary learned on
   sparse MC that misfires on exp (§8). So it is *not* an empty-void extrapolation.
4. Horizon θ-loss (E5), converged, does **not** reduce the excess and costs signal (§7).
5. ν-safety cannot be engineered geometrically (directional filter tested & rejected, §4b) —
   it rests on the MC-ν anchor + the post-fine-tune ν-efficiency check (§6, preserved).
6. SNGP implemented (§5); for this edge-of-nu excess it is only a *partial* fix — the
   fine-tune is more targeted.

## Next steps (paper pipeline, in progress)
- **Out-of-training re-measurement** of the E1-vs-fine-tuned suppression curve down to muon
  survival 1e-5 with Poisson errors (`paper_suppression_curve.py`; training removed via
  back-links, both models). 1e-6 is statistics-limited by MC muons (~1.9M muatm_2020) — not
  claimed; curve solid to ~1e-5.
- **Residual removability** (`residual_cuts.py`): rvert/Q of the residual fine-tuned false-ν
  vs kept MC-ν, and the verticality-cut tradeoff (excess removed vs ν kept) — for the paper's
  "NN halves the excess, directional cuts handle the rest" claim.
- E5-vs-E1 sweep to 1e-5 from the persisted DBs (`db_working_point_sweep.py`).
- Train SNGP; compare its distance-aware exp behaviour to the fine-tuned E1.

**Not for the paper — genuine ν-candidate visualization:** the exp events that survive BOTH
the high FT score AND the verticality cut (high FT score ∧ high rvert = vertical/up-going) are
the handful of genuine atmospheric-ν candidates (~4% of the residual survive rvert>0.9,
~6–7 events). Build event displays (3D hits colour-coded by time) for these top vertical
high-score exp_full events — they should be the network's most confident, verticality-cut-
surviving events, i.e. real ν, not the horizontal false-ν the fine-tune/cuts remove.

## Artifacts
| Path | What |
|---|---|
| `analysis/finetune_set/assemble.py` | OOD sizing; non-monotonic OOD⋄score finding; threshold sweep |
| `analysis/finetune_set/nue2_safety_check.py` | MC nue2 → background fraction by brightness |
| `analysis/finetune_set/directional_check.py` | corr(t,z) down-going filter test (nue2 protect, muatm retain) |
| `exp_finetuning/build_exp_bg_ood.py` | Builds the OOD exp-background NPY dataset (abs threshold) |
| `exp_finetuning/finetune_E1_ood2p5_rv0p7.yaml` | Fine-tune config (ExpFineTuningTrainer, output → experiments/numu) |
| `exp_bg_datasets/...@da_checkpoint_epoch_010_ood2p5_rv0p7/` | 13,989-event background dataset |
| `src/training/sngp_nu_classifier_trainer.py` + `src/models/` SNGP pieces | Full SNGP build |
| `analysis/finetune_set/mc_nu_corner.py` | §8 — corner occupied by ν; false-ν at nu-edge (kNN 3.0) |
| `analysis/model_comparison/working_point_sweep.py` | §7 — E1 vs E5 converged, matched working points |
| `analysis/model_comparison/paper_suppression_curve.py` | §6 paper curve — E1 vs FT, out-of-training, 1e-5, Poisson errors |
| `analysis/model_comparison/residual_cuts.py` | residual false-ν rvert/Q + verticality-cut tradeoff |
| `analysis/model_comparison/db_working_point_sweep.py` | E1 vs E5 sweep to 1e-5 from persisted DBs |
| `predict_mc.py --probs-h5` | new: fast persisted MC scoring via precomputed SN probs |
| `preds/…_finetuned@best_finetuned_model/` | fine-tuned model out-of-training predictions |
| `analysis/finetune_set/tables/*.csv`, `figures/*` | Per-event data and plots |
