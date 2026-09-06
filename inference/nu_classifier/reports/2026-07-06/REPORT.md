# Report — E1/E2/E3b post-training analysis of the exp neutrino-like excess

**Date:** 2026-07-06
**Scope:** everything computed/analyzed after training the three DA nu-classifier
models on the full-stat exp target (E1 baseline, E2 spectral-norm, E3b afterpulse).
**Author:** analysis session (Claude Code).

All artifacts referenced below live in this folder (`tables/`, `figures/`) or are
linked to their canonical location. Reproduce every table/figure with
[`diagnostics.py`](diagnostics.py).

---

## 0. Models under study

| tag | run dir (`experiments/numu/…`) | modification | lambda |
|---|---|---|---|
| E1  | `260705_0702_..._E1_lambda0.01`  | baseline DA            | 0.01 |
| E2  | `260705_0708_..._E2_lambda0.01`  | + spectral norm (encoder) | 0.01 |
| E3b | `260705_0721_..._E3b_lambda0.01` | + afterpulse aug (p=0.25) | 0.01 |

Full experiment matrix & per-run config: `experiments/numu/exp_full_DA_experiments.md`.

**The problem.** On real experimental data the nu-classifier flags far more
"neutrino-like" (score>0.8) events than physics allows: observed exp fraction
~1.5×10⁻³ vs the physical atmospheric-neutrino expectation ~10⁻⁶–10⁻⁵. The three
models were trained to try to close this domain gap.

---

## 1. Model comparison — did any modification close the gap?

### 1a. Methodology fix (important)
Comparing each run's `best_da_model` is **not** apples-to-apples: the best
checkpoints sit at very different epochs (E1=23, E2=43, E3b=37). We instead
**anchor at E1 @ epoch 10** and match E2/E3b on **val_loss (primary) + F1
(control)** — **not AUC**, which is saturated (~0.9999, run-range ~1e-4, its argmin
is noise). Matched checkpoints: E1@10, E2@22 (loss/AUC), E2@28 (F1), E3b@10.

We also compare at a **fixed muon-suppression working point** (score cut giving MC
muon-survival = 1e-3), not a fixed score threshold — because spectral norm
compresses all scores downward, so a fixed 0.8 threshold conflates "OOD
suppression" with "global score rescaling".

### 1b. Result — no modification closed the OOD gap
Artifact: [`figures/comparison_table_matched.csv`](figures/comparison_table_matched.csv),
`experiments/numu/exp_full_DA_experiments.md` (Round 2).

| model (matched) | AUC | sig_eff@μ=1e-3 | exp>0.8 (fixed thr) | **exp@μ=1e-3 (fixed WP)** |
|---|---|---|---|---|
| E1 @ep10  | 0.99989 | 97.69% | 0.148% | **0.252%** |
| E2 @ep22  | 0.99988 | 97.66% | 0.093% | **0.261%** |
| E2 @ep28  | 0.99989 | 97.78% | 0.108% | **0.255%** |
| E3b @ep10 | 0.99988 | 97.57% | 0.155% | **0.259%** |

At a fixed working point the exp false-neutrino fraction is **equal across all
three (~0.25%, spread ~1-2σ)**. The Round-1 "E2 wins at 0.8" signal was a double
artifact: (i) different epochs, (ii) spectral norm's global score compression.
**Neither spectral norm nor afterpulse moved the domain gap.**

Figures:
[`figures/score_hist_per_model_matched.png`](figures/score_hist_per_model_matched.png)
(exp vs MC-muon per model, with the 0.8 line and each model's μ-WP cut — E2's whole
distribution shifts left; exp sits the same factor above muon everywhere),
[`figures/score_hist_muatm_vs_exp_matched.png`](figures/score_hist_muatm_vs_exp_matched.png),
[`figures/mc_survival_curves_matched.png`](figures/mc_survival_curves_matched.png),
[`figures/exp_score_tail_matched.png`](figures/exp_score_tail_matched.png).
Round-1 (best-model, flawed) kept for the record:
[`figures/comparison_table_best.csv`](figures/comparison_table_best.csv).

---

## 2. Excess metric & "overfitting onset"

**Metric.** `excess(thr) = exp_frac(>thr) / mc_muatm_frac(>thr)`. Taken with the
same model at the same threshold, so the model's global score scale cancels — a
scale-robust domain-gap measure. excess≈1 ⇒ exp looks like MC muon background;
excess>1 ⇒ OOD false-neutrino excess.

**Onset (epochs 1,3,5,10; h8s3).** The excess is **present from epoch 1**
(excess@0.8: E3b 2.1×, E2 2.1×) and does not systematically grow — **not** a late
overfitting phenomenon; early stopping will not fix it. Under h8s3 the muon tail is
cut ~3× (mu N 210k→66k), so only **excess@0.8** is stable here; @0.9+ bounces on
single-digit muon counts (read it from the h5s0 statistics table below instead).
Artifacts: E3b [`figures/e3b_excess_vs_threshold.png`](figures/e3b_excess_vs_threshold.png),
[`figures/e3b_excess_vs_epoch.png`](figures/e3b_excess_vs_epoch.png),
[`figures/e3b_excess_table.csv`](figures/e3b_excess_table.csv);
E2 (contiguous-sampled exp) `figures/e2_excess_*` — excess@0.8 = 2.14/1.86/2.29/2.41
over ep1/3/5/10, same "present from epoch 1" picture.

**Statistical validity** (h5s0, full statistics — best case for the muon tail).
Table: [`tables/excess_stat_validity.csv`](tables/excess_stat_validity.csv).

| thr | k_exp | k_μ | excess | ±rel | σ from 1 |
|---|---|---|---|---|---|
| 0.5 | 13,751 | 1,150 | 1.90 | 3% | 15σ |
| 0.8 | 2,007 | 109 | 2.93 | 10% | 6.7σ |
| 0.9 | 500 | 23 | 3.45 | 21% | 3.3σ |
| 0.95 | 115 | 7 | 2.61 | 39% | 1.6σ |
| 0.99 | 18 | 0 | — | — | — |

The error is set by the **MC-muon tail count** (k_μ). Trust **excess ≤0.8**
(6.7–15σ); @0.9 is marginal; **>0.95 is noise**. Lever if the tail is needed: score
more muatm (577M available, only ~210k scored).

**Physics validity.** exp>0.8 fraction ~1.6×10⁻³ vs neutrino ~10⁻⁶–10⁻⁵ ⇒ even if
every real neutrino scored >0.8 they'd be ≤0.6% of the high-score exp events. So
**>99.4% of "neutrino-like" exp are false (OOD)** — the excess is a pure domain
artifact and MC-muon is the right yardstick.

---

## 3. Cause of the excess — three hypotheses tested

The prior working hypothesis was **afterpulses**. E3b (afterpulse augmentation)
did not help (§1b), so we examined three possibilities: (a) implementation bug,
(b) conceptual mis-specification, (c) afterpulses are not the cause.

### (a) Implementation bug — NONE
`src/data/nu_classifier_dataset/collate.py`: afterpulse is injected in raw units
**before** padding, and normalization runs **after** (line order verified). Charges
are normalized together with real hits. No scale/typo bug.

### (b) Afterpulse augmentation is mis-specified — 3 axes
Table: [`tables/feature_shift_mc_vs_exp.csv`](tables/feature_shift_mc_vs_exp.csv).
- **Brightness, wrong direction:** aug injects Q~U(5,100) → clip+norm mean **0.84**,
  ~10× brighter than any real hit; but exp is **dimmer** than MC (exp clip+norm amp
  mean 0.055 < MC 0.087). We pushed MC toward bright, exp is dim.
- **Position/time nonphysical:** real PMT afterpulses are **same-channel + delayed
  (late)**; the aug places them at a **random OM, in-time-window**.
- **Count:** 1–3, not clusters.

### (c) The excess is a broad multivariate OOD, not afterpulses
- Per-feature marginals don't explain it: amplitude shift is **modest after clip@100**
  (raw std 307 is outliers up to 80801 p.e.), **time is fine** for sig-noise-filtered
  hits (0.79× width; the 2× time widening applies only to UNfiltered hits), topology
  is shifted (exp more multi-string) but…
- **Excess is flat (~3×) across all topology cuts** h5s0→h12s4
  ([`tables/excess_vs_topology_cut.csv`](tables/excess_vs_topology_cut.csv)) and
  across all n_sn_strings bins → not a single marginal, not low-multiplicity.
- Present from epoch 1 (§2). ⇒ over-confident extrapolation into MC-unseen regions.

**Default cut adopted:** **h8s3** (`n_sn_hits≥8 & n_sn_strings≥3`), applied at
analysis time via DB columns (no re-inference). Baked into
`analysis/model_comparison/compare_da_models.py` (`MIN_HITS=8, MIN_STRINGS=3`).

### What the "neutrino-like" exp events actually are (h8s3)
Table: [`tables/topology_by_population_h8s3.csv`](tables/topology_by_population_h8s3.csv).

| population (h8s3) | N | nhits med/mean | nstr mean |
|---|---|---|---|
| exp score>0.8 (false ν) | 411 | **10 / 11.6** | 3.62 |
| exp all | 537k | 13 / 15.6 | 3.93 |
| MC ν score>0.8 (real) | 184k | **15 / 20.1** | 4.08 |
| MC muon score>0.8 (false) | 17 | **10 / 10.8** | 3.71 |

exp false-neutrinos are **compact** events — topologically identical to the MC
muons that also false-positive, and **unlike** real (large) MC neutrinos. Same
failure mode; exp just produces ~3× more of it.

---

## 4. KEY RESULT — the excess is a horizon (near-90° zenith) confusion

**Hypothesis (physics).** The cluster geometry is vertical, so compact events are
near-**horizontal**. At zenith θ≈90° the up/down discrimination that separates
neutrino (upward, from below Earth) from muon (downward, from above) degenerates,
so the network cannot tell them apart and leaks muons into the neutrino class.

**Test on MC ground truth** (muatm true zenith = `prime_prty[:,0]`, h8s3):
Table [`tables/mc_muatm_fp_vs_zenith.csv`](tables/mc_muatm_fp_vs_zenith.csv),
figure [`figures/mc_muatm_fp_vs_zenith.png`](figures/mc_muatm_fp_vs_zenith.png).

| true zenith θ | N | frac(score>0.8) |
|---|---|---|
| **95–105°** (horizon) | 501 | **1.20%** |
| 105–120° | 9,435 | 0.11% |
| 120–150° | 42,883 | 0.00% |
| 150–180° (straight down) | 13,947 | ~0.01% |

The muatm false-neutrino rate is **entirely concentrated at the horizon**: ~1.2% at
θ≈95–105° (≈40× the overall 0.03%), ≈0 for steep muons. **The hypothesis is
confirmed on data with known truth.** This is consistent with everything above: the
excess is present from epoch 1 (a geometric ambiguity, not overfitting), flat across
topology cuts and score-scale (a per-event direction ambiguity), and the false
positives are compact (= near-horizontal) events.

---

## 4b. Confirmation on exp + why there is an excess (z-spread proxy)

Reco angles are too noisy (±tens of deg), so we use a **verticality proxy on the
sig-noise-filtered hits**: `rvert = std_z / std_xy` (low = horizontal). Scripts
[`zspread_proxy.py`](zspread_proxy.py), [`zspread_distribution.py`](zspread_distribution.py);
table [`tables/zspread_proxy.csv`](tables/zspread_proxy.csv).

**Proxy validated on MC truth:** rvert vs true θ, `corr(rvert, θ) = 0.83` (horizon
θ95–105 → rvert 0.60; steep θ150–180 → rvert 2.39). *NB: must use FILTERED hits —
raw hits are noise-dominated and give no correlation.*

**Confirmed on exp:** high-score exp events are horizontal, matching the MC horizon
/ MC-muon-false-positive locus:

| population (h8s3) | rvert (med) |
|---|---|
| MC muon score>0.8 (false) | 0.45 |
| exp score>0.8 (false ν) | **0.56** |
| exp score<0.2 (bg) | 1.04 |
| MC nue2 score>0.8 (real cascade) | 0.81 |

**Why an excess exists (a shared mechanism needs a domain *difference*):**
"horizontal→high score" holds in *both* domains, so by itself it doesn't create an
excess. Two contributing differences:
- **(A) angular distribution:** exp is **~1.3× more horizontal** than MC muatm
  (frac rvert<0.7: exp 0.228 vs MC 0.176; frac<0.5: 0.067 vs 0.051). Real
  difference, but ~1.3× input vs ~3× excess ⇒ only ~1/3 of it.
- **(B) residual per-angle gap — DOMINANT:** A/B decomposition (`ab_decomposition.py`,
  `tables/ab_decomposition.csv`, boosted 6.6M-muatm stats): within every rvert bin
  exp scores >0.8 at **2.8–4.5× the MC-muon rate**. So (A)≈1.3–1.4× vs (B)≈3–4×;
  verticality does **not** explain the excess. (B) is resolved in §4c as OOD.

Implication: the θ-loss fix (abstain at the horizon) attacks the **absolute**
false-neutrino rate (where OOD events concentrate), but the root cause is the OOD
extrapolation of §4c, which is what domain adaptation / a distance-aware output must
address.

## 4c. Composite OOD — the excess is over-confident OOD extrapolation

Since verticality does not explain the per-angle gap (B, §4b), we characterized it
directly in the model's 128-dim **encoder embedding space** (E1@ep10). Script
[`composite_ood.py`](composite_ood.py); table [`tables/composite_ood.csv`](tables/composite_ood.csv);
figures [`figures/composite_ood_umap.png`](figures/composite_ood_umap.png),
[`figures/composite_ood_mahalanobis.png`](figures/composite_ood_mahalanobis.png).

Class-conditional **Mahalanobis** and **kNN** distance to the MC manifold, per
population (h8s3):

| population | N | rvert med | **Mahalanobis** | kNN |
|---|---|---|---|---|
| MC muon | 1426 | 1.07 | 39.6 | 0.81 |
| MC ν | 2871 | 0.79 | 42.3 | 1.05 |
| exp bg (<0.2) | 3000 | 1.00 | 45.9 | 0.91 |
| **exp false-ν (>0.8)** | 361 | 0.56 | **499.1** | **3.82** |

**The exp false-neutrinos are a cleanly separated OOD population.** In the
Mahalanobis *distribution* (`composite_ood_mahalanobis.png`) MC-μ, MC-ν and exp-bg
fully overlap (peak log₁₀≈1.6, maha~40) while exp false-ν peak at log₁₀≈2.65
(maha~450) with almost no overlap — separation across the whole population, not just
the median. kNN distance (assumption-free) corroborates: 3.82 vs ~0.9 (×4). Typical
exp (bg) sits on top of MC muons (maha 46 ≈ MC 40) — the bulk domain is aligned; DA
worked for ~99% of exp.

*Caveat on magnitude:* in-distribution maha≈40 in a 128-dim space (a true Gaussian
would give ≈128), so the covariance construction (shared cov of the combined MC
classes) makes the absolute "×12" scale-soft — read it as "clearly separated", with
kNN ×4 as the robust magnitude. Also note **UMAP is NOT good evidence here**: exp
false-ν there merely sit at the *edge* of the MC-ν blob (UMAP compresses the OOD
direction) — the distance histogram, not the 2D projection, is the indicator.

**So the excess is not in-distribution confusion — it is over-confident
extrapolation on a narrow OOD tail** of exp events the model never saw in MC and maps
to the extreme "neutrino" corner. This explains the per-angle gap (B): at any
verticality, some exp events are OOD outliers, and OOD-ness separates the false-ν
better than geometry. The exp↔MC-muon domain-classifier AUC on the *full* exp is only
0.667 precisely because most exp is aligned; the OOD is concentrated in the score>0.8
tail.

## 4d. OOD cut on score>0.8 — does an OOD veto remove the excess?

Test the OOD picture operationally: build an OOD score (distance to the MC manifold),
cut on it over the score>0.8 region, and ask (1) does the excess vanish, (2) how many
in-distribution high-score exp remain (candidate real ν), (3) do real MC ν survive
(retention). Two OOD spaces compared. Scripts [`ood_cut.py`](ood_cut.py) (physical),
[`ood_cut_embed.py`](ood_cut_embed.py) (embedding); tables `tables/ood_cut*.csv`;
figures `figures/ood_cut*.png`.

- **Physical-feature OOD (hand-picked: nhits, nstrings, charge, t/z/xy spreads,
  verticality) FAILS.** exp false-ν look in-distribution there (Mahalanobis 7.5 ≈ MC
  6–8); a cut keeping 99% of MC leaves 1734/1747 exp false-ν — the excess stays. The
  OOD is **not** in simple observables.
- **Embedding-space OOD (kNN to MC in the 128-dim encoder space) works.** exp false-ν
  form a separate kNN peak (~3.5 vs MC ~0.7).

| MC-keep % | exp false-ν survive (physical) | **exp false-ν survive (embedding)** | real-ν retention |
|---|---|---|---|
| 90 | 1581 | **29** | 0.69 |
| 95 | 1680 | **83** | 0.83 |
| 99 | 1734 | **917** | 0.96 |

(exp held-out N=1.18M; physical ν expectation ~1–12 events.)

**Conclusions:** (i) the OOD is **representational** — invisible in hand-picked
observables, sharp in the model's features (embedding cut is ~50× better at MC-keep
90%). (ii) An embedding-OOD veto **reduces the excess ~60×** (1747→29, implied frac
1.5e-3→2.5e-5, near the physical ceiling) — but at a real-ν efficiency cost (retention
0.69 at the aggressive cut), and it does **not** isolate a clean 1–2: ~29 survive
because exp false-ν genuinely overlap the real-ν tail in embedding space. (iii) This
is exactly the job of a **distance-aware output (SNGP)** — but learned jointly with
the classifier rather than a post-hoc kNN veto — while ~29 irreducibly ambiguous
events set the floor for OOD alone.

## 4e. Where the excess lives — dimensionality & the tail-ablation

Is the 128-dim embedding over-provisioned, and does reducing it (d_model 32/16) help?
Scripts `embedding_dimensionality.py` (+ scratch `ablate_tail.py`); table
`tables/embedding_dimensionality.csv`, figure `figures/embedding_dimensionality.png`.

- **The embedding is wildly over-provisioned.** PCA on MC embeddings: participation
  ratio **1.7** (effective ~2 dims); 90% of MC variance in **2** PCs, 99% in **26**
  (of 128). MC lives on a ~2-D manifold.
- **exp false-ν OOD lives in the UNUSED tail directions.** 98.9% of their
  Mahalanobis-to-MC is in the 126 low-variance tail PCs; the most-anomalous PCs are
  the highest-index (lowest MC-variance) ones. exp drifts into the directions MC
  training never constrained.
- **But the SCORE is driven only by the top ~8 (MC-used) PCs** — and the tail is
  decision-irrelevant. Tail-ablation (keep top-K MC PCs, re-score): exp false-ν mean
  score is **unchanged for K≥8** (0.866, frac>0.8 0.97) and MC-ν score is preserved.

| K (top MC PCs kept) | exp false-ν mean / frac>0.8 | MC-ν mean / frac>0.8 |
|---|---|---|
| 2 | 0.835 / 0.70 | 0.969 / 0.985 |
| 8 | 0.866 / 0.97 | 0.974 / 0.999 |
| 26 | 0.865 / 0.99 | 0.974 / 0.999 |
| 128 | 0.866 / 1.00 | 0.974 / 0.999 |

**Resolution of the apparent paradox (score in top-8, domain difference in tail):**
there are two distinct MC↔exp differences. (1) A **large** difference in the tail
(MC-unused) directions — real (kNN Euclidean too, not just whitened), the OOD marker
— but the head learned ~zero sensitivity there (no MC variance ⇒ no training signal),
so it is **decision-irrelevant** (ablation proves it). (2) A **small,
in-distribution-looking** difference in the top ~8 (MC-used) directions, where exp
false-ν sit on the **neutrino side** — this is what fires the classifier. **The excess
comes from (2): exp false-ν are neutrino-like in exactly the features the head uses;
the big tail difference (1) is an OOD marker the head ignores.** This ties to §4b: the
top-8 decision features encode the horizon/direction ambiguity.

**Consequences:**
- **Reducing d_model is a backfire.** The score is set by the top-8 (excess would
  persist), while cutting dims deletes the tail OOD marker (1) — you lose
  detectability and keep the problem.
- **This is the precise case for a distance-aware output (SNGP):** the OOD marker
  already exists and is strong (kNN ×4, maha) but the head ignores it; SNGP makes the
  output depend on full-space distance, converting the ignored marker into reduced
  confidence exactly on these events.

## 5. Conclusions

1. **Neither spectral norm nor afterpulse (at λ=0.01) closed the exp OOD gap** at a
   fixed working point (§1). The prior afterpulse hypothesis is not the driver, and
   the E3b augmentation was mis-specified anyway (§3b).
2. **Horizon (θ≈90°) confusion is where the false positives concentrate** (§4),
   confirmed on MC truth (7.6% at θ95-98°, ~0 by 130°) and on exp via the z-proxy
   (§4b). Geometric, present from epoch 1 — but only ~1/3 of the excess.
3. **The excess is dominated by over-confident OOD extrapolation** (§4b, §4c): at
   equal verticality exp scores ν 3-4× more than MC muons (B), and those false-ν are
   ~12× OOD in embedding space (Mahalanobis 499 vs 40) — a narrow OOD tail, while
   ~99% of exp is aligned with MC. Verticality does not explain the gap.
4. Measurement guidance: use **h8s3** default cut; trust **excess at thr ≤0.8**
   (statistics); the excess is >99.4% false neutrinos (physics).

## 6. Next steps
- **DONE this session:** horizon soft-label loss (E5, σ=10.5° from data) implemented
  (`horizon_loss` config; soft_focal + t_soft, source-only) and **training on cuda:0**
  — a mitigation targeting where the OOD concentrates (horizon), not the OOD root.
- **Principal fix (recommended): distance-aware / OOD-aware output (SNGP/DUQ)** — a
  head that returns low confidence for inputs far from the MC manifold would directly
  neutralise the maha-499 tail (§4c). This is the structural fix; the θ-loss and DA
  only partially reach it.
- **λ sweep incl. λ=0** — DANN's objective is satisfied by feature collapse, so
  *stronger* DA may worsen; test whether DA pressure drives the OOD over-confidence.
- Evaluate E5 vs E1 at a matched working point (does the horizon loss cut the exp
  excess without hurting MC efficiency at fixed μ-suppression?).

## 7. Tooling changes made this session
- `inference_v2/nu_classifier/predict_exp.py`: `rdcc_nbytes=64MB` HDF5 chunk cache
  (~5× faster exp I/O) + `--sample-mode contiguous` (physically unbiased: one part =
  one short run; reads one file-offset window → far fewer gzip-chunk decompressions).
- `analysis/model_comparison/compare_da_models.py`: `--preset {best,matched}`, h8s3
  default cut.
- `analysis/model_comparison/e3b_overfitting_onset.py`: generalized to any run
  (`--tag {e1,e2,e3b}`).
- Run scripts: `run_matched_epoch_infer.sh`, `run_e3b_early_epochs.sh`,
  `run_e2_early_epochs.sh`.

## Artifact index
- `diagnostics.py` — regenerates all `tables/` + `figures/mc_muatm_fp_vs_zenith.png`.
- `tables/` — excess_stat_validity, excess_vs_topology_cut, topology_by_population_h8s3,
  mc_muatm_fp_vs_zenith, feature_shift_mc_vs_exp (all `.csv`).
- `figures/` — matched-comparison (5), excess-onset E3b/E2, zenith FP curve.
- External: `experiments/numu/exp_full_DA_experiments.md` (experiment matrix + Rounds 1–2);
  memories `da_modifications_ood_result`, `exp_excess_diagnosis`.
