# Report 2026-07-07 — testing interventions on the exp neutrino-like excess

**Continues** [../2026-07-06/REPORT.md](../2026-07-06/REPORT.md), which diagnosed the
exp neutrino-like excess as **over-confident OOD extrapolation** on a narrow tail of
exp events (neutrino-like in the ~8 decision-relevant embedding dims, strongly OOD in
the directions the classifier ignores). Today we (A) test two cheap interventions
against it and (B) probe the events' temporal structure via delay embedding.

All comparisons at a **fixed muon-suppression working point** (μ-survival = 1e-3),
h8s3 cut, out-of-training exp_full held-out — the fair metric (fixed-threshold numbers
are confounded by score-scale compression, see 2026-07-06 §1b).

---

## 1. Horizon θ-loss (E5) — does abstaining at the horizon help?

Built the horizon soft-label loss (see `experiments/numu/horizon_loss_design.md`):
near true zenith θ=90° pull the MC target toward 0.5 (σ=10.5° from the muatm
false-neutrino falloff). Implemented in `src/…/dataset.py` (+theta), `collate.py`,
`da_nu_classifier_trainer.py` (`horizon_loss` config, source-only soft_focal). E5
(= E1 + horizon loss, λ=0.01) training on cuda:0.

**Preliminary (E5 @ epoch 5, still training):** table
[`tables/comparison_table_e5vse1.csv`](tables/comparison_table_e5vse1.csv).

| model | AUC | sig_eff@μ=1e-3 | exp>0.8 (fixed) | **exp@μ=1e-3 (WP)** |
|---|---|---|---|---|
| E5 horizon @ep5 | 0.99987 | 0.959 | 29 (0.006%) | **0.218%** |
| E1 @ep5 (same budget) | 0.99997 | 0.994 | 532 (0.111%) | **0.189%** |
| E1 @ep1 (val-matched) | 0.99995 | 0.989 | 550 (0.114%) | **0.181%** |

The dramatic fixed-0.8 reduction (29 vs 532) is a **score-compression artifact** (E5's
μ-suppression is 30× higher, sig_eff at 0.8 lower — everything squeezed below 0.8),
exactly like spectral norm in 2026-07-06 §1b. **At the working point the horizon loss
does NOT lower the excess** (0.218% ≥ E1's 0.18–0.19%) and costs ~3.5% signal
efficiency. **Caveat:** E5 is at epoch 5 and undertrained (AUC still climbing) — a
fair verdict needs convergence. Preliminary read: another fixed-threshold illusion, no
working-point gain.

*Why plausibly no gain:* the excess is a ratio; the horizon loss softens horizon
events in both domains, so num and denom shrink together and the ratio is preserved —
and the OOD tail (2026-07-06 §4c) is not purely horizon.

### 1b. Is E5's high-score cluster a clean confident-ν population? (bimodality probe)
The E5 exp score tail (`exp_score_tail_e5vse1.png`) looks bimodal — a dip ~0.8–0.9 and
a small cluster near 1 — suggesting the θ-loss might separate near-horizontal ambiguous
EAS (pushed down) from confident (vertical/real-ν) events (score>0.95). Tested on MC
truth (`e5_bimodal.py`, `tables/e5_bimodal_{exp,muatm}.csv`, `figures/e5_bimodal.png`).

- **Mechanism confirmed:** E5 has **0 muon false-positives above 0.9** (E1 still has some,
  and E1's >0.95 muons are 100% near-horizon, θ_med≈100°). The θ-loss pushes horizontal
  muons down to 0.6–0.8 (E5 mid-score muons: θ_med 110°).
- **E5's >0.95 is NOT empty** — 102k MC ν, **0 muons** (ν-eff 0.55): a clean ν cluster,
  so it is not pure "compression collapse".
- **But E1's >0.95 is already clean AND keeps more ν** (ν-eff **0.895**, ~0 muon
  contamination; E1's muon FPs sit at *mid* scores 0.6–0.8, contamination 5%). So on MC
  the θ-loss buys no cleaner top — E1 is already clean there — at a cost of ~45% ν
  efficiency.
- **The exp excess is OOD, not MC-muon**, and E5@ep5 has only ~1 exp event >0.95 → the
  exp bimodality can't be characterized yet. And horizon is only ~1/3 of the excess
  (2026-07-06 §4b, (B) dominates).

**Read:** the θ-loss mechanism works (removes horizontal FPs from the top) and E5's top
is a clean ν sample, but on MC E1 is already clean there with higher efficiency; a fair
verdict for exp needs E5 converged (characterise exp>0.95 verticality then). Note a
post-hoc verticality/reco-θ cut would discard horizontal events without retraining — but
only addresses the horizontal ~1/3, not the OOD remainder.

## 2. Domain adaptation on/off — does DA drive the excess?

Hypothesis (tested): DANN forces exp onto the MC manifold, possibly depositing
ambiguous exp events on the neutrino side. Used the existing **FixedDA λ=0** model
(`260531_1722_…_lambda0.0_thr0.8_FixedDA`, pure MC classifier, DA fully off, same
d_model=128 + same source MC), scored on exp_full + mc_merged, matched to E1@ep10 on
val_loss (→ ep15). Table [`tables/comparison_table_lambda0.csv`](tables/comparison_table_lambda0.csv).

| model | AUC | sig_eff@μ=1e-3 | exp>0.8 (fixed) | **exp@μ=1e-3 (WP)** |
|---|---|---|---|---|
| E1 λ0.01 @ep10 | 0.99998 | 0.9960 | 361 (0.075%) | **0.190%** |
| FixedDA λ0 @ep15 | 0.99998 | 0.9955 | 285 (0.059%) | **0.227%** |

**DA does NOT cause the excess.** With DA fully off (λ=0) the working-point excess is
**not gone — slightly higher** (0.227% vs 0.190%, ~4σ). Consistent with the
counter-argument: λ=0 = the model never sees exp → *more* unconstrained OOD
extrapolation. DA (even weak) aligns the bulk and mildly helps. (Also consistent with
2026-07-06: the excess is present from epoch 1, where λ≈0.) Matched cleanly (AUC,
sig_eff@WP nearly identical). Caveat: FixedDA is a separate older run, not a perfectly
controlled ablation, but arch/source/metrics match.

**Combined (§1+§2): neither cheap intervention removes the working-point excess.** This
strengthens the case for a **distance-aware output (SNGP)** as the principled fix
(2026-07-06 §4c/§4e).

## 3. Delay-embedding (Takens) — the events' temporal structure

Model-independent probe of the hit-*sequence* dynamics (what the transformer sees):
order each event's signal hits by time, take a scalar per-hit series, and delay-embed
with window m=3: point_i = (s_i, s_{i+1}, s_{i+2}) ∈ R³. The point cloud reconstructs
the sequence's phase portrait; its shape (from PCA of the cloud) characterises whether
the series is a structured trajectory or space-filling noise. Script
[`event_unfolding.py`](event_unfolding.py); table
[`tables/event_unfolding.csv`](tables/event_unfolding.csv); figure
[`figures/event_unfolding_clouds.png`](figures/event_unfolding_clouds.png).
(Metric definitions — participation ratio, sphericity, planarity — are in
[summary.md](summary.md).)

Median shape by population (h8s3):

| channel | metric | exp false-ν | MC ν | MC μ |
|---|---|---|---|---|
| Q (charge) | participation ratio | **2.05** | 2.55 | 2.67 |
| z (depth)  | participation ratio | **2.50** | 1.82 | 1.26 |
| z (depth)  | sphericity          | **0.31** | 0.14 | 0.04 |

- **z-series:** exp false-ν phase portrait is space-filling (PR 2.50, sphericity 0.31)
  vs a tight curve for MC (μ PR 1.26) → exp false-ν lack a **coherent depth-vs-time
  trajectory** (z jumps around with hit order; real events develop smoothly in
  depth/time). **Caveat:** z is z-scored per event, so a compact horizontal event
  (small z-range → amplified noise) also gives a disordered cloud — this signal
  **largely re-expresses the horizon/verticality** finding (2026-07-06 §4b), not
  clearly new. A clean version would embed the causal trajectory (arc-length/full
  position), not z alone.
- **Q-series:** exp false-ν are *more* ordered (PR 2.05 < MC 2.55) — a
  geometry-independent signal, but a modest separation.

**Cross-check across score bins (`unfolding_by_score.py`,
[`tables/unfolding_by_score.csv`](tables/unfolding_by_score.csv),
[`figures/unfolding_by_score.png`](figures/unfolding_by_score.png)) — the z-signal is
the horizon confound, confirmed.** Stratifying by score, z-PR of exp and MC is nearly
identical in bins 0.2–0.8 (e.g. 2.39 vs 2.39 at 0.6–0.8); the gap appears **only at
p>0.8** (exp 2.50 vs MC 1.91), and there it tracks a **verticality** gap (rvert exp
0.56 vs MC 0.80): at high score MC = vertical real ν (coherent z, low z-PR) while exp =
horizontal false-ν. In every bin z-PR mirrors rvert for *both* domains → z-PR ≡
verticality, no residual. So "exp has noisier z" is **not** a domain property — it is
horizontality (§4b, 07-06) re-expressed, visible only where high-score MC becomes
vertical-ν-rich.

**Decisive control — split MC by class, compare the two FALSE-POSITIVE populations
exp_hi vs muatm_hi** (`unfolding_by_class.py`,
[`tables/unfolding_by_class.csv`](tables/unfolding_by_class.csv),
[`figures/unfolding_by_class.png`](figures/unfolding_by_class.png); MC from the boosted
6.6M-muatm preds). The earlier comparison was against real MC ν (vertical) — trivial.
The right control is the two backgrounds that leaked past 0.8:

| population (score>0.8, h8s3) | z-PR | Q-PR | rvert | N |
|---|---|---|---|---|
| exp_hi (false ν) | 2.50 | **2.05** | 0.564 | 361 |
| muatm_hi (false ν) | 2.43 | **2.25** | 0.568 | 283 |
| nue2_hi (real ν) | 1.85 | 2.56 | 0.812 | 4000 |

**z-PR of exp_hi and muatm_hi are ~equal (2.50 vs 2.43) and their rvert is identical
(0.564 vs 0.568)** — the "noisy z" is a property of *horizontal false positives in
general* (exp and MC muons alike), NOT exp-specific. The z-signal fully collapses.
**The only exp-specific residual is Q-PR:** at equal geometry and equal false-positive
status, exp false-ν have a *more ordered* charge sequence than muon false-ν (2.05 vs
2.25). Modest (~10%), but genuinely exp-specific.

**Verdict:** properly controlled (thanks to splitting MC by class), the delay-embedding
z-"noise" is just horizontality shared by all false positives; the single genuine
exp-specific temporal signature is a mild charge-ordering (Q-PR) — weak, not a
stand-alone discriminator.

---

## 4. Formalizing "exp_hi is OOD" at the raw-event level

Route (c): represent each event in many ways and collect a rich feature set (~33
features), then a formal one-class / two-sample OOD test — no dependence on the
nu-classifier's embedding. Representations: marginal moments (Q, t, xyz), inter-hit
time gaps, position-cloud PCA shape, Takens delay-embedding shape (Q, z, principal
axis), space-time **causality** (position-along-principal-axis vs time regression
residual `speed_resid` + correlations), string topology. Script
[`raw_ood_features.py`](raw_ood_features.py); tables `tables/raw_ood_{features,summary}.csv`;
figure [`figures/raw_ood.png`](figures/raw_ood.png). Populations at score>0.8, h8s3:
exp_hi (false ν), muatm_hi (muon false ν), mc_ref (all-MC manifold).

**Formal result — the rich features DO formalize exp_hi as raw-OOD** (the coarse §4d
set failed):

| test (RF 5-fold OOD-AUC) | AUC |
|---|---|
| exp_hi vs MC-ref (general OOD) | **0.962** |
| muatm_hi vs MC-ref (control) | 0.935 |
| **exp_hi vs muatm_hi (exp-specific)** | **0.701** |

OOD scores (median Mahalanobis to MC): mc_ref 23, muatm_hi 50, exp_hi 60.

**Localisation (per-feature KS):**
- **exp_hi vs MC-ref** is carried by *time-compactness + causality*: `t_std`, `t_span`,
  `speed_resid`, `prAxis` — but these are **shared with muatm_hi** (both false
  positives are short, compact, causality-violating events). This is the
  "high-score-false-positive corner", not exp-specific.
- **exp-specific (exp_hi vs muatm_hi)** is carried entirely by **CHARGE**: `Q_std`,
  `Q_max`, `Q_mean`, `frac_Q_gt10`.

**What the charge says (direction matters):** exp_hi are **brighter**, not dimmer, than
muon false-ν — Q_mean 7.1 vs 4.7, Q_std 10.7 vs 5.3, Q_max 36 vs 19, frac(Q>10) 0.18 vs
0.13; rvert is identical (0.564 vs 0.568). So the OOD population is **near-horizontal
AND bright**. In MC that combination is absent: near-horizontal events are dim muons,
while bright events are energetic (neutrino-like). exp supplies bright horizontal events
→ OOD, and the brightness (an energy proxy) tips the classifier to "neutrino".

**Formal statement of the OOD.** exp_hi occupy a region of raw-event space that is (i)
unusual w.r.t. all MC (AUC 0.96 — the compact/causality-violating false-positive
corner, shared with muons) and (ii) exp-specifically displaced in the **charge**
distribution (AUC 0.70, brighter). A standard sigmoid extrapolates over this region
*over-confidently onto the neutrino side* (not p≈0.5) — hence a systematic excess. This
is exactly what a distance-aware output (SNGP) would neutralise. (The earlier "OOD-ness
tracks score, exp_lo maha≈46 vs exp_hi≈499" argument is dropped: exp_lo is a *different*
population — vertical EAS-like — not the same events at low score; see §5 for the
controlled test at fixed geometry.)

---

## 5. Score distribution at FIXED geometry (controlled, no score cut)

Controls the circularity: instead of comparing OOD across score (different
populations), fix the geometry (verticality rvert) — same selection on exp and muatm —
and read off the FULL score distribution. Script [`score_by_geometry.py`](score_by_geometry.py),
[`figures/score_by_geometry.png`](figures/score_by_geometry.png); a larger,
brightness-resolved follow-up is `score_vs_brightness.py`.

| selection (h8s3, any score) | frac>0.5 | frac>0.8 | median |
|---|---|---|---|
| exp / muatm VERTICAL (rvert>1) | 0.000 / 0.000 | 0.0001 / 0.0000 | 0.011 |
| exp / muatm HORIZONTAL (rvert<0.7) | 0.013 / 0.009 | 0.0020 / 0.0020 | ~0.015 |
| exp / muatm HORIZONTAL & BRIGHT | 0.024 / 0.007 | 0.0057 / 0.0035 | ~0.017 |

- **Vertical events (both) score ~0 high** → the excess comes from horizontal events.
- **Horizontal events (both) are peaked LOW** (median ~0.015), *not* uniform — the model
  correctly rejects most of them; and exp-horizontal ≈ muatm-horizontal. So horizontality
  alone does not distinguish exp from muatm. (Refutes a "muatm→U(0,1)" picture.)
- **Brightness enhances the high-score tail, more for exp** — bright-horizontal frac>0.5
  exp 0.024 vs muatm 0.007, frac>0.8 0.0057 vs 0.0035. But it is a **modest tail
  enhancement, not p→1** (median still 0.017, 0% >0.95).

**Robust follow-up (`score_vs_brightness.py`, `tables/score_vs_brightness*.csv`,
`figures/score_vs_brightness.png`; larger sample, muatm from the boosted 6.6M).**

Excess decomposition (frac>0.8): total exp/muatm = **2.5×** = **abundance ×1.43**
(exp is 1.43× more horizontal) × **per-horizontal ×1.62** (among horizontal, exp scores
>0.8 1.62× more). Roughly equal factors — cleaner than §4b's Bayes estimate which
inflated the per-angle term.

Score-vs-brightness (horizontal): Spearman corr(score,Q) is weak for both (exp 0.053,
muatm 0.036) and the curves overlap at low–mid Q; they **diverge only at the bright
extreme** (Q>15: exp mean-score 0.109 & frac>0.8 0.016 vs muatm 0.065 & **0/84**). So
the per-event exp effect is a **bright-tail** phenomenon.

**Interpretation — physics, not memorization/sparsity.** The bright-horizontal region is
NOT sparse in training (~1% of muatm ⇒ ~30k events); the net **correctly rejects
test-MC bright-horizontal muons** (0/457 at >0.8). It fails only on **exp**
bright-horizontal at matched brightness+geometry ⇒ the difference is exp's detailed
physics (charge/topology, §4 raw-OOD AUC 0.70), i.e. **exp bright-horizontal events are a
genuinely different physical population** MC never produces. The excess = physics domain
gap (input) × over-confident OOD extrapolation where the decision is unconstrained by MC
support (augmentation/validation/test-split do not fix this — only a distance-aware
output does).

---

## 6. Fine-tune fix — set assembly (poor man's SNGP)

Rather than rebuild the head (SNGP), teach the current classifier to reject the
OOD population directly: **fine-tune E1 with a labelled exp-background stream
(label 0)**, MC providing the anchor (MC already correctly rejects horizontal muons).
The selection is `encoder-OOD ∧ rvert<0.7`, with encoder-OOD the PRIMARY selector
(exactly the failure mode we treat — over-confident extrapolation in encoder space,
§4/2026-07-06 §4e) and rvert<0.7 a soft safety filter to avoid labelling any rare
steep (potential-ν) event as background.

### 6a. rvert<0.7 reliability (`rvert_theta_check.py`)
rvert=std_z/std_xy on filtered hits vs GT zenith θ (muatm, prime_prty[:,0]), N≈14k:
- **Strong monotonic rank-proxy**: Pearson 0.836, Spearman 0.894; median θ falls
  smoothly with rvert (112°→159° across the range).
- **Loose per-event cut**: rvert<0.7 selects median θ≈118° (not 90°); as a
  near-horizon (θ<110°) selector, efficiency 0.74 but purity only 0.13.

| rvert bin | N | θ median | frac θ<110° |
|---|---|---|---|
| 0.00–0.40 | 108 | 112 | 0.39 |
| 0.40–0.55 | 696 | 116 | 0.21 |
| 0.55–0.70 | 1462 | 120 | 0.08 |
| 0.70–0.90 | 2540 | 126 | 0.03 |
| 0.90–1.20 | 3232 | 134 | 0.01 |
| 1.20–2.00 | 4172 | 145 | 0.0005 |
| 2.00+ | 2060 | 159 | 0.0 |

→ rvert is reliable as a **population rank-proxy** but too loose for a per-event θ cut.
It is adequate here only because it is the SECONDARY filter: encoder-OOD⟺high-score⟺
horizontal, so rvert<0.7 is mostly redundant and only excludes the rare OOD+vertical
event (which we deliberately keep out of the background label). Fig `figures/rvert_theta.png`.

### 6b. Assembly (`analysis/finetune_set/assemble.py`) — outstanding
Sample large exp (h8s3, out-of-training, drop c02 r20/r249) → encoder embeddings +
rvert (E1@ep10); encoder-OOD = kNN distance to a MC embedding reference; threshold at
MC p99 / p99.9; report set size and score/rvert/Q distributions at each cut before
touching the trainer.

---

## Conclusions
1. **Horizon θ-loss (E5, preliminary):** no working-point improvement; fixed-0.8 gain
   is score compression. Revisit at convergence.
2. **DA is not the cause:** λ=0 gives a *slightly higher* working-point excess.
3. **Temporal geometry:** controlling exp_hi vs muatm_hi (both false positives), the
   z-"noise" is identical → it is horizontality shared by all false positives, not
   exp-specific. Only a mild charge-ordering (Q-PR 2.05 vs 2.25) is genuinely
   exp-specific — weak, not a discriminator.
4. **Raw-level OOD formalized:** exp_hi are separable from MC in raw-event space
   (AUC 0.96, rich features), largely the shared compact/causality-violating
   false-positive corner; the **exp-specific** part (AUC 0.70) is **charge** — exp_hi
   are **near-horizontal AND bright** (Q_mean 7.1 vs muon-FP 4.7), a combination MC
   lacks (its horizontal events are dim muons). The classifier extrapolates
   over-confidently onto the ν side → systematic excess.
5. **Both cheap interventions fail → the principled fix remains a distance-aware /
   OOD-aware output (SNGP)** operating on the full-space distance the classifier
   currently ignores (2026-07-06 §4e).

## Next steps
- Finish E5 and re-compare at convergence (matched working point).
- **SNGP** distance-aware head (SN encoder + RFF-GP output + Laplace covariance +
  mean-field logit) — the outstanding build.
- Cleaner unfolding: delay-embed the causal trajectory to separate "coherent
  horizontal track" from "incoherent blob"; test as a background veto.
