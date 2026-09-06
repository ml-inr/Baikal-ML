# Report 2026-07-09 — Full-statistics exp validation, bimodal super-confident ν peak, and candidates

Follows [../2026-07-08/REPORT.md](../2026-07-08/REPORT.md) (fine-tune halves the exp
excess with ν-efficiency preserved; SNGP implemented). Earlier chain:
[2026-07-06](../2026-07-06/REPORT.md) (excess = over-confident OOD extrapolation),
[2026-07-07](../2026-07-07/REPORT.md) (interventions, raw-OOD). Today: score the **full**
exp_full with both E1 (base) and the fine-tuned model, and study the high-score structure.

Reference models: **E1** `260705_0702_..._E1_lambda0.01@da_checkpoint_epoch_010`,
**FT** `...@da_checkpoint_epoch_010_ood2p5_rv0p7_finetuned@best_finetuned_model`. All h8s3
(`n_sn_hits≥8, n_sn_strings≥3`), good runs (c02 r20/r249 dropped).

---

## 1. Full exp_full inference (both models, identical event set)

Previously only a ~1% capped sample was scored. Now the **entire** exp_full is scored by both
models via `predict_exp --probs-h5` (precomputed SN probs cover all 29 parts fully). Fast:
**~20 min each** (skips the SN model; sequential whole-part reads) — not the day I first
estimated.

- **3,110,199 h8s3 good-run events**, the *same* events for E1 and FT (valid comparison).
- score>0.8: **E1 = 2,259**, **FT = 153** (≈15× fewer at fixed 0.8 — score compression + real
  reduction; at matched working points ≈2×, 2026-07-08 §6).

**Tooling:** added `--probs-h5` to `predict_mc.py` (mirrors `predict_exp`) so MC scoring can
also reuse precomputed probs and skip the on-the-fly SN model.

### 1a. Precomputed probs (a) vs on-the-fly SN (b) — which is faster
Both read the raw hits; (a) additionally reads a second (probs) h5, (b) additionally runs the
SN model. **Bulk / whole-part scans: (a) wins** (sequential probs read is cheap; skips SN
inference over millions of events — full exp 20 min via (a) vs hours via (b); muatm 7.7 h (a)
vs ~13 h (b)). **Scattered / small analysis reads: (b) wins** (random access into a
gzip-chunked probs file decompresses whole chunks for a few floats; SN on a handful of events
is instant) — and (b) has no coverage gaps (MC probs cover only 48%/17%/5% of muatm/nue2/nuatm
parts). Use (a) for bulk predict, (b) for scattered analysis / incomplete coverage.

---

## 2. Bimodality — a super-confident ν peak separates from the muon background

With the full statistics the FT high-score tail is resolved (153 events >0.8, vs 1 in the 1%
preview). FT exp vs FT MC-muatm (background), survival ratio by score:

| score> | FT exp N | FT exp/μatm |
|---|---|---|
| 0.5 | 1847 | 1.23 |
| 0.8 | 153 | 1.24 |
| 0.9 | 38 | 1.39 |
| 0.95 | 20 | **3.9** |
| 0.97 | 14 | **8.6** |
| **0.99** | **7** | **51.6** |

Up to ~0.9 exp behaves like the muon background (ratio ~1.2); above it the ratio **explodes**:
3.9× at 0.95 → 8.6× at 0.97 → **51× at 0.99**. The fine FT-exp counts in [0.85,1.0] fall
(background tail), dip around [0.94,0.98], then **rise at [0.98,1.0]** (7 events in the top bin
>0.99) — a super-confident peak, whereas muatm dies off (≈1 muon expected >0.99). This is the
bimodal separation of a genuine-ν peak from the background. Statistics are still modest (7
events >0.99), but the 51× ratio is robust. → `bimodality_full.py`,
`figures/bimodality_full.png`.

---

## 3. The super-confident events look up-going — but that alone does NOT certify ν

Cross the 38 super-confident FT events (score>0.9) with verticality (rvert) and direction
(corr(t,z): down-going μ ⇒ corr<0, calibrated on muatm at −0.945; up-going / cascade ⇒ corr>0):

- **36 of 38 are up-going-like (corr>0.3)**; 15 vertical (rvert>0.9); the 7 events >0.99 are
  all up-going (corr 0.49–1.0), spread across clusters 2/4/5/6/7 and different runs, spanning
  dim→bright (Q 2.4–23) and 8–44 hits.

Initially read as the neutrino signature. **§4 (corner cases) revises this:** MC down-going
muon false-positives are *also* 83% up-going-like, so corr(t,z) alone does not separate ν from
muon-background false positives. The candidates are a **mix** of genuine ν and muon fakes;
isolating ν needs the anti-multi-muon cut of §4. → `nu_candidates_full.py`,
`figures/nu_candidates_full.png`, `tables/nu_candidates_full.csv`.

---

## 4. Two corner cases (found in the event displays) — and what they reveal

Inspecting the top-16 candidate displays surfaced two failure modes. Using MC muatm **truth**
(per-hit labels `raw/labels`; muon multiplicity `prime_prty[:,4]`) vs the exp candidates
(geometry only), on the high-FT-score muatm false-positives (score>0.9, N=141):

**Case 2 — two weak muons (early+low, late+high) fake an up-going track. EXISTS in MC, not
exp-specific.** **79% of high-score muatm are multi-muon (n_muons≥2, truth)**, 83% are
up-going-like, 64% are both. MC muon bundles themselves fake up-going. → The classifier's
false-positives are dominated by multi-muon events; **filtering on muon multiplicity (MC
truth `prime_prty[:,4]`, or a two-cluster geometric proxy / reconstruction in exp) removes
most of them** — a powerful, physically-motivated cut. (Two-cluster gap>2: muatm 37%, exp
candidates 26%.)

**Case 1 — an isolated passed-noise hit at the top/bottom inflates verticality / flips
direction. Rare in both, ~7× enriched in exp.** 81% of high-score muatm carry a passed-noise
hit (truth), but it rarely dominates the geometry: rvert-driven-by-one-hit 0.7%,
direction-flip-on-one-hit-removal 1.4%. exp candidates: 5.3% and 7.9% (~7×). So the extreme
single-noise-hit case is partly exp-specific (a noise-model domain gap), but small.

**Would feeding the per-hit SN prob let the net hedge on such hits? Only weakly**
(`noise_hit_probs.py`). Among passed hits of high-score muatm, truth-noise probs (median
0.888) barely separate from truth-muon (0.909); for the isolated z-outlier hit, noise 0.851 vs
muon 0.865 — heavily overlapping, and **almost none >0.99** (the SN model is not confidently
wrong, but it also cannot cleanly tell an isolated noise hit from signal — the ambiguity is
inherent). So passing the SN prob as a 6th input feature (infra exists, `input_dim=6`;
current models use `input_dim=5`, blind to it) helps only modestly (isolated noise skews
<0.85: 49% vs 37%). The stronger, prob-agnostic fix is **consistency / robustness training**
(perturb or drop the marginal/isolated hit, penalise a score flip) and/or a leave-one-out
uncertainty at inference (`lo_drop`: topology hinges on one hit → lower confidence) — teach the
net to hedge on *fragile* topologies, not just on low prob values.

**What this reveals about the network.** A physics-aware net that internally reconstructs
tracks would see two separated clusters (case 2) and reject them as non-single-track. Instead
the net scores them as confident ν. So the net is **not learning track topology / multiplicity**
— it relies on coarse correlational proxies (charge distribution, coarse z–t correlation ≈
direction) that a two-track bundle happens to satisfy. This is the same root as the exp excess
(bright-horizontal → ν): the classifier learns training-distribution shortcuts, not robust
physics, and those shortcuts fail on OOD inputs and on topologies rare in training (multi-muon).
**Concrete improvements (MC truth enables them), targeting information + incentive:**
- **Case 2 (topology):** multi-task auxiliary head predicting muon multiplicity `n_muons`
  (MC truth `prime_prty[:,4]`), and/or MC multi-muon bundles as explicit hard negatives —
  forces the encoder to represent topology, rejecting two-track fakes at the source.
- **Case 1 (hit ambiguity):** the net is `input_dim=5`-blind to per-hit SN prob, but the prob
  is only weakly informative (noise/muon overlap, §above), so feeding it is a weak patch. The
  stronger fix is consistency/robustness training (penalise score flips under marginal-hit
  perturbation) + a leave-one-out `lo_drop` uncertainty at inference.

Both are cheaper and more principled than post-cuts. → `corner_cases.py`, `noise_hit_probs.py`,
`tables/corner_cases_muatm.csv`.

---

## 5. Visualisations added (for the paper)

- `paper_suppression_curve.png` — re-plotted with the **expected line** (exp = muon survival,
  i.e. no excess): panel 1 diagonal y=1/x, panel 2 the excess factor exp/μ with the =1 line.
  E1 sits at 1.9–2.5×, FT at 1.1–1.2×. (`replot_paper_curve.py`.)
- `score_hist_before_after.png` — exp score distribution E1 vs FT, and exp vs MC-muatm; the
  neutrino-like tail shrinks to the muon-background level after FT (score>0.8: E1 exp 7.5e-4 →
  FT exp 3.8e-5 ≈ FT muatm 2.0e-5). (`score_hist_before_after.py`.)
- `nu_candidates_full.png` — event displays of the super-confident candidates (3D hits,
  colour=time, size~charge; V=vertical, ↑=up-going).

---

## 6. E1-vs-FT suppression to 1e-6, and the SNGP evaluation

**Suppression curve extended to muon-rejection 1e-6.** A methodological point on the
out-of-training definition (matters for scientific honesty): the exp sample should exclude only
**label-leaked** events — the MC training set (labelled 0/1) and the FT background (labelled 0).
The **DA target exp was UNLABELLED** (used only for domain alignment; the classifier never saw a
class label for it), so those exp events are valid for scoring. Excluding them (the conservative
choice) drops the exp sample from 3.10M to 992k **and preferentially removes the super-confident
peak** (those events largely lived in the DA target) — which made an earlier version misleadingly
show "FT dips below the muon level" at strict cuts. Keeping the DA-target exp (excluding only the
13,989 FT-background events, identically for both models) gives 3.10M exp and the honest picture:

| μ (rejection) | E1 exp/μ | FT exp/μ | E1 events | FT events |
|---|---|---|---|---|
| 1e-3 | 1.74 | **1.09** | ~5400 | ~3400 |
| 1e-4 | 2.17 | **1.12** | 672 | 346 |
| 1e-5 | 3.04 | **1.36** | 94 | 42 |
| 3e-6 | 3.98 | 2.69 | 37 | 25 |
| 1e-6 | 5.81 | 5.49 | 18 | 17 |

The fine-tune removes the **moderate domain-gap excess** (1e-3–1e-4: E1 ~2× → FT ~1.1×, the
false-ν it was built to reject) with ν-efficiency preserved (≤1.5%). At the strictest cuts BOTH
models rise and **converge to ~5.5× at 1e-6** (18 vs 17 events, real statistics) — this is the
bimodal super-confident peak (§2), which the fine-tune does *not* remove and should not (it is
genuine-ν candidates and/or exp-specific accidental muon coincidences, not domain-gap background).
So the fine-tune cleanly separates "domain-gap background (removed)" from "super-confident
candidates (preserved)". → `paper_suppression_curve_daexp.py`, `figures/paper_suppression_curve_daexp.png`
(the conservative 992k version, `paper_suppression_curve.py`, is superseded — it removes the peak).

**SNGP evaluation on exp** (`sngp_exp_eval.py`; the trained SNGP, val AUC 0.9999). Predictive
variance (φᵀΣφ) and mean-field score per population:

| population | SNGP score (med) | variance (med) |
|---|---|---|
| exp false-ν (E1>0.8) | **0.833** | **0.0198** |
| MC real ν (nue2) | 0.989 | 0.0066 |
| exp bulk (E1<0.2) | 0.011 | 0.0027 |
| MC muatm (bg) | 0.009 | 0.0021 |

**SNGP does flag the exp false-ν: 3.0× the variance of real ν, and a shrunk score (0.833 vs
0.989).** This is *better* than the 2026-07-08 §8 prediction ("edge-of-nu → SNGP barely helps"),
which was measured on E1's collapsed (~1.7-dim) embedding. The from-scratch **spectral-normed
encoder keeps the embedding from collapsing**, so in its geometry the false-ν are genuinely
farther from the ν manifold (×3 variance), exactly as hypothesised. Partial, not full (0.833 is
still >0.5), but a solid distance-aware signal — and it needs **no exp-OOD labels** (unlike the
fine-tune) and would flag genuinely-far VHE ν as uncertain (safety). → `figures/sngp_exp_eval.png`.

---

## Conclusions
1. Full exp_full scored by both models (3.11M h8s3, identical events); FT reduces score>0.8
   count ~15× at fixed threshold (~2× at matched working points).
2. **Bimodality confirmed:** a super-confident peak at score>0.99 rises **51× above the (MC)
   muon background**, separated from the background-like bulk (ratio ~1.2 below 0.9). **Caveat:**
   the reference is single-shower MC muatm, which does NOT model **accidental coincidences of two
   independent muons** overlapping in the readout window — an exp reality that also fakes
   up-going. So the >0.99 excess is genuine atmospheric ν *and/or* exp-specific random
   double-muon coincidences; disentangle with the two-cluster / timing cut (and real reco).
3. The super-confident peak *looks* up-going, but **corr(t,z) alone does not certify ν**: MC
   muon false-positives are also 83% up-going-like. The candidates are a mix of ν and muon fakes.
4. **Corner cases (§4):** the classifier's high-score false-positives are dominated by
   **multi-muon events (79% of high-score muatm, truth)** that fake up-going — present in MC,
   removable by a muon-multiplicity / two-cluster cut. This reveals the net is **not learning
   track topology**, only coarse correlational shortcuts (same root as the exp excess). A
   multi-task `n_muons` auxiliary (MC truth available) is the principled fix.
5. **Suppression to 1e-6 (§6, honest out-of-training):** keeping the UNLABELLED DA-target exp
   (excluding only label-leaked events → 3.10M exp) shows FT removes the moderate domain-gap
   excess (1e-3–1e-4: E1 ~2× → FT ~1.1×, ν-eff preserved) but the **super-confident peak
   survives in both, converging to ~5.5× at 1e-6** (the candidates, not domain-gap background).
   The earlier conservative curve (992k) misleadingly showed "FT below the muon level" because it
   removed the peak (those events were in the DA target). **SNGP (§6):** flags the exp false-ν
   with 3× the variance and a shrunk score (0.833 vs 0.989 for real ν) — the SN encoder's
   un-collapsed geometry works *better* than the E1-embedding pessimism; a label-free
   distance-aware complement to the fine-tune.
6. Tooling: `predict_mc --probs-h5`; perf rule — precomputed for bulk, on-the-fly SN for
   scattered/incomplete-coverage reads.

## Next
- **Anti-multi-muon cut** on the candidate sample (n_muons in MC / two-cluster proxy in exp),
  and cross-check candidates against reconstruction + physics expectation (livetime × flux);
  disentangle genuine ν from accidental double-muon coincidences (timing).
- **Multi-task with `n_muons`** auxiliary head (or MC multi-muon hard negatives) — teach the
  net topology, removing the case-2 fakes at the source (case 1: consistency/`lo_drop`).
- **SNGP × fine-tune:** combine (SN encoder + OOD background) and/or use SNGP variance as a
  candidate-quality / VHE-safety flag.

## Artifacts
| Path | What |
|---|---|
| `predict_mc.py --probs-h5` | fast persisted MC scoring via precomputed SN probs |
| `analysis/model_comparison/bimodality_full.py` | §2 bimodal super-confident peak (exp/muatm vs score) |
| `analysis/model_comparison/nu_candidates_full.py` | §3 super-confident candidates: verticality + direction + displays |
| `analysis/model_comparison/corner_cases.py` | §4 corner cases: MC-truth multi-muon (79%) + noise-hit vs exp candidates |
| `analysis/model_comparison/noise_hit_probs.py` | §4 SN-prob of passed noise vs muon hits (prob only weakly informative) |
| `analysis/model_comparison/sngp_exp_eval.py` | §6 SNGP predictive variance/score on exp false-ν vs real ν (3× var) |
| `analysis/model_comparison/replot_paper_curve.py` | suppression curve with expected (no-excess) line |
| `analysis/model_comparison/score_hist_before_after.py` | exp score before/after FT vs muatm |
| `preds/{E1,FT}/…/exp_full_thr0p8.duckdb` | full exp_full scores (3.11M h8s3, both models) |
| `figures/*`, `tables/*` (in analysis/model_comparison) | plots and per-event tables |
