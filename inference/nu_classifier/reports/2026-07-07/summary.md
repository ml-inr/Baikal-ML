# Summary 2026-07-07 — interventions & temporal-structure probe

Follows [../2026-07-06/summary.md](../2026-07-06/summary.md) (which concluded the exp
excess is over-confident OOD extrapolation). Full detail: [REPORT.md](REPORT.md). All
comparisons at a fixed muon-suppression working point (μ-survival=1e-3), h8s3.

---

### Q1. Does a horizon θ-loss (abstain near zenith 90°) remove the excess?
**Preliminary: no.** E5 (E1 + horizon soft-label, σ=10.5°) @epoch 5: at the working
point the excess is 0.218% vs E1 0.18–0.19% — no gain, and −3.5% signal efficiency.
The huge fixed-0.8 drop (29 vs 532 events) is a score-compression artifact. E5 is
undertrained; revisit at convergence. → REPORT §1; `comparison_table_e5vse1.csv`.

### Q2. Does domain adaptation (DANN) cause the excess?
**No.** With DA fully off (FixedDA λ=0, val-matched to E1@ep10) the working-point
excess is *slightly higher* (0.227% vs 0.190%, ~4σ), not gone — λ=0 means the model
never sees exp, so it over-extrapolates more. DA mildly helps the bulk. → REPORT §2;
`comparison_table_lambda0.csv`.

### Q3. Do exp false-ν have a different temporal (hit-sequence) structure?
**It's the horizon in disguise, not a domain property — confirmed by binning on score.**
Takens delay embedding: exp false-ν have a space-filling z-depth phase portrait (PR 2.50
vs MC 1.26 at p>0.8). BUT stratified by score, exp and MC z-PR are equal in bins 0.2–0.8;
the gap is exclusive to p>0.8 and tracks a verticality gap there (high-score MC = vertical
real ν with coherent z; exp = horizontal false-ν). z-PR ≡ verticality (§4b), no residual.
Decisive control (split MC by class, compare the two false-positive populations): exp_hi
vs muatm_hi have ~equal z-PR (2.50 vs 2.43) and identical rvert (0.564 vs 0.568) → the
z-"noise" is horizontality shared by all false positives, not exp-specific. Only Q-PR is
genuinely exp-specific (exp more ordered charge, 2.05 vs 2.25) — weak. No clean new
discriminator. → REPORT §3; `unfolding_by_class.csv`.

### Q4. In what sense is exp_hi OOD, at the raw-event level?
**Formalized (route c).** A rich multi-representation feature set (marginals, PCA shape,
Takens delay embedding, space-time causality, topology) separates exp_hi from MC with
RF OOD-AUC 0.96 (the coarse §4d set failed). Most of that is the shared
compact/causality-violating false-positive corner (muatm_hi vs MC also 0.94). The
**exp-specific** deviation (exp_hi vs muatm_hi, AUC 0.70) is entirely **charge**: exp_hi
are **near-horizontal AND bright** (Q_mean 7.1 vs muon-FP 4.7, rvert identical 0.56) — a
combination MC lacks (horizontal MC events are dim muons; bright MC events are energetic
ν). → REPORT §4; `raw_ood_features.py`, `raw_ood_summary.csv`.

## Overall
Neither cheap intervention (horizon loss, DA off) removes the working-point excess; the
temporal probe finds no clean new discriminator; the raw-level OOD is formalized as
"near-horizontal + bright", a combination absent in MC on which the classifier
extrapolates over-confidently to ν. The principled fix remains a **distance-aware /
OOD-aware output (SNGP)** — outstanding build.

---

## Appendix — metric definitions (delay-embedding shape)

**Delay (Takens) embedding.** A scalar time series `s = (s₁,…,s_n)` is mapped to a point
cloud in Rᵐ by sliding a window of length m: `pᵢ = (sᵢ, sᵢ₊₁, …, sᵢ₊ₘ₋₁)`. With m=3 each
point is `(sᵢ, sᵢ₊₁, sᵢ₊₂) ∈ R³`. This "unfolds" a 1-D sequence into a geometric object
(phase portrait) whose shape encodes the sequence's dynamics — a smooth/periodic series
traces a curve; white noise fills the cube. We z-score `s` per event first (mean 0, std
1) so the *shape*, not the scale, is compared. Here `s` = the per-hit charge `Q` or
depth `z`, with hits ordered by time.

Let the point cloud have covariance matrix with eigenvalues `λ₁ ≥ λ₂ ≥ λ₃ ≥ 0`
(the variances along its principal axes), and `fᵢ = λᵢ/Σλ` the normalized ones.

- **Participation ratio** `PR = (Σλᵢ)² / Σλᵢ²`. The *effective number of dimensions* the
  cloud occupies. `PR≈1` ⇒ all variance on one axis = a 1-D curve (structured,
  predictable series). `PR≈3` ⇒ variance spread equally over all 3 axes = a
  space-filling blob (noisy, unpredictable series). It is a smooth measure of "how many
  axes matter", robust to which axis. (Same quantity we used for the encoder embedding
  in 2026-07-06 §4e, there =1.7.)
- **Sphericity** `= λ₃/λ₁` (smallest / largest eigenvalue). `≈0` ⇒ flat/elongated (a
  curve or plane); `≈1` ⇒ isotropic ball. High sphericity = the delay cloud is a
  round blob = no preferred direction = noise-like dynamics.
- **Planarity** `= f₂ − f₃`. Large ⇒ the cloud is spread in 2 axes but thin in the 3rd
  = lies near a plane. Distinguishes a 2-D sheet from a 1-D curve or a 3-D blob.
- **λ₁ (= f₁)** the fraction of variance on the leading axis: high ⇒ the series is
  dominated by one linear trend (e.g. a monotonic ramp).

Interpretation for us: a real particle traversing the detector gives an ordered
space-time sequence → its z-delay cloud is a **curve** (low PR, low sphericity). A
noise-like / incoherent event gives a **blob** (high PR, high sphericity). exp false-ν
sit toward the blob end (PR_z 2.50, sphericity_z 0.31) — but see the z-scoring
horizon-confound caveat in REPORT §3.
