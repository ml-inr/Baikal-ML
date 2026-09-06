# Summary 2026-07-08 — fine-tune fix (OOD background), ν-safety, SNGP build

Follows [../2026-07-07/summary.md](../2026-07-07/summary.md). Full detail:
[REPORT.md](REPORT.md). Reference model **E1 @ epoch 10**; all h8s3, exp = exp_full
out-of-training.

---

### The fix (interim): fine-tune E1 on exp OOD events as hard background (label 0)
"Poor man's SNGP" — teach the classifier to reject the OOD population directly, MC anchor
(1:1:2) preserving ν. Uses the existing `ExpFineTuningTrainer`; **main DA trainer
untouched**. Original = **ep10** (not "best"=ep23) for continuity: all diagnostics + the
prediction DBs + the OOD embedding space are anchored to ep10; "best" is best-on-MC-val
(saturated, blind to the exp excess).

### OOD by kNN, not Mahalanobis
The MC embedding is ~1.7-dim → its covariance is near-singular → Mahalanobis's pseudo-inverse
is noise-amplified (that's why exp_hi got maha 499, scale-soft). kNN is non-parametric,
robust, and kNN>threshold = far from *all* MC (incl. MC-ν) → safe to label 0 without hurting
ν-eff. Mahalanobis flags the excess more directly but fragilely; the principled version is
SNGP (learned Mahalanobis in RFF-GP space).

### Non-monotonic OOD⋄score — the excess sits at the MC-ν edge
kNN-OOD peaks at *mid* score (0.2–0.8, OOD ~4.4); the score>0.8 excess has OOD **3.65 < MC
p99 (4.10)** because those events sit at the MC-ν blob edge (→ high score, moderate kNN). So
`OOD>p99` captures only 20% of the false-ν. Sweep → **chose OOD>2.5 ∧ rvert<0.7** (recall
0.73). Built from the full 480k pool → **13,989 background events** (score med 0.11).

### ν-safety
- **MC nue2 test:** brighter → SAFER (feared bright→OOD mechanism is reversed; OOD falls with
  hit count). 7.7% of nue2 fall in the selection overall, ~4% for the bright tail (Q15–30).
  True VHE untestable (only 12 MC nue2 at Q≥50). rvert weak for cascades (37%<0.7).
- **Directional filter (down-going-only) — tested, REJECTED.** corr(t,z) cleanly separates
  down-going muatm (−0.945) from cascades (+0.80), and nominally protects nue2 (8.3%→0.3%).
  BUT the score>0.8 exp false-ν are themselves up-going-like (median corr **+0.357**, 3.8%
  down-going) — indistinguishable from real ν by any geometric proxy (that's *why* they fool
  the net). The filter would delete 96% of the target false-ν. → dropped.
- **Conclusion:** no cheap proxy separates exp-false-ν from real ν. ν-safety rests on the
  MC-ν anchor + the **post-fine-tune ν-efficiency check** (decisive; back off if it drops).

### RESULT — fine-tune roughly halves the excess, ν-efficiency preserved (§6)
Fine-tuned E1 on the 13,989 OOD-background events (label 0, MC anchor 1:1:2). **Final,
strictly out-of-training, to muon survival 1e-5 with Poisson errors:** the exp excess (exp
survival relative to muon survival, exp/μ) drops from E1's 1.9–2.5× to FT's **~1.1–1.2×**
across the whole range (FT/E1 ≈ 0.4–0.6), while ν-efficiency is preserved (≤1.5% vs E1 at
every point). So exp nearly behaves like the muon background, with a small honest residual
(partly real ν, partly background removable by the verticality cut). MC val AUC 0.9998
(intact). Headline, publishable result.

### Horizon θ-loss (E5), converged — does NOT help (§7)
E5@ep8 vs E1@ep10 across working points: excess ratio ≈1 (worse at 1e-3–3e-4), and E5 loses
signal efficiency at strict cuts. Confirms the 07-07 preliminary on the converged model.

### The corner is occupied by ν, not empty (§8) — a physics correction
Bright∧horizontal MC events are ν (nuatm/nue2 score ~0.98; muons rejected 0.018). The net
learned "bright+horizontal → ν" from real MC ν; exp bright-horizontal (which MC muon sim
lacks) inherits it. False-ν sit just beyond the ν edge (kNN 3.0 vs 0.8 core, 1.3 edge), not
in a void → a charge boundary learned on sparse MC misfiring on exp. Implication: SNGP only
*partially* helps (edge, not far-OOD); the fine-tune (real background in that corner) is more
targeted.

### SNGP — principled fix, implemented (§5)
SN encoder + RFF-GP head + Laplace covariance + mean-field logit. New
`src/training/sngp_nu_classifier_trainer.py` + `src/models/` pieces — DA trainer untouched.
Partial for this edge-of-nu excess (§8).

## Overall
The fine-tune fix **works**: it roughly halves the exp neutrino-like excess at physics
working points with preserved ν-efficiency, outperforming the horizon θ-loss (which fails).
The mechanism is refined from "OOD-into-void" to "charge boundary on the sparse ν-occupied
bright-horizontal edge, misfiring on exp." Paper pipeline (out-of-training suppression curve
to 1e-5 with errors + residual verticality-cut analysis) is in progress; SNGP is implemented
as a partial principled complement.
