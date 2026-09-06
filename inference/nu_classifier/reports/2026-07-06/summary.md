# Summary — why does the exp nu-classifier show a neutrino-like excess?

Executive Q→A of the 2026-07-06 investigation. Full detail + all numbers in
[REPORT.md](REPORT.md); scripts regenerate every artifact (`tables/`, `figures/`).

**The problem.** On real experimental data the nu-classifier flags ~1.5×10⁻³ of
events as neutrino-like (score>0.8) — ~100–1000× the physical atmospheric-neutrino
rate (~10⁻⁶–10⁻⁵). >99.4% of these are false. Why?

---

### Q1. Do the three trained models (E1 baseline, E2 spectral-norm, E3b afterpulse) close the gap?
**No.** At a fixed muon-suppression working point (matched epochs, not `best_da_model`),
the exp false-ν fraction is equal across all three (~0.25%). Neither spectral norm nor
afterpulse moved it; the Round-1 "E2 wins" was an artifact of different epochs + score
compression. → REPORT §1; `comparison_table_matched.csv`, `score_hist_per_model_matched.png`.

### Q2. Is it afterpulses (the prior hypothesis)?
**No.** The afterpulse augmentation is mis-specified (injects Q ~10× too bright while
exp is *dimmer*; random OM + in-time, not same-channel + delayed). And the excess is
present from epoch 1 and flat across topology cuts — not an afterpulse/late-training
artifact. → REPORT §3; `tables/feature_shift_mc_vs_exp.csv`, `excess_vs_topology_cut.csv`.

### Q3. Is it a "horizon" (near-90° zenith) up/down confusion?
**Partly — it's where the false positives concentrate, but only ~1/3 of the excess.**
In MC the label ≈ zenith (μ down θ>90, ν up θ<90); the net learned a θ≈90 boundary.
muatm false-ν rate peaks at **7.6% at θ95–98°**, ~0 by 130° (σ=10.5° from a fit).
Confirmed on exp via a z-spread verticality proxy (corr(rvert,θ)=0.83): exp false-ν
are horizontal. But A/B decomposition: exp is only ~1.3× more horizontal (A), while a
per-angle gap of ~3–4× (B) dominates. → REPORT §4, §4b; `mc_muatm_fp_vs_zenith*.csv`,
`ab_decomposition.csv`, `zspread_proxy.csv`.

### Q4. So what is the dominant (B) gap? Is it OOD?
**Yes — over-confident OOD extrapolation, seen only in the model's features.**
In the 128-dim encoder embedding, exp false-ν are a cleanly separated population
(kNN to MC ×4, Mahalanobis distribution ~450 vs ~40) while typical exp overlaps MC
(bulk aligned; DA worked for ~99%). *NB: UMAP is not evidence — it hides the OOD
direction; the distance histogram is.* → REPORT §4c; `composite_ood.csv`,
`composite_ood_mahalanobis.png`.

### Q5. Can a simple OOD cut on score>0.8 remove the excess and leave a few real ν?
**A hand-picked physical-feature OOD cut FAILS; an embedding-space cut works but not for
free.** Physical-observable Mahalanobis: exp false-ν look in-distribution (cut removes
~nothing). Embedding kNN cut: reduces the excess **~60×** (1747→29 survivors, toward
the physical ceiling) but costs ~31% real-ν efficiency, and leaves ~29 (not 1–2)
because false-ν overlap the real-ν tail. → REPORT §4d; `ood_cut.csv`,
`ood_cut_embed.csv`, `ood_cut*.png`.

### Q6. Is 128-dim over-provisioned — would reducing d_model (32/16) help?
**Over-provisioned: yes (effective dim ~2). Reducing it: no — it backfires.**
90% of MC variance is in 2 PCs, 99% in 26; exp OOD lives 98.9% in the *unused* tail
PCs. But tail-ablation shows the **score is driven only by the top ~8 PCs** (unchanged
when the tail is zeroed). So the excess comes from exp being **neutrino-like in the
decision features**, while the big tail difference is an OOD marker the head **ignores**.
Cutting dims deletes that marker and keeps the excess. → REPORT §4e;
`embedding_dimensionality.csv`, `ablate_tail.py`.

---

## Overall answer
The exp neutrino-like excess is **over-confident OOD extrapolation**: a narrow tail of
exp events that (i) are neutrino-like in the ~8 decision-relevant embedding directions
(largely the horizon/direction ambiguity) and (ii) carry a strong out-of-distribution
signal in directions the classifier learned to ignore. It is **not** afterpulses, not
a single physical observable, not curable by early stopping, working-point choice,
topology cuts, or reducing dimensionality.

## What follows
- **Built & training:** horizon soft-label loss (E5, σ=10.5° from data) — a mitigation
  targeting where the false positives concentrate (the horizon). → `horizon_loss_design.md`.
- **Principal fix: distance-aware output (SNGP)** — make the confidence depend on the
  full-space distance so the ignored OOD marker down-weights exactly these events.
  A post-hoc kNN veto already cuts the excess ~60×; SNGP does it jointly, in training.
- Floor: ~29 exp events are irreducibly neutrino-like in the decision features — OOD
  handling alone won't reach the physical ~1–12.
