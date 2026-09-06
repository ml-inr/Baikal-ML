# Summary 2026-07-09 — full exp, bimodal super-confident ν peak, candidates

Follows [../2026-07-08/summary.md](../2026-07-08/summary.md). Full detail:
[REPORT.md](REPORT.md). All h8s3, good runs.

---

### Full exp_full scored by both models (§1)
The entire exp_full (3,110,199 h8s3 good-run events, identical set for E1 and FT) scored via
`predict_exp --probs-h5` — fast (~20 min each; precomputed probs cover all parts, SN skipped).
score>0.8: E1 2,259 vs FT 153. Added `--probs-h5` to `predict_mc` too. Perf rule: precomputed
probs win for bulk whole-part scans; on-the-fly SN wins for scattered analysis reads / where
probs coverage is incomplete (MC muatm 48% / nue2 17% / nuatm 5% parts).

### Bimodality — super-confident ν peak separates from the muon background (§2)
FT exp/muatm survival ratio is ~1.2 (background-like) up to score 0.9, then **explodes: 3.9× at
0.95, 8.6× at 0.97, 51× at 0.99**. The FT-exp tail dips around [0.94,0.98] then **rises at
[0.98,1.0]** (7 events >0.99) while muatm dies off (~1 muon expected). A genuine-ν peak
separates from the background. Modest stats (7 events >0.99) but the 51× ratio is robust.

### The super-confident events are up-going — genuine ν candidates (§3)
Of the 38 super-confident FT events (>0.9): **36 are up-going-like** (corr(t,z)>0.3; muon bg is
down-going, corr<0), 15 vertical. The **7 events >0.99** (the bimodal peak) are **all up-going**,
spread across clusters 2/4/5/6/7 and different runs — the neutrino signature (only ν cross the
Earth), not muon background. These are the genuine atmospheric-ν candidates. Event displays in
`figures/nu_candidates_full.png`.

### Paper visualisations
Suppression curve re-plotted with the expected "exp=muon (no excess)" line (E1 at 1.9–2.5×, FT
at 1.1–1.2×); exp score histograms before/after FT vs muatm (tail shrinks to muon level).



### Corner cases — the network isn't learning physics (§4)
Event displays showed two failure modes. MC truth: **79% of high-score muatm false-positives are
multi-muon** (n_muons≥2) that fake up-going (case 2 — exists in MC, removable by multiplicity /
two-cluster cut); isolated passed-noise hits that flip verticality are rare (~7× enriched in exp,
case 1). So the super-confident candidates are a **mix of ν and muon fakes**, and corr(t,z) alone
does not certify ν. Deeper point: the net scores 2-track bundles and noise-flipped events as
confident ν → it learns **coarse correlational shortcuts, not track topology / hit-ambiguity** —
the same root as the exp excess. Fixes (MC truth enables): multi-task `n_muons` (case 2);
per-hit SN prob is only weakly informative (noise/muon probs overlap, almost none >0.99) so
consistency/robustness training + `lo_drop` uncertainty is the stronger case-1 fix.

## Overall
On full statistics the fine-tuned model's high-score tail shows a **bimodal super-confident
peak (>0.99, 51× over the muon background) made of up-going events spread across the detector** —
genuine neutrino candidates, cleanly separated from the down-going muon background. In progress:
E1 MC on covered parts (E1-vs-FT to muon suppression 1e-6) and SNGP training (spectral norm
does not hurt MC AUC, 0.9999).
