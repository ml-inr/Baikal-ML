# Prefilter predictions as they stood before the hit-count columns (2026-09-07)

Moved out of `inference/prefilter/preds/` when the prediction schema gained
`n_sn_hits`, `n_sn_strings`, `n_gt_sig_hits`, `n_gt_sig_strings`. These databases carry
the old three-column schema `(event_fk, score, n_hits)` and cannot take the new columns
without being rebuilt, which is what the rescore did.

They are the exact inputs of `inference/prefilter/analysis/report_260906_2310`. That
report's notebooks are committed **with their outputs**, so its numbers are preserved
regardless of these files; they are kept only so it can be re-executed unchanged.

Scores are unaffected by the schema change. What differs in the new databases is the four
columns and the part selection: the rescore covers only parts the sig-noise probabilities
file covers, and more of muatm and nuatm.
