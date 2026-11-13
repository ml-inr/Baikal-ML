"""Regenerate the two per-model analysis notebooks from one template.

**This overwrites `da__analysis.ipynb` and `control__analysis.ipynb`.**
Edit here, not in the notebooks, or the next run of this script discards the edit.

The two notebooks are the same analysis with the primary model swapped — every
figure draws the primary solid and the other dashed.  `report_lib.py` holds the
code; this file holds only the prose and the order of the cells.

Usage (from anywhere):
    python inference/prefilter/analysis/report_261111_1704/make_notebooks.py
"""
import json
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import report_lib as R                                               # noqa: E402


_cell_counter = 0


def _cell_id() -> str:
    global _cell_counter
    _cell_counter += 1
    return f"cell-{_cell_counter:02d}"


def md(src):
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {},
            "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "id": _cell_id(), "execution_count": None,
            "metadata": {}, "outputs": [], "source": src.splitlines(True)}


INTRO = """# Prefilter — {role} (primary)

Primary model of this notebook:

    trained_models/prefilter/{run}/best_da_model.pth

The companion notebook, `{companion}`, makes the other model of the pair primary.
Every figure below draws **this** model solid and the other dashed, so the
comparison is on each plot rather than only across notebooks.

## The pair

Both runs are the DA-repair prefilter retrain on hard labels. They share the
`hard_nu_labels` source dataset, a muatm-only adversarial source, `k_steps=10`,
a discriminator refresh every 5 epochs, LayerNorm + instance noise in the
discriminator, the same invariance probe, seed 42, architecture and `exp.h5`
target. They differ **only** in the adversarial strength:

| run | `lambda_scheduler` |
|---|---|
| DA | `delayed_constant`, $\\lambda=0.003$ from epoch 2 |
| control | `constant`, $\\lambda=0.0$ |

So this is domain adaptation against a matched no-DANN control. The control
still trains the discriminator (`k_steps`, refresh) and still runs the probe;
the encoder simply receives no domain gradient.

## Predictions and cuts

MC is **out-of-training**: the predictions were scored on the canonical
held-out part lists with `predict_mc.py --exclude-trained-parts`, which drops
all 952 parts the training dataset was built from (432 muatm + 187 nuatm + 333
nue2). The sample was capped at 10M muatm, 3M nuatm and all 1.29M nue2 per
model; `exp_full` is complete (133,294,745 events). The exact commands are in
`run_predictions.sh`.

Figures 1–3 use **h8s3 on MC truth** (`n_gt_sig_hits >= 8 AND
n_gt_sig_strings >= 3`). Figure 4 uses **h8s3 by the sig-noise model**
(`n_sn_hits >= 8 AND n_sn_strings >= 3` at threshold 0.8) for both `exp_full`
and muatm. Both count pairs are stored per event in the prediction databases,
copied verbatim from the probabilities file; the prefilter itself still scores
**all** raw hits, and these counts are analysis metadata, never a selection.

## Everything is read through `inference.reader`

Zenith angle, primary energy and the MC event weight are not in the catalog;
they live in `prime_prty` in `baikal_mc_merged.h5` and the reader returns them
beside each event as `prime_theta_deg`, `prime_energy_gev` and
`prime_event_weight`. `exp_full` is never materialised: its 133M scored events
are read only as aggregates through `Reader.histogram`.

## Caveats carried onto the figures

* **Weights.** `EventWeight` is applied within a class only, so the metrics stay
  defined whether or not it is comparable *between* muatm, nuatm and nue2 — that
  comparability is `assumed`, not checked. Weighted and unweighted panels are
  drawn side by side.
* **`nue2` is astrophysical nu_mu with an E^-2 spectrum**, not electron neutrinos.
* **No live-time normalisation** exists for this MC, so figure 4 is a shape
  ratio and its absolute level is not a rate.
* **The MC classes occupy disjoint zenith hemispheres** — muatm 94.2–179.9 deg,
  both neutrino classes 0.11–90.00 deg — so the separation shown is easier than
  a real sample would offer. Logged in `doc/claims_log.md`.
* **Two exp_full runs** are dropped by `reader.NOT_EXCLUDED`; the flag
  `EXCLUDE_BAD_EXP_RUNS` below turns this off.
* **The MC sample is reduced** (10M/3M/1.29M), unlike the canonical 127M
  prediction set; the figures are statistically limited accordingly.
"""


def build(primary: str, role: str, run: str, companion: str):
    other = R.OTHER[primary]
    return {
        "cells": [
            md(INTRO.format(role=role, run=run, companion=companion)),
            code('''import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd
import report_lib as R

PRIMARY = "%s"

# reader.NOT_EXCLUDED drops the two exp_full runs where channels 222/224/225/227
# emit up to 3169.8 p.e. against a median hit of 0.97 (inference/reader/spec.py).
# Set to False to keep them; the counts either way are printed below.
EXCLUDE_BAD_EXP_RUNS = True

print("primary:", R.MODELS[PRIMARY]["label"])''' % primary),
            md("""## Provenance

Which prediction database each figure rests on, and whether the reader can
resolve what that checkpoint was trained on — an unresolved checkpoint would
make `NOT_TRAINED_PARTS` raise rather than quietly return everything."""),
            code("""display(R.provenance())
display(R.da_target_is_disjoint_from_exp_full())"""),
            md("""## Loading

`load_mc` streams both models' out-of-training MC with theta, energy and
EventWeight. The figures use the **intersection** of the two models' event sets,
so the comparison cannot silently rest on different events; the table below
shows the counts and the shared size. Both models were scored on the same part
lists and caps, so the intersection should be the full set."""),
            code("""per_model = {m: R.load_mc(m) for m in R.MODELS}

shared = None
for events in per_model.values():
    ids = np.sort(events.event_fk.to_numpy())
    shared = ids if shared is None else np.intersect1d(shared, ids,
                                                       assume_unique=True)

mc = {m: events[np.isin(events.event_fk.to_numpy(), shared)]
            .sort_values("event_fk", ignore_index=True)
      for m, events in per_model.items()}
del per_model

display(pd.DataFrame({
    R.MODELS[m]["label"]: events.groupby("data_class", observed=True).size()
    for m, events in mc.items()
}).assign(shared=len(shared)))
print(f"MC events used (out-of-training, h8s3 truth): {len(shared):,}")"""),
            md("""## exp_full and muatm histograms

`exp_histograms` bins `exp_full` inside DuckDB; `muatm_signoise_histogram`
does the same for the muatm h8s3 sig-noise subset. Neither materialises the
133M scored experimental events."""),
            code("""exp_counts = R.exp_histograms(R.SCORE_EDGES,
                                  exclude_bad_runs=EXCLUDE_BAD_EXP_RUNS)
muatm_sig = {m: R.muatm_signoise_histogram(m) for m in R.MODELS}

display(pd.DataFrame({
    "exp_full h8s3 (sig-noise)": {R.MODELS[m]["label"]: int(h.sum())
                                   for m, h in exp_counts.items()},
    "muatm h8s3 (sig-noise)": {R.MODELS[m]["label"]: int(h.sum())
                                for m, h in muatm_sig.items()},
}))"""),
            md("""## 1. Score distributions

Each group is normalised to its own total, so the panels compare shapes.
`h8s3` on MC truth."""),
            code("R.plot_score_hists(mc, R.SCORE_EDGES, PRIMARY)"),
            md("""## 2. Neutrino efficiency and EAS suppression

`suppression(t)` is the fraction of muatm passing at threshold `t` — lower is
better; `efficiency(t)` is the fraction of one neutrino class passing. The
threshold grid is the muatm score quantiles at log-spaced pass fractions, so
the curves are resolved deep in the background tail. The dotted vertical line
is the primary model's working point, suppression = $10^{-3}$."""),
            code("""R.plot_threshold_and_roc(mc, PRIMARY)
display(R.working_point_table(mc))"""),
            md("""## 3. Efficiency at the working point, suppression = $10^{-3}$

Each model is evaluated at **its own** threshold that puts muatm at a pass
fraction of $10^{-3}$; comparing at one fixed score would confound a shift of
the score scale with a real change in selection power. The MC classes occupy
disjoint zenith hemispheres (muatm 94.2–179.9 deg; neutrinos 0.11–90.00 deg),
which makes this separation easier than a real sample."""),
            code("R.plot_efficiency_at_wp(mc, PRIMARY)"),
            md("""## 4. Experimental excess over atmospheric-muon MC

Left: the ratio per score bin. Right: the ratio of the fractions above a
threshold. Both normalise `exp_full` and `muatm` to their own totals. No
live-time normalisation exists for this MC, so the **shape** of the ratio
carries information and its absolute level does not: a value of 3 does not
mean three times too many events, it means the score distribution puts three
times the relative weight there."""),
            code("R.plot_exp_excess(exp_counts, muatm_sig, R.SCORE_EDGES, PRIMARY)"),
            md("""## 5. Excess vs muatm fraction

The same experimental/muatm comparison with the threshold replaced by the
**fraction of muatm above it** — the quantity a selection cut actually fixes.
The left panel is the four score survival curves; the right panel is the
`exp_full / muatm` ratio against the muatm fraction, with a 95% Wilson interval
on x and a delta-method ±1σ on y, both axes logarithmic. This mirrors the
nu-classifier's `template_score_survival.ipynb`; the populations are the h8s3
sig-noise counts, the same as figure 4. The dotted vertical line is
suppression = $10^{-3}$. The ratio is a shape ratio: no live-time
normalisation exists, so its absolute level is not a rate."""),
            code("""SURVIVAL_EDGES = np.linspace(0.0, 1.0, 1001)
survival_hists = {m: R.survival_histograms(m, SURVIVAL_EDGES)
                  for m in R.MODELS}
R.plot_excess_vs_muatm_fraction(survival_hists, SURVIVAL_EDGES, PRIMARY)"""),
            md("""## What this pair does and does not establish

Supported: a shape comparison on MC and on experiment, with the two models
compared at matched working points rather than at a matched score, and the
out-of-training guarantee at the part level.

Not supported:

* absolute event rates — no live-time normalisation for this MC exists here
  (`assumed`);
* a single combined neutrino efficiency — nuatm and nue2 are kept apart on
  purpose;
* a claim that EventWeight is comparable between classes — it is used within a
  class only;
* a per-event out-of-training guarantee finer than the part — the prefilter NPY
  dataset records no `h5_part_keys.npy`;
* full-statistics MC — this prediction set is the reduced 10M/3M/1.29M sample.

Figures are written to `figures/` beside this notebook and are regenerated by
running this notebook end to end. `run_predictions.sh` is the command that
produced the prediction databases they rest on."""),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


RUN_DA = R.MODELS["da"]["checkpoint"]
RUN_CONTROL = R.MODELS["control"]["checkpoint"]
NB_DA, NB_CONTROL = "da__analysis.ipynb", "control__analysis.ipynb"

for primary, role, run, companion, fname in [
    ("da", "DA, $\\lambda$=0.003", RUN_DA, NB_CONTROL, NB_DA),
    ("control", "no-DANN control, $\\lambda$=0.0", RUN_CONTROL,
     NB_DA, NB_CONTROL),
]:
    (OUT / fname).write_text(json.dumps(build(primary, role, run, companion),
                                        indent=1))
    print("wrote", OUT / fname)
