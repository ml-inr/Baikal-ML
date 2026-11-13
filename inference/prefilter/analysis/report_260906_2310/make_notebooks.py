"""Regenerate the two comparison notebooks from one template.

**This overwrites `compare_lambda0.05_da.ipynb` and `compare_lambda0.0_control.ipynb`.**
Edit here, not in the notebooks, or the next run of this script discards the edit.

The two notebooks are the same analysis with the primary and reference model swapped —
every figure draws the primary as a line and the other as a halo — so they are generated from one
source rather than kept in sync by hand. `report_lib.py` holds the code they call; this
file holds only their prose and the order of the cells.

Usage (from anywhere):
    python inference/prefilter/analysis/report_260906_2310/make_notebooks.py
"""
import json, pathlib

OUT = pathlib.Path(__file__).resolve().parent

def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": src.splitlines(True)}

INTRO = """# Prefilter — {lam} ({role})

Primary model of this notebook:

    trained_models/prefilter/{run}/best_da_model.pth

The companion notebook, `{companion}`, makes the other model of the pair primary.
The two training runs differ **only** in `domain_adaptation.lambda_scheduler.max_lambda`
(0.05 against 0.0): a diff of the two `da_config.yaml` files shows that key, the run name
and the training GPU, and nothing else — same seed 42, same architecture, same source NPY
dataset, same exp target, same losses and schedules. So this is domain adaptation against
a matched control.

Every figure below draws **this** model as a solid line and the other as a wide
translucent halo underneath it, so the comparison is on each plot rather than only across
notebooks. Where the two agree the halo hugs the line; where they part it peels away. A
dashed reference was tried first and does not work here — on the working-point figure the
models agree to a few tenths of a percent, and the dashes disappear under the solid line
so completely that the figure looks as though the reference had not been drawn at all.

## Everything is read through `inference.reader`

Zenith angle, primary energy and the MC event weight are **not** in the catalog, which
holds only `id, source, data_class, season, cluster, run, event_id, feature_hash`. They
live in `prime_prty` in `baikal_mc_merged.h5` and the reader returns them beside each
event as `prime_theta_deg`, `prime_energy_gev` and `prime_event_weight`
(layout per `doc/hdf5_format.md` §5).

`exp_full` is never materialised. Its 133M scored events are read only as aggregates,
through `Reader.histogram`, which bins them inside DuckDB.

## What counts as out-of-training

**MC** — `reader.NOT_TRAINED_PARTS` drops every event of every HDF5 part the training NPY
dataset was built from (432 muatm + 187 nuatm + 333 nue2 = 952 parts, read from the
builder's own `prefilter_h5_parts.json`). This is coarser than the per-event exclusion the
nu-classifier gets, and it is the finest one available: `prefilter_npy_ds_builder` writes
no `h5_part_keys.npy`, so nothing records which events *inside* those parts the builder
kept. A part the builder never opened cannot have contributed an event, so what survives
is out-of-training by construction (`verified`); it also drops events that were available
and went unused, which costs statistics and never costs correctness.

**exp_full** — the domain-adaptation target of both runs was `exp.h5`. `NOT_DA_TARGET`
cannot be used, for the same missing-back-links reason, so the claim is established
directly instead: the two files' run keys are compared below and must be disjoint.

## Two warnings carried onto the plots

**Weights.** `prime_prty[:, 5]` is `EventWeight`, and muatm, nuatm and nue2 are generated
with different spectra, so unweighted efficiency and suppression are properties of the
sample and not physical rates. Both versions are drawn side by side and each panel says
which it is. Weighting is applied **within** a class, so it stays well defined whether or
not EventWeight is comparable *between* classes — that comparability has not been checked
here and is not claimed (`assumed`).

**`nue2` is not electron neutrinos.** It is astrophysical nu_mu with an E^-2 spectrum.
Older comments in this repository get this wrong; the legends here name it honestly.
"""

def build(primary, lam, role, run, companion):
    return {
        "cells": [
            md(INTRO.format(lam=lam, role=role, run=run, companion=companion)),
            code(f'''import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd
import report_lib as R

PRIMARY = "{primary}"

# reader.NOT_EXCLUDED drops the two exp_full runs where channels 222/224/225/227 emit
# up to 3169.8 p.e. against a median hit of 0.97 (evidence in inference/reader/spec.py).
# Set to False to keep them; the counts either way are printed below.
EXCLUDE_BAD_EXP_RUNS = True

print("primary:", R.MODELS[PRIMARY]["label"])'''),
            md("""## Provenance

Which prediction database each figure rests on, and whether the reader can resolve what
that checkpoint was trained on — an unresolved checkpoint would make `NOT_TRAINED_PARTS`
raise rather than quietly return everything."""),
            code("""display(R.provenance())
display(R.da_target_is_disjoint_from_exp_full())"""),
            md("""## Loading

`load_mc` streams both models' out-of-training MC events with their truth columns;
`combine_mc` intersects them on `event_fk` and reports the agreement per class. The
figures use the intersection only, so the two models cannot be compared on different
event sets without the table below saying so.

`identical sets` is expected to be False for muatm and nuatm, and the reason is
bookkeeping rather than drift: the DA model's mc_merged database was filled earlier under
a 10M-events-per-particle-type cap, which stops **inside** a part, while `run_scoring.sh`
scored the control over the same part list without a cap and so covers those parts whole.
The extra events belong to the control alone and the intersection drops them. What would
matter is the reverse — the DA model holding events the control does not — and that shows
in the same table."""),
            code("""per_model = {m: R.load_mc(m) for m in R.MODELS}
mc, agreement = R.combine_mc(per_model)
del per_model

print(f"MC events used (out-of-training, scored by both): {len(mc):,}")
display(agreement)"""),
            code("""SCORE_EDGES = np.linspace(0.0, 1.0, 101)
exp_counts, exp_report = R.exp_histograms(SCORE_EDGES,
                                          exclude_bad_runs=EXCLUDE_BAD_EXP_RUNS)
display(exp_report)"""),
            md("""## 1. Score distributions

Each group is normalised to its own total, so the panels compare shapes. `exp_full`
carries no MC weight and is therefore the same curve in both panels."""),
            code("""R.plot_score_hists(mc, exp_counts, SCORE_EDGES, PRIMARY);"""),
            md("""## 2. Neutrino efficiency and EAS suppression

`suppression(t)` is the fraction of muatm passing at threshold `t` — lower is better —
and `efficiency(t)` is the fraction of one neutrino class passing. The threshold grid is
the muatm score quantiles at log-spaced pass fractions, so the curves are resolved deep in
the background tail rather than uniformly across a score range that is nearly empty there.

The two classes are kept as separate curves. Combining them into one "neutrino
efficiency" would need an external flux model to weight nuatm against nue2, and that
choice would then be hidden inside a single line.

The dotted vertical line is the working point used in section 3."""),
            code("""R.plot_threshold_and_roc(mc, PRIMARY);

rows = []
for weighted in (False, True):
    for model in R.MODELS:
        rows.append({"model": R.MODELS[model]["label"], "weighted": weighted,
                     **R.working_point(mc, f"score_{model}", weighted)})
display(pd.DataFrame(rows))

# Is that gap real? The models score the same events, so the comparison is paired and
# what settles it are the events they disagree about. See report_lib.paired_difference.
display(pd.concat([R.paired_difference(mc, weighted) for weighted in (False, True)],
                  ignore_index=True))"""),
            md("""## 3. Efficiency at the working point, suppression = 1e-3

Each model is evaluated at **its own** threshold that puts muatm at a pass fraction of
1e-3. Comparing the two at one fixed score instead would confound a shift of the score
scale with a real change in selection power.

Only the neutrino classes appear here, and the axis is binned over the range they occupy.
The MC productions place the classes in **disjoint zenith hemispheres**: measured on the
out-of-training parts, muatm spans 94.2-179.9 deg while both neutrino classes span
0.11-90.00 deg, the cap sitting at exactly 90.00 — a generation cut, not a physical edge.
Worth keeping in mind when reading the suppression numbers: in this MC no atmospheric muon
and no neutrino share a zenith angle, so the two populations are further apart in truth
than any real sample would be.

No error band is drawn. The binomial spread on effective counts,
`n_eff = (sum w)^2 / sum w^2`, is printed below instead: with hundreds of thousands of
events per bin it is thinner than the line, and a band nothing can be seen of is worse
than a number."""),
            code("""R.plot_efficiency_at_wp(mc, PRIMARY);

# The error the figure does not draw, at its largest over all bins and both classes.
worst = []
for weighted in (False, True):
    point = R.working_point(mc, f"score_{PRIMARY}", weighted)
    for data_class in R.NU_CLASSES:
        subset = mc[mc.data_class == data_class]
        for variable, edges in (("theta_deg", np.linspace(0, 90, 19)),
                                ("energy_gev", np.logspace(1, 7, 25))):
            profile = R.efficiency_vs(subset, f"score_{PRIMARY}", point["threshold"],
                                      variable, edges, weighted)
            worst.append({"weighted": weighted, "data_class": data_class,
                          "variable": variable,
                          "max binomial error": profile["err"].max(),
                          "min events in a bin": int(profile["n_raw"].min())})
display(pd.DataFrame(worst))"""),
            md("""## 4. Experimental excess over atmospheric-muon MC

Left: the ratio per score bin. Right: the ratio of the fractions above a threshold.

Both normalise `exp_full` and `muatm` to their own totals. No live-time normalisation
exists for this MC anywhere in the repository, so the **shape** of the ratio carries
information and its absolute level does not: a value of 3 does not mean three times too
many events, it means the score distribution puts three times the relative weight there."""),
            code("""R.plot_exp_excess(mc, exp_counts, SCORE_EDGES, PRIMARY);"""),
            md("""## What this pair does and does not establish

Supported: a shape comparison on MC and on experiment, with the two models compared at
matched working points rather than at a matched score.

Not supported:

* absolute event rates — no live-time normalisation for this MC exists here (`assumed`);
* a single combined neutrino efficiency — nuatm and nue2 are kept apart on purpose;
* any claim that EventWeight is comparable between classes — it is used within a class only;
* a per-event out-of-training guarantee finer than the part — see the note at the top.

Figures are written to `figures/` beside this notebook and are regenerated by running this
notebook end to end. `run_scoring.sh` in the same directory is the command that produced
the prediction databases they rest on."""),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }

RUN_DA = ("da_prefilter_numu_260429_0155_Qclip_softlabels_PlateauLR_lambda0.05"
          "_MC2M_dmodel128_zflipBothMCExp_SoftFocalLoss")
RUN_NODA = ("da_prefilter_numu_260428_2039_Qclip_softlabels_PlateauLR_lambda0.0"
            "_MC2M_dmodel128_zflipBothMCExp_SoftFocalLoss")
NB_DA, NB_NODA = "compare_lambda0.05_da.ipynb", "compare_lambda0.0_control.ipynb"

for primary, lam, role, run, companion, fname in [
    ("da", "max_lambda = 0.05", "domain adaptation", RUN_DA, NB_NODA, NB_DA),
    ("noda", "max_lambda = 0.0", "no-DA control", RUN_NODA, NB_DA, NB_NODA),
]:
    (OUT / fname).write_text(json.dumps(build(primary, lam, role, run, companion),
                                        indent=1))
    print("wrote", OUT / fname)
