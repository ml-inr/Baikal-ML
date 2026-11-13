"""Loading, metrics and figures for report_261111_1704, on top of `inference.reader`.

Model pair
----------
The two runs are the DA-repair prefilter retrain and its no-DANN control.  They
share hard labels (`hard_nu_labels`), a muatm-only adversarial source, k_steps=10,
a discriminator refresh every 5 epochs, LayerNorm + instance noise and the same
invariance probe; seed, architecture, source dataset and `exp.h5` target are the
same.  The only difference is the adversarial strength:

    DA       lambda_scheduler: delayed_constant, lambda=0.003 from epoch 2
    control  lambda_scheduler: constant, lambda=0.0

so this is domain adaptation against a matched no-DANN control.

Predictions
-----------
MC: held-out canonical parts only, with caps 10M muatm / 3M nuatm / all 1.29M
nue2.  Training parts are removed at scoring time by
`inference/prefilter/predict_mc.py --exclude-trained-parts` (952 parts: 432
muatm + 187 nuatm + 333 nue2).  exp_full: all 133,294,745 events.

Everything goes through `inference.reader`: zenith angle, primary energy and the
MC event weight live in `prime_prty` in `baikal_mc_merged.h5`, not in the
catalog, and `exp_full` is read only through `Reader.histogram`, never
materialised.

Conventions
-----------
suppression(t) = fraction of muatm_2020 with score >= t   (lower is better)
efficiency(t)  = fraction of one neutrino class with score >= t

Both are computed **within** a class, so they stay well defined whether or not
EventWeight is comparable *between* classes — which has not been checked here
and is not claimed.

nue2 is astrophysical nu_mu with an E^-2 spectrum, not electron neutrinos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py                                                       # noqa: E402
from inference import reader                                      # noqa: E402
from inference.reader import spec as reader_spec                  # noqa: E402

FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

MODELS = {
    "da": {
        "checkpoint": ("da_prefilter_numu_261108_1943_Qclip_hardlabels_"
                       "FocalLoss_muOnlyDA_k10_refresh5_probe"),
        "label": "DA, $\\lambda$=0.003 (delayed)",
    },
    "control": {
        "checkpoint": ("da_prefilter_numu_261109_1845_Qclip_hardlabels_"
                       "FocalLoss_lambda0_control_k10_refresh5_probe"),
        "label": "control, $\\lambda$=0.0",
    },
}
OTHER = {"da": "control", "control": "da"}

MC_CLASSES = ["muatm_2020", "nuatm_2020", "nue2_2020"]
NU_CLASSES = ["nuatm_2020", "nue2_2020"]

PRETTY = {
    "muatm_2020": "muatm — atmospheric muons (EAS)",
    "nuatm_2020": r"nuatm — atmospheric $\nu$",
    "nue2_2020": r"nue2 — astrophysical $\nu_\mu$, $E^{-2}$",
    "exp_full": "exp_full — experiment",
}
GROUP_COLOR = {"muatm_2020": "#c44e52", "nuatm_2020": "#4c72b0",
               "nue2_2020": "#55a868"}
GROUP_LABEL = {"muatm_2020": "muatm (EAS)", "nuatm_2020": r"nuatm ($\nu$ atm)",
               "nue2_2020": r"nue2 ($\nu_\mu$, $E^{-2}$)"}
MODEL_COLOR = {"da": "#4c72b0", "control": "#dd8452"}
SURVIVAL_COLOR = {**GROUP_COLOR, "exp_full": "#8172b2"}

TARGET_SUPPRESSION = 1e-3
SCORE_EDGES = np.linspace(0.0, 1.0, 101)
THETA_EDGES = np.linspace(0.0, 90.0, 19)
ENERGY_EDGES = np.logspace(1, 7, 25)


def model_line(primary: str) -> dict:
    """Primary solid, the other model dashed; the comparison is on every figure."""
    return {primary: dict(ls="-", lw=1.9),
            OTHER[primary]: dict(ls="--", lw=1.5, alpha=0.85)}


# ---------------------------------------------------------------------------
# provenance the notebooks print before anything else
# ---------------------------------------------------------------------------

def provenance() -> pd.DataFrame:
    """Which database, which checkpoint, and whether its training resolves."""
    rows = []
    for key, model in MODELS.items():
        checkpoint = reader.open_checkpoint(model["checkpoint"])
        for source in sorted(checkpoint.databases):
            rows.append({
                "model": model["label"],
                "checkpoint": checkpoint.name,
                "source": source,
                "database": str(checkpoint.database(source)
                                .relative_to(PROJECT_ROOT)),
                "training_resolves": checkpoint.config is not None,
            })
    return pd.DataFrame(rows)


def da_target_is_disjoint_from_exp_full() -> pd.DataFrame:
    """Check that no exp_full run was used as the domain-adaptation target.

    The target domain of both trainings was `exp.h5`.  `NOT_DA_TARGET` cannot
    be used here — the prefilter dataset records no `exp_h5_part_keys.npy` —
    so the claim is established directly: the two files' part keys (season +
    cluster + run) must have an empty intersection.
    """
    keys = {}
    for name in ("exp", "exp_full"):
        source = reader_spec.SOURCES[name]
        with h5py.File(source.h5, "r") as handle:
            keys[name] = set(handle[source.groups[0]]["raw"]["data"].keys())
    shared = sorted(keys["exp"] & keys["exp_full"])
    return pd.DataFrame([{
        "exp.h5 runs (DA target file)": len(keys["exp"]),
        "exp_full.h5 runs": len(keys["exp_full"]),
        "shared runs": len(shared),
        "verdict": ("disjoint — exp_full is out-of-training" if not shared
                    else f"OVERLAP: {shared[:5]}"),
    }])


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

#: Columns kept from the reader.  The rest — source, cluster, run, part_key,
#: local_idx, n_hits, phi, nucleon_n, response_muons_n — are used by no figure
#: here, and over 14M events the three string ones alone cost several GB.
_KEEP = ["event_fk", "data_class", "score", "prime_theta_deg",
         "prime_energy_gev", "prime_event_weight"]
_FLOATS = ("score", "prime_theta_deg", "prime_energy_gev", "prime_event_weight")


def _trim(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[_KEEP].copy()
    out["data_class"] = out["data_class"].astype("category")
    for column in _FLOATS:
        out[column] = out[column].astype(np.float32)
    return out.rename(columns={"prime_theta_deg": "theta_deg",
                               "prime_energy_gev": "energy_gev",
                               "prime_event_weight": "weight"})


def load_mc(model: str, cuts=None, budget_mb: int = 512) -> pd.DataFrame:
    """Out-of-training MC scored by one model, with theta, energy and EventWeight.

    `NOT_TRAINED_PARTS` drops every event of every h5 part the training NPY
    dataset was built from.  That is the finest handle available here: the
    prefilter builder writes no `h5_part_keys.npy`, so nothing records which
    events *inside* those parts were kept.  The new predictions were already
    scored with the same exclusion; applying it again is a no-op and keeps the
    query honest.
    """
    if cuts is None:
        cuts = [reader.H8S3_TRUTH, reader.NOT_TRAINED_PARTS]
    handle = reader.open("mc_merged",
                         checkpoint=MODELS[model]["checkpoint"])
    frames = []
    for chunk in handle.stream(where=cuts, with_hits=False,
                               budget_mb=budget_mb):
        frames.append(_trim(chunk.events))
    return pd.concat(frames, ignore_index=True)


def score_histogram(source: str, model: str, cuts, edges=SCORE_EDGES):
    """Counts per score bin, binned inside DuckDB (no event frame materialised)."""
    handle = reader.open(source, checkpoint=MODELS[model]["checkpoint"])
    binned = handle.histogram("score", bins=len(edges) - 1,
                              span=(edges[0], edges[-1]), where=cuts)
    return binned["n"].to_numpy(float)


def muatm_signoise_histogram(model: str, edges=SCORE_EDGES):
    """muatm with h8s3 by the sig-noise model — the denominator of figure 4."""
    return score_histogram(
        "mc_merged", model,
        [reader.H8S3, reader.NOT_TRAINED_PARTS,
         "e.data_class = 'muatm_2020'"], edges)


def exp_histograms(edges=SCORE_EDGES, exclude_bad_runs: bool = True):
    """exp_full score histograms per model, h8s3 by the sig-noise model."""
    cuts = [reader.H8S3] + ([reader.NOT_EXCLUDED] if exclude_bad_runs else [])
    return {model: score_histogram("exp_full", model, cuts, edges)
            for model in MODELS}


def survival_histograms(model: str, edges=SCORE_EDGES):
    """h8s3 sig-noise score counts for exp_full and the three MC classes.

    Same cuts as the nu-classifier survival template: the sig-noise h8s3 cut on
    every population; no hits and no fits are read.
    """
    out = {"exp_full": score_histogram(
        "exp_full", model, [reader.H8S3, reader.NOT_EXCLUDED], edges)}
    for data_class in MC_CLASSES:
        out[data_class] = score_histogram(
            "mc_merged", model,
            [reader.H8S3, reader.NOT_TRAINED_PARTS,
             f"e.data_class = '{data_class}'"], edges)
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def event_weights(subset: pd.DataFrame, weighted: bool) -> np.ndarray:
    return subset.weight.to_numpy(float) if weighted else np.ones(len(subset))


def rank_by_score(events: pd.DataFrame, weighted: bool) -> dict:
    """Per class: scores descending, with the running weight fraction beside them.

    Sorting is the only expensive step here, so it happens once per class and
    every rate below is a searchsorted into the result.
    """
    ranked = {}
    for data_class in events.data_class.unique():
        subset = events[events.data_class == data_class]
        scores = subset.score.to_numpy(float)
        weights = event_weights(subset, weighted)
        order = np.argsort(scores)[::-1]
        ranked[data_class] = (scores[order],
                              np.cumsum(weights[order]) / weights.sum())
    return ranked


def fraction_passing(descending: np.ndarray, cumulative: np.ndarray,
                     thresholds) -> np.ndarray:
    n_at_or_above = len(descending) - np.searchsorted(
        descending[::-1], thresholds, side="left")
    out = np.zeros(np.shape(thresholds))
    found = n_at_or_above > 0
    out[found] = cumulative[n_at_or_above[found] - 1]
    return out


def thresholds_at_suppression(descending: np.ndarray, cumulative: np.ndarray,
                              targets) -> np.ndarray:
    """Loosest cut meeting each target.

    Only the last score of a tie block can be a cut, since no threshold splits
    events that share a score; reading the running cumulative alone would name
    a threshold that overshoots by the weight of the tie block it landed in.
    """
    ends_a_tie = np.r_[descending[1:] != descending[:-1], True]
    candidate_score = descending[ends_a_tie]
    candidate_fraction = cumulative[ends_a_tie]
    index = np.searchsorted(candidate_fraction, targets, side="right") - 1
    return np.where(index >= 0, candidate_score[np.maximum(index, 0)],
                    np.nextafter(descending[0], np.inf))


def suppression_and_efficiency(ranked: dict, n_points: int = 400) -> pd.DataFrame:
    thresholds = np.unique(thresholds_at_suppression(
        *ranked["muatm_2020"], np.logspace(0, -6, n_points)))
    curves = {"threshold": thresholds,
              "suppression": fraction_passing(*ranked["muatm_2020"],
                                              thresholds)}
    for data_class in NU_CLASSES:
        curves[data_class] = fraction_passing(*ranked[data_class], thresholds)
    return pd.DataFrame(curves)


def working_point(ranked: dict,
                  target: float = TARGET_SUPPRESSION) -> dict:
    threshold = float(thresholds_at_suppression(
        *ranked["muatm_2020"], np.array([target]))[0])
    row = {"threshold": threshold,
           "suppression": fraction_passing(*ranked["muatm_2020"],
                                           np.array([threshold]))[0]}
    for data_class in NU_CLASSES:
        row[data_class] = fraction_passing(*ranked[data_class],
                                           np.array([threshold]))[0]
    return row


def working_point_table(mc: dict) -> pd.DataFrame:
    rows = []
    for weighted in (False, True):
        for model in mc:
            row = working_point(rank_by_score(mc[model], weighted))
            rows.append({"model": MODELS[model]["label"], "weighted": weighted,
                         **row})
    return pd.DataFrame(rows)


def efficiency_profile(events: pd.DataFrame, data_class: str, threshold: float,
                       variable: str, edges, weighted: bool) -> np.ndarray:
    subset = events[events.data_class == data_class]
    weights = event_weights(subset, weighted)
    which = np.digitize(subset[variable].to_numpy(float), edges) - 1
    inside = (which >= 0) & (which < len(edges) - 1)
    passed = inside & (subset.score.to_numpy(float) >= threshold)
    n_bins = len(edges) - 1
    total = np.bincount(which[inside], weights=weights[inside], minlength=n_bins)
    chosen = np.bincount(which[passed], weights=weights[passed], minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, chosen / total, np.nan)


def wilson_interval(k, n, z: float = 1.96) -> tuple:
    """95% Wilson interval for a binomial fraction k/n."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def shape_ratio(numerator_counts: np.ndarray,
                denominator_counts: np.ndarray) -> tuple:
    numerator = numerator_counts / numerator_counts.sum()
    denominator = denominator_counts / denominator_counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        differential = np.where(denominator > 0, numerator / denominator, np.nan)
        cumulative = np.where(
            np.cumsum(denominator[::-1])[::-1] > 0,
            np.cumsum(numerator[::-1])[::-1]
            / np.cumsum(denominator[::-1])[::-1], np.nan)
    return differential, cumulative


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def save(figure, name: str, primary: str | None = None) -> None:
    """Save with the primary model in the name, so both notebooks keep their figures."""
    path = Path(name)
    if primary is not None:
        path = path.with_name(f"{path.stem}_{primary}{path.suffix}")
    figure.savefig(FIGURES / path, dpi=140, bbox_inches="tight")
    print(FIGURES / path)


def plot_score_hists(mc: dict, edges=SCORE_EDGES,
                     primary: str = "da") -> None:
    """MC score distributions, h8s3 on MC truth; primary solid, other dashed."""
    styles = model_line(primary)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for axis, weighted in zip(axes, (False, True)):
        for model, events in mc.items():
            for data_class, label in GROUP_LABEL.items():
                subset = events[events.data_class == data_class]
                counts, _ = np.histogram(
                    subset.score, bins=edges,
                    weights=event_weights(subset, weighted))
                axis.step(centres, counts / counts.sum(), where="mid",
                          color=GROUP_COLOR[data_class], **styles[model],
                          label=label if model == primary else None)
        axis.set(yscale="log", xlabel="prefilter score",
                 title="weighted by EventWeight" if weighted else "unweighted")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("fraction of group per bin")
    axes[0].legend(fontsize=8, title="solid: primary   dashed: other",
                   title_fontsize=8)
    figure.suptitle("MC score distributions, h8s3 on MC truth", y=1.02)
    figure.tight_layout()
    save(figure, "01_score_distributions.png", primary)


def plot_threshold_and_roc(mc: dict, primary: str = "da") -> None:
    styles = model_line(primary)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    for row, weighted in enumerate((False, True)):
        versus_threshold, roc = axes[row]
        for model, events in mc.items():
            curves = suppression_and_efficiency(
                rank_by_score(events, weighted))
            versus_threshold.plot(
                curves.threshold, curves.suppression,
                color=GROUP_COLOR["muatm_2020"], **styles[model],
                label="suppression" if model == primary else None)
            for data_class in NU_CLASSES:
                versus_threshold.plot(
                    curves.threshold, curves[data_class],
                    color=GROUP_COLOR[data_class], **styles[model],
                    label=GROUP_LABEL[data_class] if model == primary else None)
                roc.plot(curves.suppression, curves[data_class],
                         color=GROUP_COLOR[data_class], **styles[model],
                         label=f"{GROUP_LABEL[data_class]} — {model}")
        versus_threshold.axvline(
            working_point(rank_by_score(mc[primary], weighted))["threshold"],
            color="0.3", lw=0.9, ls=":")
        roc.axvline(TARGET_SUPPRESSION, color="0.3", lw=0.9, ls=":")
        versus_threshold.set(
            yscale="log", xlabel="threshold", ylabel="fraction passing",
            title="weighted" if weighted else "unweighted")
        roc.set(xscale="log", xlabel="suppression (muatm passing)",
                ylabel=r"$\nu$ efficiency", title="ROC")
        for axis in (versus_threshold, roc):
            axis.grid(alpha=0.25)
        if row == 0:
            versus_threshold.legend(fontsize=8,
                                    title="solid: primary   dashed: other",
                                    title_fontsize=8)
            roc.legend(fontsize=7, loc="lower right")
    figure.suptitle("h8s3 on MC truth", y=0.995)
    figure.tight_layout()
    save(figure, "02_efficiency_suppression_roc.png", primary)


def plot_efficiency_at_wp(mc: dict, primary: str = "da") -> None:
    styles = model_line(primary)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    for row, weighted in enumerate((False, True)):
        for column, (variable, edges, xlabel, log_x) in enumerate((
                ("theta_deg", THETA_EDGES, r"primary $\theta$, deg", False),
                ("energy_gev", ENERGY_EDGES, "primary energy, GeV", True))):
            axis = axes[row, column]
            centres = (np.sqrt(edges[:-1] * edges[1:]) if log_x
                       else 0.5 * (edges[:-1] + edges[1:]))
            for model, events in mc.items():
                threshold = working_point(
                    rank_by_score(events, weighted))["threshold"]
                for data_class in NU_CLASSES:
                    axis.plot(
                        centres,
                        efficiency_profile(events, data_class, threshold,
                                           variable, edges, weighted),
                        color=GROUP_COLOR[data_class], **styles[model],
                        marker="o", ms=3,
                        label=(f"{GROUP_LABEL[data_class]} — {model}"
                               if row == 0 and column == 0 else None))
            axis.set(xlabel=xlabel, ylabel=r"$\nu$ efficiency", ylim=(0, 1.02),
                     title="weighted" if weighted else "unweighted")
            if log_x:
                axis.set_xscale("log")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    figure.suptitle(
        r"Efficiency at suppression $10^{-3}$, each model at its own threshold, "
        "h8s3 on MC truth", y=0.995)
    figure.tight_layout()
    save(figure, "03_efficiency_vs_zenith_and_energy.png", primary)


def plot_exp_excess(exp_counts: dict, muatm_counts: dict,
                    edges=SCORE_EDGES, primary: str = "da") -> None:
    styles = model_line(primary)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    centres = 0.5 * (edges[:-1] + edges[1:])
    for model in MODELS:
        differential, cumulative = shape_ratio(
            exp_counts[model], muatm_counts[model])
        axes[0].plot(centres, differential, color=MODEL_COLOR[model],
                     **styles[model], marker="o", ms=3,
                     label=MODELS[model]["label"])
        axes[1].plot(edges[:-1], cumulative, color=MODEL_COLOR[model],
                     **styles[model], marker="o", ms=3,
                     label=MODELS[model]["label"])
    for axis, (title, xlabel) in zip(
            axes, (("per score bin", "prefilter score"),
                   ("above threshold", "threshold"))):
        axis.axhline(1.0, color="0.3", lw=0.9, ls=":")
        axis.set(yscale="log", xlabel=xlabel,
                 ylabel="exp_full / muatm (shapes)", title=title)
        axis.yaxis.set_major_formatter(ticker.ScalarFormatter())
        axis.yaxis.set_minor_formatter(ticker.ScalarFormatter())
        axis.tick_params(axis="y", which="minor", labelsize=7)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Experimental excess over atmospheric-muon MC, h8s3 by the sig-noise model",
        y=1.02)
    figure.tight_layout()
    save(figure, "04_experimental_excess.png", primary)


def plot_excess_vs_muatm_fraction(histograms: dict, edges=SCORE_EDGES,
                                  primary: str = "da",
                                  min_muatm_events: int = 20,
                                  z: float = 1.96) -> None:
    """Survival curves and exp_full/muatm ratio vs the muatm pass fraction.

    Reproduces the nu-classifier survival template
    (`HIGH_SCORE_EXCESS/template_score_survival.ipynb`): x is the fraction of
    muatm above the threshold (95% Wilson interval), y is the ratio of the
    exp_full and muatm survival fractions (delta-method ±1σ); both axes are
    logarithmic.  The histogram counts are h8s3 by the sig-noise model, so the
    plot uses exactly the same populations as figure 4.
    """
    styles = model_line(primary)
    thresholds = edges[:-1]

    def survival(counts):
        counts = np.asarray(counts, dtype=float)
        total = counts.sum()
        above = np.cumsum(counts[::-1])[::-1]
        return above / total, above, total

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for model, hists in histograms.items():
        for data_class in ("exp_full",) + tuple(MC_CLASSES):
            fraction, _, _ = survival(hists[data_class])
            axes[0].step(
                thresholds, fraction, where="post",
                color=SURVIVAL_COLOR[data_class], **styles[model],
                label=(PRETTY[data_class] if model == primary else None))
    axes[0].set(yscale="log", xlabel="prefilter score threshold",
                ylabel="fraction above threshold", ylim=(1e-6, 1.0))
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8, title="solid: primary   dashed: other",
                   title_fontsize=8)

    for model, hists in histograms.items():
        s_e, _, n_e = survival(hists["exp_full"])
        s_m, k_m, n_m = survival(hists["muatm_2020"])
        ok = (s_m > 0) & (s_e > 0) & (k_m >= min_muatm_events)
        x = s_m[ok]
        ratio = s_e[ok] / s_m[ok]
        sig_e = np.sqrt(s_e[ok] * (1.0 - s_e[ok]) / n_e)
        sig_m = np.sqrt(s_m[ok] * (1.0 - s_m[ok]) / n_m)
        sig_ratio = ratio * np.sqrt((sig_e / s_e[ok]) ** 2
                                    + (sig_m / s_m[ok]) ** 2)
        x_lo, x_hi = wilson_interval(k_m[ok], n_m, z)
        order = np.argsort(x)
        xs, rs, sr = x[order], ratio[order], sig_ratio[order]
        xl, xh = x_lo[order], x_hi[order]

        axes[1].plot(xs, rs, color=MODEL_COLOR[model], **styles[model],
                     label=MODELS[model]["label"])
        axes[1].fill_between(xs, np.maximum(rs - sr, 1e-8), rs + sr,
                             color=MODEL_COLOR[model], alpha=0.12, lw=0)
        sel = np.unique(np.linspace(0, len(xs) - 1, 24).astype(int))
        axes[1].errorbar(
            xs[sel], rs[sel],
            xerr=np.vstack([xs[sel] - xl[sel], xh[sel] - xs[sel]]),
            yerr=sr[sel], fmt="none", ecolor="#555555",
            elinewidth=0.8, capsize=2, alpha=0.65)

    axes[1].axhline(1.0, color="0.3", lw=0.9, ls=":")
    axes[1].axvline(TARGET_SUPPRESSION, color="0.3", lw=0.9, ls=":")
    axes[1].set(
        xscale="log", yscale="log", xlim=(1e-5, 1.0),
        xlabel="fraction of muatm above threshold (Wilson 95% interval)",
        ylabel="exp_full / muatm  (±1σ, delta method)")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)

    figure.suptitle(
        "Score survival and experimental excess vs muatm fraction, "
        "h8s3 by the sig-noise model", y=1.02)
    figure.tight_layout()
    save(figure, "05_excess_vs_muatm_fraction.png", primary)
