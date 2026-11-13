"""Loading, metrics and figures for report_260906_2310, on top of `inference.reader`.

One place for the code behind every figure in the two notebooks, so a figure can always
be traced back to the query that produced it. Both notebooks import this and differ only
in which model they make primary.

The model pair
--------------
The two prefilter runs differ **only** in `domain_adaptation.lambda_scheduler.max_lambda`
(0.05 against 0.0); seed, architecture, data, losses and schedules are identical — a diff
of the two `da_config.yaml` files shows that key, the run name and the training GPU, and
nothing else. So the pair is domain adaptation against a matched control.

Where the data comes from
-------------------------
Everything goes through `inference.reader`, which addresses the prediction databases, the
catalog and the HDF5 stores together. In particular zenith angle, primary energy and the
MC event weight are **not** in the catalog: they live in `prime_prty` in
`baikal_mc_merged.h5`, and the reader returns them as `prime_theta_deg`,
`prime_energy_gev` and `prime_event_weight` beside each event.

`exp_full` is never materialised. 133M scored events are read only as aggregates through
`Reader.histogram`, which bins in DuckDB.

Conventions
-----------
suppression(t) = fraction of muatm_2020 with score >= t   (lower is better)
efficiency(t)  = fraction of one neutrino class with score >= t

Both are computed **within** a class, so they stay well defined whether or not EventWeight
is comparable *between* classes — which has not been checked here and is not claimed.

nue2 is astrophysical nu_mu with an E^-2 spectrum, not electron neutrinos. The labels say so.
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

MODELS = {
    "da":   {"checkpoint": "da_prefilter_numu_260429_0155",
             "label": "DA, max_lambda = 0.05"},
    "noda": {"checkpoint": "da_prefilter_numu_260428_2039",
             "label": "control, max_lambda = 0.0"},
}
OTHER = {"da": "noda", "noda": "da"}

MC_CLASSES = ["muatm_2020", "nuatm_2020", "nue2_2020"]
NU_CLASSES = ["nuatm_2020", "nue2_2020"]

#: Honest names: the h5 group name `nue2` does not mean electron neutrinos.
PRETTY = {
    "muatm_2020": "muatm — atmospheric muons (EAS)",
    "nuatm_2020": r"nuatm — atmospheric $\nu$",
    "nue2_2020":  r"nue2 — astrophysical $\nu_\mu$, $E^{-2}$",
    "exp_full":   "exp_full — experiment",
}
COLOR = {"muatm_2020": "#c44e52", "nuatm_2020": "#4c72b0",
         "nue2_2020": "#55a868", "exp_full": "#8172b2"}

#: The working point every efficiency-versus-variable figure is evaluated at.
WORKING_POINT = 1e-3


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

    The target domain of both trainings was `exp.h5`. `NOT_DA_TARGET` cannot be used
    here — the prefilter dataset records no `exp_h5_part_keys.npy` — so the claim is
    established the other way instead, by comparing the two files' part keys directly.
    A part key is season + cluster + run, so an empty intersection means the two files
    hold different detector runs and no exp_full event can have been a training target.
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

#: Columns kept from the reader. The rest — source, cluster, run, part_key, local_idx,
#: n_hits, phi, nucleon_n, response_muons_n — are used by no figure here, and over 31M
#: events the three string ones alone cost about 9 GB as pandas objects.
_KEEP = ["event_fk", "data_class", "score", "prime_theta_deg", "prime_energy_gev",
         "prime_event_weight"]
_FLOATS = ("score", "prime_theta_deg", "prime_energy_gev", "prime_event_weight")


def _trim(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[_KEEP].copy()
    out["data_class"] = out["data_class"].astype("category")
    for column in _FLOATS:
        out[column] = out[column].astype(np.float32)
    return out


def load_mc(model: str, progress=None, budget_mb: int = 512) -> pd.DataFrame:
    """Out-of-training MC scored by one model, with theta, energy and EventWeight.

    `NOT_TRAINED_PARTS` drops every event of every HDF5 part the training NPY dataset
    was built from. That is coarser than the per-event exclusion the nu-classifier gets,
    and it is the finest one available: `prefilter_npy_ds_builder` writes no
    `h5_part_keys.npy`, so there is no record of which events inside those parts the
    builder kept. A part the builder never opened cannot have contributed an event, so
    what survives is out-of-training by construction.

    Streamed rather than taken from `events()` in one piece, and trimmed to the six
    columns the figures use as each chunk arrives: the full frame carries `source`,
    `run` and `part_key` as pandas objects, which over 31M events is about 9 GB of
    strings that nothing here reads.
    """
    handle = reader.open("mc_merged", checkpoint=MODELS[model]["checkpoint"])
    pieces = [_trim(chunk.events)
              for chunk in handle.stream(where=[reader.NOT_TRAINED_PARTS],
                                         with_hits=False, budget_mb=budget_mb,
                                         progress=progress)]
    frame = pd.concat(pieces, ignore_index=True)
    return frame.sort_values("event_fk", ignore_index=True)


def combine_mc(per_model: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One frame carrying both models' scores, plus the event-set agreement report.

    The two models are only comparable on events they both scored, so the intersection
    is taken here rather than assumed. The report says how many events each model has and
    how many survive, per class — if the sets ever drift apart, that table is where it shows.
    """
    report = []
    for model, frame in per_model.items():
        counts = frame.groupby("data_class", observed=True).size()
        for data_class, n in counts.items():
            report.append({"model": MODELS[model]["label"],
                           "data_class": data_class, "n_scored": int(n)})

    left, right = per_model["da"], per_model["noda"]
    merged = left.merge(right[["event_fk", "score"]], on="event_fk",
                        suffixes=("_da", "_noda"), how="inner")
    merged = merged.rename(columns={"prime_theta_deg": "theta_deg",
                                    "prime_energy_gev": "energy_gev",
                                    "prime_event_weight": "weight"})

    both = merged.groupby("data_class", observed=True).size()
    table = pd.DataFrame(report).pivot(index="data_class", columns="model",
                                       values="n_scored")
    model_columns = list(table.columns)
    table["in both"] = both.reindex(table.index).fillna(0).astype("int64")
    # True only when neither model scored an event the other missed.
    table["identical sets"] = (
        table[model_columns].eq(table["in both"], axis=0).all(axis=1))
    return merged, table.rename_axis(columns=None).reset_index()


def exp_histograms(edges: np.ndarray, exclude_bad_runs: bool = True
                   ) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """exp_full score histograms for both models, aggregated in DuckDB.

    133M scored events are never materialised: `Reader.histogram` bins them in the
    database. `exclude_bad_runs` applies `reader.NOT_EXCLUDED`, which for this source
    drops `part_s2020_c02_r0020` and `part_s2020_c02_r0249` — the two runs where
    channels 222/224/225/227 emit up to 3169.8 p.e. against a median hit of 0.97
    (evidence recorded in `inference/reader/spec.py`). The counts are reported both
    ways so the effect of the cut is visible rather than assumed.
    """
    bins = len(edges) - 1
    span = (float(edges[0]), float(edges[-1]))
    where = [reader.NOT_EXCLUDED] if exclude_bad_runs else None

    counts, report = {}, []
    for model in MODELS:
        handle = reader.open("exp_full", checkpoint=MODELS[model]["checkpoint"])
        binned = handle.histogram("score", bins=bins, span=span, where=where)
        counts[model] = binned["n"].to_numpy(np.float64)
        report.append({"model": MODELS[model]["label"],
                       "all runs": handle.count()["events"],
                       "bad runs excluded": handle.count(
                           where=[reader.NOT_EXCLUDED])["events"],
                       "used here": int(binned["n"].sum())})
    return counts, pd.DataFrame(report)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class _Cumulative:
    """One class's scores, sorted once, with the running weight fraction beside them.

    Every quantity here — suppression at a threshold, the threshold at a suppression,
    a whole ROC curve — is a lookup in the same sorted array. Sorting once matters:
    the muatm arm alone is 21M events, and the threshold scan asks 400 questions of it.
    """

    __slots__ = ("scores", "fraction", "cut_scores", "cut_fraction")

    def __init__(self, scores: np.ndarray, weights: np.ndarray) -> None:
        scores = np.asarray(scores, np.float64)
        weights = np.asarray(weights, np.float64)
        order = np.argsort(scores)[::-1]
        self.scores = scores[order]                       # descending
        self.fraction = np.cumsum(weights[order]) / weights.sum()
        # The running sum divided by the total can land a few ulps either side of 1,
        # and one ulp above it makes a request for "keep everything" skip the lowest
        # score. Pinned, because the last entry is 1 by definition.
        self.fraction[-1] = 1.0

        # A threshold cannot split events that share a score, so the achievable cuts
        # are the DISTINCT scores, each admitting its whole tie block. Working from
        # `fraction` alone would name a threshold whose real pass fraction overshoots
        # the target by the weight of the block it lands inside. These two arrays are
        # the cuts that actually exist, and `pass_fraction(cut_scores) == cut_fraction`
        # holds exactly.
        last_of_block = np.append(np.flatnonzero(np.diff(self.scores) != 0),
                                  len(self.scores) - 1)
        self.cut_scores = self.scores[last_of_block]      # descending, distinct
        self.cut_fraction = self.fraction[last_of_block]

    def pass_fraction(self, thresholds) -> np.ndarray:
        """Weighted fraction with score >= t, for every t."""
        thresholds = np.atleast_1d(np.asarray(thresholds, np.float64))
        # self.scores descends, so count from its ascending reverse.
        index = len(self.scores) - np.searchsorted(self.scores[::-1], thresholds,
                                                  side="left")
        out = np.zeros(thresholds.shape, dtype=np.float64)
        nonzero = index > 0
        out[nonzero] = self.fraction[index[nonzero] - 1]
        return out

    def threshold_for(self, targets) -> np.ndarray:
        """The **lowest** threshold whose pass fraction is still at most each target.

        Lowest, not highest: any higher threshold also respects the target, by letting
        fewer events through, and would throw away signal for nothing. So the working
        point is the loosest cut that still meets the suppression asked for.

        `fraction` is non-decreasing along the descending scores, so the last entry at
        or below the target admits the most events and therefore sits at the lowest
        score.

        Only cuts that exist are considered — the distinct scores — so the returned
        threshold's own pass fraction respects the target rather than overshooting by
        the weight of a tie block it happened to land inside.

        When even the highest tie block already exceeds the target, the target is
        finer than the sample can express and the top score is returned. The pass
        fraction then overshoots, which is why every caller here reports the achieved
        `suppression` beside the threshold rather than the requested one. For the
        21M-event muatm arm that floor is far below the 1e-3 this report asks for.
        """
        targets = np.atleast_1d(np.asarray(targets, np.float64))
        index = np.searchsorted(self.cut_fraction, targets, side="right") - 1
        return self.cut_scores[np.clip(index, 0, len(self.cut_scores) - 1)]

    @property
    def granularity_floor(self) -> float:
        """The smallest pass fraction any threshold on this sample can reach."""
        return float(self.cut_fraction[0])


def cumulative(frame: pd.DataFrame, data_class: str, column: str,
               weighted: bool) -> _Cumulative:
    """The sorted cumulative for one class, built once per frame.

    A notebook calls `scan`, `working_point` and three plot functions on the same
    frame, which between them ask for the muatm cumulative a few dozen times; each
    build is a sort of 21M values. The cache lives in `frame.attrs` rather than in a
    module-level dict keyed by `id(frame)` — an id is reused after the frame it named
    is collected, and a stale cumulative would silently answer for the wrong data.
    """
    cache = frame.attrs.setdefault("_cumulatives", {})
    key = (data_class, column, weighted)
    found = cache.get(key)
    if found is None:
        subset = frame[frame.data_class == data_class]
        weights = (subset["weight"].to_numpy(np.float64) if weighted
                   else np.ones(len(subset)))
        found = _Cumulative(subset[column].to_numpy(np.float64), weights)
        cache[key] = found
    return found


def scan(frame: pd.DataFrame, column: str, weighted: bool,
         n_points: int = 400) -> pd.DataFrame:
    """Suppression and per-class efficiency over a grid of thresholds.

    The grid is the muatm score quantiles at log-spaced pass fractions, so the curve is
    resolved deep in the background tail instead of uniformly across a score range that
    is nearly empty there.
    """
    muatm = cumulative(frame, "muatm_2020", column, weighted)
    thresholds = np.unique(muatm.threshold_for(np.logspace(0, -6, n_points)))

    out = {"threshold": thresholds,
           "suppression": muatm.pass_fraction(thresholds)}
    for data_class in NU_CLASSES:
        out[f"eff_{data_class}"] = cumulative(
            frame, data_class, column, weighted).pass_fraction(thresholds)
    return pd.DataFrame(out)


def working_point(frame: pd.DataFrame, column: str, weighted: bool,
                  target: float = WORKING_POINT) -> dict:
    """Threshold at suppression = target, and the efficiency each class reaches there."""
    muatm = cumulative(frame, "muatm_2020", column, weighted)
    threshold = float(muatm.threshold_for(target)[0])
    out = {"threshold": threshold,
           "suppression": float(muatm.pass_fraction(threshold)[0])}
    for data_class in NU_CLASSES:
        out[f"eff_{data_class}"] = float(cumulative(
            frame, data_class, column, weighted).pass_fraction(threshold)[0])
    return out


def paired_difference(frame: pd.DataFrame, weighted: bool,
                      target: float = WORKING_POINT) -> pd.DataFrame:
    """Is the efficiency gap between the two models real? A paired test.

    The two models score the **same** events, so comparing their efficiencies with an
    unpaired binomial error is the wrong yardstick — it would treat 8.9M shared events
    as two independent samples and badly understate what the data settle. What decides
    the comparison are the events the two models disagree about: those one model selects
    at its own working point and the other does not.

    So this is McNemar's arrangement. With `b` events selected only by DA and `c` only by
    the control, the efficiency difference is `(W_b - W_c) / W`, and under the null that
    each discordant event is equally likely to fall either way its variance is the sum of
    the discordant weights squared, giving `sqrt(sum w^2) / W`. Unweighted, that reduces
    to `sqrt(b + c) / N`.

    Each model is taken at **its own** threshold for `target` suppression, matching how
    every other figure here compares them; the test therefore asks whether one model
    selects more neutrinos than the other *at equal background*, not at equal score.

    This is an MC-only statement. It says which model selects more MC neutrinos, and
    nothing about which model is preferable — the experiment-over-MC figure points the
    other way, and no test here weighs the two against each other.
    """
    thresholds = {model: working_point(frame, f"score_{model}", weighted,
                                       target)["threshold"]
                  for model in MODELS}
    rows = []
    for data_class in NU_CLASSES:
        subset = frame[frame.data_class == data_class]
        passes = {model: subset[f"score_{model}"].to_numpy(np.float64)
                  >= thresholds[model] for model in MODELS}
        weights = (subset["weight"].to_numpy(np.float64) if weighted
                   else np.ones(len(subset)))
        total = weights.sum()

        da_only = passes["da"] & ~passes["noda"]
        control_only = ~passes["da"] & passes["noda"]
        difference = (weights[da_only].sum() - weights[control_only].sum()) / total
        discordant = da_only | control_only
        error = np.sqrt(np.square(weights[discordant]).sum()) / total

        rows.append({
            "weighted": weighted,
            "data_class": data_class,
            "eff DA": weights[passes["da"]].sum() / total,
            "eff control": weights[passes["noda"]].sum() / total,
            "DA selects only": int(da_only.sum()),
            "control selects only": int(control_only.sum()),
            "difference (DA - control)": difference,
            "paired error": error,
            "z": difference / error if error > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def efficiency_vs(frame: pd.DataFrame, column: str, threshold: float, variable: str,
                  edges: np.ndarray, weighted: bool) -> pd.DataFrame:
    """Per-bin efficiency against `variable` at a fixed threshold, with its error.

    The error is the binomial spread on effective counts, n_eff = (sum w)^2 / sum w^2,
    which reduces to the usual sqrt(p(1-p)/n) when unweighted.
    """
    which = np.digitize(frame[variable].to_numpy(np.float64), edges) - 1
    weights = (frame["weight"].to_numpy(np.float64) if weighted
               else np.ones(len(frame)))
    passed = frame[column].to_numpy(np.float64) >= threshold

    rows = []
    for i in range(len(edges) - 1):
        inside = which == i
        if not inside.any():
            rows.append((np.nan, np.nan, 0))
            continue
        w, p_mask = weights[inside], passed[inside]
        total, selected = w.sum(), w[p_mask].sum()
        p = selected / total if total > 0 else np.nan
        n_eff = total ** 2 / np.square(w).sum() if total > 0 else 0.0
        err = np.sqrt(max(p * (1 - p), 0.0) / n_eff) if n_eff > 0 else np.nan
        rows.append((p, err, int(inside.sum())))

    return pd.DataFrame(rows, columns=["eff", "err", "n_raw"]).assign(
        lo=edges[:-1], hi=edges[1:], mid=0.5 * (edges[:-1] + edges[1:]))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    try:
        shown = path.relative_to(PROJECT_ROOT)
    except ValueError:                     # FIGURES redirected outside the repo
        shown = path
    print(f"saved {shown}")
    return path


def _provenance(fig, primary: str, extra: str = "") -> None:
    checkpoint = reader.open_checkpoint(MODELS[primary]["checkpoint"])
    text = (f"primary: {checkpoint.name}  |  read through inference.reader  |  "
            f"MC: NOT_TRAINED_PARTS")
    if extra:
        text += f"  |  {extra}"
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=6, color="0.35")


def _styles(primary: str):
    """The other model as a wide translucent halo, the primary as a line on top.

    Drawn reference-first so the primary sits over it. A dashed reference was tried
    and is unreadable here: on the working-point figure the two models agree to a few
    tenths of a percent, so the dashes vanish under the solid line and the figure looks
    as though the reference were missing. As a halo, agreement reads as a band hugging
    the line and disagreement reads as the halo peeling away — the two cases look
    different, which is the whole point of putting them on one axis.
    """
    return ((OTHER[primary], dict(lw=4.5, alpha=0.28, solid_capstyle="round")),
            (primary, dict(lw=1.7, alpha=1.0)))


def plot_score_hists(mc: pd.DataFrame, exp_counts: dict[str, np.ndarray],
                     edges: np.ndarray, primary: str) -> Path:
    """Score histograms per group, primary model solid, the other as a halo."""
    mid = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    for ax, weighted in zip(axes, (False, True)):
        for data_class in MC_CLASSES:
            subset = mc[mc.data_class == data_class]
            weights = (subset["weight"].to_numpy(np.float64) if weighted else None)
            for model, style in _styles(primary):
                counts, _ = np.histogram(subset[f"score_{model}"], bins=edges,
                                         weights=weights)
                ax.step(mid, counts / counts.sum(), where="mid",
                        color=COLOR[data_class], **style,
                        label=PRETTY[data_class] if model == primary else None)
        # exp_full carries no MC weight, so it is the same curve in both panels
        for model, style in _styles(primary):
            counts = exp_counts[model]
            ax.step(mid, counts / counts.sum(), where="mid",
                    color=COLOR["exp_full"], **style,
                    label=PRETTY["exp_full"] if model == primary else None)
        ax.set_yscale("log")
        ax.set_xlabel("prefilter score")
        ax.grid(alpha=0.25)
        ax.set_title("weighted by EventWeight, within class" if weighted
                     else "unweighted — a property of the sample, not a rate")
    axes[0].set_ylabel("fraction of the group per bin")
    axes[0].legend(fontsize=8, loc="upper center")

    fig.suptitle(f"Prefilter score distributions — solid: {MODELS[primary]['label']}, "
                 f"halo: {MODELS[OTHER[primary]]['label']}", y=1.03)
    _provenance(fig, primary, "exp_full unweighted in both panels")
    return _save(fig, f"01_score_hists_{primary}.png")


def plot_threshold_and_roc(mc: pd.DataFrame, primary: str) -> Path:
    """Efficiency and suppression against threshold, and against each other."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, weighted in enumerate((False, True)):
        ax_threshold, ax_roc = axes[row]
        for model, style in _styles(primary):
            curves = scan(mc, f"score_{model}", weighted)
            ax_threshold.plot(
                curves.threshold, curves.suppression, color=COLOR["muatm_2020"],
                label="suppression — muatm passing" if model == primary else None,
                **style)
            for data_class in NU_CLASSES:
                ax_threshold.plot(
                    curves.threshold, curves[f"eff_{data_class}"],
                    color=COLOR[data_class], **style,
                    label=PRETTY[data_class] if model == primary else None)
                ax_roc.plot(
                    curves.suppression, curves[f"eff_{data_class}"],
                    color=COLOR[data_class], **style,
                    label=PRETTY[data_class] if model == primary else None)

        point = working_point(mc, f"score_{primary}", weighted)
        ax_threshold.axvline(point["threshold"], color="0.3", lw=0.9, ls=":")
        ax_roc.axvline(WORKING_POINT, color="0.3", lw=0.9, ls=":")
        ax_threshold.set_yscale("log")
        ax_threshold.set_xlabel("threshold")
        ax_threshold.set_ylabel("fraction passing")
        ax_roc.set_xscale("log")
        ax_roc.set_xlabel("suppression — fraction of muatm passing")
        ax_roc.set_ylabel(r"$\nu$ efficiency")
        tag = "weighted by EventWeight, within class" if weighted else "unweighted"
        ax_threshold.set_title(
            f"threshold scan — {tag}\nworking point t = {point['threshold']:.5f}")
        ax_roc.set_title(f"ROC — {tag}")
        for ax in (ax_threshold, ax_roc):
            ax.grid(alpha=0.25)
        if row == 0:
            ax_threshold.legend(fontsize=8)
            ax_roc.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"MC metrics — solid: {MODELS[primary]['label']}, "
                 f"halo: {MODELS[OTHER[primary]]['label']}", y=0.995)
    fig.tight_layout()
    _provenance(fig, primary)
    return _save(fig, f"02_threshold_roc_{primary}.png")


def plot_efficiency_at_wp(mc: pd.DataFrame, primary: str) -> Path:
    """Efficiency against zenith angle and against energy, at suppression = 1e-3."""
    # Binned over the range the plotted classes actually occupy. The MC productions
    # put atmospheric muons and neutrinos in disjoint theta hemispheres — measured on
    # out-of-training parts, muatm spans 94.2-179.9 deg and both neutrino classes
    # 0.11-90.00 deg, the cap sitting at exactly 90.00, which is a generation cut and
    # not a physical edge. Only neutrinos appear on this figure, so a 0-180 axis would
    # be half empty and would suggest the curves had simply stopped.
    nu_only = mc[mc.data_class.isin(NU_CLASSES)]
    theta_hi = float(np.ceil(nu_only["theta_deg"].max()))
    energy = nu_only["energy_gev"]
    energy_edges = np.logspace(np.floor(np.log10(max(energy.min(), 1e-3))),
                               np.ceil(np.log10(energy.max())), 25)
    axes_spec = (
        ("theta_deg", np.linspace(0.0, theta_hi, 19),
         r"primary zenith angle $\theta$, deg", False),
        ("energy_gev", energy_edges, "primary energy, GeV", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, weighted in enumerate((False, True)):
        for column_i, (variable, edges, xlabel, logx) in enumerate(axes_spec):
            ax = axes[row, column_i]
            for model, style in _styles(primary):
                point = working_point(mc, f"score_{model}", weighted)
                for data_class in NU_CLASSES:
                    subset = mc[mc.data_class == data_class]
                    profile = efficiency_vs(subset, f"score_{model}",
                                            point["threshold"], variable, edges,
                                            weighted)
                    marker = dict(marker="o", ms=3) if model == primary else {}
                    ax.plot(profile["mid"], profile["eff"], color=COLOR[data_class],
                            **style, **marker,
                            label=PRETTY[data_class] if model == primary else None)
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\nu$ efficiency at suppression $10^{-3}$")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            ax.set_title("weighted by EventWeight, within class" if weighted
                         else "unweighted")
            if row == 0 and column_i == 0:
                ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(f"Efficiency at the working point (muatm passing = $10^{{-3}}$) — "
                 f"solid: {MODELS[primary]['label']}, "
                 f"halo: {MODELS[OTHER[primary]]['label']}", y=0.995)
    fig.tight_layout()
    _provenance(fig, primary,
                "each model at its own threshold at suppression 1e-3; "
                "binomial errors are printed in the notebook, not drawn")
    return _save(fig, f"03_efficiency_at_wp_{primary}.png")


def plot_exp_excess(mc: pd.DataFrame, exp_counts: dict[str, np.ndarray],
                    edges: np.ndarray, primary: str) -> Path:
    """exp_full over muatm along the score, differential and cumulative.

    Both are normalised to their own totals. MC is not normalised to live time anywhere
    in this repository, so only the SHAPE of the ratio carries information: a flat line
    at 1 would mean the two distributions agree, and the level itself is not a rate.
    """
    mid = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for model, style in _styles(primary):
        muatm = mc[mc.data_class == "muatm_2020"]
        h_mu, _ = np.histogram(muatm[f"score_{model}"], bins=edges)
        f_mu = h_mu / h_mu.sum()
        f_exp = exp_counts[model] / exp_counts[model].sum()

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(f_mu > 0, f_exp / f_mu, np.nan)
            cumulative_mu = np.cumsum(f_mu[::-1])[::-1]
            cumulative_exp = np.cumsum(f_exp[::-1])[::-1]
            cumulative = np.where(cumulative_mu > 0,
                                  cumulative_exp / cumulative_mu, np.nan)

        marker = dict(marker="o", ms=3) if model == primary else {}
        axes[0].plot(mid, ratio, color=COLOR["exp_full"], **style, **marker,
                     label=MODELS[model]["label"])
        axes[1].plot(edges[:-1], cumulative, color=COLOR["exp_full"], **style,
                     **marker, label=MODELS[model]["label"])

    for ax, (title, xlabel) in zip(axes, (("differential — per score bin",
                                           "prefilter score"),
                                          ("cumulative — above the threshold",
                                           "threshold"))):
        ax.axhline(1.0, color="0.3", lw=0.9, ls=":")
        ax.set_yscale("log")
        # The ratio spans well under one decade, where the default log formatter
        # labels only the powers of ten and leaves the axis with a single number on it.
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis="y", which="minor", labelsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("exp_full / muatm, each normalised to 1")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Experimental excess over atmospheric-muon MC along the score", y=1.03)
    fig.tight_layout()
    _provenance(fig, primary,
                "shape ratio only — no live-time normalisation exists for this MC")
    return _save(fig, f"04_exp_excess_{primary}.png")
