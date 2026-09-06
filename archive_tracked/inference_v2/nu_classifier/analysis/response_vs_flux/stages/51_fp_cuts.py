"""Stage 51 -- cuts that remove the false-positive muons, and what they cost.

See PROTOCOL.md, test 5.  The design is deliberately arranged so that the two
numbers that matter cannot be tuned:

* **experiment is never seen while fitting.**  Rules are fitted on MC only, and
  the experimental sample enters exactly one function, :func:`evaluate`.
* **neutrinos are never seen while fitting** either, so their retention is a
  prediction rather than an objective.

What is fitted: muatm the classifier wrongly accepts (xi > 0.5) against muatm it
confidently rejects (xi < 0.01).  Both are the same particles from the same
generator, so the discriminator learns *confusability* and not "muon versus
neutrino", which the classifier already does.

Balance comes from subsampling, not weights, in **disjoint** replicates -- an
overlapping draw shares events with its neighbours and makes a rule look stabler
than it is.  A rule counts only if it survives every replicate.

Four families, because they answer different questions: one-sided single cuts
say whether one quantity carries the difference monotonically, **interval cuts**
say whether it carries it in a *band*, a shallow tree says whether a few
quantities do, and gradient boosting bounds what any rule in this feature space
could achieve.

Interval cuts are not decoration.  The false positives sit near the horizon --
accepted muons fit to a zenith of about 74 deg -- while genuine neutrinos spread
over the whole up-going hemisphere.  A one-sided cut on zenith has to remove that
entire hemisphere and takes vertical neutrinos with it; a band cut can remove the
horizon strip and leave them.  A search restricted to one-sided rules cannot see
that solution even if it exists, which is why the first version of this stage
reported a recall cost far worse than may be necessary.

Usage:
    python stages/51_fp_cuts.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import h5io                                                    # noqa: E402
import provenance                                              # noqa: E402

LOG = logging.getLogger("stage51")
NOT_FEATURES = {"event_fk", "score", "data_class", "source", "part_key",
                "from_target", "from_uniform", "half"}
# Every candidate is evaluated, not a shortlist.  The first version ranked
# candidates by how well they preserve *clean muons* and pushed only the top four
# through the neutrino measurement -- so `extent_m`, which ranks seventh by that
# criterion and keeps 80% of neutrinos where the leaders keep 0.01%, was never
# looked at.  The ranking criterion was not aligned with what matters, and the
# fix is not a better criterion but to stop pre-filtering: measuring what a cut
# does to neutrinos is evaluation, not design, and contaminates nothing.
TOP_SINGLE = 1000
INTERVAL_BINS = 48      # quantile grid per feature for the two-sided search


class _IntervalCut:
    """Keep the events whose feature falls inside (or outside) a band.

    Represented as a mask rather than a score: a band is not a monotone rule, and
    forcing it through a threshold would misrepresent what it does.  Events with
    a missing value are removed, which is the conservative reading.
    """

    def __init__(self, index: int, lo: float, hi: float, inside: bool) -> None:
        self.index, self.lo, self.hi, self.inside = index, lo, hi, inside

    def keep(self, values: np.ndarray) -> np.ndarray:
        column = values[:, self.index]
        within = (column >= self.lo) & (column <= self.hi)
        return np.where(np.isfinite(column), within if self.inside else ~within,
                        False)

    def describe(self, name: str) -> str:
        if self.inside:
            return f"{self.lo:.4g} <= {name} <= {self.hi:.4g}"
        return f"{name} < {self.lo:.4g} or {name} > {self.hi:.4g}"


class _SingleCut:
    """One feature, one direction -- presented like the fitted models.

    Scoring by the signed value means a single cut goes through exactly the same
    threshold-setting and evaluation path as a tree or a boosted ensemble, so the
    three levels are compared on equal terms rather than by different code.
    """

    def __init__(self, index: int, sign: int) -> None:
        self.index, self.sign = index, sign

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        column = self.sign * values[:, self.index]
        return np.column_stack([-column, column])


def load(cfg: h5io.Config) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_parquet(HERE / "data" / "50_fp_features.parquet")
    table["half"] = [crc32(str(p).encode()) % 2 for p in table.part_key]
    columns = [c for c in table.columns
               if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(table[c])]
    return table, columns


def populations(table: pd.DataFrame, cfg: h5io.Config) -> dict[str, pd.DataFrame]:
    """The six samples the report is built from, kept strictly apart."""
    high = float(cfg["stage50"]["score_min"])
    low = float(cfg["stage50"]["score_clean"])
    muatm = table[table.data_class == "muatm_2020"]
    exp = table[table.data_class == "exp_full"]
    return {
        # target and negative class -- MC only, used for fitting
        "muatm_fp": muatm[muatm.from_target & (muatm.score > high)],
        "muatm_clean": muatm[muatm.from_uniform & (muatm.score < low)],
        # denominators, never fitted on
        "muatm_all": muatm[muatm.from_uniform],
        "exp_accepted": exp[exp.from_target & (exp.score > high)],
        "exp_clean": exp[exp.from_uniform & (exp.score < low)],
        "exp_all": exp[exp.from_uniform],
        "nuatm": table[table.data_class == "nuatm_2020"],
        "nue2": table[table.data_class == "nue2_2020"],
    }


def as_arrays(pops: dict[str, pd.DataFrame],
              columns: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each population as a feature matrix plus its half labels, converted once.

    Converting inside the evaluation loop meant rebuilding a 14-million-element
    matrix for every one of ~1,500 rules, which is what made the full scan too
    slow to finish.
    """
    out = {}
    for name, frame in pops.items():
        matrix = frame[columns].to_numpy(np.float64)
        halves = (frame.half.to_numpy() if "half" in frame
                  else np.zeros(len(frame), dtype=int))
        out[name] = (matrix, halves)
    return out


def evaluate(mask, arrays: dict[str, tuple[np.ndarray, np.ndarray]],
             census: dict) -> dict:
    """Retention of every population under one frozen rule.

    ``mask`` maps a feature matrix to a boolean "kept" array.  Every rule family
    -- one-sided, interval, tree, boosted -- is reduced to that same interface,
    so the four are compared by identical code rather than by four code paths.

    This is the only place the experimental and neutrino samples are touched.
    """
    def kept(name: str, half: int | None = None) -> float:
        matrix, halves = arrays[name]
        if half is not None:
            matrix = matrix[halves == half]
        if len(matrix) == 0:
            return np.nan
        return float(mask(matrix).mean())

    out = {
        # MC populations are read on the half the rule was not fitted on
        "muatm_fp_kept": kept("muatm_fp", 1),
        "muatm_clean_kept": kept("muatm_clean", 1),
        "muatm_all_kept": kept("muatm_all", 1),
        # experiment is held out entirely by construction
        "exp_accepted_kept": kept("exp_accepted"),
        "exp_clean_kept": kept("exp_clean"),
        "exp_all_kept": kept("exp_all"),
        "nuatm_kept": kept("nuatm"),
        "nue2_kept": kept("nue2"),
        # neutrinos split by MC part too: if a cut is later chosen *because* it
        # spares neutrinos, that choice biases the number, and only agreement
        # between two halves shows the retention is real rather than fitted
        "nuatm_kept_a": kept("nuatm", 0), "nuatm_kept_b": kept("nuatm", 1),
        "nue2_kept_a": kept("nue2", 0), "nue2_kept_b": kept("nue2", 1),
    }
    out["n_exp_accepted_left"] = int(round(out["exp_accepted_kept"]
                                           * len(arrays["exp_accepted"][0])))
    out["mc_selectivity"] = (out["muatm_clean_kept"] / out["muatm_fp_kept"]
                             if out["muatm_fp_kept"] > 0 else np.inf)
    out["exp_selectivity"] = (out["exp_clean_kept"] / out["exp_accepted_kept"]
                              if out["exp_accepted_kept"] > 0 else np.inf)
    for name in ("nuatm_2020", "nue2_2020"):
        accepted_frac = census[f"{name}_above"] / census[f"{name}_quality_total"]
        short = name.split("_")[0]
        out[f"{short}_absolute_efficiency"] = accepted_frac * out[f"{short}_kept"]
    return out


def interval_cuts(design: pd.DataFrame, columns: list[str], target: float,
                  bins: int = INTERVAL_BINS) -> pd.DataFrame:
    """Best band per feature: keep `target` of clean muons removed as few of them
    as possible while leaving at most `1 - target` of the false positives.

    Both readings are scanned -- keep what is inside the band, and keep what is
    outside it.  Counting is done on a quantile grid with cumulative sums, so all
    ~1,100 bands per feature cost one pass over the data rather than 1,100.
    """
    is_fp = design.is_fp.to_numpy()
    rows = []
    for column in columns:
        values = design[column].to_numpy(np.float64)
        good = np.isfinite(values)
        if good.sum() < 200:
            continue
        edges = np.unique(np.nanquantile(values[good], np.linspace(0, 1, bins + 1)))
        if len(edges) < 4:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        index = np.digitize(values, edges) - 1
        index = np.clip(index, 0, len(edges) - 2)
        n_bins = len(edges) - 1
        fp_hist = np.bincount(index[good & is_fp], minlength=n_bins)
        clean_hist = np.bincount(index[good & ~is_fp], minlength=n_bins)
        fp_cum = np.concatenate([[0], np.cumsum(fp_hist)])
        clean_cum = np.concatenate([[0], np.cumsum(clean_hist)])
        total_fp = max(int(is_fp.sum()), 1)
        total_clean = max(int((~is_fp).sum()), 1)
        best = None
        for i in range(n_bins):
            j = np.arange(i + 1, n_bins + 1)
            fp_in = (fp_cum[j] - fp_cum[i]) / total_fp
            clean_in = (clean_cum[j] - clean_cum[i]) / total_clean
            for inside, fp_kept, clean_kept in ((True, fp_in, clean_in),
                                                (False, 1 - fp_in, 1 - clean_in)):
                ok = fp_kept <= (1 - target) + 1e-9
                if not ok.any():
                    continue
                k = int(np.argmax(np.where(ok, clean_kept, -np.inf)))
                candidate = (clean_kept[k], fp_kept[k], inside, edges[i],
                             edges[j[k]])
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            continue
        clean_kept, fp_kept, inside, lo, hi = best
        rows.append({"feature": column, "inside": bool(inside),
                     "lo": float(lo), "hi": float(hi),
                     "design_fp_kept": float(fp_kept),
                     "design_clean_kept": float(clean_kept),
                     "selectivity": float(clean_kept / max(fp_kept, 1e-9))})
    return pd.DataFrame(rows).sort_values("design_clean_kept", ascending=False)


def single_cuts(design: pd.DataFrame, columns: list[str],
                target: float) -> pd.DataFrame:
    """For every feature, the threshold removing `target` of the false positives."""
    is_fp = design.is_fp.to_numpy()
    rows = []
    for column in columns:
        values = design[column].to_numpy(np.float64)
        good = np.isfinite(values)
        if good.sum() < 100:
            continue
        for sign in (+1, -1):
            signed = sign * values
            keep_quantile = np.nanquantile(signed[good & is_fp], 1 - target)
            kept_fp = np.nanmean(signed[good & is_fp] <= keep_quantile)
            kept_clean = np.nanmean(signed[good & ~is_fp] <= keep_quantile)
            rows.append({"feature": column, "direction": "<=" if sign > 0 else ">=",
                         "threshold": sign * keep_quantile,
                         "design_fp_kept": float(kept_fp),
                         "design_clean_kept": float(kept_clean),
                         "selectivity": float(kept_clean / max(kept_fp, 1e-9))})
    return pd.DataFrame(rows).sort_values("selectivity", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    cfg = h5io.load_config(HERE)
    data, config_path = HERE / "data", HERE / "config.yaml"
    settings = cfg["stage51"]

    table, columns = load(cfg)
    census = provenance.read_meta(data / "50_fp_features.parquet")["notes"]["census"]
    pops = populations(table, cfg)
    arrays = as_arrays(pops, columns)
    LOG.info("%d features; populations: %s", len(columns),
             {k: len(v) for k, v in pops.items()})

    fp_design = pops["muatm_fp"][pops["muatm_fp"].half == 0]
    clean_pool = pops["muatm_clean"][pops["muatm_clean"].half == 0]
    n_rep = int(cfg["stage50"]["n_replicates"])
    size = len(fp_design)
    LOG.info("design half: %d false positives, %d clean available -> %d disjoint "
             "replicates of %d", size, len(clean_pool),
             min(n_rep, len(clean_pool) // max(size, 1)), size)

    order = np.random.default_rng(cfg["seed"]).permutation(len(clean_pool))
    replicates = [clean_pool.iloc[order[i * size:(i + 1) * size]]
                  for i in range(min(n_rep, len(clean_pool) // max(size, 1)))]

    singles, intervals, rules = [], [], []
    for index, negatives in enumerate(replicates):
        design = pd.concat([fp_design.assign(is_fp=True),
                            negatives.assign(is_fp=False)], ignore_index=True)
        x = design[columns].to_numpy(np.float64)
        y = design.is_fp.to_numpy(int)

        for target in settings["removal_targets"]:
            target = float(target)
            found = single_cuts(design, columns, target)
            found["replicate"], found["removal_target"] = index, target
            singles.append(found)
            band = interval_cuts(design, columns, target)
            band["replicate"], band["removal_target"] = index, target
            intervals.append(band)

            # the readable answers of both families, carried into evaluation
            for _, cut in found.head(TOP_SINGLE).iterrows():
                model = _SingleCut(columns.index(cut.feature),
                                   +1 if cut.direction == "<=" else -1)
                scores = model.predict_proba(
                    np.nan_to_num(fp_design[columns].to_numpy(np.float64),
                                  nan=-999))[:, 1]
                threshold = float(np.quantile(scores, 1 - target))
                rules.append((f"single:{cut.feature} {cut.direction}", index,
                              target,
                              lambda v, m=model, t=threshold:
                              m.predict_proba(np.nan_to_num(v, nan=-999))[:, 1] <= t,
                              f"{cut.feature} {cut.direction} {cut.threshold:.4g}"))
            for _, cut in band.head(TOP_SINGLE).iterrows():
                model = _IntervalCut(columns.index(cut.feature), cut.lo, cut.hi,
                                     bool(cut.inside))
                rules.append((f"band:{cut.feature}", index, target,
                              lambda v, m=model: m.keep(v),
                              model.describe(cut.feature)))

        for depth in settings["tree_depths"]:
            tree = DecisionTreeClassifier(max_depth=depth,
                                          min_samples_leaf=int(settings["min_leaf"]),
                                          random_state=0)
            tree.fit(np.nan_to_num(x, nan=-999), y)
            scores = tree.predict_proba(
                np.nan_to_num(fp_design[columns].to_numpy(np.float64), nan=-999))[:, 1]
            for target in settings["removal_targets"]:
                threshold = float(np.quantile(scores, 1 - float(target)))
                rules.append((f"tree{depth}", index, float(target),
                              lambda v, m=tree, t=threshold:
                              m.predict_proba(np.nan_to_num(v, nan=-999))[:, 1] <= t,
                              export_text(tree, feature_names=columns,
                                          max_depth=depth)))
        gbm = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                             learning_rate=0.06, random_state=0)
        gbm.fit(x, y)
        scores = gbm.predict_proba(fp_design[columns].to_numpy(np.float64))[:, 1]
        for target in settings["removal_targets"]:
            threshold = float(np.quantile(scores, 1 - float(target)))
            rules.append(("gbm", index, float(target),
                          lambda v, m=gbm, t=threshold:
                          m.predict_proba(v)[:, 1] <= t, ""))
        LOG.info("replicate %d fitted", index)

    provenance.write(pd.concat(singles, ignore_index=True),
                     data / "51_single_cuts.parquet", stage="51_fp_cuts",
                     config_path=config_path,
                     inputs=[data / "50_fp_features.parquet"], started=started)
    provenance.write(pd.concat(intervals, ignore_index=True),
                     data / "51_interval_cuts.parquet", stage="51_fp_cuts",
                     config_path=config_path,
                     inputs=[data / "50_fp_features.parquet"], started=started)

    results, shown = [], set()
    for name, index, target, mask, text in rules:
        row = evaluate(mask, arrays, census)
        row.update({"rule": name, "replicate": index, "removal_target": target,
                    "description": text.splitlines()[0] if text else ""})
        results.append(row)
        if index == 0 and name not in shown and text and not text.startswith("|"):
            shown.add(name)
            LOG.info("%-34s %s", name, text.splitlines()[0])
    frame = pd.DataFrame(results)
    provenance.write(frame, data / "51_cut_results.parquet", stage="51_fp_cuts",
                     config_path=config_path,
                     inputs=[data / "50_fp_features.parquet"], started=started)

    # stability: does the same rule survive every replicate?
    stability = frame.groupby(["rule", "removal_target"]).agg(
        replicates=("replicate", "nunique"),
        muatm_fp_kept=("muatm_fp_kept", "mean"),
        muatm_fp_kept_spread=("muatm_fp_kept", "std"),
        exp_accepted_kept=("exp_accepted_kept", "mean"),
        exp_accepted_kept_spread=("exp_accepted_kept", "std"),
        nuatm_kept=("nuatm_kept", "mean"),
        nuatm_kept_spread=("nuatm_kept", "std"),
        mc_selectivity=("mc_selectivity", "mean"),
        exp_selectivity=("exp_selectivity", "mean")).reset_index()
    provenance.write(stability, data / "51_stability.parquet", stage="51_fp_cuts",
                     config_path=config_path,
                     inputs=[data / "50_fp_features.parquet"], started=started)

    for target in settings["removal_targets"]:
        sub = frame[frame.removal_target == float(target)]
        LOG.info("=== removal target %.0f%% ===", 100 * float(target))
        for name, group in sub.groupby("rule"):
            LOG.info("%-6s muatm_fp %.3f  muatm_all %.3f | exp_acc %.3f "
                     "exp_clean %.3f | nuatm %.3f nue2 %.3f | sel MC %.1f exp %.1f",
                     name, group.muatm_fp_kept.mean(), group.muatm_all_kept.mean(),
                     group.exp_accepted_kept.mean(), group.exp_clean_kept.mean(),
                     group.nuatm_kept.mean(), group.nue2_kept.mean(),
                     group.mc_selectivity.mean(), group.exp_selectivity.mean())
    LOG.info("stage 51 done in %.1f s", time.time() - started)


if __name__ == "__main__":
    main()
