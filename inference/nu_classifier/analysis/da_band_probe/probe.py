"""Probe the residual MC<->exp domain separation as a function of the nu-classifier score.

Question. The DANN domain adaptation aligns MC and exp in the pooled feature space on
average. The experimental excess lives in the high-score tail. Two hypotheses:

  H1 (tail attention): the residual domain separability is concentrated in the tail.
      Then a score-conditional / tail-weighted DA objective has something to attack.
  H2 (energy/composition): the residual separability is explained by gross event size
      (an energy proxy) rather than by anything the classifier reads.

What this measures, per score band:
  * a linear probe on the stored 128-d embeddings, MC-muon vs exp-holdout (5-fold CV
    over events and a part-disjoint split; the latter is the honest one, events inside
    a part share conditions);
  * the same probe on `scalars` (size/charge/topology summaries) -- how much of the
    separation needs no encoder at all;
  * n_sig_hits alone (single-feature AUC) -- the crudest energy proxy;
  * the embeddings probe within n_sig_hits terciles -- H2 as a conditional test.

Controls: mc-muon vs mc-muon (null, gives the noise floor at this n) and
mc-muon vs mc-nuatm in one mid band (sanity: the probe must find real structure).

Populations come from the `splits` table of the FIXED_sn256 prediction databases:
  mc_*  -- data_class=... AND NOT used_for_labels (out of training);
  exp   -- NOT was_da_target AND NOT excluded (holdout).
These are the same conditions `inference.reader` recomputes from the NPY back-links.

Usage:
    python inference/nu_classifier/analysis/da_band_probe/probe.py [--cache DIR]

Writes results/<name>.csv / .json / .png and logs/probe_<timestamp>.log next to this
file. A cache of the extracted arrays is kept under /tmp so reruns skip the DuckDB
reads (the mc_merged embeddings table is ~50 GB; a full fetch takes minutes).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

BASE = ROOT / ("inference/nu_classifier/preds/"
               "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256"
               "@best_da_model")
MC_DB = BASE / "mc_merged_thr0p8.duckdb"
EXP_DB = BASE / "exp_full_thr0p8.duckdb"

#: Score bands. Pooled tail is the union of the last three.
BANDS = [
    ("b0_lt0.05", 0.00, 0.05),
    ("b1_0.05_0.2", 0.05, 0.20),
    ("b2_0.2_0.5", 0.20, 0.50),
    ("b3_0.5_0.8", 0.50, 0.80),
    ("b4_0.8_0.9", 0.80, 0.90),
    ("b5_0.9_0.95", 0.90, 0.95),
    ("b6_gt0.95", 0.95, 1.01),
]
TAIL = ["b4_0.8_0.9", "b5_0.9_0.95", "b6_gt0.95"]

#: Per (population, band): modular subsample (event_fk % mod == 0); None = take all.
#: Chosen from the per-band counts so every sample stays <= ~40k events.
POPULATIONS = {
    "mc_muatm": dict(db="mc", cls="muatm_2020", mods={
        "b0_lt0.05": 2048, "b1_0.05_0.2": 64, "b2_0.2_0.5": 32, "b3_0.5_0.8": 8,
        "b4_0.8_0.9": None, "b5_0.9_0.95": None, "b6_gt0.95": None}),
    "mc_nuatm": dict(db="mc", cls="nuatm_2020", mods={
        "b0_lt0.05": None, "b1_0.05_0.2": None, "b2_0.2_0.5": None,
        "b3_0.5_0.8": 4, "b4_0.8_0.9": 8, "b5_0.9_0.95": 8, "b6_gt0.95": 512}),
    "mc_nue2": dict(db="mc", cls="nue2_2020", mods={
        "b0_lt0.05": None, "b1_0.05_0.2": None, "b2_0.2_0.5": None,
        "b3_0.5_0.8": 4, "b4_0.8_0.9": 8, "b5_0.9_0.95": 8, "b6_gt0.95": 128}),
    "exp": dict(db="exp", cls=None, mods={
        "b0_lt0.05": 64, "b1_0.05_0.2": None, "b2_0.2_0.5": None,
        "b3_0.5_0.8": None, "b4_0.8_0.9": None, "b5_0.9_0.95": None,
        "b6_gt0.95": None}),
}

SCALAR_COLS = [
    "n_sig_hits", "n_sig_strings", "n_channels", "q_mean", "q_std", "q_total",
    "q_max", "q_frac_max", "r_vert", "t_span", "t_span_core", "z_span", "xy_span",
    "x_c", "y_c", "z_c", "prob_mean", "prob_min",
]
#: log1p before standardising -- heavy-tailed, strictly positive.
LOG_COLS = ["n_sig_hits", "n_sig_strings", "n_channels", "q_mean", "q_std", "q_total",
            "q_max", "q_frac_max", "r_vert", "t_span", "t_span_core", "z_span",
            "xy_span", "prob_mean"]

SEED = 42
CAP_LOGISTIC_C = 1.0

logger = logging.getLogger("probe")


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def _where(db: str, cls: str | None) -> str:
    if db == "mc":
        return f"s.data_class = '{cls}' AND NOT s.used_for_labels"
    return "NOT s.was_da_target AND NOT s.excluded"


def extract(connection: duckdb.DuckDBPyConnection, name: str,
            spec: dict) -> pd.DataFrame:
    """Rows: event_fk, score, band, part_key, embedding, <scalar columns>."""
    db = spec["db"]
    pieces = []
    for band, lo, hi in BANDS:
        mod = spec["mods"].get(band)
        p_filt = f"WHERE (event_fk % {mod}) = 0" if mod else ""
        s_filt = f"WHERE (event_fk % {mod}) = 0" if mod else ""
        pieces.append(f"""
        SELECT p.event_fk, p.score, '{band}' AS band, s.part_key
        FROM (SELECT event_fk, score FROM {db}.predictions {p_filt}) p
        JOIN (SELECT event_fk, data_class, used_for_labels, was_da_target, excluded,
                     part_key
              FROM {db}.splits {s_filt}) s USING (event_fk)
        WHERE {_where(db, spec['cls'])}
          AND p.score >= {lo} AND p.score < {hi}""")
    want_sql = " UNION ALL ".join(pieces)

    t0 = time.time()
    query = f"""
    WITH want AS ({want_sql})
    SELECT w.event_fk, w.score, w.band, w.part_key, e.embedding,
           {', '.join('sc.' + c for c in SCALAR_COLS)}
    FROM want w
    JOIN {db}.embeddings e USING (event_fk)
    JOIN {db}.scalars    sc USING (event_fk)
    """
    frame = connection.execute(query).df()
    logger.info("[%s] extracted %s rows in %.1fs", name, f"{len(frame):,}",
                time.time() - t0)
    if len(frame) != frame.event_fk.nunique():
        raise RuntimeError(f"{name}: duplicate event_fk in extraction")
    return frame


def extract_all(connection: duckdb.DuckDBPyConnection, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name, spec in POPULATIONS.items():
        path = cache / f"{name}.parquet"
        if path.exists():
            logger.info("[%s] cache hit: %s", name, path)
            frames[name] = pd.read_parquet(path)
            continue
        frame = extract(connection, name, spec)
        frame.reset_index(drop=True).to_parquet(path, index=False)
        frames[name] = frame
    return frames


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------

def _matrix(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "emb":
        return np.stack(frame.embedding.values).astype(np.float32)
    X = frame[SCALAR_COLS].to_numpy(dtype=np.float64)
    for column in LOG_COLS:
        j = SCALAR_COLS.index(column)
        X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def auc_event_cv(X: np.ndarray, y: np.ndarray, folds: int = 5) -> tuple[float, float]:
    """5-fold CV over events; the optimistic number (events within a part correlate)."""
    splits = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in splits.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=CAP_LOGISTIC_C, max_iter=2000)
        clf.fit(scaler.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(scaler.transform(X[te]))))
    return float(np.mean(aucs)), float(np.std(aucs))


def auc_part_split(X: np.ndarray, y: np.ndarray, parts: np.ndarray,
                   ) -> tuple[float, float]:
    """Part-disjoint split: half the parts of each side train, the other half tests.

    Both directions are averaged. This is the conservative number -- same-condition
    events never sit on both sides of the split.
    """
    rng = np.random.default_rng(SEED)
    train = np.zeros(len(y), dtype=bool)
    for label in (0, 1):
        unique = np.unique(parts[y == label])
        rng.shuffle(unique)
        keep = set(unique[: max(1, len(unique) // 2)].tolist())
        idx = np.where(y == label)[0]
        inside = np.array([p in keep for p in parts[idx]])
        train[idx[inside]] = True
    out = []
    for flip in (False, True):
        tr = ~train if flip else train
        te = train if flip else ~train
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=CAP_LOGISTIC_C, max_iter=2000)
        clf.fit(scaler.transform(X[tr]), y[tr])
        out.append(roc_auc_score(y[te], clf.decision_function(scaler.transform(X[te]))))
    if not out:
        return float("nan"), float("nan")
    return float(np.mean(out)), float(np.std(out))


def single_feature_auc(values: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, values))


def probe_band(mu: pd.DataFrame, exp: pd.DataFrame, tag: str) -> dict:
    y = np.concatenate([np.zeros(len(mu)), np.ones(len(exp))])
    parts = np.concatenate([mu.part_key.values, exp.part_key.values])
    rows = {"band": tag, "n_muatm": len(mu), "n_exp": len(exp)}

    for kind in ("emb", "scal"):
        X = np.vstack([_matrix(mu, kind), _matrix(exp, kind)])
        mean, std = auc_event_cv(X, y)
        pmean, pstd = auc_part_split(X, y, parts)
        rows[f"auc_{kind}_cv"] = round(mean, 4)
        rows[f"auc_{kind}_cv_std"] = round(std, 4)
        rows[f"auc_{kind}_part"] = round(pmean, 4)
        rows[f"auc_{kind}_part_std"] = round(pstd, 4)

    rows["auc_nsighits"] = round(single_feature_auc(
        np.concatenate([mu.n_sig_hits.values, exp.n_sig_hits.values]).astype(float), y), 4)
    rows["med_nsighits_muatm"] = float(np.median(mu.n_sig_hits))
    rows["med_nsighits_exp"] = float(np.median(exp.n_sig_hits))
    return rows


def probe_strata(mu: pd.DataFrame, exp: pd.DataFrame, tag: str) -> list[dict]:
    """Embeddings probe within terciles of n_sig_hits (pooled over both sides)."""
    hits = np.concatenate([mu.n_sig_hits.values, exp.n_sig_hits.values]).astype(float)
    edges = np.quantile(hits, [0, 1 / 3, 2 / 3, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for i in range(3):
        m = (hits >= edges[i]) & (hits < edges[i + 1])
        n_mu, n_exp = int((m[:len(mu)]).sum()), int((m[len(mu):]).sum())
        if n_mu < 200 or n_exp < 200:
            out.append({"band": tag, "stratum": i, "n_muatm": n_mu, "n_exp": n_exp,
                        "hits_lo": edges[i], "hits_hi": edges[i + 1],
                        "auc_emb_cv": np.nan})
            continue
        X = np.vstack([_matrix(mu[m[:len(mu)]], "emb"), _matrix(exp[m[len(mu):]], "emb")])
        y = np.concatenate([np.zeros(n_mu), np.ones(n_exp)])
        mean, std = auc_event_cv(X, y)
        out.append({"band": tag, "stratum": i, "n_muatm": n_mu, "n_exp": n_exp,
                    "hits_lo": round(float(edges[i]), 1),
                    "hits_hi": round(float(edges[i + 1]), 1),
                    "auc_emb_cv": round(mean, 4), "auc_emb_cv_std": round(std, 4)})
    return out


def probe_null(mu: pd.DataFrame) -> float:
    """mc-muon vs mc-muon: the noise floor of the probe at this sample size."""
    X = _matrix(mu, "emb")
    rng = np.random.default_rng(SEED)
    y = rng.permutation(np.concatenate([np.zeros(len(mu) // 2), np.ones(len(mu) - len(mu) // 2)]))
    mean, _ = auc_event_cv(X, y)
    return round(mean, 4)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="/tmp/da_band_probe_cache",
                        help="directory for the extracted parquet frames")
    args = parser.parse_args()

    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(HERE / "logs" / f"probe_{stamp}.log"),
                  logging.StreamHandler(sys.stdout)])

    logger.info("Attaching %s and %s", MC_DB.name, EXP_DB.name)
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    connection.execute(f"ATTACH '{MC_DB}' AS mc (READ_ONLY)")
    connection.execute(f"ATTACH '{EXP_DB}' AS exp (READ_ONLY)")

    frames = extract_all(connection, Path(args.cache))

    def by_band(frame: pd.DataFrame, band: str) -> pd.DataFrame:
        return frame[frame.band == band]

    exp_all = frames["exp"]
    band_rows, strata_rows = [], []

    for band, _, _ in BANDS:
        label = f"muatm vs exp | {band}"
        logger.info("probing %s", label)
        mu, ex = by_band(frames["mc_muatm"], band), by_band(exp_all, band)
        row = probe_band(mu, ex, band)
        row["auc_emb_null_muatm"] = probe_null(mu)
        band_rows.append(row)
        strata_rows += probe_strata(mu, ex, band)

    # Pooled tail
    logger.info("probing pooled tail (score >= 0.8)")
    tail = lambda f: f[f.band.isin(TAIL)]  # noqa: E731
    mu_t, exp_t = tail(frames["mc_muatm"]), tail(exp_all)
    row = probe_band(mu_t, exp_t, "tail_ge0.8")
    row["auc_emb_null_muatm"] = probe_null(mu_t)
    band_rows.append(row)
    strata_rows += probe_strata(mu_t, exp_t, "tail_ge0.8")

    # Sanity: the probe must find real structure between the MC classes
    logger.info("sanity: muatm vs nue2 in b3")
    sanity = probe_band(by_band(frames["mc_muatm"], "b3_0.5_0.8"),
                        by_band(frames["mc_nue2"], "b3_0.5_0.8"), "sanity_muatm_vs_nue2_b3")

    table = pd.DataFrame(band_rows)
    strata = pd.DataFrame(strata_rows)
    table.to_csv(HERE / "results" / f"band_probe_{stamp}.csv", index=False)
    strata.to_csv(HERE / "results" / f"band_probe_strata_{stamp}.csv", index=False)
    summary = {
        "timestamp": stamp,
        "checkpoint": BASE.name,
        "n_events": {k: int(len(v)) for k, v in frames.items()},
        "sanity": sanity,
        "bands": band_rows,
        "strata": strata_rows,
    }
    (HERE / "results" / f"band_probe_{stamp}.json").write_text(
        json.dumps(summary, indent=2, default=str))

    logger.info("\n%s", table.to_string(index=False))
    logger.info("\n%s", strata.to_string(index=False))
    logger.info("sanity (%s): %s", sanity["band"], {k: v for k, v in sanity.items() if k.startswith("auc")})
    logger.info("done; results in %s", HERE / "results")


if __name__ == "__main__":
    main()
