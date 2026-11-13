"""Verification and decomposition companion to probe.py.

Four checks, in order of what they protect against:

1. **What the stored embeddings are.** Apply the checkpoint's classifier head to the
   cached embeddings and compare with the stored scores: if the head reproduces the
   score exactly, the embeddings are precisely the features the classifier consumes
   and every probe below is measuring the right thing.

2. **That the probe can find real structure (honest sanity).** MC muatm vs MC nuatm on
   a *naturally sampled* pair (no band stratification -- stratifying by score destroys
   the class separation by construction, which is how an earlier version of this check
   produced a misleading 0.90). Expected: score AUC ~0.9999, embeddings probe ~1.00.

3. **What the domain signal is made of.** Logistic probes per feature group
   (counts / charge / geometry / sig-noise) and per single feature, per band.

4. **What differs between the domains event by event.** Medians of the main scalars
   per band, exp vs MC-muatm, with ratios.

Usage:
    python inference/nu_classifier/analysis/da_band_probe/verify.py [--cache DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / ("inference/nu_classifier/preds/"
               "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256"
               "@best_da_model")
CHECKPOINT = ROOT / ("trained_models/classifier/"
                     "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256/"
                     "best_da_model.pth")
TAIL = ["b4_0.8_0.9", "b5_0.9_0.95", "b6_gt0.95"]

GROUPS = {
    "counts": ["n_sig_hits", "n_sig_strings", "n_channels"],
    "charge": ["q_mean", "q_std", "q_total", "q_max", "q_frac_max"],
    "geometry": ["r_vert", "t_span", "t_span_core", "z_span", "xy_span",
                 "x_c", "y_c", "z_c"],
    "signoise": ["prob_mean", "prob_min"],
}
LOG_COLS = {"n_sig_hits", "n_sig_strings", "n_channels", "q_mean", "q_std", "q_total",
            "q_max", "q_frac_max", "r_vert", "t_span", "t_span_core", "z_span",
            "xy_span", "prob_mean"}
SINGLES = ["q_max", "q_total", "q_mean", "n_sig_hits", "xy_span", "z_c", "r_vert",
           "prob_mean"]
MEDIAN_COLS = ["n_sig_hits", "n_sig_strings", "n_channels", "q_mean", "q_total",
               "q_max", "prob_mean", "r_vert", "t_span", "z_c", "z_span", "xy_span"]
SEED = 42

logger = logging.getLogger("verify")


def prep(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].to_numpy(float).copy()
    for j, c in enumerate(cols):
        if c in LOG_COLS:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def auc_cv(X: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        s = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=2000).fit(s.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], c.decision_function(s.transform(X[te]))))
    return float(np.mean(aucs))


def check_embeddings_are_head_input(mu: pd.DataFrame, nu: pd.DataFrame) -> dict:
    mu_s = mu.sample(min(20000, len(mu)), random_state=0)
    nu_s = nu.sample(min(20000, len(nu)), random_state=0)
    E = np.vstack([np.stack(mu_s.embedding.values),
                   np.stack(nu_s.embedding.values)]).astype(np.float32)
    stored = np.concatenate([mu_s.score.values, nu_s.score.values])
    y = np.concatenate([np.zeros(len(mu_s)), np.ones(len(nu_s))])

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    head = {k.replace("classifier.", ""): v
            for k, v in ckpt["base_model_state_dict"].items()
            if k.startswith("classifier.")}
    clf = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.0),
                        nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.0), nn.Linear(16, 1))
    clf.load_state_dict(head)
    clf.eval()
    with torch.no_grad():
        rec = torch.sigmoid(clf(torch.from_numpy(E))).numpy().ravel()
    return {
        "corr_head_vs_stored_muatm": round(float(np.corrcoef(
            rec[:len(mu_s)], stored[:len(mu_s)])[0, 1]), 8),
        "max_abs_diff": round(float(np.abs(rec - stored).max()), 8),
        "auc_stored": round(roc_auc_score(y, stored), 6),
        "auc_head_reconstructed": round(roc_auc_score(y, rec), 6),
    }


def check_natural_sanity(connection: duckdb.DuckDBPyConnection) -> dict:
    def natural(cls: str, mod: int, n: int) -> pd.DataFrame:
        return connection.execute(f"""
          WITH p AS (SELECT event_fk, score FROM mc.predictions WHERE (event_fk % {mod}) = 0),
               s AS (SELECT event_fk, data_class, used_for_labels FROM mc.splits
                     WHERE (event_fk % {mod}) = 0)
          SELECT p.score, e.embedding
          FROM p JOIN s USING (event_fk) JOIN mc.embeddings e USING (event_fk)
          WHERE s.data_class = '{cls}' AND NOT s.used_for_labels
          LIMIT {n}""").df()

    mu, nu = natural("muatm_2020", 4096, 20000), natural("nuatm_2020", 1024, 20000)
    X = np.vstack([np.stack(mu.embedding.values),
                   np.stack(nu.embedding.values)]).astype(np.float32)
    y = np.concatenate([np.zeros(len(mu)), np.ones(len(nu))])
    s = StandardScaler().fit(X)
    c = LogisticRegression(max_iter=3000).fit(s.transform(X), y)
    return {
        "n_muatm": len(mu), "n_nuatm": len(nu),
        "auc_score": round(roc_auc_score(y, np.concatenate([mu.score, nu.score])), 6),
        "auc_embeddings_probe": round(roc_auc_score(y, c.decision_function(s.transform(X))), 6),
    }


def decompose(frames: dict) -> list[dict]:
    mu, ex = frames["mc_muatm"], frames["exp"]
    rows = []
    for band in ["b0_lt0.05", "b3_0.5_0.8", "tail_ge0.8"]:
        if band == "tail_ge0.8":
            m, e = mu[mu.band.isin(TAIL)], ex[ex.band.isin(TAIL)]
        else:
            m, e = mu[mu.band == band], ex[ex.band == band]
        y = np.concatenate([np.zeros(len(m)), np.ones(len(e))])
        row = {"band": band, "n_muatm": len(m), "n_exp": len(e)}
        for group, cols in GROUPS.items():
            row[f"auc_{group}"] = round(auc_cv(np.vstack([prep(m, cols), prep(e, cols)]), y), 4)
        for col in SINGLES:
            v = np.concatenate([prep(m, [col]).ravel(), prep(e, [col]).ravel()])
            row[f"auc_1_{col}"] = round(roc_auc_score(y, v), 4)
        rows.append(row)
    return rows


def scalar_medians(frames: dict) -> list[dict]:
    mu, ex = frames["mc_muatm"], frames["exp"]
    rows = []
    for band in ["b0_lt0.05", "b1_0.05_0.2", "b2_0.2_0.5", "b3_0.5_0.8",
                 "b4_0.8_0.9", "b5_0.9_0.95", "b6_gt0.95"]:
        m, e = mu[mu.band == band], ex[ex.band == band]
        row = {"band": band, "n_muatm": len(m), "n_exp": len(e)}
        for col in MEDIAN_COLS:
            row[f"muatm_{col}"] = round(float(np.median(m[col])), 3)
            row[f"exp_{col}"] = round(float(np.median(e[col])), 3)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="/tmp/da_band_probe_cache")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(HERE / "logs" / f"verify_{stamp}.log"),
                  logging.StreamHandler(sys.stdout)])

    cache = Path(args.cache)
    frames = {name: pd.read_parquet(cache / f"{name}.parquet")
              for name in ("mc_muatm", "mc_nuatm", "mc_nue2", "exp")}

    logger.info("1. embeddings == classifier head input?")
    head_check = check_embeddings_are_head_input(frames["mc_muatm"], frames["mc_nuatm"])
    logger.info("   %s", head_check)

    logger.info("2. honest sanity (natural sampling, muatm vs nuatm)")
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    connection.execute(f"ATTACH '{BASE / 'mc_merged_thr0p8.duckdb'}' AS mc (READ_ONLY)")
    sanity = check_natural_sanity(connection)
    logger.info("   %s", sanity)

    logger.info("3. feature-group decomposition")
    groups = pd.DataFrame(decompose(frames))
    logger.info("\n%s", groups.to_string(index=False))

    logger.info("4. scalar medians per band")
    medians = pd.DataFrame(scalar_medians(frames))
    logger.info("\n%s", medians.to_string(index=False))

    (HERE / "results" / f"verify_{stamp}.json").write_text(json.dumps(
        {"timestamp": stamp, "head_check": head_check, "sanity": sanity,
         "decomposition": decompose(frames), "medians": scalar_medians(frames)},
        indent=2, default=str))
    groups.to_csv(HERE / "results" / f"decomposition_{stamp}.csv", index=False)
    medians.to_csv(HERE / "results" / f"scalar_medians_{stamp}.csv", index=False)
    logger.info("done; results in %s", HERE / "results")


if __name__ == "__main__":
    main()
