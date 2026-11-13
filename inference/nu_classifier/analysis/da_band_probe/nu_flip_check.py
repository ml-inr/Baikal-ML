"""Where does a z-flipped MC neutrino land: with the muons, with the data, or nowhere?

The working picture in the DA design debate (Albert, 2026-10-02):

  * in the *direction* readout a z-flipped neutrino looks like a muon / like the
    data (all down-going), so orientation stops being a domain cue;
  * the *brightness/energy* subspace is where the domains differ, and it is the
    "bias" the network should not be deciding on.

This script measures the picture. It collects *natural* samples (no score-band
stratification -- band sampling distorts the comparison) of MC muatm, nuatm,
nue2 and exp-holdout, encodes each as-is and with z -> -z, and reports:

  * the score (the model's direction readout) per set and orientation;
  * a cross-probe matrix: which pairs are linearly separable in the 128-d
    embedding, which is the operational version of "coincide".

Usage:
    python inference/nu_classifier/analysis/da_band_probe/nu_flip_check.py [--n 2500]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from inference import reader                                         # noqa: E402
from inference.shared.model_utils import (load_model,               # noqa: E402
                                          predict_scores_and_embeddings)

CHECKPOINT = ("260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256"
              "@best_da_model")
CHECKPOINT_PATH = ROOT / ("trained_models/classifier/"
                          "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256/"
                          "best_da_model.pth")
SEED = 42
logger = logging.getLogger("nu_flip_check")

#: name -> (source, extra where clause)
SETS = {
    "muatm": ("mc_merged", "e.data_class = 'muatm_2020'"),
    "nuatm": ("mc_merged", "e.data_class = 'nuatm_2020'"),
    "nue2":  ("mc_merged", "e.data_class = 'nue2_2020'"),
    "exp":   ("exp_full", None),
}


def collect(name: str, n: int):
    """Hits of `n` naturally-sampled events (no band stratification)."""
    source, extra = SETS[name]
    where = ([reader.NOT_TRAINED, extra] if source == "mc_merged"
             else [reader.NOT_DA_TARGET, reader.NOT_EXCLUDED])
    r = reader.open(source, checkpoint=CHECKPOINT)
    feats, q_total, q_max, stream_score = [], [], [], []
    t0 = time.time()
    for chunk in r.stream(where=where, with_hits=True, budget_mb=128):
        if len(chunk.hits):
            r.verify(chunk)
        sig = chunk.hits[chunk.hits["is_sig"]]
        grouped = {pos: sub for pos, sub in sig.groupby("event")}
        for pos in range(len(chunk.events)):
            sub = grouped.get(pos)
            if sub is None:
                continue
            q = sub["q"].to_numpy(np.float32)
            feats.append(sub[["q", "t", "x", "y", "z"]].to_numpy(np.float32))
            q_total.append(float(q.sum()))
            q_max.append(float(q.max()))
            stream_score.append(float(chunk.events.score.iloc[pos]))
            if len(feats) >= n:
                break
        if len(feats) >= n:
            break
    logger.info("[%s] collected %d events (%.0fs)", name, len(feats), time.time() - t0)
    return (feats, np.array(q_total, np.float32), np.array(q_max, np.float32),
            np.array(stream_score, np.float32))


def encode(model, norm, feats, flip: bool):
    if flip:
        feats = [np.column_stack([f[:, :4], -f[:, 4]]) for f in feats]
    scores, embs = predict_scores_and_embeddings(
        model, feats, norm, batch_size=512, device="cpu", feats_with_probs=False)
    return scores, embs


def auc_cv(A, B, folds: int = 5) -> float:
    X = np.vstack([A, B]).astype(np.float32)
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        s = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=3000).fit(s.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], c.decision_function(s.transform(X[te]))))
    return float(np.mean(aucs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2500, help="per set")
    parser.add_argument("--cache", default="/tmp/da_band_probe_cache")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(HERE / "logs" / f"nu_flip_check_{stamp}.log"),
                  logging.StreamHandler(sys.stdout)])

    model, norm, _ = load_model(str(CHECKPOINT_PATH), device="cpu")

    data = {}
    for name in SETS:
        feats, qt, qm, sc = collect(name, args.n)
        su, eu = encode(model, norm, feats, flip=False)
        sf, ef = encode(model, norm, feats, flip=True)
        data[name] = dict(n=len(feats), score_u=su, score_f=sf, emb_u=eu, emb_f=ef,
                          q_total=qt, q_max=qm, stream_score=sc)
        logger.info("[%s] score median %.4f (as-is) -> %.4f (flipped)",
                    name, float(np.median(su)), float(np.median(sf)))

    # --- score table ---------------------------------------------------------
    rows = []
    for name, d in data.items():
        rows.append({
            "set": name, "n": d["n"],
            "score_med_u": round(float(np.median(d["score_u"])), 4),
            "score_med_f": round(float(np.median(d["score_f"])), 4),
            "frac_gt0.8_u": round(float((d["score_u"] > 0.8).mean()), 4),
            "frac_gt0.8_f": round(float((d["score_f"] > 0.8).mean()), 4),
            "q_total_med": round(float(np.median(d["q_total"])), 2),
            "q_max_med": round(float(np.median(d["q_max"])), 2),
            "n_sig_hits_med": round(float(np.median([len(f) for f in
                                                     [d["score_u"]]])), 0) if False else None,
        })
    table = pd.DataFrame(rows)

    # --- cross-probe matrix --------------------------------------------------
    names = list(data)
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                continue
            matrix.loc[a, b] = round(auc_cv(data[a]["emb_u"], data[b]["emb_u"]), 4)
    flipped = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                continue
            flipped.loc[a, b] = round(auc_cv(data[a]["emb_f"], data[b]["emb_f"]), 4)

    logger.info("\n%s", table.to_string(index=False))
    logger.info("\nAUC, both as-is (rows vs cols):\n%s", matrix.to_string())
    logger.info("\nAUC, both flipped:\n%s", flipped.to_string())

    (HERE / "results" / f"nu_flip_check_{stamp}.json").write_text(json.dumps(
        {"timestamp": stamp, "checkpoint": CHECKPOINT, "n_per_set": args.n,
         "scores": rows,
         "auc_as_is": matrix.where(pd.notna(matrix), None).to_dict(),
         "auc_flipped": flipped.where(pd.notna(flipped), None).to_dict()},
        indent=2, default=str))
    table.to_csv(HERE / "results" / f"nu_flip_check_{stamp}.csv", index=False)

    arrays = {}
    for name, d in data.items():
        for key in ("score_u", "score_f", "emb_u", "emb_f", "q_total", "q_max",
                    "stream_score"):
            arrays[f"{name}|{key}"] = d[key]
    np.savez_compressed(Path(args.cache) / f"natural_flip_arrays_{stamp}.npz", **arrays)
    logger.info("done; results in %s", HERE / "results")


if __name__ == "__main__":
    main()
