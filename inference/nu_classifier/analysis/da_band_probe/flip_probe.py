"""Does z-flip neutralize the domain cue the DANN actually trained against?

Why this exists. The FIXED_sn256 training log ends with `domain_loss` at 0.692 and
`domain_accuracy` ~0.51 -- the trained discriminator could not tell the domains
apart. Yet a fresh linear probe on the *stored* (unflipped) embeddings separates
MC-muon from exp-holdout at AUC 0.74-0.78 in every score band (see probe.py).

The reconciling hypothesis: the discriminator only ever saw z-flip-augmented
batches -- both domains mixed 50/50 in vertical orientation, so the orientation
coded part of the class/domain cue was neutralized *within the augmented batch*,
while the unflipped events (what inference sees) stayed separable.

This script tests it directly: read the hits of sampled events, encode them twice
-- as stored and with z -> -z -- and probe the domains in both representations.
If the flipped AUC collapses toward 0.5 while the unflipped stays ~0.75, the
invariance the DA achieved is flip-conditional, and a new adversarial objective
must be built on the unflipped distribution (or the flip mechanism revisited).

Byproduct: score(flipped) vs score(unflipped) per class and band -- how much the
model's verdict rides on the vertical direction at all.

The addressing chain is checked per chunk with `reader.verify()` (recomputes
n_sn_hits / n_sn_strings from the HDF5 and compares with the database); the
encoder path is checked against the stored embeddings on a subsample.

Usage:
    python inference/nu_classifier/analysis/da_band_probe/flip_probe.py [--n 2000]
Writes results/flip_probe_<stamp>.json and logs/flip_probe_<stamp>.log.
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

import duckdb
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

from inference import reader                                        # noqa: E402
from inference.shared.model_utils import (load_model,              # noqa: E402
                                          predict_scores_and_embeddings)

CHECKPOINT = ("260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256"
              "@best_da_model")
CHECKPOINT_PATH = ROOT / ("trained_models/classifier/"
                          "260816_2250_da_nu_classifier_exp_full_E1_lambda0.01_FIXED_sn256/"
                          "best_da_model.pth")
PREDS = ROOT / f"inference/nu_classifier/preds/{CHECKPOINT}"

#: (name, lo, hi) -- bulk / mid / tail; the flip question needs the contrast, and
#: the intermediate bands add little over probe.py's monotone gradient.
BANDS = [
    ("b0_lt0.05", 0.00, 0.05),
    ("b3_0.5_0.8", 0.50, 0.80),
    ("tail_ge0.8", 0.80, 1.01),
]
SEED = 42

logger = logging.getLogger("flip_probe")


def top_tail_parts(k: int) -> list[str]:
    """The `k` muatm parts with the most high-score false positives.

    The muon tail is sparse (0.05%); a plain stream finds ~3.6 tail events per
    part and scans hundreds of parts to fill a sample. These parts carry ~3x the
    average tail rate, so a part-filtered stream fills the sample in seconds.
    The tail arm is therefore drawn from part-enriched events -- irrelevant for
    the flip comparison, which is paired on identical events, but it does mean
    the absolute AUC level of the tail arm is not the population value.
    """
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    connection.execute(f"ATTACH '{PREDS / 'mc_merged_thr0p8.duckdb'}' AS mc (READ_ONLY)")
    parts = connection.execute(f"""
      SELECT s.part_key
      FROM mc.predictions p JOIN mc.splits s USING (event_fk)
      WHERE s.data_class='muatm_2020' AND NOT s.used_for_labels AND p.score >= 0.8
      GROUP BY 1 ORDER BY count(*) DESC LIMIT {k}""").df().part_key.tolist()
    connection.close()
    return parts


def collect(source: str, band: tuple, n: int,
            parts: list[str] | None = None) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Hits of up to `n` events of one domain inside one score band."""
    name, lo, hi = band
    if source == "exp_full":
        where = [reader.NOT_DA_TARGET, reader.NOT_EXCLUDED]
    else:
        where = [reader.NOT_TRAINED, "e.data_class = 'muatm_2020'"]
    if parts:
        listed = ", ".join(f"'{p}'" for p in parts)
        where.append(f"l.part_key IN ({listed})")
    where += [f"p.score >= {lo}", f"p.score < {hi}"]

    r = reader.open(source, checkpoint=CHECKPOINT)
    feats: list[np.ndarray] = []
    rows: list[dict] = []
    t0 = time.time()
    for chunk in r.stream(where=where, with_hits=True, budget_mb=128):
        if len(chunk.hits):
            r.verify(chunk)                       # addressing self-check, raises
        events = chunk.events
        sig = chunk.hits[chunk.hits["is_sig"]]
        grouped = {pos: sub for pos, sub in sig.groupby("event")}
        for pos in range(len(events)):
            sub = grouped.get(pos)
            if sub is None:                       # no sig hits -> not a scored event
                continue
            feats.append(sub[["q", "t", "x", "y", "z"]].to_numpy(np.float32))
            rows.append({"event_fk": int(events.event_fk.iloc[pos]),
                         "score": float(events.score.iloc[pos])})
            if len(feats) >= n:
                break
        logger.info("[%s %s] %d/%d events (%.0fs)", source, name, len(feats), n,
                    time.time() - t0)
        if len(feats) >= n:
            break
    return feats, pd.DataFrame(rows)


def encode(model, norm, feats: list[np.ndarray], flip: bool):
    if flip:
        flipped = []
        for f in feats:
            g = f.copy()
            g[:, 4] = -g[:, 4]
            flipped.append(g)
        feats = flipped
    scores, embs = predict_scores_and_embeddings(
        model, feats, norm, batch_size=512, device="cpu", feats_with_probs=False)
    return scores, embs


def auc_cv(X: np.ndarray, y: np.ndarray, folds: int = 5) -> tuple[float, float]:
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        s = StandardScaler().fit(X[tr])
        c = LogisticRegression(max_iter=3000).fit(s.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], c.decision_function(s.transform(X[te]))))
    return float(np.mean(aucs)), float(np.std(aucs))


def stored_embeddings(fks: list[int]) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute(f"ATTACH '{PREDS / 'mc_merged_thr0p8.duckdb'}' AS mc (READ_ONLY)")
    con.execute(f"ATTACH '{PREDS / 'exp_full_thr0p8.duckdb'}' AS exp (READ_ONLY)")
    con.register("want", pd.DataFrame({"event_fk": fks}))
    out = []
    for alias in ("mc", "exp"):
        out.append(con.execute(
            f"SELECT e.event_fk, e.embedding FROM {alias}.embeddings e "
            f"JOIN want w USING (event_fk)").df())
    return pd.concat(out, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1500,
                        help="events per band per domain (tail up to 2x)")
    parser.add_argument("--verify-stored", type=int, default=1500,
                        help="how many events to cross-check against stored embeddings")
    parser.add_argument("--mu-tail-parts", type=int, default=100,
                        help="parts to draw the muatm tail arm from (FP-enriched)")
    parser.add_argument("--cache", default="/tmp/da_band_probe_cache",
                        help="where to write the raw-array npz")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(HERE / "logs" / f"flip_probe_{stamp}.log"),
                  logging.StreamHandler(sys.stdout)])

    model, norm_config, _ = load_model(str(CHECKPOINT_PATH), device="cpu")

    tail_parts = top_tail_parts(args.mu_tail_parts)
    logger.info("muatm tail drawn from %d FP-enriched parts", len(tail_parts))

    data: dict[str, dict] = {}
    for domain, source in (("muatm", "mc_merged"), ("exp", "exp_full")):
        for band in BANDS:
            n = 2 * args.n if band[0] == "tail_ge0.8" else args.n
            parts = tail_parts if (domain == "muatm" and band[0] == "tail_ge0.8") else None
            feats, frame = collect(source, band, n, parts=parts)
            logger.info("[%s %s] collected %d events", domain, band[0], len(feats))
            score_u, emb_u = encode(model, norm_config, feats, flip=False)
            score_f, emb_f = encode(model, norm_config, feats, flip=True)
            data[domain] = data.get(domain, {})
            data[domain][band[0]] = {
                "feats_n": len(feats), "frame": frame,
                "score_u": score_u, "score_f": score_f,
                "emb_u": emb_u, "emb_f": emb_f}

    # --- validation: my unflipped encoding vs the stored one -------------------
    rng = np.random.default_rng(SEED)
    all_fks, all_emb, owner = [], [], []
    for domain in data:
        for band in data[domain]:
            d = data[domain][band]
            all_fks += d["frame"].event_fk.tolist()
            all_emb.append(d["emb_u"])
            owner += [(domain, band)] * len(d["frame"])
    mine = np.vstack(all_emb)
    take = rng.choice(len(all_fks), size=min(args.verify_stored, len(all_fks)),
                      replace=False)
    stored = stored_embeddings([all_fks[i] for i in take])
    merged = pd.DataFrame({"event_fk": [all_fks[i] for i in take],
                           "mine_idx": take}).merge(stored, on="event_fk")
    A = mine[merged.mine_idx.to_numpy()]
    B = np.stack(merged.embedding.values)
    cos = np.sum(A * B, axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
    check = {"n": len(merged), "cosine_median": float(np.median(cos)),
             "cosine_p01": float(np.percentile(cos, 1)),
             "cosine_min": float(cos.min())}

    # --- probes ---------------------------------------------------------------
    rows = []
    for band in [b[0] for b in BANDS]:
        mu, ex = data["muatm"][band], data["exp"][band]
        y = np.concatenate([np.zeros(mu["feats_n"]), np.ones(ex["feats_n"])])
        row = {"band": band, "n_muatm": mu["feats_n"], "n_exp": ex["feats_n"]}
        for tag, key in (("unflipped", "emb_u"), ("flipped", "emb_f")):
            X = np.vstack([mu[key], ex[key]])
            mean, std = auc_cv(X, y)
            row[f"auc_{tag}"] = round(mean, 4)
            row[f"auc_{tag}_std"] = round(std, 4)
        # The mixture the domain discriminator actually saw: both orientations of
        # both domains in one batch. Note: the trainer's source half was class
        # balanced (half neutrinos), which this muatm-only pool does not reproduce.
        X = np.vstack([mu["emb_u"], mu["emb_f"], ex["emb_u"], ex["emb_f"]])
        y_mixed = np.concatenate([np.zeros(2 * mu["feats_n"]),
                                  np.ones(2 * ex["feats_n"])])
        mean, std = auc_cv(X, y_mixed)
        row["auc_mixed"] = round(mean, 4)
        row["auc_mixed_std"] = round(std, 4)
        rows.append(row)

    # --- score behaviour under the flip --------------------------------------
    score_rows = []
    for domain in ("muatm", "exp"):
        for band in [b[0] for b in BANDS]:
            d = data[domain][band]
            su, sf = d["score_u"], d["score_f"]
            score_rows.append({
                "domain": domain, "band": band, "n": d["feats_n"],
                "med_score_unflipped": round(float(np.median(su)), 4),
                "med_score_flipped": round(float(np.median(sf)), 4),
                "frac_gt0.8_unflipped": round(float((su > 0.8).mean()), 4),
                "frac_gt0.8_flipped": round(float((sf > 0.8).mean()), 4),
            })

    table = pd.DataFrame(rows)
    scores = pd.DataFrame(score_rows)
    logger.info("\n%s", table.to_string(index=False))
    logger.info("\n%s", scores.to_string(index=False))
    logger.info("encoder-path check vs stored embeddings: %s", check)

    (HERE / "results" / f"flip_probe_{stamp}.json").write_text(json.dumps(
        {"timestamp": stamp, "n_per_band": args.n, "check_vs_stored": check,
         "probes": rows, "scores": score_rows}, indent=2, default=str))

    # Raw arrays, so no follow-up question needs another six-minute collection.
    arrays = {}
    for domain in data:
        for band in data[domain]:
            d = data[domain][band]
            for kind in ("emb_u", "emb_f", "score_u", "score_f"):
                arrays[f"{domain}|{band}|{kind}"] = d[kind]
            arrays[f"{domain}|{band}|fk"] = d["frame"].event_fk.to_numpy()
    np.savez_compressed(Path(args.cache) / f"flip_arrays_{stamp}.npz", **arrays)
    logger.info("raw arrays -> %s", Path(args.cache) / f"flip_arrays_{stamp}.npz")
    table.to_csv(HERE / "results" / f"flip_probe_{stamp}.csv", index=False)
    scores.to_csv(HERE / "results" / f"flip_probe_scores_{stamp}.csv", index=False)
    logger.info("done; results in %s", HERE / "results")


if __name__ == "__main__":
    main()
