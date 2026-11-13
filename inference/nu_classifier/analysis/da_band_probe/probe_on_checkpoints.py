"""Run the invariance probe over a run's epoch checkpoints, building its trajectory.

`probe_auc_*` grows while a model learns (the features get more discriminative, and
the domain signal comes with them), so a single number -- the final checkpoint -- is
not comparable to a young run's. This script applies the trainer's probe (same fixed
sample, same normalizer, same folds and seed) to every saved checkpoint of a run, so
two runs can be compared at matched epochs.

Usage:
    python inference/nu_classifier/analysis/da_band_probe/probe_on_checkpoints.py \
        --run-dir trained_models/classifier/260816_2250_..._FIXED_sn256 \
        --out results/probe_trajectory_FIXED.csv
"""

from __future__ import annotations

import argparse
import glob
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from inference.shared.model_utils import load_model                  # noqa: E402

logger = logging.getLogger("probe_on_checkpoints")


def encode(model, norm, hits, offsets) -> np.ndarray:
    means = torch.tensor(norm["means"], dtype=torch.float32)
    stds = torch.tensor(norm["stds"], dtype=torch.float32)
    model.eval()
    chunks, n_ev = [], len(offsets) - 1
    with torch.no_grad():
        for start in range(0, n_ev, 512):
            end = min(start + 512, n_ev)
            lengths = torch.tensor(offsets[start + 1:end + 1] - offsets[start:end],
                                   dtype=torch.long)
            max_len = int(lengths.max())
            padded = torch.zeros(len(lengths), max_len, hits.shape[1])
            for i, (lo, hi) in enumerate(zip(offsets[start:end], offsets[start + 1:end + 1])):
                padded[i, :hi - lo] = torch.from_numpy(hits[lo:hi])
            mask = torch.arange(max_len)[None, :] < lengths[:, None]
            padded = torch.where(mask.unsqueeze(-1), (padded - means) / (stds + 1e-8), padded)
            chunks.append(model.get_feature_representation(
                {"features": padded, "lengths": lengths, "mask": mask}).numpy())
    return np.concatenate(chunks)


def probe(emb, y, folds: int = 3) -> float:
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=42).split(emb, y):
        s = StandardScaler().fit(emb[tr])
        c = LogisticRegression(max_iter=2000).fit(s.transform(emb[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], c.decision_function(s.transform(emb[te]))))
    return float(np.mean(aucs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sample", default=str(
        HERE.parents[3] / "data_manager/probe_sample/probe_sample.npz"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    sample = np.load(args.sample)
    hits, offsets = sample["hits"], sample["offsets"]
    group, band = sample["group"], sample["band"]
    is_exp = ((group >= 2) & (group <= 3)).astype(int)  # 4,5 = nu, not exp

    rows = []
    for path in sorted(glob.glob(str(Path(args.run_dir) / "da_checkpoint_epoch_*.pth"))):
        epoch = int(re.search(r"epoch_(\d+)", path).group(1))
        model, norm, _ = load_model(path, device="cpu")
        emb = encode(model, norm, hits, offsets)
        rows.append({
            "epoch": epoch,
            "probe_auc_all": round(probe(emb, is_exp), 4),
            "probe_auc_bulk": round(probe(emb[band == 0], is_exp[band == 0]), 4),
            "probe_auc_tail": round(probe(emb[band == 2], is_exp[band == 2]), 4),
        })
        logger.info("epoch %2d: all=%.4f bulk=%.4f tail=%.4f", epoch,
                    rows[-1]["probe_auc_all"], rows[-1]["probe_auc_bulk"],
                    rows[-1]["probe_auc_tail"])
        del model

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    logger.info("saved %s", args.out)


if __name__ == "__main__":
    main()
