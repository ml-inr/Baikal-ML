#!/usr/bin/env bash
# Scoring for report_260907_0820 — prefilter DA (lambda 0.05) against the no-DA control.
#
# Rescores from scratch because the prediction schema gained four columns: n_sn_hits /
# n_sn_strings (sig-noise model at threshold 0.8) and n_gt_sig_hits / n_gt_sig_strings
# (MC truth, raw/labels != 0). Both are copied from the probabilities file, which is where
# the nu-classifier's own counts come from, so h8s3 means the same thing in both pipelines.
# The prefilter still scores ALL raw hits — the counts are analysis metadata, not a cut.
#
# Part selection: only parts the probabilities file covers, because a part without counts
# cannot carry an h8s3 cut. That coverage is 10,100 of the 20,004 muatm parts and all of
# nuatm and nue2. Training parts are excluded here rather than at analysis time so the
# scoring itself is out-of-training.
#
# Usage (from project root):  bash inference/prefilter/analysis/report_260907_0820/run_scoring.sh

set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID          # cuda:N follows nvidia-smi only with this
DEVICE="cuda:0"

HERE="inference/prefilter/analysis/report_260907_0820"
ROOT="trained_models/prefilter"
DA="$ROOT/da_prefilter_numu_260429_0155_Qclip_softlabels_PlateauLR_lambda0.05_MC2M_dmodel128_zflipBothMCExp_SoftFocalLoss/best_da_model.pth"
NODA="$ROOT/da_prefilter_numu_260428_2039_Qclip_softlabels_PlateauLR_lambda0.0_MC2M_dmodel128_zflipBothMCExp_SoftFocalLoss/best_da_model.pth"

PREDS="inference/prefilter/preds"
CATALOG="data_manager/catalog_v2.duckdb"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
EXP_H5="data_manager/data/h5datasets/exp_full.h5"
BATCH=2048
PARTS_DIR="$HERE/scored_parts"

# Events per class, chosen from the h8s3 survival rates measured on out-of-training parts
# (muatm 9.7% by sig-noise, nuatm 1.1%, nue2 25.9%): enough muatm left in the tail to
# resolve a 1e-3 working point, and enough nuatm to fill the theta and energy profiles.
MUATM_PARTS=2000
NUATM_PARTS=0        # 0 = every out-of-training part the probabilities file covers
NUE2_PARTS=0

mkdir -p "$PARTS_DIR"
python - "$MC_H5" "$PARTS_DIR" "$MUATM_PARTS" "$NUATM_PARTS" "$NUE2_PARTS" <<'PY'
"""Pick the parts to score: out-of-training, covered by the probabilities file."""
import json, sys
from pathlib import Path

import h5py

mc_h5, out_dir, *limits = sys.argv[1:]
limits = dict(zip(("muatm_2020", "nuatm_2020", "nue2_2020"), map(int, limits)))
probs = Path(mc_h5).with_name(
    f"{Path(mc_h5).stem}_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5")
listed = json.loads(Path(
    "data_manager/datasets/baikal_mc2020_prefilter/prefilter_h5_parts.json").read_text())
short = {"muatm_2020": "muatm", "nuatm_2020": "nuatm", "nue2_2020": "nue2"}

with h5py.File(probs, "r") as handle:
    for data_class, limit in limits.items():
        covered = set(handle[data_class]["n_sig_hits_0.8"].keys())
        trained = {f"part_{n}" for n in listed[short[data_class]]}
        usable = sorted(covered - trained, key=lambda p: int(p.split("_")[1]))
        chosen = usable[:limit] if limit else usable
        Path(f"{out_dir}/{data_class}.txt").write_text(",".join(chosen))
        print(f"{data_class}: {len(covered)} with probs, {len(trained)} trained, "
              f"{len(usable)} usable -> scoring {len(chosen)}")
PY

for PTYPE in muatm_2020 nuatm_2020 nue2_2020; do
    for CKPT in "$DA" "$NODA"; do
        echo "=== mc_merged / $PTYPE / $(basename "$(dirname "$CKPT")") ==="
        python inference/prefilter/predict_mc.py \
            --checkpoint       "$CKPT" \
            --mc-h5            "$MC_H5" \
            --probs-h5         auto \
            --source           mc_merged \
            --ptypes           "$PTYPE" \
            --parts            "$(cat "$PARTS_DIR/$PTYPE.txt")" \
            --batch-size       "$BATCH" \
            --min-batch-events 32768 \
            --device           "$DEVICE" \
            --output-dir       "$PREDS" \
            --catalog          "$CATALOG" \
            --skip-done-parts
    done
done

for CKPT in "$DA" "$NODA"; do
    echo "=== exp_full / $(basename "$(dirname "$CKPT")") ==="
    python inference/prefilter/predict_exp.py \
        --checkpoint   "$CKPT" \
        --exp-h5       "$EXP_H5" \
        --probs-h5     auto \
        --batch-size   "$BATCH" \
        --chunk-events 200000 \
        --device       "$DEVICE" \
        --output-dir   "$PREDS" \
        --catalog      "$CATALOG" \
        --skip-done-parts
done

echo "ALL DONE"
