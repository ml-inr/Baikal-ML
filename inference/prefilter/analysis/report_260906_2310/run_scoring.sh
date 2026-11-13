#!/usr/bin/env bash
# Scoring driver for report_260906_2310 — prefilter DA (lambda 0.05) vs control (lambda 0.0).
#
# The two runs differ only in domain_adaptation.lambda_scheduler.max_lambda; everything
# else in da_config.yaml (seed 42 included) matches, so the comparison is DA vs no-DA.
#
# Both models must see the SAME events. mc_merged was already scored for lambda 0.05 on a
# specific set of h5 parts, so the lambda 0.0 run is restricted to exactly those parts,
# read back from the existing predictions DB rather than re-derived from wrapper settings.
# exp_full is scored in full (all 29 parts, 133.3M events) by both.
#
# Usage (from project root):  bash inference/prefilter/analysis/report_260906_2310/run_scoring.sh

set -euo pipefail

# PyTorch's cuda:N does not follow nvidia-smi's numbering unless this is set before
# torch is imported. See memory note cuda_device_ordering.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DEVICE="cuda:0"                      # nvidia-smi 0 = RTX 5000 Ada

ROOT="trained_models/prefilter"
DA="$ROOT/da_prefilter_numu_260429_0155_Qclip_softlabels_PlateauLR_lambda0.05_MC2M_dmodel128_zflipBothMCExp_SoftFocalLoss/best_da_model.pth"
NODA="$ROOT/da_prefilter_numu_260428_2039_Qclip_softlabels_PlateauLR_lambda0.0_MC2M_dmodel128_zflipBothMCExp_SoftFocalLoss/best_da_model.pth"

PREDS="inference/prefilter/preds"
CATALOG="data_manager/catalog_v2.duckdb"
MC_H5="data_manager/data/h5datasets/baikal_mc_merged.h5"
EXP_FULL_H5="data_manager/data/h5datasets/exp_full.h5"
BATCH=2048
PARTS_DIR="inference/prefilter/analysis/report_260906_2310/scored_parts"

# ---------------------------------------------------------------------------
# 1. mc_merged for the lambda 0.0 control, on the parts lambda 0.05 already covers
# ---------------------------------------------------------------------------
mkdir -p "$PARTS_DIR"
python - "$PREDS" "$CATALOG" "$PARTS_DIR" <<'PY'
import glob, sys, duckdb
preds, catalog, out_dir = sys.argv[1:4]
db = glob.glob(f"{preds}/*lambda0.05*@best_da_model/mc_merged_allhits.duckdb")[0]
conn = duckdb.connect(db, read_only=True)
conn.execute(f"ATTACH '{catalog}' AS cat (READ_ONLY)")
rows = conn.execute("""
    SELECT e.data_class, e.run, count(*) AS n
    FROM predictions p JOIN cat.events e ON p.event_fk = e.id
    GROUP BY 1, 2
""").fetchall()
by_class: dict = {}
for data_class, run, n in rows:
    by_class.setdefault(data_class, []).append(run)
for data_class, parts in sorted(by_class.items()):
    with open(f"{out_dir}/{data_class}.txt", "w") as fh:
        fh.write(",".join(sorted(parts)))
    print(f"{data_class}: {len(parts)} parts")
PY

for PTYPE in muatm_2020 nuatm_2020 nue2_2020; do
    echo "=== mc_merged / lambda 0.0 / $PTYPE ==="
    python inference/prefilter/predict_mc.py \
        --checkpoint       "$NODA" \
        --mc-h5            "$MC_H5" \
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

# ---------------------------------------------------------------------------
# 2. exp_full for both models — all 29 parts, all 133.3M events
# ---------------------------------------------------------------------------
for CKPT in "$DA" "$NODA"; do
    echo "=== exp_full / $(basename "$(dirname "$CKPT")") ==="
    python inference/prefilter/predict_exp.py \
        --checkpoint  "$CKPT" \
        --exp-h5      "$EXP_FULL_H5" \
        --batch-size  "$BATCH" \
        --chunk-events 200000 \
        --device      "$DEVICE" \
        --output-dir  "$PREDS" \
        --catalog     "$CATALOG" \
        --skip-done-parts
done

echo "ALL DONE"
