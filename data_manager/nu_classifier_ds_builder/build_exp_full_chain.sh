#!/usr/bin/env bash
# Wait for the exp_full probs precompute to finish, then run the fast NPY build.
set -uo pipefail
cd /home/albert/Baikal2025
PROBS_LOG=data_manager/data/h5datasets/predict_exp_full_probs.log
BUILD_LOG=data_manager/nu_classifier_ds_builder/build_exp_full_npy.log

echo "[chain] waiting for predict_exp_h5 to finish ..."
while pgrep -f predict_exp_h5 >/dev/null; do sleep 60; done

if ! grep -q "Done in" "$PROBS_LOG"; then
  echo "[chain] ABORT: probs precompute did not report 'Done in' — check $PROBS_LOG"
  exit 1
fi
echo "[chain] probs done; starting NPY build (reads probs, no SN) ..."
python data_manager/build_exp_nu_classifier_dataset.py \
    --config data_manager/nu_classifier_ds_builder/exp_full_config.yaml > "$BUILD_LOG" 2>&1
echo "[chain] NPY build exit=$?  (log: $BUILD_LOG)"
tail -5 "$BUILD_LOG"
