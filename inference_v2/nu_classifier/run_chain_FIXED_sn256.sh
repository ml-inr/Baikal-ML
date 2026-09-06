#!/usr/bin/env bash
# Wait for the running MC scoring to finish, then score exp on the same GPU.
#
# Sequential on purpose: both jobs want cuda:0, and sharing it would slow each without
# finishing either sooner. Exp only starts if MC actually succeeded — a failed MC run
# followed by a successful exp run would leave the two sides out of step, which is the
# exact failure mode this whole rebuild exists to remove.

set -uo pipefail
cd /home/albert/Baikal2025

MC_LOG=inference_v2/nu_classifier/predict_mc_FIXED_sn256.log
EXP_LOG=inference_v2/nu_classifier/predict_exp_FIXED_sn256.log

echo "[chain] waiting for predict_mc to finish ..."
while pgrep -f "predict_mc.py --checkpoint .*FIXED_sn256" >/dev/null; do sleep 60; done

if ! grep -q "^Done:" "$MC_LOG"; then
    echo "[chain] ABORT: predict_mc did not report Done — check $MC_LOG"
    tail -20 "$MC_LOG"
    exit 1
fi
echo "[chain] MC finished:"
grep "^Done:" "$MC_LOG" | tail -3

echo "[chain] starting exp scoring ..."
bash inference_v2/nu_classifier/run_predict_exp_FIXED_sn256.sh > "$EXP_LOG" 2>&1
echo "[chain] exp exit=$?"
tail -5 "$EXP_LOG"
