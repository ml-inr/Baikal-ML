#!/usr/bin/env bash
# Launch TensorBoard to monitor training runs.
#
# Runs write logs to experiments/numu/<experiment_name>/tensorboard/, so pointing
# TensorBoard at experiments/numu picks up every run. Use the run-filter regex in
# the TB UI (top-left) to focus, e.g. "exp_full" or a specific experiment name.
#
# Usage:
#   bash tb.sh                 # logdir=experiments/numu, port=6006
#   bash tb.sh <logdir>        # custom logdir (e.g. a single run's dir)
#   bash tb.sh <logdir> <port>
#
# Remote access (server): on your laptop run
#   ssh -N -L 6006:localhost:6006 <user>@<this-host>
# then open http://localhost:6006

set -euo pipefail
cd "$(dirname "$0")"

LOGDIR="${1:-experiments/numu}"
PORT="${2:-6006}"

# Call the baikal25 env's tensorboard binary DIRECTLY. Do NOT use `conda run` —
# it interferes with TensorBoard's rustboard data-server subprocess (gRPC over a
# local socket), which leaves the dashboards empty even though the server binds.
TB_BIN="/home/albert/miniconda3/envs/baikal25/bin/tensorboard"
[ -x "$TB_BIN" ] || TB_BIN="tensorboard"   # fallback if env moved / already active

echo "TensorBoard"
echo "  logdir : $LOGDIR"
echo "  url    : http://localhost:$PORT"
echo "  (Ctrl-C to stop; filter runs by regex in the UI, e.g. 'exp_full')"

exec "$TB_BIN" --logdir "$LOGDIR" --port "$PORT"
