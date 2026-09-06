#!/bin/bash
# Quick TensorBoard launcher
# Usage: ./tb.sh [experiment_name] [port]
#   ./tb.sh                                          # all experiments
#   ./tb.sh da_prefilter_numu_baseline               # specific experiment
#   ./tb.sh da_prefilter_numu_baseline 6007           # custom port

LOGDIR="experiments/numu/${1:-}/tensorboard"
PORT="${2:-6007}"

# If no experiment specified, point to all
if [ -z "$1" ]; then
    LOGDIR="experiments/numu"
fi

echo "TensorBoard: http://localhost:${PORT}"
echo "Logdir: ${LOGDIR}"
tensorboard --logdir="$LOGDIR" --port="$PORT" --bind_all
