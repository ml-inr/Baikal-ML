#!/bin/bash
# Quick TensorBoard launcher
# Usage: ./tb.sh [<задача>/<прогон>] [port]
#   ./tb.sh                                          # all experiments
#   ./tb.sh prefilter/da_prefilter_numu_260429_0155_...   # конкретный прогон
#   ./tb.sh classifier/260531_1720_... 6007              # свой порт

LOGDIR="trained_models/${1:-}/tensorboard"
PORT="${2:-6007}"

# If no experiment specified, point to all
if [ -z "$1" ]; then
    LOGDIR="trained_models"
fi

echo "TensorBoard: http://localhost:${PORT}"
echo "Logdir: ${LOGDIR}"
tensorboard --logdir="$LOGDIR" --port="$PORT" --bind_all
