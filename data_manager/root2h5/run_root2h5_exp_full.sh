#!/usr/bin/env bash
# Convert full-statistics exp ROOT files -> exp_full.h5 (cluster c01 excluded
# via config). Runs detached via nohup; logs to root2h5/logs/.
#
# Usage (from project root):
#   bash data_manager/root2h5/run_root2h5_exp_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/root2h5_config_exp_full.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/exp_full_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

echo "Converting exp_full (config: $CONFIG)"
echo "  Log : $LOG_FILE"

nohup conda run -n baikal25 --no-capture-output python -u \
    data_manager/root2h5/root2h5_exp_full.py --config "$CONFIG" \
    > "$LOG_FILE" 2>&1 &

echo "  PID : $!"
echo "  Tail: tail -f $LOG_FILE"
