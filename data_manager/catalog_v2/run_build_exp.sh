#!/usr/bin/env bash
# Launch exp/exp_reco catalog_v2 build via nohup.
# Logs to data_manager/catalog_v2/logs/<source>_build_<timestamp>.log
#
# Usage (from project root):
#   bash data_manager/catalog_v2/run_build_exp.sh exp
#   bash data_manager/catalog_v2/run_build_exp.sh exp_reco

set -euo pipefail

SOURCE="${1:-}"
if [[ -z "$SOURCE" ]]; then
    echo "Usage: $0 <source>  (exp or exp_reco)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${SOURCE}_build_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

if [[ "$SOURCE" == "exp" ]]; then
    H5_PATH="data_manager/data/h5datasets/exp.h5"
elif [[ "$SOURCE" == "exp_reco" ]]; then
    H5_PATH="data_manager/data/h5datasets/exp_reco.h5"
elif [[ "$SOURCE" == "exp_full" ]]; then
    H5_PATH="data_manager/data/h5datasets/exp_full.h5"
else
    echo "Unknown source: $SOURCE (use 'exp', 'exp_reco' or 'exp_full')"
    exit 1
fi

echo "Starting catalog_v2 build: source=$SOURCE"
echo "  Log : $LOG_FILE"

nohup conda run -n baikal25 --no-capture-output python -u -m data_manager.catalog_v2.build_exp \
    --h5-path  "$H5_PATH" \
    --source   "$SOURCE" \
    --catalog  data_manager/catalog_v2.duckdb \
    > "$LOG_FILE" 2>&1 &

echo "  PID : $!"
echo "  Tail: tail -f $LOG_FILE"
