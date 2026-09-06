#!/usr/bin/env bash
# Launch mc_reco catalog_v2 build via nohup.
# Logs to data_manager/catalog_v2/logs/mc_reco_build_<timestamp>.log
# Run from the project root: bash data_manager/catalog_v2/run_build_mc_reco.sh
#
# Optional: pass --particles to process a subset, e.g.:
#   bash data_manager/catalog_v2/run_build_mc_reco.sh --particles muatm nuatm_conv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/mc_reco_build_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

echo "Starting mc_reco catalog_v2 build"
echo "  Log : $LOG_FILE"
echo "  Args: $*"

nohup python -u -m data_manager.catalog_v2.build_mc_reco \
    --h5-path  data_manager/data/h5datasets/baikal_mc_reco.h5 \
    --catalog  data_manager/catalog_v2.duckdb \
    "$@" \
    > "$LOG_FILE" 2>&1 &

echo "  PID : $!"
echo "  Tail: tail -f $LOG_FILE"
