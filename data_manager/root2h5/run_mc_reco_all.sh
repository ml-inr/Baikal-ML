#!/usr/bin/env bash
# Convert all MC reco particle types to HDF5 one by one.
# Run from data_manager/root2h5/:
#   bash run_mc_reco_all.sh

set -e
cd "$(dirname "$0")"

CONFIG="root2h5_config_mc_reco.yaml"
ROOT_BASE="/home/albert/Baikal2025/data_manager/data/mc_reco_root"

declare -A PARTICLES=(
    ["muatm"]="$ROOT_BASE/muatm_files"
    ["nue2"]="$ROOT_BASE/nue2_files"
    ["nuatm_conv"]="$ROOT_BASE/nuatm_conv_files"
    ["nuatm_prompt"]="$ROOT_BASE/nuatm_prompt_files"
)

for PARTICLE in muatm nue2 nuatm_conv nuatm_prompt; do
    ROOT_DIR="${PARTICLES[$PARTICLE]}"
    echo "========================================"
    echo "Starting: $PARTICLE"
    echo "Root dir: $ROOT_DIR"
    echo "========================================"

    # Patch particle and root_dir_path in-place, run, restore
    cp "$CONFIG" "${CONFIG}.bak"
    sed -i "s|^  particle:.*|  particle: \"$PARTICLE\"|" "$CONFIG"
    sed -i "s|^  root_dir_path:.*|  root_dir_path: \"$ROOT_DIR\"|" "$CONFIG"

    python root2h5_mc_reco.py 2>&1 | tee "mc_reco_${PARTICLE}.log"

    mv "${CONFIG}.bak" "$CONFIG"
    echo "Done: $PARTICLE"
    echo
done

echo "All particle types converted."
