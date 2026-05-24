"""Build the nu-classifier NPY dataset from baikal_mc_merged.h5.

Runs the two-pass sig-noise filtering pipeline:
  Pass 1 — inference on all parts, collect per-event metadata
  Pass 2 — write prob-filtered hit features + per-event arrays to .npy files

All paths in the config are resolved relative to the project root
(the parent of this script's directory), regardless of cwd.

Usage (from data_manager/ directory):
    nohup python build_nu_classifier_dataset.py \
        --config nu_classifier_ds_builder/default_config.yaml \
        > nu_classifier_ds_builder/build.log 2>&1 &
    echo "PID: $!"

    # Follow progress:
    tail -f nu_classifier_ds_builder/build.log
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_manager.nu_classifier_ds_builder.build import build_nu_classifier_npy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="Path to YAML config (relative to data_manager/ or absolute)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Resolve config path relative to cwd, then switch to project root
    # so that all relative paths inside the config work correctly.
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    cfg_path = cfg_path.resolve()

    if not cfg_path.exists():
        logging.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    os.chdir(PROJECT_ROOT)
    logging.info(f"Working dir: {PROJECT_ROOT}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Ensure output dir exists before we start (so log redirection works too)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output dir: {out_dir.resolve()}")
    logging.info(f"Config:     {cfg_path}")

    build_nu_classifier_npy(cfg)


if __name__ == "__main__":
    main()
