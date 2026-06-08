"""Entry point for exp-background supervised fine-tuning.

Loads a YAML config and launches ExpFineTuningTrainer.

Usage (from project root):
    python inference_v2/nu_classifier/exp_finetuning/run_finetune.py \\
        --config inference_v2/nu_classifier/exp_finetuning/finetune_seed32.yaml \\
        [--debug]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--debug",  action="store_true",
                        help="Enable debug mode (fewer batches per epoch)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.debug:
        config.setdefault("debug", {})["enabled"] = True
        config["debug"]["max_batches_per_epoch"] = 5
        config["epochs"] = 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    from src.training.exp_finetuning_trainer import ExpFineTuningTrainer

    with ExpFineTuningTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main()
