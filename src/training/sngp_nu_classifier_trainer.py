"""SNGP trainer for the nu-classifier task (distance-aware / OOD-aware output).

Adapted from ``da_nu_classifier_trainer.py`` — the DANN machinery (domain
discriminator, GRL, lambda schedule, z-flip target stream) is removed. Training
is plain supervised classification on the MC source NPY dataset, but with:

  * a spectral-normed encoder (bi-Lipschitz -> embedding distance tracks input
    distance), and
  * an RFF-GP output head (``RandomFeatureGPHead``) whose Laplace covariance is
    accumulated over each epoch; at inference the mean-field logit
    ``logit/√(1+λ·var)`` shrinks OOD inputs toward score 0.5.

This is the principled fix for the exp neutrino-like excess (over-confident OOD
extrapolation): exp events far from the MC manifold get high predictive variance
and are flagged uncertain rather than confidently ν. The main DA trainer is
untouched.

Usage:
    python src/training/sngp_nu_classifier_trainer.py \
        --config experiments/sngp_nu_classifier_baseline.yaml [--debug]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_models import create_model
from src.utils.training import set_reproducible_seeds
from src.data.nu_classifier_dataset import (
    NuClassifierNpyDataset,
    create_nu_classifier_dataloader,
)
from src.training.metrics import (
    MetricsTracker,
    calculate_class_weights,
    binary_cross_entropy_with_logits_weighted,
    focal_loss_with_logits_weighted,
    soft_focal_loss_with_logits,
)

logger = logging.getLogger(__name__)


class NuClassifierSNGPTrainer:
    """Supervised SNGP trainer: MC source NPY, SN encoder + RFF-GP head."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device(
            config["data"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        set_reproducible_seeds(config["experiment"].get("seed", 42))
        if config.get("reproducibility", {}).get("deterministic", True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if config["data"].get("normalization", {}).get("enabled", True):
            self.normalization_config = self._load_normalization_config()
        else:
            self.normalization_config = None

        include_probs = config["data"].get("include_probs", False)
        if include_probs and self.normalization_config is not None:
            self.normalization_config_collate = {
                "means": self.normalization_config["means"] + [0.75],
                "stds":  self.normalization_config["stds"]  + [0.25],
            }
        else:
            self.normalization_config_collate = self.normalization_config

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader:   Optional[DataLoader] = None

        self.current_epoch = 0
        es_mode = config["training"]["early_stopping"]["mode"]
        self.best_metric_for_es = float("-inf") if es_mode == "max" else float("inf")
        self.best_epoch = 0
        self.early_stopping_counter = 0

        self.metrics_tracker = MetricsTracker(device=self.device)

        self.output_dir = Path(config["logging"]["output_dir"]) / config["experiment"]["name"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config["logging"].get("tensorboard", True):
            tb_dir = self.output_dir / "tensorboard"; tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
        else:
            self.writer = None

        logger.info(f"Initialized NuClassifierSNGPTrainer on {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if getattr(self, "writer", None) is not None:
            self.writer.close()

    # -- Data -------------------------------------------------------------

    def _load_normalization_config(self) -> Dict:
        stats_path = project_root / "data_manager" / "stats_dict" / "default_mc.yaml"
        if not stats_path.exists():
            logger.warning(f"Normalization stats not found: {stats_path}")
            return {"means": [0.0] * 5, "stds": [1.0] * 5}
        with open(stats_path) as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Loaded normalization config from {stats_path}")
        return cfg

    def prepare_data(self) -> None:
        logger.info("Preparing data loaders …")
        data_cfg = self.config["data"]
        source_cfg = data_cfg["source_domain"]
        seed = self.config["experiment"].get("seed", 42)
        include_probs = data_cfg.get("include_probs", False)

        source_ds = NuClassifierNpyDataset(
            npy_dir=source_cfg["npy_dir"],
            max_hits=data_cfg.get("max_hits"),
            max_events=source_cfg.get("max_events"),
            seed=seed,
            include_probs=include_probs,
        )
        train_ds, val_ds = source_ds.split(source_cfg.get("train_split", 0.9), seed=seed)
        logger.info(f"Source (MC NPY): {len(train_ds):,} train, {len(val_ds):,} val")

        self.class_weights = calculate_class_weights(
            torch.tensor((train_ds.get_all_labels() > 0.5).astype(np.float32), device=self.device)
        )

        bs = self.config["training"].get("batch_size", 512)
        dl = self.config.get("dataloader", {})
        nw, pm = dl.get("num_workers", 0), dl.get("pin_memory", False)
        aug = dl.get("augmentation")

        self.train_loader = create_nu_classifier_dataloader(
            train_ds, bs, shuffle=True,
            normalization_config=self.normalization_config_collate,
            augmentation_config=aug, shuffle_batch=True,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        self.val_loader = create_nu_classifier_dataloader(
            val_ds, bs, shuffle=False,
            normalization_config=self.normalization_config_collate,
            augmentation_config=None, shuffle_batch=False,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        logger.info("Data preparation complete")

    # -- Model ------------------------------------------------------------

    def prepare_model(self) -> None:
        logger.info("Preparing SNGP model …")
        model_config = dict(self.config["model"])
        model_config.setdefault("type", "numu_sngp")

        raw_clip = model_config.get("amp_clip")
        if raw_clip is not None and self.normalization_config is not None:
            m, s = self.normalization_config["means"][0], self.normalization_config["stds"][0]
            model_config["amp_clip"] = (raw_clip - m) / s
            logger.info(f"amp_clip: Q={raw_clip} PE → normalized={model_config['amp_clip']:.4f}")

        self.model = create_model(model_config).to(self.device)
        logger.info(f"Model parameters: {self.model.count_parameters():,}")

        # Optional warm-start from a pretrained encoder (e.g. an E1 checkpoint):
        # loads matching encoder weights, ignores the GP head.
        warm = self.config["training"].get("warm_start_checkpoint")
        if warm:
            self._warm_start_encoder(warm)

        if self.config["logging"].get("save_model_summary", True):
            with open(self.output_dir / "model_summary.txt", "w") as f:
                f.write("=== SNGP Model ===\n\n" + str(self.model))

        tcfg = self.config["training"]
        opt = tcfg.get("optimizer", "adamw")
        if opt == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=tcfg["learning_rate"],
                weight_decay=tcfg.get("weight_decay", 0.01),
            )
        elif opt == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=tcfg["learning_rate"],
                weight_decay=tcfg.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt}")
        self._create_scheduler(tcfg)
        logger.info("Model preparation complete")

    def _warm_start_encoder(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("base_model_state_dict", ckpt)
        enc = {k[len("feature_extractor."):]: v for k, v in sd.items()
               if k.startswith("feature_extractor.")}
        missing, unexpected = self.model.feature_extractor.load_state_dict(enc, strict=False)
        logger.info(
            f"Warm-started encoder from {ckpt_path} "
            f"({len(enc)} tensors, missing={len(missing)}, unexpected={len(unexpected)})"
        )

    def _create_scheduler(self, tcfg: Dict) -> None:
        stype = tcfg.get("scheduler")
        if stype == "cosine":
            sp = tcfg.get("scheduler_params", {})
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, int(sp.get("T_max", tcfg["epochs"])), float(sp.get("eta_min", 1e-6)),
            )
        elif stype == "step":
            sp = tcfg.get("scheduler_params", {})
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, int(sp.get("step_size", 30)), float(sp.get("gamma", 0.1)),
            )
        elif stype == "plateau":
            sp = tcfg.get("scheduler_params", {})
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=sp.get("mode", "min"), factor=float(sp.get("factor", 0.5)),
                patience=int(sp.get("patience", 5)), min_lr=float(sp.get("min_lr", 1e-6)),
            )
        else:
            self.scheduler = None

    # -- Loss -------------------------------------------------------------

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        tcfg = self.config["training"]
        pos_weight = None
        if hasattr(self, "class_weights") and tcfg.get("class_weights") is not None:
            if tcfg.get("class_weights") == "auto" or tcfg.get("class_weights") is True:
                w = self.class_weights.to(self.device)
            else:
                w = torch.tensor(tcfg["class_weights"], device=self.device)
            pos_weight = w[1] / w[0]
        loss_fn = tcfg.get("classification_loss", "focal")
        gamma = float(tcfg.get("focal_gamma", 2.0))
        if loss_fn == "bce":
            return binary_cross_entropy_with_logits_weighted(logits, labels, pos_weight)
        if loss_fn == "focal":
            return focal_loss_with_logits_weighted(logits, labels, gamma, pos_weight)
        if loss_fn == "soft_focal":
            return soft_focal_loss_with_logits(logits, labels, gamma, pos_weight)
        raise ValueError(f"Unknown classification_loss: '{loss_fn}'")

    # -- Train / val ------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.metrics_tracker.reset_epoch()
        head = self.model.classifier
        head.reset_precision()                 # fresh Laplace precision each epoch

        log_every = self.config["logging"].get("log_every", 50)
        total = len(self.train_loader)
        run_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            feats = self.model.encode(batch)
            logits = head(feats)               # raw logits in train mode
            labels = batch["labels"].float()
            loss = self._classification_loss(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            head.update_precision(feats.detach())   # accumulate Laplace precision

            self.metrics_tracker.update_train(logits.detach(), labels, loss)
            run_loss += loss.item()
            if batch_idx % log_every == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{total}, "
                            f"Loss {loss.item():.4f}")
            if self.config.get("debug", {}).get("enabled", False):
                if batch_idx >= self.config["debug"].get("max_batches_per_epoch", 10):
                    break

        head.update_covariance()               # invert precision -> covariance for eval

        train_metrics = self.metrics_tracker.train_metrics.compute()
        train_metrics["classification_loss"] = run_loss / max(total, 1)
        if self.writer:
            for k, tb in [("loss", "Train/Loss"), ("auc", "Train/AUC"),
                          ("f1", "Train/F1"), ("accuracy", "Train/Accuracy")]:
                if k in train_metrics:
                    self.writer.add_scalar(tb, train_metrics[k], self.current_epoch)
        return train_metrics

    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        head = self.model.classifier
        var_sum, var_n = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                feats = self.model.encode(batch)
                logits = head(feats)           # mean-field adjusted (eval)
                labels = batch["labels"].float()
                loss = self._classification_loss(logits, labels)
                self.metrics_tracker.update_val(logits, labels, loss)
                # monitor predictive variance (OOD signal) on MC val
                if bool(head.cov_valid.item()):
                    phi = head._phi(feats)
                    var = torch.einsum("bi,ij,bj->b", phi, head.covariance, phi).clamp_min(0.0)
                    var_sum += var.sum().item(); var_n += var.numel()

        val_metrics = self.metrics_tracker.val_metrics.compute()
        val_metrics["gp_var_mean"] = var_sum / max(var_n, 1)
        logger.info(f"  Val — loss={val_metrics.get('loss', 0):.4f}, "
                    f"auc={val_metrics.get('auc', 0):.4f}, f1={val_metrics.get('f1', 0):.4f}, "
                    f"gp_var={val_metrics['gp_var_mean']:.4f}")
        if self.writer:
            for k, tb in [("loss", "Val/Loss"), ("auc", "Val/AUC"), ("f1", "Val/F1"),
                          ("accuracy", "Val/Accuracy"), ("gp_var_mean", "Val/GP_Var")]:
                if k in val_metrics:
                    self.writer.add_scalar(tb, val_metrics[k], self.current_epoch)
        return val_metrics

    # -- Checkpoint / early stop -----------------------------------------

    def save_checkpoint(self, epoch: int, is_best: bool, history: Optional[list] = None) -> None:
        ckpt = {
            "epoch": epoch,
            "base_model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric_for_es,
            "best_epoch": self.best_epoch,
            "config": self.config,
            "normalization_config": self.normalization_config,
            "training_history": history[: epoch + 1] if history else [],
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(ckpt, self.output_dir / f"sngp_checkpoint_epoch_{epoch:03d}.pth")
        torch.save(ckpt, self.output_dir / "latest_sngp_checkpoint.pth")
        if is_best:
            torch.save(ckpt, self.output_dir / "best_sngp_model.pth")
            logger.info(f"New best model saved at epoch {epoch}")

    def check_early_stopping(self, metric: float) -> bool:
        es = self.config["training"]["early_stopping"]
        if not es.get("enabled", True):
            return False
        mode, patience, delta = es.get("mode", "max"), es.get("patience", 10), es.get("min_delta", 1e-3)
        improved = (metric > self.best_metric_for_es + delta if mode == "max"
                    else metric < self.best_metric_for_es - delta)
        if improved:
            self.best_metric_for_es = metric
            self.best_epoch = self.current_epoch
            self.early_stopping_counter = 0
            return False
        self.early_stopping_counter += 1
        if self.early_stopping_counter >= patience:
            logger.info(f"Early stopping after {patience} epochs without improvement")
            return True
        return False

    # -- Main loop --------------------------------------------------------

    def train(self) -> None:
        logger.info("Starting nu-classifier SNGP training …")
        self.prepare_data()
        self.prepare_model()

        if self.config["logging"].get("save_config", True):
            with open(self.output_dir / "sngp_config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            if self.normalization_config:
                with open(self.output_dir / "normalization_config.yaml", "w") as f:
                    yaml.dump(self.normalization_config, f, default_flow_style=False)

        num_epochs = self.config["training"]["epochs"]
        validate_every = self.config["training"]["validate_every"]
        save_every = self.config["training"]["save_every"]
        monitor = self.config["training"]["early_stopping"]["monitor"].replace("val_", "")
        es_mode = self.config["training"]["early_stopping"].get("mode", "max")
        history: list = []

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            t0 = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch() if epoch % validate_every == 0 else {}

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(monitor, train_metrics.get(monitor, 0)))
                else:
                    self.scheduler.step()

            self.metrics_tracker.save_epoch_metrics(epoch)
            history.append({
                "epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time": time.time() - t0, **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })
            self._save_history_csv(history)
            logger.info(f"Epoch {epoch}/{num_epochs-1} [{time.time()-t0:.1f}s] — "
                        f"loss={train_metrics.get('loss', 0):.4f}, auc={train_metrics.get('auc', 0):.4f}")

            if not val_metrics:
                continue
            current_metric = val_metrics[monitor]
            is_best = (current_metric > self.best_metric_for_es if es_mode == "max"
                       else current_metric < self.best_metric_for_es)
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best, history)
            if self.check_early_stopping(current_metric):
                logger.info(f"Training stopped early at epoch {epoch}")
                break

        self._save_results(history)
        self.close()
        logger.info(f"Training complete. Best {monitor}: {self.best_metric_for_es:.4f} "
                    f"@ epoch {self.best_epoch}")

    def _save_history_csv(self, history: list) -> None:
        try:
            import pandas as pd
            pd.DataFrame(history).to_csv(self.output_dir / "sngp_training_history.csv", index=False)
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    def _save_results(self, history: list) -> None:
        self._save_history_csv(history)
        summary = {
            "best_metric": self.best_metric_for_es, "best_epoch": self.best_epoch,
            "total_epochs": self.current_epoch + 1,
            "final_train": self.metrics_tracker.train_metrics.compute(),
            "final_val": self.metrics_tracker.val_metrics.compute(),
            "model_parameters": self.model.count_parameters(),
        }
        with open(self.output_dir / "sngp_training_summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SNGP nu-classifier (distance-aware output)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.debug:
        config.setdefault("debug", {})["enabled"] = True
        config["training"]["epochs"] = 3
        logger.info("Debug mode enabled")

    logging.basicConfig(
        level=getattr(logging, config["logging"].get("level", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    with NuClassifierSNGPTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main()
