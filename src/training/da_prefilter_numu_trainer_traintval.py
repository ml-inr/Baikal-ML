"""Domain Adaptation trainer for the prefilter neutrino classification task.

EXPERIMENT VARIANT: validate_epoch uses train loaders instead of val loaders,
to diagnose whether train/val domain loss gap is due to data distribution or
overfitting. Val loaders are still created but unused in validate_epoch.


Source domain: memory-mapped NPY arrays (MC, produced by prefilter_npy_ds_builder).
Target domain: in-memory exp.h5 (~300–650K events, fits in RAM).

Soft labels in [0, 1]; background defined as label < 0.5.

Usage:
    python src/training/da_prefilter_numu_trainer.py \
        --config experiments/da_prefilter_numu_baseline.yaml [--debug]
"""

import argparse
import logging
import math
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

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_models import create_model
from src.models.domain_discriminator import create_da_model
from src.data.prefilter_npy_dataset import (
    PrefilterNpyDataset,
    ExpDataset,
    create_prefilter_npy_dataloader,
)
from src.training.metrics import (
    MetricsTracker,
    calculate_class_weights,
    binary_cross_entropy_with_logits_weighted,
)

logger = logging.getLogger(__name__)


def set_reproducible_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PrefilterDomainAdaptationTrainer:
    """DANN trainer: NPY source (MC) + in-memory target (Exp).

    Key differences from ``DomainAdaptationTrainer`` (hcut variant):
    * Source is ``PrefilterNpyDataset`` (mmap .npy, zero-copy).
    * Target is ``ExpDataset`` (in-memory from exp.h5).
    * Labels are float32 soft labels — domain adaptation uses muon-only source (particle_type==0).
    * Signal-hit metrics use precomputed ``signal_hit_counts`` tensor.
    """

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device(
            config["data"].get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        seed = config["experiment"].get("seed", 42)
        set_reproducible_seeds(seed)

        if config["reproducibility"].get("deterministic", True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if config["data"].get("normalization", {}).get("enabled", True):
            self.normalization_config = self._load_normalization_config()
        else:
            self.normalization_config = None

        # Components (set by prepare_*)
        self.base_model = None
        self.da_model = None
        self.optimizer_feature = None
        self.optimizer_classifier = None
        self.optimizer_discriminator = None
        self.scheduler_feature = None
        self.scheduler_classifier = None
        self.scheduler_discriminator = None

        # Data loaders
        self.source_train_loader: Optional[DataLoader] = None
        self.source_val_loader: Optional[DataLoader] = None
        self.target_train_loader: Optional[DataLoader] = None
        self.target_val_loader: Optional[DataLoader] = None

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.early_stopping_counter = 0

        self.metrics_tracker = MetricsTracker(device=self.device)

        self.output_dir = (
            Path(config["logging"]["output_dir"]) / config["experiment"]["name"]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config["logging"].get("tensorboard", True):
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
        else:
            self.writer = None

        self.lambda_scheduler_config = config["domain_adaptation"].get(
            "lambda_scheduler", {}
        )

        logger.info(f"Initialized PrefilterDomainAdaptationTrainer on {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

    # -- Context manager --------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()

    # -- Data preparation -------------------------------------------------

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
        target_cfg = data_cfg["target_domain"]
        seed = self.config["experiment"].get("seed", 42)

        # --- Source domain: memory-mapped NPY (MC) ---
        source_ds = PrefilterNpyDataset(
            npy_dir=source_cfg["npy_dir"],
            max_hits=data_cfg.get("max_hits"),
            max_events=source_cfg.get("max_events"),
            seed=seed,
        )
        source_train_ds, source_val_ds = source_ds.split(
            source_cfg.get("train_split", 0.85), seed=seed
        )
        logger.info(
            f"Source (MC NPY): {len(source_train_ds):,} train, "
            f"{len(source_val_ds):,} val"
        )

        # --- Target domain: in-memory exp.h5 ---
        target_ds = ExpDataset(
            h5_path=target_cfg["h5_path"],
            max_events=target_cfg.get("max_events"),
            max_hits=data_cfg.get("max_hits"),
            seed=seed,
        )
        target_train_ds, target_val_ds = target_ds.split(
            target_cfg.get("train_split", 0.85), seed=seed
        )
        logger.info(
            f"Target (Exp H5): {len(target_train_ds):,} train, "
            f"{len(target_val_ds):,} val"
        )

        # Save target train event IDs used in this experiment
        ids_path = self.output_dir / "used_target_event_ids.txt"
        target_ids = target_ds.get_event_ids()
        ids_path.write_text("\n".join(target_ids))
        logger.info(f"Saved {len(target_ids):,} target train & val event IDs → {ids_path}")

        # Class weights from source training labels
        self.class_weights = self._calculate_class_weights(source_train_ds)

        # --- Data loaders ---
        src_bs = self.config["training"].get("source_batch_size", 512)
        tgt_bs = self.config["training"].get("target_batch_size", 256)
        dl_cfg = self.config.get("dataloader", {})
        nw = dl_cfg.get("num_workers", 0)
        pm = dl_cfg.get("pin_memory", False)
        aug = dl_cfg.get("augmentation")

        self.source_train_loader = create_prefilter_npy_dataloader(
            source_train_ds, src_bs, shuffle=True,
            normalization_config=self.normalization_config,
            augmentation_config=aug, shuffle_batch=True,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        self.source_val_loader = create_prefilter_npy_dataloader(
            source_val_ds, src_bs, shuffle=False,
            normalization_config=self.normalization_config,
            augmentation_config=None, shuffle_batch=False,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        self.target_train_loader = create_prefilter_npy_dataloader(
            target_train_ds, tgt_bs, shuffle=True,
            normalization_config=self.normalization_config,
            augmentation_config=aug, shuffle_batch=True,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        self.target_val_loader = create_prefilter_npy_dataloader(
            target_val_ds, tgt_bs, shuffle=False,
            normalization_config=self.normalization_config,
            augmentation_config=None, shuffle_batch=False,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        logger.info("Data preparation complete")

    def _calculate_class_weights(self, dataset) -> torch.Tensor:
        logger.info("Calculating class weights from source training data …")
        labels = dataset.get_all_labels()
        binary = torch.tensor((labels > 0.5).astype(np.float32), device=self.device)
        return calculate_class_weights(binary)

    # -- Model preparation ------------------------------------------------

    def prepare_model(self) -> None:
        logger.info("Preparing model …")
        self.base_model = create_model(self.config["model"])
        self.base_model.to(self.device)

        self.da_model = create_da_model(
            self.base_model, self.config["domain_adaptation"]
        )
        self.da_model.to(self.device)

        pc = self.da_model.count_parameters()
        logger.info(
            f"Model: base={pc['base_model']:,}, "
            f"discriminator={pc['domain_discriminator']:,}, "
            f"total={pc['total']:,}"
        )

        if self.config["logging"].get("save_model_summary", True):
            with open(self.output_dir / "model_summary.txt", "w") as f:
                f.write("=== Domain Adaptation Model ===\n\n")
                f.write(str(self.base_model))
                f.write(f"\n\nDomain Discriminator:\n")
                f.write(str(self.da_model.domain_discriminator))
                f.write(f"\n\nParameters:\n")
                for k, v in pc.items():
                    f.write(f"  {k}: {v:,}\n")

        tcfg = self.config["training"]
        self.optimizer_feature = self._create_optimizer(
            self.da_model.base_model.feature_extractor.parameters(),
            tcfg.get("feature_optimizer", tcfg),
        )
        self.optimizer_classifier = self._create_optimizer(
            self.da_model.base_model.classifier.parameters(),
            tcfg.get("classifier_optimizer", tcfg),
        )
        self.optimizer_discriminator = self._create_optimizer(
            self.da_model.domain_discriminator.parameters(),
            tcfg.get("discriminator_optimizer", tcfg),
        )
        self._create_schedulers(tcfg)
        logger.info("Model preparation complete")

    def _create_optimizer(self, params, cfg: Dict) -> optim.Optimizer:
        if cfg["optimizer"] == "adamw":
            return optim.AdamW(
                params, lr=cfg["learning_rate"],
                weight_decay=cfg.get("weight_decay", 0.01),
            )
        elif cfg["optimizer"] == "adam":
            return optim.Adam(
                params, lr=cfg["learning_rate"],
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    def _create_schedulers(self, tcfg: Dict) -> None:
        stype = tcfg.get("scheduler")
        if stype == "cosine":
            sp = tcfg.get("scheduler_params", {})
            T_max = int(sp.get("T_max", tcfg["epochs"]))
            eta_min = float(sp.get("eta_min", 1e-6))
            self.scheduler_feature = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_feature, T_max=T_max, eta_min=eta_min,
            )
            self.scheduler_classifier = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_classifier, T_max=T_max, eta_min=eta_min,
            )
            self.scheduler_discriminator = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_discriminator, T_max=T_max, eta_min=eta_min,
            )
        elif stype == "step":
            sp = tcfg.get("scheduler_params", {})
            ss = int(sp.get("step_size", 30))
            g = float(sp.get("gamma", 0.1))
            self.scheduler_feature = optim.lr_scheduler.StepLR(
                self.optimizer_feature, step_size=ss, gamma=g,
            )
            self.scheduler_classifier = optim.lr_scheduler.StepLR(
                self.optimizer_classifier, step_size=ss, gamma=g,
            )
            self.scheduler_discriminator = optim.lr_scheduler.StepLR(
                self.optimizer_discriminator, step_size=ss, gamma=g,
            )

    # -- Lambda scheduling ------------------------------------------------

    def _get_lambda_factor(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> float:
        sc = self.lambda_scheduler_config
        by = sc.get("schedule_by", "epoch")
        if by == "step" and step is not None and total_steps:
            p = step / total_steps
        else:
            p = (epoch or 0) / max(total_epochs or 1, 1)

        stype = sc.get("type", "constant")
        if stype == "progressive":
            max_l = sc.get("max_lambda", 1.0)
            gamma = sc.get("gamma", 10.0)
            return (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0) * max_l
        elif stype == "constant":
            return sc.get("lambda", 1.0)
        elif stype == "linear":
            return sc.get("start_lambda", 0.0) + (
                sc.get("end_lambda", 1.0) - sc.get("start_lambda", 0.0)
            ) * p
        return 1.0

    # -- Classification loss ----------------------------------------------

    def _calculate_classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        tcfg = self.config["training"]
        pos_weight = None
        if hasattr(self, "class_weights") and tcfg.get("class_weights") != "auto":
            if tcfg.get("class_weights") is not None:
                w = torch.tensor(tcfg["class_weights"], device=self.device)
            else:
                w = self.class_weights.to(self.device)
            pos_weight = w[1] / w[0]
        return binary_cross_entropy_with_logits_weighted(logits, labels, pos_weight)

    # -- Signal-hits subset metrics ---------------------------------------

    def _calculate_signal_hits_metrics(
        self,
        all_logits: List[torch.Tensor],
        all_labels: List[torch.Tensor],
        all_signal_hits: List[torch.Tensor],
    ) -> Dict[str, float]:
        """AUC for events with >8 and >16 signal hits."""
        metrics: Dict[str, float] = {}
        if not all_logits:
            return metrics
        try:
            logits = torch.cat(all_logits, dim=0)
            labels = (torch.cat(all_labels, dim=0).numpy() > 0.5).astype(int)
            probs = torch.sigmoid(logits).numpy().flatten()
            sig_hits = torch.cat(all_signal_hits, dim=0).numpy()

            from sklearn.metrics import roc_auc_score

            for threshold, tag in [(8, "8hits"), (16, "16hits")]:
                mask = sig_hits > threshold
                n = int(mask.sum())
                metrics[f"cls_events_{tag}"] = n
                if n > 0 and len(np.unique(labels[mask])) > 1:
                    metrics[f"cls_auc_{tag}"] = float(
                        roc_auc_score(labels[mask], probs[mask])
                    )
                else:
                    metrics[f"cls_auc_{tag}"] = 0.0
        except Exception as e:
            logger.warning(f"Failed to calc signal-hits metrics: {e}")
            for tag in ("8hits", "16hits"):
                metrics[f"cls_auc_{tag}"] = 0.0
                metrics[f"cls_events_{tag}"] = 0
        return metrics

    # -- Training epoch ---------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        self.da_model.train()
        self.metrics_tracker.reset_epoch()

        total_epochs = self.config["training"]["epochs"]
        total_steps = total_epochs * min(
            len(self.source_train_loader), len(self.target_train_loader)
        )
        log_every = self.config["logging"].get("log_every", 10)

        source_iter = iter(self.source_train_loader)
        target_iter = iter(self.target_train_loader)
        total_batches = min(
            len(self.source_train_loader), len(self.target_train_loader)
        )

        epoch_metrics = {
            "classification_loss": 0.0,
            "domain_loss": 0.0,
            "total_loss": 0.0,
            "domain_accuracy": 0.0,
            "lambda_factor": 0.0,
            "domain_utilization": 0.0,
            "source_background_avg": 0.0,
            "target_events_avg": 0.0,
            "domain_batch_size_avg": 0.0,
        }

        domain_predictions: List[float] = []
        domain_labels_true: List[float] = []

        all_source_logits: List[torch.Tensor] = []
        all_source_labels: List[torch.Tensor] = []
        all_source_signal_hits: List[torch.Tensor] = []

        for batch_idx in range(total_batches):
            try:
                source_batch = next(source_iter)
                target_batch = next(target_iter)
            except StopIteration:
                break

            # Move to device
            for b in (source_batch, target_batch):
                for k in b:
                    if isinstance(b[k], torch.Tensor):
                        b[k] = b[k].to(self.device)

            lambda_factor = self._get_lambda_factor(
                step=self.current_step, epoch=self.current_epoch,
                total_steps=total_steps, total_epochs=total_epochs,
            )
            self.da_model.set_lambda(lambda_factor)

            # === Classification forward (source, all events) ===
            source_class_logits, _ = self.da_model.classification_forward(
                source_batch
            )
            source_labels = source_batch["labels"].float()

            all_source_logits.append(source_class_logits.detach().cpu())
            all_source_labels.append(source_labels.detach().cpu())
            all_source_signal_hits.append(
                source_batch["signal_hit_counts"].detach().cpu()
            )

            classification_loss = self._calculate_classification_loss(
                source_class_logits, source_labels,
            )

            self.optimizer_feature.zero_grad()
            self.optimizer_classifier.zero_grad()
            self.optimizer_discriminator.zero_grad()

            classification_loss.backward()

            # === Domain adaptation (muon events only) ===
            # Filter by particle type (0=muatm), not soft label — low-signal
            # neutrino events (label < 0.5) must not enter domain training since
            # exp data contains no neutrinos.
            source_bg_mask = (source_batch["particle_types"] == 0).squeeze()
            n_source_bg = int(source_bg_mask.sum().item())
            n_target = target_batch["features"].shape[0]

            if n_source_bg == 0:
                # Rare: no background in this source batch — skip domain step
                epoch_metrics["classification_loss"] += classification_loss.item()
                self.optimizer_feature.step()
                self.optimizer_classifier.step()
                self.optimizer_discriminator.step()
                self.current_step += 1
                continue

            domain_bs = min(n_source_bg, n_target)

            # Sample balanced domain sub-batches
            src_bg_idx = torch.where(source_bg_mask)[0]
            if n_source_bg > domain_bs:
                sel_src = src_bg_idx[
                    torch.randperm(n_source_bg, device=self.device)[:domain_bs]
                ]
            else:
                sel_src = src_bg_idx

            if n_target > domain_bs:
                sel_tgt = torch.randperm(n_target, device=self.device)[:domain_bs]
            else:
                sel_tgt = torch.arange(n_target, device=self.device)

            src_dom_batch = {
                "features": source_batch["features"][sel_src],
                "lengths": source_batch["lengths"][sel_src],
                "mask": source_batch["mask"][sel_src],
            }
            tgt_dom_batch = {
                "features": target_batch["features"][sel_tgt],
                "lengths": target_batch["lengths"][sel_tgt],
                "mask": target_batch["mask"][sel_tgt],
            }

            src_dom_logits, _ = self.da_model.domain_forward(src_dom_batch)
            tgt_dom_logits, _ = self.da_model.domain_forward(tgt_dom_batch)

            src_dom_labels = torch.zeros(domain_bs, 1, device=self.device)
            tgt_dom_labels = torch.ones(domain_bs, 1, device=self.device)

            domain_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    src_dom_logits, src_dom_labels,
                )
                + nn.functional.binary_cross_entropy_with_logits(
                    tgt_dom_logits, tgt_dom_labels,
                )
            ) / 2

            # Normalize by utilization rate
            src_util = domain_bs / n_source_bg
            tgt_util = domain_bs / n_target
            avg_util = (src_util + tgt_util) / 2
            norm_domain_loss = domain_loss / avg_util

            norm_domain_loss.backward()

            # Single combined optimizer step
            self.optimizer_feature.step()
            self.optimizer_classifier.step()
            self.optimizer_discriminator.step()

            # Domain accuracy
            with torch.no_grad():
                src_correct = (torch.sigmoid(src_dom_logits) < 0.5).float().mean()
                tgt_correct = (torch.sigmoid(tgt_dom_logits) >= 0.5).float().mean()
                dom_acc = (src_correct + tgt_correct) / 2

                domain_predictions.extend(
                    torch.sigmoid(src_dom_logits).cpu().flatten().tolist()
                )
                domain_predictions.extend(
                    torch.sigmoid(tgt_dom_logits).cpu().flatten().tolist()
                )
                domain_labels_true.extend([0.0] * domain_bs)
                domain_labels_true.extend([1.0] * domain_bs)

            self.metrics_tracker.update_train(
                source_class_logits, source_labels, classification_loss,
            )

            epoch_metrics["classification_loss"] += classification_loss.item()
            epoch_metrics["domain_loss"] += norm_domain_loss.item()
            epoch_metrics["total_loss"] += (
                classification_loss.item() + norm_domain_loss.item()
            )
            epoch_metrics["domain_accuracy"] += dom_acc.item()
            epoch_metrics["lambda_factor"] = lambda_factor
            epoch_metrics["domain_utilization"] += avg_util
            epoch_metrics["source_background_avg"] += n_source_bg
            epoch_metrics["target_events_avg"] += n_target
            epoch_metrics["domain_batch_size_avg"] += domain_bs

            if batch_idx % log_every == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, "
                    f"Batch {batch_idx}/{total_batches}, "
                    f"Class Loss: {classification_loss.item():.4f}, "
                    f"Domain Loss: {norm_domain_loss.item():.4f}, "
                    f"Domain Acc: {dom_acc.item():.4f}, "
                    f"Lambda: {lambda_factor:.4f}"
                )

            self.current_step += 1

            if self.config["debug"].get("enabled", False):
                if batch_idx >= self.config["debug"].get(
                    "max_batches_per_epoch", 10
                ):
                    break

        # Average epoch metrics
        for key in [
            "classification_loss", "domain_loss", "total_loss",
            "domain_accuracy", "domain_utilization",
            "source_background_avg", "target_events_avg",
            "domain_batch_size_avg",
        ]:
            epoch_metrics[key] /= max(total_batches, 1)

        # Domain AUC
        epoch_metrics["domain_auc"] = 0.0
        if domain_predictions and len(set(domain_labels_true)) > 1:
            try:
                from sklearn.metrics import roc_auc_score

                epoch_metrics["domain_auc"] = roc_auc_score(
                    domain_labels_true, domain_predictions,
                )
            except Exception:
                pass

        # Signal-hits subset metrics
        sig_metrics = self._calculate_signal_hits_metrics(
            all_source_logits, all_source_labels, all_source_signal_hits,
        )
        train_metrics = self.metrics_tracker.train_metrics.compute()
        train_metrics.update(epoch_metrics)
        train_metrics.update(sig_metrics)

        if self.writer:
            self._log_tensorboard_train(train_metrics, self.current_epoch)

        return train_metrics

    # -- Validation epoch -------------------------------------------------

    def validate_epoch(self) -> Dict[str, float]:
        self.da_model.eval()

        val_domain_loss = 0.0
        val_domain_acc = 0.0
        val_domain_batches = 0
        val_domain_preds: List[float] = []
        val_domain_labels: List[float] = []

        all_val_logits: List[torch.Tensor] = []
        all_val_labels: List[torch.Tensor] = []
        all_val_signal_hits: List[torch.Tensor] = []

        with torch.no_grad():
            # EXPERIMENT: validate on train data to check train/val domain loss gap
            src_domain_logits: List[torch.Tensor] = []
            tgt_domain_logits: List[torch.Tensor] = []

            for batch in self.source_train_loader:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)

                logits, _ = self.da_model.classification_forward(batch)
                labels = batch["labels"].float()
                loss = self._calculate_classification_loss(logits, labels)
                self.metrics_tracker.update_val(logits, labels, loss)

                all_val_logits.append(logits.cpu())
                all_val_labels.append(labels.cpu())
                all_val_signal_hits.append(batch["signal_hit_counts"].cpu())

                # Collect muon domain logits in the same pass
                src_bg_mask = (batch["particle_types"] == 0).squeeze()
                if src_bg_mask.any():
                    src_idx = torch.where(src_bg_mask)[0]
                    src_dl, _ = self.da_model.domain_forward({
                        "features": batch["features"][src_idx],
                        "lengths": batch["lengths"][src_idx],
                        "mask": batch["mask"][src_idx],
                    })
                    src_domain_logits.append(src_dl.cpu())

            # Target domain pass (EXPERIMENT: use train loader)
            for tb in self.target_train_loader:
                for k in tb:
                    if isinstance(tb[k], torch.Tensor):
                        tb[k] = tb[k].to(self.device)
                tgt_dl, _ = self.da_model.domain_forward({
                    "features": tb["features"],
                    "lengths": tb["lengths"],
                    "mask": tb["mask"],
                })
                tgt_domain_logits.append(tgt_dl.cpu())

            n_src = sum(t.shape[0] for t in src_domain_logits)
            n_tgt = sum(t.shape[0] for t in tgt_domain_logits)
            if n_src > 0 and n_tgt > 0:
                src_logits_t = torch.cat(src_domain_logits)  # (n_src, 1)
                tgt_logits_t = torch.cat(tgt_domain_logits)  # (n_tgt, 1)

                # Subsample the larger set by random index selection
                n_bal = min(n_src, n_tgt)
                rng_bal = np.random.default_rng(self.current_epoch)
                if n_src > n_bal:
                    sel = rng_bal.choice(n_src, size=n_bal, replace=False)
                    src_logits_t = src_logits_t[sel]
                if n_tgt > n_bal:
                    sel = rng_bal.choice(n_tgt, size=n_bal, replace=False)
                    tgt_logits_t = tgt_logits_t[sel]

                val_domain_loss = (
                    nn.functional.binary_cross_entropy_with_logits(
                        src_logits_t, torch.zeros(n_bal, 1),
                    )
                    + nn.functional.binary_cross_entropy_with_logits(
                        tgt_logits_t, torch.ones(n_bal, 1),
                    )
                ).item() / 2
                src_preds_t = torch.sigmoid(src_logits_t).flatten()
                tgt_preds_t = torch.sigmoid(tgt_logits_t).flatten()
                val_domain_acc = (
                    (src_preds_t < 0.5).float().mean()
                    + (tgt_preds_t >= 0.5).float().mean()
                ).item() / 2
                val_domain_preds = src_preds_t.tolist() + tgt_preds_t.tolist()
                val_domain_labels = [0.0] * n_bal + [1.0] * n_bal
                val_domain_batches = 1

        val_metrics = self.metrics_tracker.val_metrics.compute()

        if val_domain_batches > 0:
            val_metrics["domain_loss"] = val_domain_loss
            val_metrics["domain_accuracy"] = val_domain_acc
        else:
            val_metrics["domain_loss"] = 0.0
            val_metrics["domain_accuracy"] = 0.5

        val_metrics["domain_auc"] = 0.0
        if val_domain_preds and len(set(val_domain_labels)) > 1:
            try:
                from sklearn.metrics import roc_auc_score

                val_metrics["domain_auc"] = roc_auc_score(
                    val_domain_labels, val_domain_preds,
                )
            except Exception:
                pass

        logger.info(
            f"  Val cls  — loss={val_metrics.get('loss', 0):.4f}, "
            f"auc={val_metrics.get('auc', 0):.4f}, "
            f"f1={val_metrics.get('f1', 0):.4f}"
        )
        logger.info(
            f"  Val dom  — loss={val_metrics['domain_loss']:.4f}, "
            f"acc={val_metrics['domain_accuracy']:.4f}, "
            f"auc={val_metrics['domain_auc']:.4f} "
            f"(src={n_src:,} mu, tgt={n_tgt:,} exp, balanced={min(n_src, n_tgt) if n_src > 0 and n_tgt > 0 else 0:,})"
        )

        val_sig = self._calculate_signal_hits_metrics(
            all_val_logits, all_val_labels, all_val_signal_hits,
        )
        for k, v in val_sig.items():
            val_metrics[f"val_{k}"] = v

        if self.writer:
            self._log_tensorboard_val(val_metrics, self.current_epoch)

        return val_metrics

    # -- Checkpoint -------------------------------------------------------

    def save_checkpoint(
        self, epoch: int, is_best: bool = False, history: Optional[list] = None,
    ) -> None:
        ckpt = {
            "epoch": epoch,
            "base_model_state_dict": self.da_model.base_model.state_dict(),
            "domain_discriminator_state_dict": (
                self.da_model.domain_discriminator.state_dict()
            ),
            "optimizer_feature_state_dict": self.optimizer_feature.state_dict(),
            "optimizer_classifier_state_dict": (
                self.optimizer_classifier.state_dict()
            ),
            "optimizer_discriminator_state_dict": (
                self.optimizer_discriminator.state_dict()
            ),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "config": self.config,
            "normalization_config": self.normalization_config,
            "training_history": history[: epoch + 1] if history else [],
        }
        if self.scheduler_feature is not None:
            ckpt["scheduler_feature_state_dict"] = (
                self.scheduler_feature.state_dict()
            )
            ckpt["scheduler_classifier_state_dict"] = (
                self.scheduler_classifier.state_dict()
            )
            ckpt["scheduler_discriminator_state_dict"] = (
                self.scheduler_discriminator.state_dict()
            )

        torch.save(
            ckpt, self.output_dir / f"da_checkpoint_epoch_{epoch:03d}.pth",
        )
        torch.save(ckpt, self.output_dir / "latest_da_checkpoint.pth")
        if is_best:
            torch.save(ckpt, self.output_dir / "best_da_model.pth")
            logger.info(f"New best model saved at epoch {epoch}")

    # -- Early stopping ---------------------------------------------------

    def check_early_stopping(self, metric: float) -> bool:
        es = self.config["training"]["early_stopping"]
        if not es.get("enabled", True):
            return False
        mode = es.get("mode", "max")
        patience = es.get("patience", 10)
        delta = es.get("min_delta", 0.001)

        improved = (
            metric > self.best_metric + delta
            if mode == "max"
            else metric < self.best_metric - delta
        )
        if improved:
            self.best_metric = metric
            self.best_epoch = self.current_epoch
            self.early_stopping_counter = 0
            return False
        self.early_stopping_counter += 1
        if self.early_stopping_counter >= patience:
            logger.info(
                f"Early stopping after {patience} epochs without improvement"
            )
            return True
        return False

    # -- Main training loop -----------------------------------------------

    def train(self) -> None:
        logger.info("Starting prefilter DA training …")
        self.prepare_data()
        self.prepare_model()

        if self.config["logging"].get("save_config", True):
            with open(self.output_dir / "da_config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            if self.normalization_config:
                with open(
                    self.output_dir / "normalization_config.yaml", "w"
                ) as f:
                    yaml.dump(
                        self.normalization_config, f, default_flow_style=False,
                    )

        num_epochs = self.config["training"]["epochs"]
        validate_every = self.config["training"].get("validate_every", 1)
        save_every = self.config["training"].get("save_every", 5)
        monitor = self.config["training"]["early_stopping"].get(
            "monitor", "val_auc"
        )
        history: list = []

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch()

            val_metrics: Dict[str, float] = {}
            if epoch % validate_every == 0:
                val_metrics = self.validate_epoch()

            # Scheduler step
            if self.scheduler_feature is not None:
                self.scheduler_feature.step()
                self.scheduler_classifier.step()
                self.scheduler_discriminator.step()

            if self.writer:
                self._log_tensorboard_lr(epoch)

            self.metrics_tracker.save_epoch_metrics(epoch)

            epoch_entry = {
                "epoch": epoch,
                "lr_feature": self.optimizer_feature.param_groups[0]["lr"],
                "lr_classifier": self.optimizer_classifier.param_groups[0]["lr"],
                "lr_discriminator": (
                    self.optimizer_discriminator.param_groups[0]["lr"]
                ),
                "epoch_time": time.time() - t0,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(epoch_entry)
            self._save_history_csv(history)

            # Logging
            logger.info(
                f"Epoch {epoch}/{num_epochs-1} [{time.time()-t0:.1f}s] — "
                f"cls_loss={train_metrics.get('classification_loss', 0):.4f}, "
                f"dom_loss={train_metrics.get('domain_loss', 0):.4f}, "
                f"auc={train_metrics.get('auc', 0):.4f}"
            )
            if val_metrics:
                logger.info(
                    f"  val — loss={val_metrics.get('loss', 0):.4f}, "
                    f"auc={val_metrics.get('auc', 0):.4f}, "
                    f"dom_auc={val_metrics.get('domain_auc', 0):.4f}"
                )

            # Best model check
            current_metric = val_metrics.get(
                monitor.replace("val_", ""),
                train_metrics.get(monitor, 0),
            )
            is_best = False
            if es_mode := self.config["training"]["early_stopping"].get(
                "mode", "max"
            ):
                if es_mode == "max":
                    is_best = current_metric > self.best_metric
                else:
                    is_best = current_metric < self.best_metric

            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best, history)

            if val_metrics and self.check_early_stopping(current_metric):
                logger.info(f"Training stopped early at epoch {epoch}")
                break

            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch

        self._save_results(history)
        self.close()
        logger.info(
            f"Training complete. Best {monitor}: {self.best_metric:.4f} "
            f"@ epoch {self.best_epoch}"
        )

    # -- Logging helpers --------------------------------------------------

    def _save_history_csv(self, history: list) -> None:
        try:
            import pandas as pd

            pd.DataFrame(history).to_csv(
                self.output_dir / "da_training_history.csv", index=False,
            )
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    def _save_results(self, history: list) -> None:
        self._save_history_csv(history)
        summary = {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "total_epochs": self.current_epoch + 1,
            "final_train": self.metrics_tracker.train_metrics.compute(),
            "final_val": self.metrics_tracker.val_metrics.compute(),
            "model_parameters": self.da_model.count_parameters(),
        }
        with open(self.output_dir / "da_training_summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

    def _log_tensorboard_train(self, m: Dict, epoch: int) -> None:
        w = self.writer
        for k, tb in [
            ("loss", "Train/Classification_Loss"),
            ("accuracy", "Train/Classification_Accuracy"),
            ("f1", "Train/Classification_F1"),
            ("auc", "Train/Classification_AUC"),
            ("classification_loss", "Train/DA_Classification_Loss"),
            ("domain_loss", "Train/DA_Domain_Loss"),
            ("total_loss", "Train/DA_Total_Loss"),
            ("domain_accuracy", "Train/DA_Domain_Accuracy"),
            ("domain_auc", "Train/DA_Domain_AUC"),
            ("lambda_factor", "Train/DA_Lambda_Factor"),
            ("domain_utilization", "Train/Domain_Utilization"),
            ("domain_batch_size_avg", "Train/Domain_Batch_Size"),
            ("cls_auc_8hits", "Train/Classification_AUC_8hits"),
            ("cls_auc_16hits", "Train/Classification_AUC_16hits"),
        ]:
            if k in m:
                w.add_scalar(tb, m[k], epoch)

    def _log_tensorboard_val(self, m: Dict, epoch: int) -> None:
        w = self.writer
        for k, tb in [
            ("loss", "Val/Classification_Loss"),
            ("accuracy", "Val/Classification_Accuracy"),
            ("f1", "Val/Classification_F1"),
            ("auc", "Val/Classification_AUC"),
            ("domain_loss", "Val/Domain_Loss"),
            ("domain_accuracy", "Val/Domain_Accuracy"),
            ("domain_auc", "Val/Domain_AUC"),
            ("val_cls_auc_8hits", "Val/Classification_AUC_8hits"),
            ("val_cls_auc_16hits", "Val/Classification_AUC_16hits"),
        ]:
            if k in m:
                w.add_scalar(tb, m[k], epoch)

    def _log_tensorboard_lr(self, epoch: int) -> None:
        w = self.writer
        w.add_scalar(
            "LearningRate/Feature_Extractor",
            self.optimizer_feature.param_groups[0]["lr"],
            epoch,
        )
        w.add_scalar(
            "LearningRate/Classifier",
            self.optimizer_classifier.param_groups[0]["lr"],
            epoch,
        )
        w.add_scalar(
            "LearningRate/Discriminator",
            self.optimizer_discriminator.param_groups[0]["lr"],
            epoch,
        )


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train prefilter DA model for neutrino detection",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.debug:
        config["debug"]["enabled"] = True
        config["training"]["epochs"] = 3
        logger.info("Debug mode enabled")

    logging.basicConfig(
        level=getattr(logging, config["logging"].get("level", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with PrefilterDomainAdaptationTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main()
