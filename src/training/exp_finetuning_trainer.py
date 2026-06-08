"""Supervised fine-tuning trainer using pseudo-labeled exp background events.

Loads a pretrained nu-classifier checkpoint (DA or otherwise), then continues
supervised training using two data sources:
  - MC NPY dataset (label 0/1 from NPY arrays)
  - Exp BG NPY dataset (hard label 0 — low-score exp events, almost certainly muons)

No DANN.  Pure classification loss only.

Per-epoch MC resampling: at the start of each epoch a fresh random MC subset
of size N = len(exp_bg_train) is drawn from the full MC training pool.
This gives a 1:1 per-event exposure ratio by construction and avoids memorising
specific MC events.

Validation: combined MC val + exp BG val set, single val_loss / val_auc monitored
for early stopping.  If the model overfits to the exp BG training subset, val_loss
will rise because exp BG val events provide an independent held-out background signal.

Usage:
    python inference_v2/nu_classifier/exp_finetuning/run_finetune.py \\
        --config inference_v2/nu_classifier/exp_finetuning/finetune_config.yaml
"""

import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_models import create_model
from src.utils.training import set_reproducible_seeds
from src.data.nu_classifier_dataset import (
    NuClassifierNpyDataset,
    NuClassifierExpNpyDataset,
    create_nu_classifier_dataloader,
)
from src.training.metrics import (
    MetricsTracker,
    calculate_class_weights,
    focal_loss_with_logits_weighted,
    binary_cross_entropy_with_logits_weighted,
    soft_focal_loss_with_logits,
)

logger = logging.getLogger(__name__)


class ExpFineTuningTrainer:
    """Supervised fine-tuner: MC NPY + exp-background NPY, no DANN."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

        seed = config.get("seed", 42)
        set_reproducible_seeds(seed)

        self.normalization_config: Optional[Dict] = None
        self.base_model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None

        self.mc_train_ds:   Optional[NuClassifierNpyDataset]    = None
        self.bg_train_ds:   Optional[NuClassifierExpNpyDataset] = None
        self.val_loader:    Optional[DataLoader]                = None

        self._mc_train_full_indices: Optional[np.ndarray] = None

        self.current_epoch = 0
        es_mode = config.get("early_stopping", {}).get("mode", "min")
        self.best_metric = float("inf") if es_mode == "min" else float("-inf")
        self.best_epoch  = 0
        self.es_counter  = 0

        self.metrics_tracker = MetricsTracker(device=self.device)

        out_root = Path(config.get("output_dir", "inference_v2/nu_classifier/exp_finetuning/finetuned_models"))
        exp_name = config.get("experiment_name", "finetuned")
        self.output_dir = out_root / exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config.get("tensorboard", True):
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
        else:
            self.writer = None

        logger.info(f"ExpFineTuningTrainer on {self.device}")
        logger.info(f"Output dir: {self.output_dir}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    # -- Data preparation ---------------------------------------------------

    def prepare_data(self) -> None:
        cfg  = self.config
        seed = cfg.get("seed", 42)

        # MC source
        mc_ds = NuClassifierNpyDataset(
            npy_dir=cfg["mc_npy_dir"],
            max_hits=cfg.get("max_hits"),
            seed=seed,
        )
        mc_train_ds, mc_val_ds = mc_ds.split(cfg.get("mc_train_split", 0.9), seed=seed)
        self._mc_train_full_indices = mc_train_ds._indices.copy()
        self.mc_train_ds = mc_train_ds
        logger.info(
            f"MC: {len(mc_train_ds):,} train, {len(mc_val_ds):,} val "
            f"(total {len(mc_ds):,})"
        )

        # Exp BG source
        bg_ds = NuClassifierExpNpyDataset(
            npy_dir=cfg["exp_bg_npy_dir"],
            max_hits=cfg.get("max_hits"),
            seed=seed,
            prefix="exp_bg",
        )
        bg_train_ds, bg_val_ds = bg_ds.split(cfg.get("bg_train_split", 0.8), seed=seed)
        self.bg_train_ds = bg_train_ds
        logger.info(
            f"Exp BG: {len(bg_train_ds):,} train, {len(bg_val_ds):,} val "
            f"(total {len(bg_ds):,})"
        )

        # Class weights from MC train (used for BCE; focal doesn't need them)
        mc_labels = mc_train_ds.get_all_labels()
        binary    = torch.tensor((mc_labels > 0.5).astype(np.float32), device=self.device)
        self.class_weights = calculate_class_weights(binary)

        # Split MC train indices by label for exp:mc_mu:mc_nu = 1:1:2 epoch sampling
        self._mc_mu_pos = np.where(mc_labels < 0.5)[0]
        self._mc_nu_pos = np.where(mc_labels >= 0.5)[0]
        logger.info(
            f"MC train label split: {len(self._mc_mu_pos):,} mu (label 0), "
            f"{len(self._mc_nu_pos):,} nu (label 1)"
        )

        dl_cfg = cfg.get("dataloader", {})
        nw     = dl_cfg.get("num_workers", 0)
        pm     = dl_cfg.get("pin_memory", False)
        bs     = cfg.get("batch_size", 512)

        # Combined val DataLoader (MC val + exp BG val)
        val_combined = ConcatDataset([mc_val_ds, bg_val_ds])
        self.val_loader = create_nu_classifier_dataloader(
            val_combined, bs, shuffle=False,
            normalization_config=self.normalization_config,
            augmentation_config=None, shuffle_batch=False,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        logger.info(f"Val: {len(val_combined):,} events (MC + exp BG combined)")
        logger.info("Data preparation complete")

    # -- Model preparation --------------------------------------------------

    def prepare_model(self) -> None:
        ckpt_path = self.config["pretrained_checkpoint"]
        logger.info(f"Loading pretrained checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=str(self.device), weights_only=False)
        train_config = ckpt["config"]
        self._pretrained_config = train_config   # saved into finetuned checkpoints for load_model compat
        self.normalization_config = ckpt["normalization_config"]

        model_config = dict(train_config["model"])
        raw_clip = model_config.get("amp_clip")
        if raw_clip is not None and self.normalization_config is not None:
            mean_amp = self.normalization_config["means"][0]
            std_amp  = self.normalization_config["stds"][0]
            model_config["amp_clip"] = (raw_clip - mean_amp) / std_amp

        self.base_model = create_model(model_config)
        self.base_model.load_state_dict(ckpt["base_model_state_dict"])
        self.base_model.to(self.device)

        n_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        logger.info(
            f"Model loaded (epoch {ckpt.get('epoch', '?')}, "
            f"best_metric={ckpt.get('best_metric', '?'):.4f}), "
            f"{n_params:,} trainable params"
        )

        tcfg = self.config
        lr   = float(tcfg.get("learning_rate", 5e-5))
        wd   = float(tcfg.get("weight_decay", 0.01))
        self.optimizer = optim.AdamW(self.base_model.parameters(), lr=lr, weight_decay=wd)

        stype = tcfg.get("scheduler", "plateau")
        if stype == "plateau":
            sp = tcfg.get("scheduler_params", {})
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode      = sp.get("mode",    "min"),
                factor    = float(sp.get("factor",   0.5)),
                patience  = int(sp.get("patience", 3)),
                min_lr    = float(sp.get("min_lr",   1e-7)),
            )
        elif stype == "cosine":
            sp   = tcfg.get("scheduler_params", {})
            T    = int(sp.get("T_max", tcfg.get("epochs", 50)))
            eta  = float(sp.get("eta_min", 1e-7))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T, eta)

        logger.info("Model preparation complete")

    # -- Classification loss -----------------------------------------------

    def _classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        tcfg     = self.config
        pos_weight = None
        if tcfg.get("class_weights", True) and self.class_weights is not None:
            pos_weight = (self.class_weights[1] / self.class_weights[0]).to(self.device)

        loss_fn = tcfg.get("classification_loss", "focal")
        if loss_fn == "bce":
            return binary_cross_entropy_with_logits_weighted(logits, labels, pos_weight)
        elif loss_fn == "focal":
            gamma = float(tcfg.get("focal_gamma", 2.0))
            return focal_loss_with_logits_weighted(logits, labels, gamma, pos_weight)
        elif loss_fn == "soft_focal":
            gamma = float(tcfg.get("focal_gamma", 2.0))
            return soft_focal_loss_with_logits(logits, labels, gamma, pos_weight)
        raise ValueError(f"Unknown classification_loss: {loss_fn!r}")

    # -- Augmentation -------------------------------------------------------

    def _augment(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aug_cfg = self.config.get("dataloader", {}).get("augmentation")
        if aug_cfg is None:
            return features, mask

        features   = features.clone()
        batch_size = features.shape[0]

        if aug_cfg.get("rotation_enabled", False):
            angles = torch.rand(batch_size, device=self.device) * 2 * math.pi
            cos_a  = torch.cos(angles)
            sin_a  = torch.sin(angles)
            x      = features[:, :, 2].clone()
            y      = features[:, :, 3].clone()
            features[:, :, 2] = torch.where(mask, cos_a[:, None] * x - sin_a[:, None] * y, features[:, :, 2])
            features[:, :, 3] = torch.where(mask, sin_a[:, None] * x + cos_a[:, None] * y, features[:, :, 3])

        noise_std_list = aug_cfg.get("noise_std")
        if noise_std_list is not None:
            feature_dim = features.shape[2]
            if len(noise_std_list) != feature_dim:
                raise ValueError(
                    f"augmentation.noise_std has {len(noise_std_list)} entries "
                    f"but features have {feature_dim} dims"
                )
            noise_std = torch.tensor(noise_std_list, dtype=torch.float32, device=self.device)
            if self.normalization_config is not None:
                norm_stds = torch.tensor(
                    self.normalization_config["stds"], dtype=torch.float32, device=self.device
                )
                if len(norm_stds) != feature_dim:
                    raise ValueError(
                        f"normalization_config.stds has {len(norm_stds)} entries "
                        f"but features have {feature_dim} dims"
                    )
                noise_std = noise_std / norm_stds
            noise    = torch.randn_like(features) * noise_std
            features = torch.where(mask.unsqueeze(-1), features + noise, features)

            time_for_sort = features[:, :, 1].masked_fill(~mask, float("inf"))
            sort_idx      = time_for_sort.argsort(dim=1)
            sort_idx_exp  = sort_idx.unsqueeze(-1).expand(-1, -1, features.shape[2])
            features      = features.gather(dim=1, index=sort_idx_exp)
            mask          = mask.gather(dim=1, index=sort_idx)

        return features, mask

    # -- MC resample --------------------------------------------------------

    def _resample_mc_loader(self) -> DataLoader:
        """Return a DataLoader over a fresh MC epoch sample (N mu + 2N nu events).

        N = len(bg_train_ds).  Combined with bg_loader (N exp events) the per-epoch
        ratio is exp:mc_mu:mc_nu = 1:1:2.  Batch size is 3/4 of total batch_size so
        that mc and bg loaders produce the same number of batches per epoch.
        """
        n_bg = len(self.bg_train_ds)
        rng  = np.random.default_rng(self.current_epoch)

        replace_mu = len(self._mc_mu_pos) < n_bg
        replace_nu = len(self._mc_nu_pos) < 2 * n_bg
        if replace_mu:
            logger.warning(
                f"MC mu pool ({len(self._mc_mu_pos):,}) < n_bg ({n_bg:,}), sampling with replacement"
            )
        if replace_nu:
            logger.warning(
                f"MC nu pool ({len(self._mc_nu_pos):,}) < 2×n_bg ({2*n_bg:,}), sampling with replacement"
            )

        mu_sel = rng.choice(self._mc_mu_pos, size=n_bg,       replace=replace_mu)
        nu_sel = rng.choice(self._mc_nu_pos, size=2 * n_bg,   replace=replace_nu)
        combined_pos = np.concatenate([mu_sel, nu_sel])
        combined_pos.sort()
        self.mc_train_ds._indices = self._mc_train_full_indices[combined_pos]

        cfg    = self.config
        dl_cfg = cfg.get("dataloader", {})
        bs     = cfg.get("batch_size", 512)
        nw     = dl_cfg.get("num_workers", 0)
        pm     = dl_cfg.get("pin_memory", False)
        mc_bs  = max(1, 3 * bs // 4)

        return create_nu_classifier_dataloader(
            self.mc_train_ds, mc_bs, shuffle=True,
            normalization_config=self.normalization_config,
            augmentation_config=None, shuffle_batch=True,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )

    def _bg_train_loader(self) -> DataLoader:
        cfg    = self.config
        dl_cfg = cfg.get("dataloader", {})
        bs     = cfg.get("batch_size", 512)
        nw     = dl_cfg.get("num_workers", 0)
        pm     = dl_cfg.get("pin_memory", False)
        bg_bs  = max(1, bs // 4)

        return create_nu_classifier_dataloader(
            self.bg_train_ds, bg_bs, shuffle=True,
            normalization_config=self.normalization_config,
            augmentation_config=None, shuffle_batch=True,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )

    # -- Batch alignment helper --------------------------------------------

    @staticmethod
    def _align_seq_lengths(
        feat_a: torch.Tensor, mask_a: torch.Tensor,
        feat_b: torch.Tensor, mask_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad the shorter (B, L, F)/(B, L) pair so both have the same L."""
        la, lb = feat_a.shape[1], feat_b.shape[1]
        if la == lb:
            return feat_a, mask_a, feat_b, mask_b
        if la < lb:
            diff = lb - la
            feat_a = torch.cat([feat_a, feat_a.new_zeros(feat_a.shape[0], diff, feat_a.shape[2])], dim=1)
            mask_a = torch.cat([mask_a, mask_a.new_zeros(mask_a.shape[0], diff)], dim=1)
        else:
            diff = la - lb
            feat_b = torch.cat([feat_b, feat_b.new_zeros(feat_b.shape[0], diff, feat_b.shape[2])], dim=1)
            mask_b = torch.cat([mask_b, mask_b.new_zeros(mask_b.shape[0], diff)], dim=1)
        return feat_a, mask_a, feat_b, mask_b

    # -- Training epoch -----------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        self.base_model.train()
        self.metrics_tracker.reset_epoch()

        import itertools

        mc_loader  = self._resample_mc_loader()
        bg_loader  = self._bg_train_loader()

        n_mc = len(mc_loader)
        n_bg = len(bg_loader)
        total_batches = max(n_mc, n_bg)

        if n_mc >= n_bg:
            mc_iter = iter(mc_loader)
            bg_iter = itertools.cycle(bg_loader)
        else:
            mc_iter = itertools.cycle(mc_loader)
            bg_iter = iter(bg_loader)

        log_every = self.config.get("log_every", 50)
        running_loss = 0.0

        for batch_idx in range(total_batches):
            mc_batch = next(mc_iter)
            bg_batch = next(bg_iter)

            for b in (mc_batch, bg_batch):
                for k, v in b.items():
                    if isinstance(v, torch.Tensor):
                        b[k] = v.to(self.device)

            # Align sequence lengths (each batch is padded to its own max independently)
            mc_feat, mc_mask, bg_feat, bg_mask = self._align_seq_lengths(
                mc_batch["features"], mc_batch["mask"],
                bg_batch["features"], bg_batch["mask"],
            )

            # Combine MC + bg into one batch
            combined_features = torch.cat([mc_feat, bg_feat], dim=0)
            combined_lengths  = torch.cat([mc_batch["lengths"], bg_batch["lengths"]], dim=0)
            combined_mask     = torch.cat([mc_mask, bg_mask], dim=0)
            combined_labels   = torch.cat(
                [mc_batch["labels"].float(), bg_batch["labels"].float()], dim=0
            )

            combined_features, combined_mask = self._augment(combined_features, combined_mask)

            # Shuffle combined batch to avoid order effects
            perm     = torch.randperm(combined_features.shape[0], device=self.device)
            combined_features = combined_features[perm]
            combined_lengths  = combined_lengths[perm]
            combined_mask     = combined_mask[perm]
            combined_labels   = combined_labels[perm]

            batch_dict = {
                "features": combined_features,
                "lengths":  combined_lengths,
                "mask":     combined_mask,
            }
            logits = self.base_model(batch_dict)
            loss   = self._classification_loss(logits, combined_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.metrics_tracker.update_train(logits, combined_labels, loss)
            running_loss += loss.item()

            if batch_idx % log_every == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, "
                    f"Batch {batch_idx}/{total_batches}, "
                    f"Loss: {loss.item():.4f}"
                )

            if self.config.get("debug", {}).get("enabled", False):
                if batch_idx >= self.config["debug"].get("max_batches_per_epoch", 10):
                    break

        train_metrics = self.metrics_tracker.train_metrics.compute()
        train_metrics["loss_mean"] = running_loss / max(total_batches, 1)

        if self.writer:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, self.current_epoch)

        return train_metrics

    # -- Validation epoch ---------------------------------------------------

    def validate_epoch(self) -> Dict[str, float]:
        self.base_model.eval()
        self.metrics_tracker.reset_epoch()

        with torch.no_grad():
            for batch in self.val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                logits = self.base_model(batch)
                labels = batch["labels"].float()
                loss   = self._classification_loss(logits, labels)
                self.metrics_tracker.update_val(logits, labels, loss)

        val_metrics = self.metrics_tracker.val_metrics.compute()
        logger.info(
            f"  Val — loss={val_metrics.get('loss', 0):.4f}, "
            f"auc={val_metrics.get('auc', 0):.4f}, "
            f"f1={val_metrics.get('f1', 0):.4f}"
        )

        if self.writer:
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, self.current_epoch)

        return val_metrics

    # -- Checkpoint ---------------------------------------------------------

    def save_checkpoint(
        self, epoch: int, is_best: bool = False, history: Optional[list] = None,
    ) -> None:
        ckpt = {
            "epoch":                  epoch,
            "base_model_state_dict":  self.base_model.state_dict(),
            "optimizer_state_dict":   self.optimizer.state_dict(),
            "best_metric":            self.best_metric,
            "best_epoch":             self.best_epoch,
            "config":                 self.config,
            "pretrained_config":      self._pretrained_config,
            "normalization_config":   self.normalization_config,
            "training_history":       history[: epoch + 1] if history else [],
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(ckpt, self.output_dir / "latest_finetuned_checkpoint.pth")
        if self.config.get("save_epoch_checkpoints", False):
            torch.save(ckpt, self.output_dir / f"checkpoint_epoch_{epoch:04d}.pth")
        if is_best:
            torch.save(ckpt, self.output_dir / "best_finetuned_model.pth")
            logger.info(f"New best model saved at epoch {epoch}")

    # -- Early stopping -----------------------------------------------------

    def check_early_stopping(self, metric: float) -> bool:
        es = self.config.get("early_stopping", {})
        if not es.get("enabled", True):
            return False
        mode    = es.get("mode", "min")
        patience = es.get("patience", 8)
        delta   = es.get("min_delta", 1e-4)

        improved = (
            metric < self.best_metric - delta if mode == "min"
            else metric > self.best_metric + delta
        )
        if improved:
            self.best_metric = metric
            self.best_epoch  = self.current_epoch
            self.es_counter  = 0
            return False
        self.es_counter += 1
        if self.es_counter >= patience:
            logger.info(f"Early stopping after {patience} epochs without improvement")
            return True
        return False

    # -- Main training loop ------------------------------------------------

    def train(self) -> None:
        logger.info("Starting supervised fine-tuning …")
        self.prepare_data()
        self.prepare_model()

        # Update val_loader normalization after model load
        cfg    = self.config
        dl_cfg = cfg.get("dataloader", {})
        bs     = cfg.get("batch_size", 512)
        nw     = dl_cfg.get("num_workers", 0)
        pm     = dl_cfg.get("pin_memory", False)

        # Recreate val_loader now that normalization_config is known
        mc_ds  = NuClassifierNpyDataset(npy_dir=cfg["mc_npy_dir"],   max_hits=cfg.get("max_hits"), seed=cfg.get("seed", 42))
        bg_ds  = NuClassifierExpNpyDataset(npy_dir=cfg["exp_bg_npy_dir"], max_hits=cfg.get("max_hits"), seed=cfg.get("seed", 42), prefix="exp_bg")
        _, mc_val = mc_ds.split(cfg.get("mc_train_split", 0.9), seed=cfg.get("seed", 42))
        _, bg_val = bg_ds.split(cfg.get("bg_train_split", 0.8),  seed=cfg.get("seed", 42))
        val_combined = ConcatDataset([mc_val, bg_val])
        self.val_loader = create_nu_classifier_dataloader(
            val_combined, bs, shuffle=False,
            normalization_config=self.normalization_config,
            augmentation_config=None, shuffle_batch=False,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )

        if cfg.get("save_config", True):
            with open(self.output_dir / "finetune_config.yaml", "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            if self.normalization_config:
                with open(self.output_dir / "normalization_config.yaml", "w") as f:
                    yaml.dump(self.normalization_config, f, default_flow_style=False)

        num_epochs     = cfg.get("epochs", 50)
        validate_every = cfg.get("validate_every", 1)
        save_every     = cfg.get("save_every", 5)
        monitor        = cfg.get("early_stopping", {}).get("monitor", "loss")
        es_mode        = cfg.get("early_stopping", {}).get("mode", "min")
        history: list  = []

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch()

            val_metrics: Dict[str, float] = {}
            if epoch % validate_every == 0:
                val_metrics = self.validate_epoch()

            if self.scheduler is not None and val_metrics:
                plateau_metric = val_metrics.get(monitor, train_metrics.get(monitor, 0))
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(plateau_metric)
                else:
                    self.scheduler.step()

            if self.writer:
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch
                )

            epoch_entry = {
                "epoch":       epoch,
                "lr":          self.optimizer.param_groups[0]["lr"],
                "epoch_time":  time.time() - t0,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(epoch_entry)
            self._save_history_csv(history)

            current_metric = val_metrics.get(monitor, float("inf"))
            is_best = (
                current_metric < self.best_metric if es_mode == "min"
                else current_metric > self.best_metric
            )

            logger.info(
                f"Epoch {epoch}/{num_epochs-1} [{time.time()-t0:.1f}s] — "
                f"train_loss={train_metrics.get('loss', 0):.4f}, "
                f"val_loss={val_metrics.get('loss', 0):.4f}, "
                f"val_auc={val_metrics.get('auc', 0):.4f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best, history)

            if val_metrics and self.check_early_stopping(current_metric):
                logger.info(f"Training stopped early at epoch {epoch}")
                break

            if is_best:
                self.best_metric = current_metric
                self.best_epoch  = epoch

        self._save_history_csv(history)
        self.close()
        logger.info(
            f"Fine-tuning complete. Best {monitor}: {self.best_metric:.4f} "
            f"@ epoch {self.best_epoch}"
        )

    # -- Helpers ------------------------------------------------------------

    def _save_history_csv(self, history: list) -> None:
        try:
            import pandas as pd
            pd.DataFrame(history).to_csv(
                self.output_dir / "finetuning_history.csv", index=False,
            )
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
