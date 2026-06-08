"""Domain separability trainer — works for both nu-classifier and prefilter feature spaces.

Select the data source via ``data.source_type`` in the config:
  "nu_classifier"  (default) — NuClassifierNpyDataset + NuClassifierExpNpyDataset
  "prefilter"               — PrefilterNpyDataset     + ExpDataset (H5)

Two run modes (``data.control_mode`` in the config):

  control_mode: false  (default)
    Train encoder + discriminator to distinguish MC muatm (label=0) from
    real Exp data (label=1).
    val_domain_auc >> 0.5 → domains are separable; DANN has meaningful work.

  control_mode: true
    Randomly split MC muatm 50/50 into MC-A (label=0) and MC-B (label=1).
    Both subsets are fixed for all epochs (no per-epoch resampling).
    Expected result: val_domain_auc ≈ 0.5.
    Use as sanity check: if AUC stays near 0.5, the MC/Exp result is credible.

Usage:
    python src/training/domain_sep_trainer.py --config experiments/domain_sep_nu_classifier.yaml
    python src/training/domain_sep_trainer.py --config experiments/domain_sep_prefilter.yaml
    python src/training/domain_sep_trainer.py --config experiments/domain_sep_prefilter_control.yaml --debug
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_models import create_model
from src.models.domain_discriminator import create_da_model
from src.utils.training import set_reproducible_seeds
from src.data.nu_classifier_dataset import (
    NuClassifierNpyDataset,
    NuClassifierExpNpyDataset,
    create_nu_classifier_dataloader,
)
from src.data.prefilter_npy_dataset import (
    PrefilterNpyDataset,
    create_prefilter_npy_dataloader,
)
from src.data.prefilter_npy_dataset.exp_dataset import ExpDataset

logger = logging.getLogger(__name__)


class DomainSeparabilityTrainer:
    """Train encoder + domain discriminator to maximise MC muatm / Exp separability."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device(config["data"].get("device", "cpu"))

        seed = config.get("experiment", {}).get("seed", 42)
        set_reproducible_seeds(seed)

        self.da_model   = None
        self.optimizer  = None
        self.scheduler  = None
        self.normalization_config: Optional[Dict] = None

        self.source_type:  str  = "nu_classifier"
        self.control_mode: bool = False
        self.mc_train_ds:  Optional[NuClassifierNpyDataset] = None
        self.exp_train_ds: Optional[object] = None  # varies by source_type / control_mode
        self.val_mc_loader:  Optional[object] = None
        self.val_exp_loader: Optional[object] = None

        self._mc_train_full_indices: Optional[np.ndarray] = None

        self.current_epoch = 0
        es_mode = config.get("training", {}).get("early_stopping", {}).get("mode", "min")
        self.best_metric = float("inf") if es_mode == "min" else float("-inf")
        self.best_epoch  = 0
        self.es_counter  = 0

        log_cfg  = config.get("logging", {})
        out_root = Path(log_cfg.get("output_dir", "experiments/domain_sep"))
        exp_name = config.get("experiment", {}).get("name", "domain_sep")
        self.output_dir = out_root / exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if log_cfg.get("tensorboard", True):
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
        else:
            self.writer = None

        logger.info(f"DomainSeparabilityTrainer  device={self.device}")
        logger.info(f"Output dir: {self.output_dir}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    # -- Normalization config -----------------------------------------------

    def _load_normalization_config(self) -> Dict:
        stats_path = project_root / "data_manager" / "stats_dict" / "default_mc.yaml"
        if not stats_path.exists():
            logger.warning(f"Normalization stats not found at {stats_path}, using identity")
            return {"means": [0.0] * 5, "stds": [1.0] * 5}
        with open(stats_path) as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Loaded normalization config from {stats_path}")
        return cfg

    # -- Dataset / dataloader factories (dispatch on source_type) -------------

    def _make_mc_dataset(self, src_cfg: Dict, max_hits: Optional[int], seed: int):
        """Instantiate the MC source dataset for the configured source_type."""
        kw = dict(npy_dir=src_cfg["npy_dir"], max_hits=max_hits,
                  max_events=src_cfg.get("max_events"), seed=seed)
        if self.source_type == "prefilter":
            return PrefilterNpyDataset(**kw)
        return NuClassifierNpyDataset(**kw)

    def _make_exp_dataset(self, tgt_cfg: Dict, max_hits: Optional[int], seed: int):
        """Instantiate the Exp target dataset for the configured source_type."""
        if self.source_type == "prefilter":
            return ExpDataset(
                h5_path=tgt_cfg["h5_path"],
                max_events=tgt_cfg.get("max_events"),
                max_hits=max_hits,
                seed=seed,
            )
        return NuClassifierExpNpyDataset(
            npy_dir=tgt_cfg["npy_dir"],
            max_hits=max_hits,
            max_events=tgt_cfg.get("max_events"),
            seed=seed,
        )

    def _make_dataloader(self, dataset, batch_size: int, shuffle: bool,
                         augmentation_config=None):
        """Create a DataLoader using the correct factory for self.source_type."""
        dl_cfg = self.config.get("dataloader", {})
        kw = dict(
            normalization_config=self.normalization_config,
            augmentation_config=augmentation_config,
            shuffle_batch=shuffle,
            device=str(self.device),
            num_workers=dl_cfg.get("num_workers", 0),
            pin_memory=dl_cfg.get("pin_memory", False),
        )
        if self.source_type == "prefilter":
            return create_prefilter_npy_dataloader(dataset, batch_size, shuffle=shuffle, **kw)
        return create_nu_classifier_dataloader(dataset, batch_size, shuffle=shuffle, **kw)

    # -- Data preparation --------------------------------------------------

    def prepare_data(self) -> None:
        data_cfg = self.config["data"]
        src_cfg  = data_cfg["source_domain"]
        seed     = self.config.get("experiment", {}).get("seed", 42)
        max_hits = data_cfg.get("max_hits")

        self.source_type  = data_cfg.get("source_type", "nu_classifier")
        self.control_mode = data_cfg.get("control_mode", False)
        self.normalization_config = self._load_normalization_config()

        # MC muatm only — label convention is identical for both source types
        mc_ds = self._make_mc_dataset(src_cfg, max_hits, seed)
        mc_ds._indices = mc_ds._indices[mc_ds.get_all_labels() < 0.5]
        logger.info(f"MC muatm: {len(mc_ds):,} events after label=0 filter  "
                    f"[source_type={self.source_type}]")

        train_split = src_cfg.get("train_split", 0.9)

        if self.control_mode:
            # Split MC muatm 50/50: MC-A (label=0) vs MC-B (label=1).
            # Both subsets are fixed for all epochs (no per-epoch resampling).
            rng_split = np.random.default_rng(seed + 1000)
            all_muatm = mc_ds._indices.copy()
            rng_split.shuffle(all_muatm)
            half = len(all_muatm) // 2

            mc_ds._indices = all_muatm[:half]
            mc_a_train, mc_a_val = mc_ds.split(train_split, seed=seed)
            self._mc_train_full_indices = mc_a_train._indices.copy()
            self.mc_train_ds = mc_a_train
            logger.info(f"MC-A (label=0): {len(mc_a_train):,} train  /  {len(mc_a_val):,} val")

            mc_b_ds = self._make_mc_dataset(
                {**src_cfg, "max_events": None}, max_hits, seed
            )
            mc_b_ds._indices = all_muatm[half:]
            mc_b_train, mc_b_val = mc_b_ds.split(train_split, seed=seed)
            self.exp_train_ds = mc_b_train
            logger.info(f"MC-B (label=1): {len(mc_b_train):,} train  /  {len(mc_b_val):,} val")

            val_source_ds = mc_a_val
            val_target_ds = mc_b_val

        else:
            tgt_cfg = data_cfg["target_domain"]

            mc_train_ds, mc_val_ds = mc_ds.split(train_split, seed=seed)
            self._mc_train_full_indices = mc_train_ds._indices.copy()
            self.mc_train_ds = mc_train_ds
            logger.info(f"MC: {len(mc_train_ds):,} train  /  {len(mc_val_ds):,} val")

            exp_ds = self._make_exp_dataset(tgt_cfg, max_hits, seed)
            exp_train_ds, exp_val_ds = exp_ds.split(tgt_cfg.get("train_split", 0.8), seed=seed)
            self.exp_train_ds = exp_train_ds
            logger.info(f"Exp: {len(exp_train_ds):,} train  /  {len(exp_val_ds):,} val")

            val_source_ds = mc_val_ds
            val_target_ds = exp_val_ds

        bs = self.config["training"].get("batch_size", 512)
        self.val_mc_loader  = self._make_dataloader(val_source_ds, bs // 2, shuffle=False)
        self.val_exp_loader = self._make_dataloader(val_target_ds, bs // 2, shuffle=False)
        logger.info("Data preparation complete")

    # -- Model preparation -------------------------------------------------

    def prepare_model(self) -> None:
        model_cfg = dict(self.config["model"])
        disc_cfg  = model_cfg.pop("domain_discriminator", {})

        raw_clip = model_cfg.get("amp_clip")
        if raw_clip is not None and self.normalization_config is not None:
            mean_amp = self.normalization_config["means"][0]
            std_amp  = self.normalization_config["stds"][0]
            model_cfg = dict(model_cfg)
            model_cfg["amp_clip"] = (raw_clip - mean_amp) / std_amp

        base_model  = create_model(model_cfg)
        self.da_model = create_da_model(base_model, {"domain_discriminator": disc_cfg})
        self.da_model.to(self.device)

        pc = self.da_model.count_parameters()
        logger.info(
            f"Model  encoder={pc['base_model']:,}  discriminator={pc['domain_discriminator']:,} params"
        )

        # Optimise encoder + discriminator MLP only (classification head excluded)
        tcfg   = self.config["training"]
        lr     = float(tcfg.get("learning_rate", 3e-4))
        wd     = float(tcfg.get("weight_decay", 0.01))
        params = (
            list(self.da_model.base_model.feature_extractor.parameters())
            + list(self.da_model.domain_discriminator.discriminator.parameters())
        )
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)

        stype = tcfg.get("scheduler", "plateau")
        if stype == "plateau":
            sp = tcfg.get("scheduler_params", {})
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode     = sp.get("mode",     "min"),
                factor   = float(sp.get("factor",   0.5)),
                patience = int(sp.get("patience",  5)),
                min_lr   = float(sp.get("min_lr",   1e-6)),
            )
        logger.info("Model preparation complete")

    # -- MC resample -------------------------------------------------------

    def _resample_mc_loader(self):
        """Return an MC train loader.

        Normal mode: resample MC to len(exp_train_ds) each epoch.
        Control mode: use the fixed MC-A set every epoch (no resampling).
        """
        if self.control_mode:
            self.mc_train_ds._indices = self._mc_train_full_indices
        else:
            n_exp   = len(self.exp_train_ds)
            rng     = np.random.default_rng(self.current_epoch)
            replace = len(self._mc_train_full_indices) < n_exp
            if replace:
                logger.warning(
                    f"MC pool ({len(self._mc_train_full_indices):,}) < n_exp ({n_exp:,}), "
                    "sampling with replacement"
                )
            sel = rng.choice(len(self._mc_train_full_indices), size=n_exp, replace=replace)
            sel.sort()
            self.mc_train_ds._indices = self._mc_train_full_indices[sel]

        aug = self.config.get("dataloader", {}).get("augmentation")
        bs  = self.config["training"].get("batch_size", 512)
        return self._make_dataloader(self.mc_train_ds, bs // 2, shuffle=True,
                                     augmentation_config=aug)

    def _exp_train_loader(self):
        aug = self.config.get("dataloader", {}).get("augmentation")
        bs  = self.config["training"].get("batch_size", 512)
        return self._make_dataloader(self.exp_train_ds, bs // 2, shuffle=True,
                                     augmentation_config=aug)

    # -- Training epoch ----------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        import itertools

        self.da_model.train()

        mc_loader  = self._resample_mc_loader()
        exp_loader = self._exp_train_loader()

        n_mc  = len(mc_loader)
        n_exp = len(exp_loader)
        total_batches = max(n_mc, n_exp)

        if n_mc >= n_exp:
            mc_iter  = iter(mc_loader)
            exp_iter = itertools.cycle(exp_loader)
        else:
            mc_iter  = itertools.cycle(mc_loader)
            exp_iter = iter(exp_loader)

        log_every    = self.config.get("logging", {}).get("log_every", 50)
        running_loss = 0.0
        actual_batches = 0
        mc_logits_list:  List[torch.Tensor] = []
        exp_logits_list: List[torch.Tensor] = []

        for batch_idx in range(total_batches):
            mc_batch  = next(mc_iter)
            exp_batch = next(exp_iter)

            for b in (mc_batch, exp_batch):
                for k, v in b.items():
                    if isinstance(v, torch.Tensor):
                        b[k] = v.to(self.device)

            # Two separate forward passes — avoids sequence-length alignment issues
            mc_feats  = self.da_model.base_model.get_feature_representation(mc_batch)
            mc_logits = self.da_model.domain_discriminator.discriminator(mc_feats)
            mc_loss   = F.binary_cross_entropy_with_logits(
                mc_logits, torch.zeros(mc_feats.shape[0], 1, device=self.device)
            )

            exp_feats  = self.da_model.base_model.get_feature_representation(exp_batch)
            exp_logits = self.da_model.domain_discriminator.discriminator(exp_feats)
            exp_loss   = F.binary_cross_entropy_with_logits(
                exp_logits, torch.ones(exp_feats.shape[0], 1, device=self.device)
            )

            loss = (mc_loss + exp_loss) / 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss  += loss.item()
            actual_batches += 1
            mc_logits_list.append(mc_logits.detach().cpu())
            exp_logits_list.append(exp_logits.detach().cpu())

            if batch_idx % log_every == 0:
                logger.info(
                    f"Epoch {self.current_epoch}  "
                    f"Batch {batch_idx}/{total_batches}  "
                    f"Loss: {loss.item():.4f}"
                )

            if self.config.get("debug", {}).get("enabled", False):
                if batch_idx >= self.config["debug"].get("max_batches_per_epoch", 10):
                    break

        train_loss = running_loss / actual_batches

        # Train accuracy and AUC from accumulated logits
        mc_probs_t  = torch.sigmoid(torch.cat(mc_logits_list)).flatten()
        exp_probs_t = torch.sigmoid(torch.cat(exp_logits_list)).flatten()
        train_acc   = (
            (mc_probs_t  < 0.5).float().mean()
            + (exp_probs_t >= 0.5).float().mean()
        ).item() / 2
        all_probs  = mc_probs_t.tolist()  + exp_probs_t.tolist()
        all_labels = [0.0] * len(mc_probs_t) + [1.0] * len(exp_probs_t)
        train_auc  = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

        if self.writer:
            self.writer.add_scalar("train/domain_loss",     train_loss, self.current_epoch)
            self.writer.add_scalar("train/domain_accuracy", train_acc,  self.current_epoch)
            self.writer.add_scalar("train/domain_auc",      train_auc,  self.current_epoch)
        return {"domain_loss": train_loss, "domain_accuracy": train_acc, "domain_auc": train_auc}

    # -- Validation epoch --------------------------------------------------

    def validate_epoch(self) -> Dict[str, float]:
        self.da_model.eval()

        mc_logits_list:  List[torch.Tensor] = []
        exp_logits_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.val_mc_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                feats  = self.da_model.base_model.get_feature_representation(batch)
                logits = self.da_model.domain_discriminator.discriminator(feats)
                mc_logits_list.append(logits.cpu())

            for batch in self.val_exp_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                feats  = self.da_model.base_model.get_feature_representation(batch)
                logits = self.da_model.domain_discriminator.discriminator(feats)
                exp_logits_list.append(logits.cpu())

        mc_logits  = torch.cat(mc_logits_list)
        exp_logits = torch.cat(exp_logits_list)

        # Balance for fair loss / AUC
        n_mc_val  = mc_logits.shape[0]
        n_exp_val = exp_logits.shape[0]
        n_bal     = min(n_mc_val, n_exp_val)
        rng       = np.random.default_rng(self.current_epoch)
        if n_mc_val > n_bal:
            mc_logits = mc_logits[rng.choice(n_mc_val, n_bal, replace=False)]
        if n_exp_val > n_bal:
            exp_logits = exp_logits[rng.choice(n_exp_val, n_bal, replace=False)]

        mc_labels  = torch.zeros(n_bal, 1)
        exp_labels = torch.ones(n_bal,  1)

        val_loss = (
            F.binary_cross_entropy_with_logits(mc_logits, mc_labels)
            + F.binary_cross_entropy_with_logits(exp_logits, exp_labels)
        ).item() / 2

        mc_probs  = torch.sigmoid(mc_logits).flatten()
        exp_probs = torch.sigmoid(exp_logits).flatten()
        val_acc   = (
            (mc_probs  < 0.5).float().mean()
            + (exp_probs >= 0.5).float().mean()
        ).item() / 2

        all_probs  = mc_probs.tolist()  + exp_probs.tolist()
        all_labels = [0.0] * n_bal      + [1.0] * n_bal
        val_auc    = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

        src_lbl, tgt_lbl = ("MC-A", "MC-B") if self.control_mode else ("MC", "Exp")
        logger.info(
            f"  Val — loss={val_loss:.4f}  acc={val_acc:.4f}  auc={val_auc:.4f}"
            f"  ({src_lbl} {n_mc_val:,}  {tgt_lbl} {n_exp_val:,})"
        )

        if self.writer:
            self.writer.add_scalar("val/domain_loss",     val_loss, self.current_epoch)
            self.writer.add_scalar("val/domain_accuracy", val_acc,  self.current_epoch)
            self.writer.add_scalar("val/domain_auc",      val_auc,  self.current_epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.current_epoch)

        return {"domain_loss": val_loss, "domain_accuracy": val_acc, "domain_auc": val_auc}

    # -- Checkpointing -----------------------------------------------------

    def save_checkpoint(self, epoch: int, is_best: bool = False, history: Optional[list] = None) -> None:
        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     self.da_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric":          self.best_metric,
            "best_epoch":           self.best_epoch,
            "config":               self.config,
            "normalization_config": self.normalization_config,
            "training_history":     history[:epoch + 1] if history else [],
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(ckpt, self.output_dir / "latest_checkpoint.pth")
        if is_best:
            torch.save(ckpt, self.output_dir / "best_model.pth")
            logger.info(f"New best model saved at epoch {epoch}")

    # -- Early stopping ----------------------------------------------------

    def check_early_stopping(self, metric: float) -> bool:
        es = self.config.get("training", {}).get("early_stopping", {})
        if not es.get("enabled", True):
            return False
        mode     = es.get("mode",     "min")
        patience = es.get("patience", 15)
        delta    = es.get("min_delta", 1e-4)

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
            logger.info(f"Early stopping triggered (patience={patience})")
            return True
        return False

    # -- Main training loop ------------------------------------------------

    def train(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.output_dir / "train.log"),
            ],
        )
        logger.info("Starting domain separability training…")
        self.prepare_data()
        self.prepare_model()

        if self.config.get("logging", {}).get("save_config", True):
            with open(self.output_dir / "config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)

        tcfg           = self.config["training"]
        num_epochs     = tcfg.get("epochs", 100)
        validate_every = tcfg.get("validate_every", 1)
        save_every     = tcfg.get("save_every", 5)
        monitor        = tcfg.get("early_stopping", {}).get("monitor", "domain_loss")
        es_mode        = tcfg.get("early_stopping", {}).get("mode", "min")
        history: list  = []

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch()
            val_metrics: Dict[str, float] = {}
            if epoch % validate_every == 0:
                val_metrics = self.validate_epoch()

            if self.scheduler is not None and val_metrics:
                m = val_metrics.get(monitor, train_metrics.get("loss", 0))
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(m)
                else:
                    self.scheduler.step()

            current_metric = val_metrics.get(monitor, float("inf"))
            is_best = (
                current_metric < self.best_metric if es_mode == "min"
                else current_metric > self.best_metric
            )

            logger.info(
                f"Epoch {epoch}/{num_epochs - 1} [{time.time() - t0:.1f}s] — "
                f"train_loss={train_metrics.get('domain_loss', 0):.4f}  "
                f"train_auc={train_metrics.get('domain_auc', 0):.4f}  "
                f"val_loss={val_metrics.get('domain_loss', 0):.4f}  "
                f"val_auc={val_metrics.get('domain_auc', 0):.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            epoch_entry = {
                "epoch": epoch,
                "lr":    self.optimizer.param_groups[0]["lr"],
                "time":  time.time() - t0,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(epoch_entry)
            self._save_history_csv(history)

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
            f"Done.  Best {monitor}: {self.best_metric:.4f} @ epoch {self.best_epoch}"
        )

    def _save_history_csv(self, history: list) -> None:
        try:
            import pandas as pd
            pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)
        except Exception as e:
            logger.error(f"Failed to save history CSV: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--debug",  action="store_true",
                        help="Debug mode: limit to 10 batches per epoch")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.debug:
        config.setdefault("debug", {})["enabled"] = True
        config["debug"].setdefault("max_batches_per_epoch", 10)

    with DomainSeparabilityTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main()
