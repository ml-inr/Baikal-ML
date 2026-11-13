"""Domain Adaptation trainer for the nu-classifier task — discriminator-pool variant.

A copy of `da_nu_classifier_trainer_da_repair.py` (itself a copy of the untouched
baseline `da_nu_classifier_trainer.py`) that trains the domain branch against a
**pool of M discriminators** instead of one, and measures whether the members stay
diverse while the encoder keeps moving.

Why: a single critic constrains the encoder along one readout; the encoder can
satisfy it by re-coordinating the features (hiding the domain in the critic's
blind directions) rather than by removing the domain — the diagnosis of
2026-10-02, see doc/learning/dann_encoder_shortcut/. An ensemble only helps if
its members keep differing: on a frozen feature distribution they converge to the
same function, and the question is whether the live game keeps them apart. This
run exists to measure that, so every epoch it logs:

  * `pool_cos_mean` / `pool_cos_min` — pairwise cosine between the members'
    reading directions (mean gradient of the member's loss w.r.t. the features);
    1.0 means the pool has collapsed into one critic in M copies;
  * `pool_disagree` — fraction of events where members disagree on the domain label;
  * `pool_auc_min` / `pool_auc_max` — per-member domain AUC on a fixed batch;
  * `pool_sel_share` — how often the member selected by the `min` aggregation was
    the same one (1.0 = the game always fights one member).

Training differs from the da_repair run in two documented ways:
  1. M members, each trained on its own loss; the encoder maximises the
     aggregate over the pool (`discriminator_aggregation`: "min" = fool the
     currently-best member, "mean" = the average critic). Members are updated
     *before* the encoder step, so the encoder's gradient is computed against a
     freshly updated opponent (a Gauss-Seidel step rather than Jacobi);
  2. nothing else — same data, seed, losses, lambda schedule, probe and refresh.

Source domain: memory-mapped NPY arrays (MC, produced by nu_classifier_ds_builder).
Target domain: memory-mapped NPY arrays (exp, produced by nu_classifier_ds_builder/exp_builder).

Both domains use sig-noise-filtered hits.  The MC source has hard labels
(1.0 = neutrino, 0.0 = muatm).  The exp target is unlabeled.

Note on checkpoints: `domain_discriminator_state_dict` now holds the *pool*
(keys "0.discriminator...", "1.discriminator...", ...), so the inference tooling
of the single-discriminator runs cannot load it as-is.

Usage:
    python src/training/da_nu_classifier_trainer_ensemble.py \
        --config src/training/configs/classifier/da_nu_classifier_exp_full_E1_ensemble10.yaml [--debug]
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_models import create_model
from src.utils.training import set_reproducible_seeds


class _EnsembleDAModel(nn.Module):
    """Носитель пула из M дискриминаторов домена.

    Интерфейс совпадает с DomainAdaptationModel там, где его использует тренер
    (`base_model`, `domain_discriminator`, `get_features`, `classification_forward`,
    `domain_forward`, `set_lambda`, `count_parameters`); дополнительно есть
    `domain_forward_multi` — логиты каждого члена пула отдельно.

    Поле `domain_discriminator` здесь — ModuleList: так существующие вызовы
    (`str()`, `.parameters()`, `.state_dict()`) работают без правок и при этом
    видят всех членов, а не первого.
    """

    def __init__(self, base_model: nn.Module, da_config: Dict, n_discriminators: int):
        super().__init__()
        self.base_model = base_model
        self.n_discriminators = int(n_discriminators)
        cfg = da_config.get("domain_discriminator", da_config)
        feature_dim = base_model.feature_extractor.feature_dim

        self.discriminators = nn.ModuleList([
            DomainDiscriminator(
                input_dim=feature_dim,
                hidden_dims=cfg.get("hidden_dims", [64, 32]),
                dropout=cfg.get("dropout", 0.2),
                use_batch_norm=cfg.get("use_batch_norm", True),
                use_gradient_reversal=cfg.get("use_gradient_reversal", True),
                lambda_factor=cfg.get("lambda_factor", 1.0),
                hidden_layer_norm=cfg.get("hidden_layer_norm", False),
                instance_noise=cfg.get("instance_noise", 0.0),
            )
            for _ in range(self.n_discriminators)
        ])
        self.domain_discriminator = self.discriminators
        logger.info("Created ensemble DA model: %d discriminators, feature_dim=%d",
                    self.n_discriminators, feature_dim)

    def get_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Признаки без градиентного разворота."""
        return self.base_model.get_feature_representation(batch)

    def classification_forward(self, batch: Dict[str, torch.Tensor]):
        features = self.get_features(batch)
        return self.base_model.classifier(features), features

    def domain_forward(self, batch: Dict[str, torch.Tensor]):
        """Одиночный путь для валидации: логиты усредняются по пулу.

        Среднее по логитам — это «средний критик» пула; для разброса по членам
        служит отдельная диагностика (`_pool_diagnostics`).
        """
        features = self.get_features(batch)
        logits = torch.stack([d(features) for d in self.discriminators], dim=0).mean(0)
        return logits, features

    def domain_forward_multi(self, batch: Dict[str, torch.Tensor]):
        """Логиты каждого члена пула (список длины M) и общие признаки."""
        features = self.get_features(batch)
        return [d(features) for d in self.discriminators], features

    def set_lambda(self, lambda_factor: float) -> None:
        for d in self.discriminators:
            d.set_lambda(lambda_factor)

    def count_parameters(self) -> Dict[str, int]:
        base = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        dom = sum(p.numel() for p in self.discriminators.parameters() if p.requires_grad)
        return {"base_model": base, "domain_discriminator": dom, "total": base + dom}

from src.models.domain_discriminator import DomainDiscriminator
from src.data.nu_classifier_dataset import (
    NuClassifierNpyDataset,
    NuClassifierExpNpyDataset,
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




class NuClassifierDomainAdaptationTrainer:
    """DANN trainer: NPY source (MC) + NPY target (sig-noise-filtered exp).

    Key difference from prefilter DA trainer:
    * Target domain is ``NuClassifierExpNpyDataset`` (NPY mmap from exp_builder),
      not ``ExpDataset`` (in-memory from raw exp.h5).  The sig-noise model has
      already been applied to both source and target hits.
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

        include_probs = config["data"].get("include_probs", False)
        if include_probs and self.normalization_config is not None:
            self.normalization_config_collate = {
                "means": self.normalization_config["means"] + [0.75],
                "stds":  self.normalization_config["stds"]  + [0.25],
            }
        else:
            self.normalization_config_collate = self.normalization_config

        self.base_model = None
        self.da_model = None
        self.optimizer_feature = None
        self.optimizer_classifier = None
        self.optimizer_discriminator = None
        self.scheduler_feature = None
        self.scheduler_classifier = None
        self.scheduler_discriminator = None

        self.source_train_loader: Optional[DataLoader] = None
        self.source_val_loader:   Optional[DataLoader] = None
        self.target_train_loader: Optional[DataLoader] = None
        self.target_val_loader:   Optional[DataLoader] = None

        self.current_epoch = 0
        self.current_step  = 0
        if config['training']['early_stopping']['mode'] == 'max':
            self.best_metric_for_es   = float("-inf") 
        elif config['training']['early_stopping']['mode'] == 'min':
            self.best_metric_for_es = float('inf')
        else:
            raise ValueError("Unexpected Early Stopping mode.")
        self.best_epoch    = 0
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

        logger.info(f"Initialized NuClassifierDomainAdaptationTrainer on {self.device}")
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
        data_cfg      = self.config["data"]
        source_cfg    = data_cfg["source_domain"]
        target_cfg    = data_cfg["target_domain"]
        seed          = self.config["experiment"].get("seed", 42)
        include_probs = data_cfg.get("include_probs", False)

        # --- Source domain: MC NPY ---
        weights_file = source_cfg.get("weights_file")
        if weights_file:
            logger.info(f"Source weights: {weights_file}")
        source_ds = NuClassifierNpyDataset(
            npy_dir=source_cfg["npy_dir"],
            max_hits=data_cfg.get("max_hits"),
            max_events=source_cfg.get("max_events"),
            seed=seed,
            include_probs=include_probs,
            weights_file=weights_file,
        )
        source_train_ds, source_val_ds = source_ds.split(
            source_cfg.get("train_split", 0.85), seed=seed
        )
        logger.info(
            f"Source (MC NPY): {len(source_train_ds):,} train, "
            f"{len(source_val_ds):,} val"
        )

        # --- Target domain: exp NPY (sig-noise filtered) ---
        target_ds = NuClassifierExpNpyDataset(
            npy_dir=target_cfg["npy_dir"],
            max_hits=data_cfg.get("max_hits"),
            max_events=target_cfg.get("max_events"),
            seed=seed,
            include_probs=include_probs,
        )
        target_train_ds, target_val_ds = target_ds.split(
            target_cfg.get("train_split", 0.85), seed=seed
        )
        logger.info(
            f"Target (Exp NPY): {len(target_train_ds):,} train, "
            f"{len(target_val_ds):,} val"
        )

        self.class_weights = self._calculate_class_weights(source_train_ds)

        src_bs = self.config["training"].get("source_batch_size", 512)
        tgt_bs = self.config["training"].get("target_batch_size", 256)
        dl_cfg = self.config.get("dataloader", {})
        nw     = dl_cfg.get("num_workers", 0)
        pm     = dl_cfg.get("pin_memory", False)
        aug    = dl_cfg.get("augmentation")
        # Afterpulse is injected ONLY into the MC source (exp already has real
        # afterpulses). Build the OM-position pool from the source features and
        # attach it to the source aug; strip afterpulse from the target aug.
        aug_source, aug_target = self._build_train_augs(aug, source_cfg["npy_dir"], seed)

        self.source_train_loader = create_nu_classifier_dataloader(
            source_train_ds, src_bs, shuffle=True,
            normalization_config=self.normalization_config_collate,
            augmentation_config=aug_source, shuffle_batch=True,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        self.source_val_loader = create_nu_classifier_dataloader(
            source_val_ds, src_bs, shuffle=False,
            normalization_config=self.normalization_config_collate,
            augmentation_config=None, shuffle_batch=False,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        self.target_train_loader = create_nu_classifier_dataloader(
            target_train_ds, tgt_bs, shuffle=True,
            normalization_config=self.normalization_config_collate,
            augmentation_config=aug_target, shuffle_batch=True,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        self.target_val_loader = create_nu_classifier_dataloader(
            target_val_ds, tgt_bs, shuffle=False,
            normalization_config=self.normalization_config_collate,
            augmentation_config=None, shuffle_batch=False,
            device=str(self.device), num_workers=nw, pin_memory=pm,
        )
        logger.info("Data preparation complete")

    @staticmethod
    def _build_om_pool(
        npy_dir: str, seed: int, n_sample: int = 1_000_000, round_to: float = 2.0,
    ) -> np.ndarray:
        """Unique OM (x, y, z) positions sampled from the source hit features.

        Deduped on a coarse grid so afterpulse positions are ~uniform over OMs
        (no bias toward frequently-hit modules).
        """
        feats = np.load(Path(npy_dir) / "features.npy", mmap_mode="r")
        n   = len(feats)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=min(n_sample, n), replace=False))
        xyz = np.asarray(feats[idx, 2:5], dtype=np.float32)
        pool = np.unique(np.round(xyz / round_to) * round_to, axis=0).astype(np.float32)
        logger.info(
            f"OM pool for afterpulse: {len(pool):,} unique positions "
            f"(from {len(idx):,} sampled hits)"
        )
        return pool

    def _build_train_augs(self, aug, source_npy_dir: str, seed: int):
        """Split the train augmentation into (source, target).

        Afterpulse is injected ONLY into the MC source; the (expensive) OM pool
        is built only when afterpulse is enabled. Target keeps rotation/noise.
        """
        if aug is None:
            return None, None
        ap = aug.get("afterpulse")
        aug_target = {k: v for k, v in aug.items() if k != "afterpulse"}
        if ap and ap.get("enabled", False):
            pool = self._build_om_pool(source_npy_dir, seed)
            aug_source = dict(aug)
            aug_source["afterpulse"] = {**ap, "om_pool": pool}
        else:
            aug_source = aug_target
        return aug_source, aug_target

    def _calculate_class_weights(self, dataset) -> torch.Tensor:
        logger.info("Calculating class weights from source training data …")
        labels = dataset.get_all_labels()
        binary = torch.tensor((labels > 0.5).astype(np.float32), device=self.device)
        return calculate_class_weights(binary)

    # -- Model preparation ------------------------------------------------

    def prepare_model(self) -> None:
        logger.info("Preparing model …")
        model_config = dict(self.config["model"])

        raw_clip = model_config.get("amp_clip")
        if raw_clip is not None and self.normalization_config is not None:
            mean_amp = self.normalization_config["means"][0]
            std_amp  = self.normalization_config["stds"][0]
            model_config["amp_clip"] = (raw_clip - mean_amp) / std_amp
            logger.info(
                f"amp_clip: Q={raw_clip} PE → normalized={model_config['amp_clip']:.4f} "
                f"(mean={mean_amp}, std={std_amp})"
            )

        self.base_model = create_model(model_config)
        self.base_model.to(self.device)

        n_disc = int(self.config["domain_adaptation"].get("n_discriminators", 1))
        self.pool_aggregation = str(
            self.config["domain_adaptation"].get("discriminator_aggregation", "min"))
        self._pool_selection = np.zeros(n_disc, dtype=np.int64)
        self._pool_member_loss = np.zeros(n_disc, dtype=np.float64)
        logger.info("Discriminator pool: M=%d, aggregation=%s", n_disc, self.pool_aggregation)
        self.da_model = _EnsembleDAModel(
            self.base_model, self.config["domain_adaptation"], n_disc
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
                f.write("\n\nDomain Discriminator:\n")
                f.write(str(self.da_model.domain_discriminator))
                f.write("\n\nParameters:\n")
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
        elif cfg["optimizer"] == "sgd":
            return optim.SGD(
                params, lr=cfg["learning_rate"],
                momentum=cfg.get("momentum", 0.0),
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    def _create_schedulers(self, tcfg: Dict) -> None:
        stype = tcfg.get("scheduler")
        if stype == "cosine":
            sp    = tcfg.get("scheduler_params", {})
            T_max = int(sp.get("T_max", tcfg["epochs"]))
            eta   = float(sp.get("eta_min", 1e-6))
            for attr, opt in [
                ("scheduler_feature",       self.optimizer_feature),
                ("scheduler_classifier",    self.optimizer_classifier),
                ("scheduler_discriminator", self.optimizer_discriminator),
            ]:
                setattr(self, attr, optim.lr_scheduler.CosineAnnealingLR(opt, T_max, eta))
        elif stype == "step":
            sp = tcfg.get("scheduler_params", {})
            ss = int(sp.get("step_size", 30))
            g  = float(sp.get("gamma", 0.1))
            for attr, opt in [
                ("scheduler_feature",       self.optimizer_feature),
                ("scheduler_classifier",    self.optimizer_classifier),
                ("scheduler_discriminator", self.optimizer_discriminator),
            ]:
                setattr(self, attr, optim.lr_scheduler.StepLR(opt, ss, g))
        elif stype == "plateau":
            sp      = tcfg.get("scheduler_params", {})
            mode    = sp.get("mode", "max")
            factor  = float(sp.get("factor", 0.5))
            patience = int(sp.get("patience", 5))
            min_lr  = float(sp.get("min_lr", 1e-6))
            for attr, opt in [
                ("scheduler_feature",       self.optimizer_feature),
                ("scheduler_classifier",    self.optimizer_classifier),
                ("scheduler_discriminator", self.optimizer_discriminator),
            ]:
                setattr(self, attr, optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode=mode, factor=factor, patience=patience, min_lr=min_lr,
                ))

    # -- Lambda scheduling ------------------------------------------------

    def _get_lambda_factor(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> float:
        sc   = self.lambda_scheduler_config
        by   = sc.get("schedule_by", "epoch")
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
        elif stype == "delayed_constant":
            # λ=0 до start_epoch, с start_epoch — константа lambda.
            # Смысл: дать классовому лоссу осесть до включения состязательной игры.
            start = int(sc.get("start_epoch", 0))
            return sc.get("lambda", 1.0) if (epoch or 0) >= start else 0.0
        elif stype == "linear":
            return sc.get("start_lambda", 0.0) + (
                sc.get("end_lambda", 1.0) - sc.get("start_lambda", 0.0)
            ) * p
        return 1.0

    # -- Classification loss ----------------------------------------------

    def _calculate_classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tcfg = self.config["training"]
        pos_weight = None
        if hasattr(self, "class_weights") and tcfg.get("class_weights") != "auto":
            if tcfg.get("class_weights") is not None:
                w = torch.tensor(tcfg["class_weights"], device=self.device)
            else:
                w = self.class_weights.to(self.device)
            pos_weight = w[1] / w[0]

        # Horizon-aware soft labels: near true zenith θ=90° the up/down cue (which
        # separates ν from μ in MC) degenerates, so pull the target toward 0.5 there
        # — teach the net to abstain where direction is unknowable. Source-only (θ is
        # NaN → weight 0 for datasets without GT theta). σ from the muatm FP falloff
        # (≈10.5°); see doc/horizon_loss_design.md.
        hcfg = tcfg.get("horizon_loss", {})
        if hcfg.get("enabled", False) and theta is not None:
            sigma    = float(hcfg.get("sigma_deg", 10.5))
            max_soft = float(hcfg.get("max_soften", 1.0))
            w_h = max_soft * torch.exp(-((theta - 90.0) ** 2) / (2.0 * sigma ** 2))
            w_h = torch.nan_to_num(w_h, nan=0.0)      # no GT θ → no softening
            soft_labels = labels * (1.0 - w_h) + 0.5 * w_h
            gamma = float(tcfg.get("focal_gamma", 2.0))
            return soft_focal_loss_with_logits(logits, soft_labels, gamma, pos_weight)

        loss_fn = tcfg.get("classification_loss", "bce")
        if loss_fn == "bce":
            return binary_cross_entropy_with_logits_weighted(
                logits, labels, pos_weight, sample_weight)
        elif loss_fn == "focal":
            gamma = float(tcfg.get("focal_gamma", 2.0))
            return focal_loss_with_logits_weighted(
                logits, labels, gamma, pos_weight, sample_weight)
        elif loss_fn == "soft_focal":
            # per-event weights are not plumbed through the soft-label variant;
            # no config in use combines them (weights come with hard-label runs)
            gamma = float(tcfg.get("focal_gamma", 2.0))
            return soft_focal_loss_with_logits(logits, labels, gamma, pos_weight)
        raise ValueError(f"Unknown classification_loss: '{loss_fn}'")

    # -- Signal-hits subset metrics ---------------------------------------

    def _calculate_signal_hits_metrics(
        self,
        all_logits: List[torch.Tensor],
        all_labels: List[torch.Tensor],
        all_signal_hits: List[torch.Tensor],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not all_logits:
            return metrics
        try:
            logits    = torch.cat(all_logits, dim=0)
            labels    = (torch.cat(all_labels, dim=0).numpy() > 0.5).astype(int)
            probs     = torch.sigmoid(logits).numpy().flatten()
            sig_hits  = torch.cat(all_signal_hits, dim=0).numpy()

            from sklearn.metrics import roc_auc_score

            for threshold, tag in [(8, "8hits"), (16, "16hits")]:
                mask = sig_hits > threshold
                n    = int(mask.sum())
                metrics[f"cls_events_{tag}"] = n
                if n > 0 and len(np.unique(labels[mask])) > 1:
                    metrics[f"cls_auc_{tag}"] = float(roc_auc_score(labels[mask], probs[mask]))
                else:
                    metrics[f"cls_auc_{tag}"] = 0.0
        except Exception as e:
            logger.warning(f"Failed to calc signal-hits metrics: {e}")
            for tag in ("8hits", "16hits"):
                metrics[f"cls_auc_{tag}"]    = 0.0
                metrics[f"cls_events_{tag}"] = 0
        return metrics

    # -- Augmentation helper ----------------------------------------------

    def _augment_features(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ):
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
            x_rot  = cos_a[:, None] * x - sin_a[:, None] * y
            y_rot  = sin_a[:, None] * x + cos_a[:, None] * y
            features[:, :, 2] = torch.where(mask, x_rot, features[:, :, 2])
            features[:, :, 3] = torch.where(mask, y_rot, features[:, :, 3])

        noise_std_list = aug_cfg.get("noise_std")
        if noise_std_list is not None:
            noise_std = torch.tensor(
                noise_std_list, dtype=torch.float32, device=self.device
            )
            noise    = torch.randn_like(features) * noise_std
            features = torch.where(mask.unsqueeze(-1), features + noise, features)

            time_for_sort = features[:, :, 1].masked_fill(~mask, float("inf"))
            sort_idx      = time_for_sort.argsort(dim=1)
            sort_idx_exp  = sort_idx.unsqueeze(-1).expand(-1, -1, features.shape[2])
            features      = features.gather(dim=1, index=sort_idx_exp)
            mask          = mask.gather(dim=1, index=sort_idx)

        return features, mask

    # -- Domain batches (shared by the training step and the refresh) -----

    def _build_domain_batches(self, source_batch: Dict, target_batch: Dict):
        """z-flip both domains and align the two halves to one size.

        Both domains get a 50/50 vertical-orientation mix: the source by flipping a
        random half of its events, the target by concatenating every event with its
        mirror. Orientation therefore carries no domain cue into the discriminator.

        Returns (src_dom_batch, tgt_dom_batch, n_source, n_target, domain_bs).
        """
        n_source = source_batch["features"].shape[0]

        tgt_feat_orig     = target_batch["features"]
        tgt_mask          = target_batch["mask"]
        tgt_feat_flip_aug = tgt_feat_orig.clone()
        tgt_feat_flip_aug[:, :, 4] = torch.where(
            tgt_mask, -tgt_feat_orig[:, :, 4], tgt_feat_orig[:, :, 4]
        )
        tgt_mask_flip_aug = tgt_mask.clone()

        tgt_features_cat = torch.cat([tgt_feat_orig, tgt_feat_flip_aug], dim=0)
        tgt_lengths_cat  = torch.cat([target_batch["lengths"], target_batch["lengths"]], dim=0)
        tgt_mask_cat     = torch.cat([tgt_mask, tgt_mask_flip_aug], dim=0)
        n_target         = tgt_features_cat.shape[0]

        domain_bs = min(n_source, n_target)

        sel_src = (
            torch.randperm(n_source, device=self.device)[:domain_bs]
            if n_source > domain_bs
            else torch.arange(n_source, device=self.device)
        )
        sel_tgt = (
            torch.randperm(n_target, device=self.device)[:domain_bs]
            if n_target > domain_bs
            else torch.arange(n_target, device=self.device)
        )

        src_features = source_batch["features"][sel_src].clone()
        src_mask     = source_batch["mask"][sel_src]

        n_flip   = domain_bs // 2
        flip_idx = torch.randperm(domain_bs, device=self.device)[:n_flip]
        src_features[flip_idx, :, 4] = torch.where(
            src_mask[flip_idx],
            -src_features[flip_idx, :, 4],
            src_features[flip_idx, :, 4],
        )

        src_dom_batch = {
            "features": src_features,
            "lengths":  source_batch["lengths"][sel_src],
            "mask":     src_mask,
        }
        tgt_dom_batch = {
            "features": tgt_features_cat[sel_tgt],
            "lengths":  tgt_lengths_cat[sel_tgt],
            "mask":     tgt_mask_cat[sel_tgt],
        }
        return src_dom_batch, tgt_dom_batch, n_source, n_target, domain_bs

    # -- Pool diversity ----------------------------------------------------

    def _pool_probe_batch(self):
        """Фиксированный доменный батч для диагностики пула (строится один раз)."""
        if getattr(self, "_pool_probe", None) is None:
            src_batch = next(iter(self.source_val_loader))
            tgt_batch = next(iter(self.target_val_loader))
            for b in (src_batch, tgt_batch):
                for k in b:
                    if isinstance(b[k], torch.Tensor):
                        b[k] = b[k].to(self.device)
            src_dom, tgt_dom, _, _, n = self._build_domain_batches(src_batch, tgt_batch)
            self._pool_probe = (src_dom, tgt_dom, int(n))
            logger.info("Pool probe batch: %d source + %d target (frozen for the run)",
                        int(n), int(n))
        return self._pool_probe

    @staticmethod
    def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        """AUC по формуле Манна—Уитни (без зависимости от sklearn)."""
        order = np.argsort(scores)
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        pos = labels > 0.5
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def _pool_diagnostics(self) -> Dict[str, float]:
        """Разнообразие пула на фиксированном батче.

        Читающее направление члена — средний по батчу градиент его лосса по
        признакам (источник и цель вместе). Косинусы между этими векторами
        отвечают на вопрос «смотрят ли члены в разные стороны», а не «одинаковы
        ли их веса»: именно первое нужно ансамблю, чтобы сузить слепую зону
        энкодера.
        """
        model = self.da_model
        was_training = model.training
        model.eval()
        src_dom, tgt_dom, n = self._pool_probe_batch()
        zeros = torch.zeros(n, 1, device=self.device)
        ones = torch.ones(n, 1, device=self.device)

        with torch.no_grad():
            f_src = model.get_features(src_dom)
            f_tgt = model.get_features(tgt_dom)

        directions, probs = [], []
        for disc in model.discriminators:
            fs = f_src.detach().requires_grad_(True)
            ft = f_tgt.detach().requires_grad_(True)
            loss = 0.5 * (
                nn.functional.binary_cross_entropy_with_logits(disc(fs), zeros)
                + nn.functional.binary_cross_entropy_with_logits(disc(ft), ones))
            g_src, g_tgt = torch.autograd.grad(loss, [fs, ft])
            # Знак градиента у всех членов переворачивает GRL одинаково, поэтому
            # косинусы между направлениями от разворота не зависят.
            directions.append(torch.cat([g_src.mean(0), g_tgt.mean(0)]))
            probs.append(torch.cat([torch.sigmoid(disc(f_src)),
                                    torch.sigmoid(disc(f_tgt))]).flatten().detach())
        if was_training:
            model.train()

        m = len(directions)
        dirs = torch.stack(directions)
        unit = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cos = unit @ unit.t()
        pr = torch.stack(probs).detach()
        hard = pr >= 0.5
        labels = np.r_[np.zeros(n), np.ones(n)]

        pair_cos, pair_dis, aucs = [], [], []
        for i in range(m):
            aucs.append(self._rank_auc(pr[i].cpu().numpy(), labels))
            for j in range(i + 1, m):
                pair_cos.append(float(cos[i, j].item()))
                pair_dis.append(float((hard[i] != hard[j]).float().mean().item()))

        total_sel = int(self._pool_selection.sum())
        member_loss_mean = self._pool_member_loss / max(self._epoch_steps, 1)
        metrics = {
            "pool_cos_mean": float(np.mean(pair_cos)) if pair_cos else float("nan"),
            "pool_cos_min":  float(np.min(pair_cos)) if pair_cos else float("nan"),
            "pool_cos_max":  float(np.max(pair_cos)) if pair_cos else float("nan"),
            "pool_disagree": float(np.mean(pair_dis)) if pair_dis else float("nan"),
            "pool_auc_mean": float(np.nanmean(aucs)),
            "pool_auc_min":  float(np.nanmin(aucs)),
            "pool_auc_max":  float(np.nanmax(aucs)),
            "pool_loss_spread": float(member_loss_mean.max() - member_loss_mean.min()),
            "pool_sel_share": (float(self._pool_selection.max() / total_sel)
                               if total_sel else float("nan")),
        }
        return metrics

    # -- Discriminator refresh --------------------------------------------

    def _refresh_discriminator(self, steps: int) -> None:
        """Re-fit the discriminator alone, on frozen features.

        The FIXED_sn256 run ended with a discriminator that could not separate the
        domains although the features were still linearly separable (an independent
        probe on the same inputs: AUC 0.72 against its 0.50). From that state its
        reversal gradient to the encoder is ~zero, so the domain branch does nothing.
        Freezing the encoder and re-fitting the discriminator restores a live
        opponent; whether the encoder then responds is what the probe measures.
        """
        import itertools

        model = self.da_model
        for param in model.base_model.parameters():
            param.requires_grad_(False)
        model.base_model.eval()
        model.domain_discriminator.train()

        source_iter = itertools.cycle(self.source_train_loader)
        target_iter = itertools.cycle(self.target_train_loader)
        loss_value = float("nan")
        for _ in range(steps):
            source_batch = next(source_iter)
            target_batch = next(target_iter)
            for b in (source_batch, target_batch):
                for k in b:
                    if isinstance(b[k], torch.Tensor):
                        b[k] = b[k].to(self.device)
            src_dom_batch, tgt_dom_batch, _, _, _ = self._build_domain_batches(
                source_batch, target_batch)
            with torch.no_grad():
                f_src = model.get_features(src_dom_batch).detach()
                f_tgt = model.get_features(tgt_dom_batch).detach()
            # Каждого члена пула переобучаем на одних и тех же замороженных
            # фичах: это и есть проверка «сходятся ли члены к одной функции».
            loss_value = 0.0
            for disc in model.discriminators:
                logits_src = disc(f_src)
                logits_tgt = disc(f_tgt)
                loss = (nn.functional.binary_cross_entropy_with_logits(
                            logits_src, torch.zeros_like(logits_src))
                        + nn.functional.binary_cross_entropy_with_logits(
                            logits_tgt, torch.ones_like(logits_tgt))) / 2
                self.optimizer_discriminator.zero_grad()
                loss.backward()
                self.optimizer_discriminator.step()
                loss_value += loss.item() / model.n_discriminators

        for param in model.base_model.parameters():
            param.requires_grad_(True)
        model.train()
        logger.info("  discriminator pool refresh done (%d steps x %d members, "
                    "mean last loss %.4f)",
                    steps, model.n_discriminators, loss_value)

    # -- Invariance probe --------------------------------------------------

    def _probe_sample(self) -> Dict:
        """Fixed held-out events built by data_manager/probe_sample/build_probe_sample.py."""
        if getattr(self, "_probe_arrays", None) is None:
            path = self.config["training"]["invariance_probe"]["path"]
            with np.load(path) as npz:
                self._probe_arrays = {k: npz[k]
                                      for k in ("hits", "offsets", "group", "band")}
            logger.info("Invariance probe sample: %s (%d events)",
                        path, len(self._probe_arrays["group"]))
        return self._probe_arrays

    def _probe_embeddings(
        self, hits: np.ndarray, offsets: np.ndarray, flip_z: bool = False
    ) -> np.ndarray:
        """Encoder features of the probe sample, normalized and clipped as in training.

        flip_z=True negates the (normalized) z coordinate of every hit -- the same
        operation `_build_domain_batches` applies to the domain batches -- so the
        mirrored embeddings can feed the game-style probe below.
        """
        means = torch.tensor(self.normalization_config["means"],
                             dtype=torch.float32, device=self.device)
        stds = torch.tensor(self.normalization_config["stds"],
                            dtype=torch.float32, device=self.device)
        model = self.da_model.base_model
        was_training = model.training
        model.eval()
        n_events = len(offsets) - 1
        chunks = []
        with torch.no_grad():
            for start in range(0, n_events, 512):
                end = min(start + 512, n_events)
                lows  = offsets[start:end]
                highs = offsets[start + 1:end + 1]
                lengths = torch.tensor(highs - lows, dtype=torch.long,
                                       device=self.device)
                max_len = int(lengths.max().item())
                padded = torch.zeros(len(lengths), max_len, hits.shape[1],
                                     dtype=torch.float32, device=self.device)
                for i, (lo, hi) in enumerate(zip(lows, highs)):
                    padded[i, :hi - lo] = torch.from_numpy(hits[lo:hi]).to(self.device)
                mask = torch.arange(max_len, device=self.device)[None, :] < \
                    lengths[:, None]
                padded = torch.where(mask.unsqueeze(-1),
                                     (padded - means) / (stds + 1e-8), padded)
                if flip_z:
                    padded[:, :, 4] = torch.where(
                        mask, -padded[:, :, 4], padded[:, :, 4])
                feats = model.get_feature_representation(
                    {"features": padded, "lengths": lengths, "mask": mask})
                chunks.append(feats.cpu().numpy())
        if was_training:
            model.train()
        return np.concatenate(chunks, axis=0)

    def _invariance_probe(self) -> Dict[str, float]:
        """Fresh logistic probes on frozen features: two meters, one task each.

        * ``probe_auc_*`` -- muatm vs exp (sample groups 0-3), overall and per
          band.  Historical meter; comparable with runs before 2026-10-10 only
          because the sample's first four groups are byte-frozen.
        * ``probe_auc_game_*`` -- the *in-game* task: the natural nu+mu mixture
          (groups 0,1,4,5) against exp, built exactly as the domain batches are
          -- half of the MC side z-flipped, the exp side doubled with its mirror.
          This is the number to compare against ``Val dom``.  The flipped half
          is fixed by seed 42, so the meter is reproducible across epochs.

        A fresh probe, not the in-game discriminator, is the meter: a
        discriminator driven into an uninformative state reports chance exactly
        as genuinely invariant features would (measured on the FIXED_sn256
        checkpoint -- its discriminator scored 0.50 where a fresh probe scored
        0.72).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        sample = self._probe_sample()
        emb = self._probe_embeddings(sample["hits"], sample["offsets"])
        emb_flip = self._probe_embeddings(sample["hits"], sample["offsets"],
                                          flip_z=True)
        group = sample["group"]   # 0,1 = muatm; 2,3 = exp; 4,5 = nu
        band = sample["band"]     # 0 = bulk, 1 = mid, 2 = tail

        metrics: Dict[str, float] = {}

        def cv_auc(X: np.ndarray, y: np.ndarray) -> Optional[float]:
            if len(np.unique(y)) < 2 or min(np.bincount(y)) < 50:
                return None
            aucs = []
            for tr, te in StratifiedKFold(3, shuffle=True,
                                          random_state=42).split(X, y):
                scaler = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000).fit(
                    scaler.transform(X[tr]), y[tr])
                aucs.append(roc_auc_score(
                    y[te], clf.decision_function(scaler.transform(X[te]))))
            return float(np.mean(aucs))

        hist = group <= 3
        for mask, tag in [(hist, "all"),
                          (hist & (band == 0), "bulk"),
                          (hist & (band == 2), "tail")]:
            auc = cv_auc(emb[mask], (group[mask] >= 2).astype(int))
            if auc is not None:
                metrics[f"probe_auc_{tag}"] = auc

        mc = np.flatnonzero((group <= 1) | (group >= 4))
        exp = np.flatnonzero((group >= 2) & (group <= 3))
        flipped = np.zeros(len(group), dtype=bool)
        flipped[np.random.default_rng(42).choice(
            mc, size=len(mc) // 2, replace=False)] = True
        src_rows = np.where(flipped[mc, None], emb_flip[mc], emb[mc])
        tgt_rows = np.concatenate([emb[exp], emb_flip[exp]], axis=0)
        X_game = np.concatenate([src_rows, tgt_rows], axis=0)
        y_game = np.concatenate([np.zeros(len(src_rows), int),
                                 np.ones(len(tgt_rows), int)])
        band_game = np.concatenate([band[mc], band[exp], band[exp]])
        for mask, tag in [(np.ones(len(y_game), dtype=bool), "all"),
                          (band_game == 0, "bulk"),
                          (band_game == 2, "tail")]:
            auc = cv_auc(X_game[mask], y_game[mask])
            if auc is not None:
                metrics[f"probe_auc_game_{tag}"] = auc
        return metrics

    # -- Training epoch ---------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        self.da_model.train()
        self.metrics_tracker.reset_epoch()

        # Счётчики пула на эпоху: их читает диагностика после валидации.
        self._pool_selection[:] = 0
        self._pool_member_loss[:] = 0.0
        self._epoch_steps = 0

        import itertools

        total_epochs  = self.config["training"]["epochs"]
        n_src         = len(self.source_train_loader)
        n_tgt         = len(self.target_train_loader)
        total_batches = max(n_src, n_tgt)
        total_steps   = total_epochs * total_batches
        log_every     = self.config["logging"].get("log_every", 10)

        if n_src >= n_tgt:
            source_iter = iter(self.source_train_loader)
            target_iter = itertools.cycle(self.target_train_loader)
        else:
            source_iter = itertools.cycle(self.source_train_loader)
            target_iter = iter(self.target_train_loader)

        epoch_metrics = {
            "classification_loss":   0.0,
            "domain_loss":           0.0,
            "total_loss":            0.0,
            "domain_accuracy":       0.0,
            "lambda_factor":         0.0,
            "domain_utilization":    0.0,
            "source_background_avg": 0.0,
            "target_events_avg":     0.0,
            "domain_batch_size_avg": 0.0,
        }

        domain_predictions:  List[float] = []
        domain_labels_true:  List[float] = []
        all_source_logits:   List[torch.Tensor] = []
        all_source_labels:   List[torch.Tensor] = []
        all_source_sig_hits: List[torch.Tensor] = []

        for batch_idx in range(total_batches):
            source_batch = next(source_iter)
            target_batch = next(target_iter)

            for b in (source_batch, target_batch):
                for k in b:
                    if isinstance(b[k], torch.Tensor):
                        b[k] = b[k].to(self.device)

            lambda_factor = self._get_lambda_factor(
                step=self.current_step, epoch=self.current_epoch,
                total_steps=total_steps, total_epochs=total_epochs,
            )
            self.da_model.set_lambda(lambda_factor)

            # === Classification (source) ===
            source_class_logits, _ = self.da_model.classification_forward(source_batch)
            source_labels = source_batch["labels"].float()

            all_source_logits.append(source_class_logits.detach().cpu())
            all_source_labels.append(source_labels.detach().cpu())
            all_source_sig_hits.append(source_batch["signal_hit_counts"].detach().cpu())

            classification_loss = self._calculate_classification_loss(
                source_class_logits, source_labels,
                theta=source_batch.get("theta"),
                sample_weight=source_batch.get("weights"),
            )

            self.optimizer_feature.zero_grad()
            self.optimizer_classifier.zero_grad()
            self.optimizer_discriminator.zero_grad()
            classification_loss.backward()

            # === Domain adaptation: pool of M discriminators (z-flip both) ===
            src_dom_batch, tgt_dom_batch, n_source, n_target, domain_bs = (
                self._build_domain_batches(source_batch, target_batch))

            src_dom_labels = torch.zeros(domain_bs, 1, device=self.device)
            tgt_dom_labels = torch.ones(domain_bs,  1, device=self.device)

            def _dom_loss(logits_src: torch.Tensor, logits_tgt: torch.Tensor) -> torch.Tensor:
                return (nn.functional.binary_cross_entropy_with_logits(
                            logits_src, src_dom_labels)
                        + nn.functional.binary_cross_entropy_with_logits(
                            logits_tgt, tgt_dom_labels)) / 2

            # (1) Шаг каждого члена пула — на замороженных (detached) фичах, чтобы
            # эти шаги не трогали энкодер и не смешивались с доменным градиентом.
            with torch.no_grad():
                f_src_frozen = self.da_model.get_features(src_dom_batch)
                f_tgt_frozen = self.da_model.get_features(tgt_dom_batch)
            member_losses = []
            for disc in self.da_model.discriminators:
                loss_m = _dom_loss(disc(f_src_frozen), disc(f_tgt_frozen))
                self.optimizer_discriminator.zero_grad()
                loss_m.backward()
                self.optimizer_discriminator.step()
                member_losses.append(float(loss_m.item()))
            self._pool_member_loss += np.asarray(member_losses)

            # (2) Шаг энкодера: он максимизирует агрегат по пулу (градиент
            # реверсный, через GRL выбранных членов). Голова пула считается по
            # уже обновлённым членам — энкодер видит свежего противника.
            f_src = self.da_model.get_features(src_dom_batch)
            f_tgt = self.da_model.get_features(tgt_dom_batch)
            member_logits = []
            for disc in self.da_model.discriminators:
                member_logits.append((disc(f_src), disc(f_tgt)))
            pool_losses = torch.stack([_dom_loss(ls, lt) for ls, lt in member_logits])
            domain_loss = pool_losses.mean()          # для метрики — средний критик
            if self.pool_aggregation == "mean":
                agg_loss = domain_loss
            else:                                     # "min": обмануть всех
                idx = int(torch.argmin(pool_losses).item())
                self._pool_selection[idx] += 1
                agg_loss = pool_losses[idx]

            src_util        = domain_bs / n_source
            tgt_util        = domain_bs / n_target
            avg_util        = (src_util + tgt_util) / 2
            norm_domain_loss = agg_loss / avg_util
            # Доменные градиенты в пул не применяем: члены уже сделали свой шаг.
            self.optimizer_discriminator.zero_grad()
            norm_domain_loss.backward()

            self.optimizer_feature.step()
            self.optimizer_classifier.step()

            with torch.no_grad():
                src_dom_logits = torch.stack([ls for ls, _ in member_logits]).mean(0)
                tgt_dom_logits = torch.stack([lt for _, lt in member_logits]).mean(0)
                src_correct = (torch.sigmoid(src_dom_logits) < 0.5).float().mean()
                tgt_correct = (torch.sigmoid(tgt_dom_logits) >= 0.5).float().mean()
                dom_acc     = (src_correct + tgt_correct) / 2

                domain_predictions.extend(torch.sigmoid(src_dom_logits).cpu().flatten().tolist())
                domain_predictions.extend(torch.sigmoid(tgt_dom_logits).cpu().flatten().tolist())
                domain_labels_true.extend([0.0] * domain_bs)
                domain_labels_true.extend([1.0] * domain_bs)

            self.metrics_tracker.update_train(source_class_logits, source_labels, classification_loss)

            epoch_metrics["classification_loss"]   += classification_loss.item()
            epoch_metrics["domain_loss"]           += norm_domain_loss.item()
            epoch_metrics["total_loss"]            += classification_loss.item() + norm_domain_loss.item()
            epoch_metrics["domain_accuracy"]       += dom_acc.item()
            epoch_metrics["lambda_factor"]          = lambda_factor
            epoch_metrics["domain_utilization"]    += avg_util
            epoch_metrics["source_background_avg"] += n_source
            epoch_metrics["target_events_avg"]     += n_target
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
                if batch_idx >= self.config["debug"].get("max_batches_per_epoch", 10):
                    break

        for key in [
            "classification_loss", "domain_loss", "total_loss",
            "domain_accuracy", "domain_utilization",
            "source_background_avg", "target_events_avg", "domain_batch_size_avg",
        ]:
            epoch_metrics[key] /= max(total_batches, 1)

        self._epoch_steps = total_batches

        epoch_metrics["domain_auc"] = 0.0
        if domain_predictions and len(set(domain_labels_true)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                epoch_metrics["domain_auc"] = roc_auc_score(domain_labels_true, domain_predictions)
            except Exception:
                pass

        sig_metrics  = self._calculate_signal_hits_metrics(
            all_source_logits, all_source_labels, all_source_sig_hits,
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

        all_val_logits:   List[torch.Tensor] = []
        all_val_labels:   List[torch.Tensor] = []
        all_val_sig_hits: List[torch.Tensor] = []

        src_domain_logits: List[torch.Tensor] = []
        tgt_domain_logits: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.source_val_loader:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)

                logits, _ = self.da_model.classification_forward(batch)
                labels    = batch["labels"].float()
                loss      = self._calculate_classification_loss(
                    logits, labels, sample_weight=batch.get("weights"))
                self.metrics_tracker.update_val(logits, labels, loss)

                all_val_logits.append(logits.cpu())
                all_val_labels.append(labels.cpu())
                all_val_sig_hits.append(batch["signal_hit_counts"].cpu())

                src_feat = batch["features"].clone()
                src_mask = batch["mask"]
                bs       = src_feat.shape[0]
                n_flip   = bs // 2
                flip_idx = torch.randperm(bs, device=self.device)[:n_flip]
                src_feat[flip_idx, :, 4] = torch.where(
                    src_mask[flip_idx],
                    -src_feat[flip_idx, :, 4],
                    src_feat[flip_idx, :, 4],
                )
                src_dl, _ = self.da_model.domain_forward({
                    "features": src_feat,
                    "lengths":  batch["lengths"],
                    "mask":     src_mask,
                })
                src_domain_logits.append(src_dl.cpu())

            for tb in self.target_val_loader:
                for k in tb:
                    if isinstance(tb[k], torch.Tensor):
                        tb[k] = tb[k].to(self.device)

                tgt_feat_orig     = tb["features"]
                tgt_mask          = tb["mask"]
                tgt_feat_flip_aug = tgt_feat_orig.clone()
                tgt_feat_flip_aug[:, :, 4] = torch.where(
                    tgt_mask, -tgt_feat_orig[:, :, 4], tgt_feat_orig[:, :, 4]
                )
                tgt_mask_flip_aug = tgt_mask.clone()
                tgt_features_cat = torch.cat([tgt_feat_orig, tgt_feat_flip_aug], dim=0)
                tgt_lengths_cat  = torch.cat([tb["lengths"], tb["lengths"]], dim=0)
                tgt_mask_cat     = torch.cat([tgt_mask, tgt_mask_flip_aug], dim=0)

                tgt_dl, _ = self.da_model.domain_forward({
                    "features": tgt_features_cat,
                    "lengths":  tgt_lengths_cat,
                    "mask":     tgt_mask_cat,
                })
                tgt_domain_logits.append(tgt_dl.cpu())

        n_src = sum(t.shape[0] for t in src_domain_logits)
        n_tgt = sum(t.shape[0] for t in tgt_domain_logits)

        val_domain_loss = 0.0
        val_domain_acc  = 0.5
        val_domain_preds:  List[float] = []
        val_domain_labels: List[float] = []

        if n_src > 0 and n_tgt > 0:
            src_logits_t = torch.cat(src_domain_logits)
            tgt_logits_t = torch.cat(tgt_domain_logits)
            n_bal        = min(n_src, n_tgt)
            rng_bal      = np.random.default_rng(self.current_epoch)
            if n_src > n_bal:
                src_logits_t = src_logits_t[rng_bal.choice(n_src, size=n_bal, replace=False)]
            if n_tgt > n_bal:
                tgt_logits_t = tgt_logits_t[rng_bal.choice(n_tgt, size=n_bal, replace=False)]

            val_domain_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    src_logits_t, torch.zeros(n_bal, 1),
                )
                + nn.functional.binary_cross_entropy_with_logits(
                    tgt_logits_t, torch.ones(n_bal, 1),
                )
            ).item() / 2
            src_preds_t   = torch.sigmoid(src_logits_t).flatten()
            tgt_preds_t   = torch.sigmoid(tgt_logits_t).flatten()
            val_domain_acc = (
                (src_preds_t < 0.5).float().mean()
                + (tgt_preds_t >= 0.5).float().mean()
            ).item() / 2
            val_domain_preds  = src_preds_t.tolist() + tgt_preds_t.tolist()
            val_domain_labels = [0.0] * n_bal + [1.0] * n_bal

        val_metrics = self.metrics_tracker.val_metrics.compute()
        val_metrics["domain_loss"]     = val_domain_loss
        val_metrics["domain_accuracy"] = val_domain_acc

        val_metrics["domain_auc"] = 0.0
        if val_domain_preds and len(set(val_domain_labels)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                val_metrics["domain_auc"] = roc_auc_score(val_domain_labels, val_domain_preds)
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
            f"(src={n_src:,} mc, tgt={n_tgt:,} exp+zflip, "
            f"balanced={min(n_src, n_tgt) if n_src > 0 and n_tgt > 0 else 0:,})"
        )

        val_sig = self._calculate_signal_hits_metrics(
            all_val_logits, all_val_labels, all_val_sig_hits,
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
            "domain_discriminator_state_dict": self.da_model.domain_discriminator.state_dict(),
            "optimizer_feature_state_dict":       self.optimizer_feature.state_dict(),
            "optimizer_classifier_state_dict":    self.optimizer_classifier.state_dict(),
            "optimizer_discriminator_state_dict": self.optimizer_discriminator.state_dict(),
            "best_metric":    self.best_metric_for_es,
            "best_epoch":     self.best_epoch,
            "config":         self.config,
            "normalization_config": self.normalization_config,
            "training_history": history[: epoch + 1] if history else [],
        }
        if self.scheduler_feature is not None:
            ckpt["scheduler_feature_state_dict"]       = self.scheduler_feature.state_dict()
            ckpt["scheduler_classifier_state_dict"]    = self.scheduler_classifier.state_dict()
            ckpt["scheduler_discriminator_state_dict"] = self.scheduler_discriminator.state_dict()

        torch.save(ckpt, self.output_dir / f"da_checkpoint_epoch_{epoch:03d}.pth")
        torch.save(ckpt, self.output_dir / "latest_da_checkpoint.pth")
        if is_best:
            torch.save(ckpt, self.output_dir / "best_da_model.pth")
            logger.info(f"New best model saved at epoch {epoch}")

    # -- Early stopping ---------------------------------------------------

    def check_early_stopping(self, metric: float) -> bool:
        es      = self.config["training"]["early_stopping"]
        if not es.get("enabled", True):
            return False
        mode    = es.get("mode", "max")
        patience = es.get("patience", 10)
        delta   = es.get("min_delta", 0.001)

        improved = (
            metric > self.best_metric_for_es + delta if mode == "max"
            else metric < self.best_metric_for_es - delta
        )
        if improved:
            self.best_metric_for_es = metric
            self.best_epoch  = self.current_epoch
            self.early_stopping_counter = 0
            return False
        self.early_stopping_counter += 1
        if self.early_stopping_counter >= patience:
            logger.info(f"Early stopping after {patience} epochs without improvement")
            return True
        return False

    # -- Main training loop -----------------------------------------------

    def train(self) -> None:
        logger.info("Starting nu-classifier DA training …")
        self.prepare_data()
        self.prepare_model()

        if self.config["logging"].get("save_config", True):
            with open(self.output_dir / "da_config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            if self.normalization_config:
                with open(self.output_dir / "normalization_config.yaml", "w") as f:
                    yaml.dump(self.normalization_config, f, default_flow_style=False)

        num_epochs    = self.config["training"]["epochs"]
        validate_every = self.config["training"]['validate_every']
        save_every    = self.config["training"]['save_every']
        monitor       = self.config["training"]["early_stopping"]["monitor"]
        history: list = []

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch()

            val_metrics: Dict[str, float] = {}
            if epoch % validate_every == 0:
                val_metrics = self.validate_epoch()

            probe_metrics: Dict[str, float] = {}
            probe_every = int(self.config["training"].get(
                "invariance_probe", {}).get("every_n_epochs", 0) or 0)
            if probe_every and epoch % probe_every == 0:
                probe_metrics = self._invariance_probe()
                logger.info("  Probe   — " + ", ".join(
                    f"{key.replace('probe_auc_', 'auc_')}={value:.4f}"
                    for key, value in probe_metrics.items()))
                if self.writer:
                    for key, value in probe_metrics.items():
                        self.writer.add_scalar(f"Val/{key}", value, epoch)

            pool_metrics: Dict[str, float] = {}
            if len(self.da_model.discriminators) > 1:
                pool_metrics = self._pool_diagnostics()
                logger.info("  Pool    — " + ", ".join(
                    f"{key.replace('pool_', '')}={value:.4f}"
                    for key, value in pool_metrics.items()))
                if self.writer:
                    for key, value in pool_metrics.items():
                        self.writer.add_scalar(f"Pool/{key}", value, epoch)

            refresh_cfg = self.config["domain_adaptation"].get(
                "discriminator_refresh", {})
            refresh_every = int(refresh_cfg.get("every_epochs", 0) or 0)
            if refresh_every and (epoch + 1) % refresh_every == 0:
                self._refresh_discriminator(int(refresh_cfg.get("steps", 300)))

            if self.scheduler_feature is not None:
                if isinstance(self.scheduler_feature, optim.lr_scheduler.ReduceLROnPlateau):
                    plateau_metric = val_metrics.get(
                        monitor.replace("val_", ""),
                        train_metrics.get(monitor, 0),
                    )
                    self.scheduler_feature.step(plateau_metric)
                    self.scheduler_classifier.step(plateau_metric)
                    self.scheduler_discriminator.step(plateau_metric)
                else:
                    self.scheduler_feature.step()
                    self.scheduler_classifier.step()
                    self.scheduler_discriminator.step()

            if self.writer:
                self._log_tensorboard_lr(epoch)

            self.metrics_tracker.save_epoch_metrics(epoch)

            epoch_entry = {
                "epoch":            epoch,
                "lr_feature":       self.optimizer_feature.param_groups[0]["lr"],
                "lr_classifier":    self.optimizer_classifier.param_groups[0]["lr"],
                "lr_discriminator": self.optimizer_discriminator.param_groups[0]["lr"],
                "epoch_time":       time.time() - t0,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **probe_metrics,
                **pool_metrics,
            }
            history.append(epoch_entry)
            self._save_history_csv(history)

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

            current_metric = val_metrics[monitor.replace("val_", "")]
            is_best = False
            es_mode = self.config["training"]["early_stopping"].get("mode", "max")
            if es_mode == "max":
                is_best = current_metric > self.best_metric_for_es
            else:
                is_best = current_metric < self.best_metric_for_es

            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best, history)

            if val_metrics and self.check_early_stopping(current_metric):
                logger.info(f"Training stopped early at epoch {epoch}")
                break

            if is_best:
                self.best_metric_for_es = current_metric
                self.best_epoch  = epoch

        self._save_results(history)
        self.close()
        logger.info(
            f"Training complete. Best {monitor}: {self.best_metric_for_es:.4f} "
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
            "best_metric":       self.best_metric_for_es,
            "best_epoch":        self.best_epoch,
            "total_epochs":      self.current_epoch + 1,
            "final_train":       self.metrics_tracker.train_metrics.compute(),
            "final_val":         self.metrics_tracker.val_metrics.compute(),
            "model_parameters":  self.da_model.count_parameters(),
        }
        with open(self.output_dir / "da_training_summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

    def _log_tensorboard_train(self, m: Dict, epoch: int) -> None:
        w = self.writer
        for k, tb in [
            ("loss",                    "Train/Classification_Loss"),
            ("accuracy",                "Train/Classification_Accuracy"),
            ("f1",                      "Train/Classification_F1"),
            ("auc",                     "Train/Classification_AUC"),
            ("classification_loss",     "Train/DA_Classification_Loss"),
            ("domain_loss",             "Train/DA_Domain_Loss"),
            ("total_loss",              "Train/DA_Total_Loss"),
            ("domain_accuracy",         "Train/DA_Domain_Accuracy"),
            ("domain_auc",              "Train/DA_Domain_AUC"),
            ("lambda_factor",           "Train/DA_Lambda_Factor"),
            ("domain_utilization",      "Train/Domain_Utilization"),
            ("domain_batch_size_avg",   "Train/Domain_Batch_Size"),
            ("cls_auc_8hits",           "Train/Classification_AUC_8hits"),
            ("cls_auc_16hits",          "Train/Classification_AUC_16hits"),
        ]:
            if k in m:
                w.add_scalar(tb, m[k], epoch)

    def _log_tensorboard_val(self, m: Dict, epoch: int) -> None:
        w = self.writer
        for k, tb in [
            ("loss",              "Val/Classification_Loss"),
            ("accuracy",          "Val/Classification_Accuracy"),
            ("f1",                "Val/Classification_F1"),
            ("auc",               "Val/Classification_AUC"),
            ("domain_loss",       "Val/Domain_Loss"),
            ("domain_accuracy",   "Val/Domain_Accuracy"),
            ("domain_auc",        "Val/Domain_AUC"),
            ("val_cls_auc_8hits",  "Val/Classification_AUC_8hits"),
            ("val_cls_auc_16hits", "Val/Classification_AUC_16hits"),
        ]:
            if k in m:
                w.add_scalar(tb, m[k], epoch)

    def _log_tensorboard_lr(self, epoch: int) -> None:
        w = self.writer
        w.add_scalar("LearningRate/Feature_Extractor",
                     self.optimizer_feature.param_groups[0]["lr"], epoch)
        w.add_scalar("LearningRate/Classifier",
                     self.optimizer_classifier.param_groups[0]["lr"], epoch)
        w.add_scalar("LearningRate/Discriminator",
                     self.optimizer_discriminator.param_groups[0]["lr"], epoch)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train nu-classifier DA model for neutrino detection",
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

    with NuClassifierDomainAdaptationTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main()
