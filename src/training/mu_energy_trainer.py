"""Baseline trainer for muon-energy regression (no domain adaptation).

Task: predict ``log10(E_mu / GeV)`` at entry into the cluster's sensitive volume from the
sig-noise-selected hits.  The data come from ``data_manager/mu_energy_ds_builder``: the
dataset contains the flat-selected train+val population only.  Train/val is a training
YAML field (``data.train_split`` / ``data.split_seed``, like the nu-classifier); the test
complement is not stored and is predicted from the full source.

The loss is a per-event-weighted Huber on ``log10 E``; the flat-selection weights are 1 for
the bulk and up to 20 for the sparse tails.  Validation is reported both unweighted and
weighted.  No DA, no afterpulse/rotation augmentation in this baseline.

Usage:
    python -m src.training.mu_energy_trainer --config <config.yaml>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.mu_energy_dataset import (  # noqa: E402
    ExpEnergyDomainDataset,
    McEnergyDomainDataset,
    MuEnergyNpyDataset,
    create_mu_energy_dataloader,
    part_split_indices,
)
from src.models.domain_discriminator import DomainDiscriminator  # noqa: E402
from src.models.mu_energy_model import create_mu_energy_model  # noqa: E402
from src.utils.training import set_reproducible_seeds  # noqa: E402

logger = logging.getLogger(__name__)

_LOG10_2 = float(np.log10(2.0))
_LOG10_5 = float(np.log10(5.0))


def _new_accumulator() -> Dict[str, Any]:
    return {
        "n": 0,
        "loss_num": 0.0,
        "loss_unweighted_num": 0.0,
        "weight_sum": 0.0,
        "abs_sum": 0.0,
        "abs_weighted_sum": 0.0,
        "sq_sum": 0.0,
        "bias_sum": 0.0,
        "within2": 0,
        "within5": 0,
        "within10": 0,
        "per_class": {},
    }


def _update_accumulator(
    acc: Dict[str, Any],
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    particle_type: torch.Tensor,
    loss_sum: float,
    loss_unweighted_sum: float,
) -> None:
    resid = (pred - target).detach().cpu()
    w = weight.detach().cpu()
    ptype = particle_type.detach().cpu()
    n = len(resid)
    abs_r = resid.abs()

    acc["n"] += n
    acc["loss_num"] += float(loss_sum)
    acc["loss_unweighted_num"] += float(loss_unweighted_sum)
    acc["weight_sum"] += float(w.sum())
    acc["abs_sum"] += float(abs_r.sum())
    acc["abs_weighted_sum"] += float((abs_r * w).sum())
    acc["sq_sum"] += float((resid ** 2).sum())
    acc["bias_sum"] += float(resid.sum())
    acc["within2"] += int((abs_r <= _LOG10_2).sum())
    acc["within5"] += int((abs_r <= _LOG10_5).sum())
    acc["within10"] += int((abs_r <= 1.0).sum())

    for code in torch.unique(ptype).tolist():
        m = ptype == code
        pc = acc["per_class"].setdefault(
            int(code), {"n": 0, "abs_sum": 0.0, "sq_sum": 0.0, "bias_sum": 0.0}
        )
        pc["n"] += int(m.sum())
        pc["abs_sum"] += float(abs_r[m].sum())
        pc["sq_sum"] += float((resid[m] ** 2).sum())
        pc["bias_sum"] += float(resid[m].sum())


def _new_uncertainty_accumulator() -> Dict[str, Any]:
    return {
        "w_sum": 0.0,
        "nll_w_sum": 0.0,
        "sigma_w_sum": 0.0,
        "sigma_sq_w_sum": 0.0,
        "abs_err_w_sum": 0.0,
        "err_sq_w_sum": 0.0,
        "sigma_abs_err_w_sum": 0.0,
        "cov1_w": 0.0,
        "cov2_w": 0.0,
        "cov68_w": 0.0,
        "cov95_w": 0.0,
        "pit_hist": torch.zeros(20, dtype=torch.float64),
    }


def _update_uncertainty(
    acc: Dict[str, Any],
    pred_dex: torch.Tensor,
    target_dex: torch.Tensor,
    sigma_dex: torch.Tensor,
    nll: torch.Tensor,
    weight: torch.Tensor,
    distribution: str = "gaussian",
    student_t_nu: float = 5.0,
) -> None:
    pred = pred_dex.detach().float().cpu()
    target = target_dex.detach().float().cpu()
    sigma = sigma_dex.detach().float().cpu().clamp_min(1e-9)
    w = weight.detach().float().cpu()
    nll = nll.detach().float().cpu()
    err = pred - target
    abs_err = err.abs()
    with torch.no_grad():
        acc["w_sum"] += float(w.sum().item())
        acc["nll_w_sum"] += float((nll * w).sum().item())
        acc["sigma_w_sum"] += float((sigma * w).sum().item())
        acc["sigma_sq_w_sum"] += float((sigma * sigma * w).sum().item())
        acc["abs_err_w_sum"] += float((abs_err * w).sum().item())
        acc["err_sq_w_sum"] += float((err * err * w).sum().item())
        acc["sigma_abs_err_w_sum"] += float((sigma * abs_err * w).sum().item())
        acc["cov1_w"] += float(((abs_err <= sigma).float() * w).sum().item())
        acc["cov2_w"] += float(((abs_err <= 2.0 * sigma).float() * w).sum().item())
        z = err / sigma
        if distribution == "student_t":
            from scipy.stats import t as student_t
            pit = torch.from_numpy(
                student_t.cdf(z.numpy(), df=student_t_nu).astype("float32")
            )
        else:
            pit = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        acc["cov68_w"] += float(
            (((pit >= 0.16) & (pit <= 0.84)).float() * w).sum().item()
        )
        acc["cov95_w"] += float(
            (((pit >= 0.025) & (pit <= 0.975)).float() * w).sum().item()
        )
        bins = (pit * 20.0).long().clamp(0, 19)
        acc["pit_hist"] += torch.bincount(
            bins, weights=w.double(), minlength=20
        ).double()


def _finalize_uncertainty(acc: Dict[str, Any]) -> Dict[str, float]:
    w = max(acc["w_sum"], 1e-12)
    mean_sigma = acc["sigma_w_sum"] / w
    mean_abs = acc["abs_err_w_sum"] / w
    cov = acc["sigma_abs_err_w_sum"] / w - mean_sigma * mean_abs
    var_sigma = acc["sigma_sq_w_sum"] / w - mean_sigma ** 2
    var_err = acc["err_sq_w_sum"] / w - mean_abs ** 2
    corr = 0.0
    if var_sigma > 0.0 and var_err > 0.0:
        corr = cov / math.sqrt(var_sigma * var_err)
    hist = acc["pit_hist"]
    total = float(hist.sum().item())
    if total > 0.0:
        cum = torch.cumsum(hist / total, dim=0).numpy()
        uniform = np.arange(1, 21, dtype=np.float64) / 20.0
        pit_ks = float(np.max(np.abs(cum - uniform)))
    else:
        pit_ks = 0.0
    return {
        "unc_nll": acc["nll_w_sum"] / w,
        "unc_mean_sigma_dex": mean_sigma,
        "unc_coverage_1sigma": acc["cov1_w"] / w,
        "unc_coverage_2sigma": acc["cov2_w"] / w,
        "unc_coverage_68": acc["cov68_w"] / w,
        "unc_coverage_95": acc["cov95_w"] / w,
        "unc_sigma_error_corr": corr,
        "unc_pit_ks": pit_ks,
    }


def _finalize(acc: Dict[str, Any]) -> Dict[str, float]:
    n = max(acc["n"], 1)
    mae = acc["abs_sum"] / n
    out = {
        "loss": acc["loss_num"] / max(acc["weight_sum"], 1e-12),
        "loss_unweighted": acc["loss_unweighted_num"] / n,
        "mae_log10": mae,
        "rmse_log10": float(np.sqrt(acc["sq_sum"] / n)),
        "bias_log10": acc["bias_sum"] / n,
        "mae_log10_weighted": acc["abs_weighted_sum"] / max(acc["weight_sum"], 1e-12),
        "mae_factor": float(10.0 ** mae),
        "within_factor_2": acc["within2"] / n,
        "within_factor_5": acc["within5"] / n,
        "within_factor_10": acc["within10"] / n,
        "n_events": float(acc["n"]),
    }
    for code, pc in acc["per_class"].items():
        m = max(pc["n"], 1)
        out[f"class_{code}_mae_log10"] = pc["abs_sum"] / m
        out[f"class_{code}_rmse_log10"] = float(np.sqrt(pc["sq_sum"] / m))
        out[f"class_{code}_bias_log10"] = pc["bias_sum"] / m
        out[f"class_{code}_n"] = float(pc["n"])
    return out


class MuEnergyTrainer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device(
            config["data"].get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        seed = int(config["experiment"].get("seed", 42))
        set_reproducible_seeds(seed)
        rep = config.get("reproducibility", {})
        if rep.get("deterministic", True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.normalization_config: Optional[Dict] = None
        data_cfg = config["data"]
        if data_cfg.get("normalization", {}).get("enabled", True):
            stats_path = Path(
                data_cfg["normalization"].get(
                    "stats_file", "data_manager/stats_dict/default_mc.yaml"
                )
            )
            if not stats_path.is_absolute():
                stats_path = project_root / stats_path
            with open(stats_path) as f:
                self.normalization_config = yaml.safe_load(f)
            logger.info("Normalization from %s", stats_path)

        self.output_dir = (
            Path(config["logging"]["output_dir"]) / config["experiment"]["name"]
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        if config["logging"].get("tensorboard", True):
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir))

        self.model = None
        self.optimizer = None
        self.optimizer_sigma = None
        self.scheduler = None
        self.scaler = None
        self._main_params: List[torch.nn.Parameter] = []
        self._sigma_params: List[torch.nn.Parameter] = []
        self.train_loader = None
        self.val_loader = None
        self.cal_loader = None
        self.calibration_metrics: Dict[str, float] = {}
        self.class_names: Dict[int, str] = {}

        self.da_cfg: Dict = config.get("domain_adaptation", {}) or {}
        self.da_enabled = bool(self.da_cfg.get("enabled", False))
        self.discriminator = None
        self.optimizer_discriminator = None
        self.source_da_loader = None
        self.target_da_loader = None
        self.source_probe_loader = None
        self.target_probe_loader = None
        self._da_iters: Dict[str, Any] = {}
        self._da_metrics: Dict[str, float] = {}

        self.best_metric = float("inf")
        self.best_epoch = -1
        self.early_stopping_counter = 0
        self.current_epoch = 0
        self.start_epoch = 0
        self.resume_history: List[Dict] = []
        self.resumed_from: Optional[str] = None

        loss_cfg = config["training"].get("loss", {})
        self.loss_name = str(loss_cfg.get("name", "huber")).lower()
        self.huber_delta_dex = float(loss_cfg.get("huber_delta", 1.0))
        self.huber_delta = self.huber_delta_dex

        hcfg = config["model"].get("heteroscedastic", {}) or {}
        self.heteroscedastic = bool(hcfg.get("enabled", False))
        self.hcfg = hcfg
        self.sigma_min_dex = float(hcfg.get("sigma_min_dex", 0.05))
        self.sigma_max_dex = float(hcfg.get("sigma_max_dex", 5.0))
        self.init_sigma_dex = float(hcfg.get("init_sigma_dex", 0.5))
        self.beta_nll = float(hcfg.get("beta_nll", 0.5))
        self.nll_const = 0.5 * math.log(2.0 * math.pi)
        dist = str(hcfg.get("distribution", "gaussian")).lower().replace("-", "_")
        if dist in ("student", "studentt", "t"):
            dist = "student_t"
        if dist not in ("gaussian", "student_t"):
            raise ValueError(
                f"heteroscedastic.distribution must be gaussian or student_t, got {dist!r}"
            )
        self.distribution = dist
        self.student_t_nu = float(hcfg.get("student_t_nu", 5.0))
        if self.distribution == "student_t" and self.student_t_nu <= 2.0:
            raise ValueError("student_t_nu must be > 2 for a finite variance")
        if self.distribution == "student_t" and self.beta_nll > 0.0:
            logger.warning(
                "beta_nll > 0 is a Gaussian stabiliser; for student_t it is "
                "applied only as a scale prefactor and is not theoretically grounded"
            )
        self.warmup_epochs = int(hcfg.get("warmup_epochs", 0))
        self.warmup_loss_name = str(hcfg.get("warmup_loss", "huber")).lower()
        self.log_var_min = -math.inf
        self.log_var_max = math.inf
        self.experiment_id = str(
            config.get("experiment", {}).get("id", config["experiment"]["name"])
        )
        self.sigma_strategy = "none"
        if self.heteroscedastic:
            strategy = str(
                config["training"].get("sigma_strategy", "joint")
            ).lower()
            aliases = {
                "joint": "joint",
                "joint_single_opt": "joint",
                "joint_one_opt": "joint",
                "full_two_opt": "full_two_opt",
                "full": "full_two_opt",
                "detach": "detach",
                "detached": "detach",
                "frozen": "frozen",
                "frozen_head": "frozen",
                "head_only": "frozen",
            }
            if strategy not in aliases:
                raise ValueError(
                    "training.sigma_strategy must be one of joint, full_two_opt, "
                    f"detach, frozen; got {strategy!r}"
                )
            self.sigma_strategy = aliases[strategy]
        self.sigma_opt_cfg: Dict[str, Any] = dict(
            config["training"].get("sigma_optimizer", {}) or {}
        )
        if self.sigma_strategy == "frozen" and self.da_enabled:
            raise ValueError(
                "domain adaptation cannot train a frozen base; use an already "
                "DA-adapted base (AH-*) or a joint sigma strategy (AD-*)"
            )
        if self.heteroscedastic and self.loss_name not in ("gaussian_nll", "nll"):
            raise ValueError(
                "heteroscedastic model requires training.loss.name: gaussian_nll"
            )
        self.target_normalization: Optional[Dict] = None
        self.target_mean = 0.0
        self.target_std = 1.0
        self.use_amp = bool(config["training"].get("amp", False)) and (
            self.device.type == "cuda"
        )
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        if self.da_enabled and self.use_amp:
            raise ValueError(
                "domain adaptation with training.amp=true is not implemented; "
                "set training.amp: false for DA runs"
            )
        if self.use_amp and self.sigma_strategy in (
            "detach", "full_two_opt", "frozen"
        ):
            raise ValueError(
                "training.amp=true is not implemented for detached/head-only "
                "sigma strategies; set training.amp: false"
            )

        logger.info("Device: %s", self.device)
        if self.da_enabled:
            logger.info("Domain adaptation: enabled")
        logger.info("Output: %s", self.output_dir)

    def __enter__(self) -> "MuEnergyTrainer":
        return self

    def __exit__(self, *exc) -> None:
        if self.writer is not None:
            self.writer.close()

    # -- data -----------------------------------------------------------

    def prepare_data(self) -> None:
        data_cfg = self.config["data"]
        npy_dir = Path(data_cfg["npy_dir"])
        max_hits = data_cfg.get("max_hits")

        base_dataset = MuEnergyNpyDataset(str(npy_dir), max_hits=max_hits)
        train_frac = float(data_cfg.get("train_split", 0.9))
        split_seed = int(
            data_cfg.get("split_seed", self.config["experiment"].get("seed", 42))
        )
        train_ds, val_ds = base_dataset.split(train_frac, seed=split_seed)
        logger.info(
            "Dataset: %s events; train/val split %.3f/%.3f from YAML (seed=%d): "
            "%s train, %s val",
            f"{len(base_dataset):,}",
            train_frac, 1.0 - train_frac, split_seed,
            f"{len(train_ds):,}", f"{len(val_ds):,}",
        )

        if data_cfg.get("max_train_events"):
            n = int(data_cfg["max_train_events"])
            rng = np.random.RandomState(int(self.config["experiment"]["seed"]))
            sel = np.sort(
                rng.choice(len(train_ds), size=min(n, len(train_ds)), replace=False)
            )
            train_ds = MuEnergyNpyDataset(
                str(npy_dir),
                indices=train_ds.indices[sel],
                weights=train_ds.weights[sel],
                max_hits=max_hits,
            )
            logger.info("max_train_events: using %s train events", f"{len(train_ds):,}")

        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.cal_dataset = None
        cal_fraction = float(data_cfg.get("calibration_fraction", 0.0) or 0.0)
        if 0.0 < cal_fraction < 0.5:
            val_indices = np.asarray(val_ds.indices)
            val_weights = np.asarray(val_ds.weights)
            rng = np.random.RandomState(split_seed + 1)
            perm = rng.permutation(len(val_indices))
            n_cal = int(round(len(val_indices) * cal_fraction))
            cal_sel = perm[:n_cal]
            keep_sel = perm[n_cal:]
            self.cal_dataset = MuEnergyNpyDataset(
                str(npy_dir),
                indices=val_indices[cal_sel],
                weights=val_weights[cal_sel],
                max_hits=max_hits,
            )
            self.val_dataset = MuEnergyNpyDataset(
                str(npy_dir),
                indices=val_indices[keep_sel],
                weights=val_weights[keep_sel],
                max_hits=max_hits,
            )
            logger.info(
                "Calibration split: %s val-select, %s calibration (fraction %.2f)",
                f"{len(self.val_dataset):,}", f"{len(self.cal_dataset):,}",
                cal_fraction,
            )

        info = json.load(open(npy_dir / "dataset_info.json"))
        self.class_names = {
            int(code): name for name, code in info["particle_encode"].items()
        }

        dl_cfg = self.config.get("dataloader", {})
        num_workers = int(dl_cfg.get("num_workers", 0))
        pin_memory = bool(dl_cfg.get("pin_memory", self.device.type == "cuda"))
        bs = int(self.config["training"]["batch_size"])

        tn_cfg = data_cfg.get("target_normalization") or {}
        if tn_cfg.get("enabled", False):
            if "mean" not in tn_cfg or "std" not in tn_cfg:
                raise ValueError(
                    "data.target_normalization is enabled but mean/std are missing"
                )
            self.target_normalization = dict(tn_cfg)
            self.target_mean = float(tn_cfg["mean"])
            self.target_std = float(tn_cfg["std"])
            if "min" in tn_cfg and "max" in tn_cfg:
                logger.info(
                    "Target normalization: z=(log10E - %.6f)/%.6f "
                    "(min %.6f, max %.6f)",
                    self.target_mean, self.target_std,
                    float(tn_cfg["min"]), float(tn_cfg["max"]),
                )
            else:
                logger.info(
                    "Target normalization: z=(log10E - %.6f)/%.6f",
                    self.target_mean, self.target_std,
                )
        else:
            self.target_normalization = None
            self.target_mean, self.target_std = 0.0, 1.0

        augmentation = dl_cfg.get("augmentation")
        self.train_loader = create_mu_energy_dataloader(
            self.train_dataset, bs, shuffle=True,
            normalization_config=self.normalization_config,
            target_normalization=self.target_normalization,
            augmentation_config=augmentation,
            num_workers=num_workers, pin_memory=pin_memory,
            shuffle_batch=bool(dl_cfg.get("shuffle_batch", False)),
        )
        self.val_loader = create_mu_energy_dataloader(
            self.val_dataset, bs, shuffle=False,
            normalization_config=self.normalization_config,
            target_normalization=self.target_normalization,
            augmentation_config=None,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        if self.cal_dataset is not None:
            self.cal_loader = create_mu_energy_dataloader(
                self.cal_dataset, bs, shuffle=False,
                normalization_config=self.normalization_config,
                target_normalization=self.target_normalization,
                augmentation_config=None,
                num_workers=num_workers, pin_memory=pin_memory,
            )
        logger.info(
            "Data: train %s, val %s (batches %s / %s), classes %s",
            f"{len(self.train_dataset):,}", f"{len(self.val_dataset):,}",
            f"{len(self.train_loader):,}", f"{len(self.val_loader):,}",
            self.class_names,
        )
        logger.info("Train augmentation: %s", augmentation or "none")

        if self.da_enabled:
            self._prepare_domain_data(
                npy_dir, max_hits, num_workers, pin_memory, augmentation,
            )

    def _prepare_domain_data(
        self,
        npy_dir,
        max_hits,
        num_workers: int,
        pin_memory: bool,
        augmentation,
    ) -> None:
        """Build the unlabeled source/target streams and their held-out probes."""
        src_cfg = self.da_cfg.get("source_da", {}) or {}
        tgt_cfg = self.da_cfg.get("target_da", {}) or {}
        probe_cfg = self.da_cfg.get("probe", {}) or {}
        seed = int(
            self.da_cfg.get(
                "split_seed", self.config["experiment"].get("seed", 42)
            )
        )
        probe_max = int(probe_cfg.get("max_events", 5000))

        if "npy_dir" not in src_cfg or "npy_dir" not in tgt_cfg:
            raise ValueError("domain_adaptation needs source_da.npy_dir and target_da.npy_dir")

        src_full = McEnergyDomainDataset(
            src_cfg["npy_dir"],
            particle_types=src_cfg.get("particle_types", [0]),
            max_hits=max_hits,
            max_events=None,
            seed=seed,
            track_counts_file=src_cfg.get("track_counts_file"),
            track_min=src_cfg.get("track_min"),
        )
        tgt_full = ExpEnergyDomainDataset(
            tgt_cfg["npy_dir"],
            prefix=str(tgt_cfg.get("prefix", "exp")),
            max_hits=max_hits,
            max_events=None,
            seed=seed,
        )

        src_tr, src_probe = part_split_indices(
            src_full.part_keys, float(src_cfg.get("probe_fraction", 0.1)), seed,
        )
        tgt_tr, tgt_probe = part_split_indices(
            tgt_full.part_keys, float(tgt_cfg.get("holdout_fraction", 0.2)), seed,
        )
        src_tr = np.intersect1d(src_tr, src_full.all_indices)
        src_probe = np.intersect1d(src_probe, src_full.all_indices)
        tgt_tr = np.intersect1d(tgt_tr, tgt_full.all_indices)
        tgt_probe = np.intersect1d(tgt_probe, tgt_full.all_indices)
        if len(src_tr) == 0 or len(tgt_tr) == 0:
            raise ValueError("empty domain train split; check npy_dir/holdout")
        if len(src_probe) == 0 or len(tgt_probe) == 0:
            raise ValueError("empty domain probe split; lower holdout_fraction")

        def mc_view(rows, max_events=None):
            return McEnergyDomainDataset(
                src_cfg["npy_dir"],
                particle_types=src_cfg.get("particle_types", [0]),
                indices=rows,
                max_hits=max_hits,
                max_events=max_events,
                seed=seed,
                track_counts_file=src_cfg.get("track_counts_file"),
                track_min=src_cfg.get("track_min"),
            )

        def exp_view(rows, max_events=None):
            return ExpEnergyDomainDataset(
                tgt_cfg["npy_dir"],
                prefix=str(tgt_cfg.get("prefix", "exp")),
                indices=rows,
                max_hits=max_hits,
                max_events=max_events,
                seed=seed,
            )

        src_train = mc_view(src_tr, src_cfg.get("max_events"))
        tgt_train = exp_view(tgt_tr, tgt_cfg.get("max_events"))
        src_probe_ds = mc_view(src_probe, probe_max)
        tgt_probe_ds = exp_view(tgt_probe, probe_max)

        src_bs = int(src_cfg.get("batch_size", self.config["training"]["batch_size"]))
        tgt_bs = int(tgt_cfg.get("batch_size", self.config["training"]["batch_size"]))
        self.source_da_loader = create_mu_energy_dataloader(
            src_train, src_bs, shuffle=True,
            normalization_config=self.normalization_config,
            target_normalization=None,
            augmentation_config=augmentation,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.target_da_loader = create_mu_energy_dataloader(
            tgt_train, tgt_bs, shuffle=True,
            normalization_config=self.normalization_config,
            target_normalization=None,
            augmentation_config=augmentation,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.source_probe_loader = create_mu_energy_dataloader(
            src_probe_ds, src_bs, shuffle=False,
            normalization_config=self.normalization_config,
            target_normalization=None,
            augmentation_config=None,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.target_probe_loader = create_mu_energy_dataloader(
            tgt_probe_ds, tgt_bs, shuffle=False,
            normalization_config=self.normalization_config,
            target_normalization=None,
            augmentation_config=None,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        src_parts = len({str(k) for k in src_full.part_keys[src_tr]})
        tgt_parts = len({str(k) for k in tgt_full.part_keys[tgt_tr]})
        logger.info(
            "DA source (MC): %s train (%s parts), %s probe (held-out parts)",
            f"{len(src_train):,}", src_parts, f"{len(src_probe_ds):,}",
        )
        logger.info(
            "DA target (exp): %s train (%s parts), %s probe (held-out parts)",
            f"{len(tgt_train):,}", tgt_parts, f"{len(tgt_probe_ds):,}",
        )

    # -- model ----------------------------------------------------------

    @staticmethod
    def _make_optimizer(
        params: List[torch.nn.Parameter],
        name: str,
        lr: float,
        weight_decay: float,
    ) -> Optional[optim.Optimizer]:
        """Build an optimizer over an explicit parameter list.

        ``None`` is returned for an empty list.  This is used by the frozen-head
        strategy, where the base has no trainable parameters.
        """
        if not params:
            return None
        name = name.lower()
        if name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if name == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        raise ValueError(f"unknown optimizer {name}")

    @staticmethod
    def _clip_params(
        params: List[torch.nn.Parameter], grad_clip: float
    ) -> None:
        if grad_clip > 0 and params:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

    def _set_train_mode(self, train: bool) -> None:
        """Set train/eval mode without unfreezing the frozen post-hoc base."""
        if self.sigma_strategy == "frozen":
            self.model.eval()
            if self.model.variance_head is not None:
                self.model.variance_head.train(train)
            return
        self.model.train(train)

    def prepare_model(self) -> None:
        model_cfg = dict(self.config["model"])
        raw_clip = model_cfg.get("amp_clip")
        if raw_clip is not None and self.normalization_config is not None:
            mean_amp = self.normalization_config["means"][0]
            std_amp = self.normalization_config["stds"][0]
            model_cfg["amp_clip"] = (float(raw_clip) - mean_amp) / std_amp
            logger.info(
                "amp_clip: Q=%s PE -> normalized %.4f", raw_clip, model_cfg["amp_clip"]
            )
        if self.loss_name == "huber" or (
            self.heteroscedastic and self.warmup_loss_name == "huber"
        ):
            self.huber_delta = self.huber_delta_dex / self.target_std
            logger.info(
                "Huber delta: %.4f dex -> %.6f in z units (target std %.6f)",
                self.huber_delta_dex, self.huber_delta, self.target_std,
            )
        self.model = create_mu_energy_model(model_cfg).to(self.device)

        if self.heteroscedastic:
            sigma_min_z = max(self.sigma_min_dex / self.target_std, 1e-6)
            sigma_max_z = max(self.sigma_max_dex / self.target_std, sigma_min_z)
            init_sigma_z = max(self.init_sigma_dex / self.target_std, 1e-6)
            self.log_var_min = 2.0 * math.log(sigma_min_z)
            self.log_var_max = 2.0 * math.log(sigma_max_z)
            self.model.initialize_variance_head(
                2.0 * math.log(init_sigma_z)
            )
            logger.info(
                "Heteroscedastic head: init sigma %.3f dex (%.4f z), "
                "sigma range [%.3f, %.3f] dex, beta_nll=%.2f, warmup=%d epochs",
                self.init_sigma_dex, init_sigma_z,
                self.sigma_min_dex, self.sigma_max_dex,
                self.beta_nll, self.warmup_epochs,
            )

        tcfg = self.config["training"]
        opt_cfg = tcfg.get("optimizer", {})
        opt_name = str(opt_cfg.get("name", "adamw")).lower()
        lr = float(opt_cfg.get("learning_rate", 1e-3))
        wd = float(opt_cfg.get("weight_decay", 0.01))
        sigma_opt = self.sigma_opt_cfg
        sigma_opt_name = str(sigma_opt.get("name", opt_name)).lower()
        sigma_lr = float(sigma_opt.get("learning_rate", lr))
        sigma_wd = float(sigma_opt.get("weight_decay", wd))

        # Load the model state first.  The frozen-head strategies need the base
        # weights before freezing; the optimizer(s) are built afterwards over
        # explicit parameter groups.
        resume_cfg = self.config.get("resume", {}) or {}
        resume_path = resume_cfg.get("from_checkpoint")
        reset_training_state = bool(
            resume_cfg.get("reset_training_state", False)
        )
        ckpt = None
        checkpoint_epoch = -1
        resume_optimizer_compatible = True
        self._resume_has_variance = not self.heteroscedastic
        if resume_path:
            path = Path(resume_path)
            if not path.is_absolute():
                path = project_root / path
            if not path.exists():
                raise FileNotFoundError(f"resume checkpoint not found: {path}")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt["model_state_dict"]
            if self.heteroscedastic:
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                bad_missing = [
                    k for k in missing if not k.startswith("variance_head.")
                ]
                if bad_missing or unexpected:
                    raise RuntimeError(
                        f"checkpoint/model mismatch: missing={bad_missing}, "
                        f"unexpected={unexpected}"
                    )
                if missing:
                    resume_optimizer_compatible = False
                    logger.info(
                        "Loaded a deterministic base checkpoint; variance head "
                        "keeps its initialized values; optimizer state is skipped"
                    )
                else:
                    self._resume_has_variance = True
            else:
                state = {
                    k: v for k, v in state.items()
                    if not k.startswith("variance_head.")
                }
                self.model.load_state_dict(state, strict=True)
            self.resumed_from = str(path)
            if reset_training_state:
                checkpoint_epoch = -1
                self.start_epoch = 0
                self.resume_history = []
                self.best_metric = float("inf")
                self.best_epoch = -1
                resume_optimizer_compatible = False
                logger.info(
                    "Initialized model from %s; training state reset "
                    "(base initialization, not continuation)",
                    path,
                )
            else:
                checkpoint_epoch = int(ckpt.get("epoch", -1))
                self.start_epoch = checkpoint_epoch + 1
                self.resume_history = list(ckpt.get("training_history") or [])
                logger.info(
                    "Resumed from %s: checkpoint epoch %d, start epoch %d, "
                    "history entries %d",
                    path, checkpoint_epoch, self.start_epoch,
                    len(self.resume_history),
                )

        if self.sigma_strategy == "frozen":
            for name, param in self.model.named_parameters():
                if not name.startswith("variance_head."):
                    param.requires_grad_(False)
            logger.info("Frozen base: trainable only variance_head")

        variance_ids = set()
        if self.model.variance_head is not None:
            variance_ids = {id(p) for p in self.model.variance_head.parameters()}
        if self.sigma_strategy == "frozen":
            self._main_params = []
            self._sigma_params = [
                p for p in self.model.variance_head.parameters()
                if p.requires_grad
            ]
            self.optimizer = self._make_optimizer(
                self._sigma_params, opt_name, lr, wd
            )
            self.optimizer_sigma = None
        elif self.sigma_strategy in ("detach", "full_two_opt"):
            self._main_params = [
                p for p in self.model.parameters()
                if p.requires_grad and id(p) not in variance_ids
            ]
            self._sigma_params = [
                p for p in self.model.variance_head.parameters()
                if p.requires_grad
            ]
            self.optimizer = self._make_optimizer(
                self._main_params, opt_name, lr, wd
            )
            self.optimizer_sigma = self._make_optimizer(
                self._sigma_params, sigma_opt_name, sigma_lr, sigma_wd
            )
        else:
            self._main_params = [
                p for p in self.model.parameters() if p.requires_grad
            ]
            self._sigma_params = []
            self.optimizer = self._make_optimizer(
                self._main_params, opt_name, lr, wd
            )
            self.optimizer_sigma = None
        if self.optimizer is None:
            raise RuntimeError("no trainable parameters for the main optimizer")

        if (
            ckpt is not None
            and not reset_training_state
            and resume_optimizer_compatible
            and resume_cfg.get("load_optimizer", True)
            and ckpt.get("optimizer_state_dict")
        ):
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for group in self.optimizer.param_groups:
                group["lr"] = lr
                group["initial_lr"] = lr
        if (
            self.optimizer_sigma is not None
            and ckpt is not None
            and not reset_training_state
            and resume_optimizer_compatible
            and resume_cfg.get("load_optimizer", True)
            and ckpt.get("optimizer_sigma_state_dict")
        ):
            self.optimizer_sigma.load_state_dict(
                ckpt["optimizer_sigma_state_dict"]
            )
            for group in self.optimizer_sigma.param_groups:
                group["lr"] = sigma_lr
                group["initial_lr"] = sigma_lr

        if self.config["logging"].get("save_model_summary", True):
            with open(self.output_dir / "model_summary.txt", "w") as f:
                f.write(str(self.model))
                f.write(
                    f"\n\nTrainable parameters: "
                    f"{sum(p.numel() for p in self._main_params) + sum(p.numel() for p in self._sigma_params):,}\n"
                )
                f.write(
                    f"Main optimizer parameters: "
                    f"{sum(p.numel() for p in self._main_params):,}\n"
                )
                f.write(
                    f"Sigma optimizer parameters: "
                    f"{sum(p.numel() for p in self._sigma_params):,}\n"
                )

        sched_cfg = tcfg.get("scheduler", {})
        sched_name = str(sched_cfg.get("name", "plateau")).lower()
        sp = sched_cfg.get("params", {})
        if sched_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=str(sp.get("mode", "min")),
                factor=float(sp.get("factor", 0.5)),
                patience=int(sp.get("patience", 3)),
                threshold=float(sp.get("threshold", 1e-4)),
                threshold_mode=str(sp.get("threshold_mode", "rel")),
                cooldown=int(sp.get("cooldown", 0)),
                min_lr=float(sp.get("min_lr", 1e-6)),
                eps=float(sp.get("eps", 1e-8)),
            )
        elif sched_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(sp.get("T_max", tcfg["epochs"])),
                eta_min=float(sp.get("eta_min", 1e-6)),
            )
        elif sched_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(sp.get("step_size", 20)),
                gamma=float(sp.get("gamma", 0.1)),
            )
        elif sched_name == "none":
            self.scheduler = None
        else:
            raise ValueError(f"unknown scheduler {sched_name}")

        if (
            ckpt is not None
            and resume_optimizer_compatible
            and resume_cfg.get("load_scheduler", False)
        ):
            state = ckpt.get("scheduler_state_dict")
            if state is not None:
                self.scheduler.load_state_dict(state)
                logger.info("Loaded scheduler state from the checkpoint")

        if ckpt is not None:
            monitor = str(
                tcfg.get("early_stopping", {}).get("monitor", "val_loss")
            )
            hist_key = monitor if monitor.startswith("val_") else f"val_{monitor}"
            incumbent = None
            for entry in self.resume_history:
                if int(entry.get("epoch", -1)) == checkpoint_epoch:
                    if hist_key in entry:
                        incumbent = float(entry[hist_key])
                    break
            if incumbent is not None:
                self.best_metric = incumbent
                self.best_epoch = checkpoint_epoch
                logger.info(
                    "Resume monitor %s = %.6f at epoch %d",
                    monitor, self.best_metric, self.best_epoch,
                )
            else:
                logger.warning(
                    "Resume history has no %s for epoch %d; best metric starts at inf",
                    hist_key, checkpoint_epoch,
                )

        if self.config["logging"].get("save_config", True):
            with open(self.output_dir / "train_config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            if self.normalization_config:
                with open(self.output_dir / "normalization_config.yaml", "w") as f:
                    yaml.dump(self.normalization_config, f, default_flow_style=False)

        if self.da_enabled:
            dcfg = self.da_cfg.get("discriminator", {}) or {}
            gcfg = self.da_cfg.get("grl", {}) or {}
            self.discriminator = DomainDiscriminator(
                input_dim=self.model.feature_extractor.feature_dim,
                hidden_dims=list(dcfg.get("hidden_dims", [128, 128])),
                dropout=float(dcfg.get("dropout", 0.1)),
                use_batch_norm=bool(dcfg.get("use_batch_norm", False)),
                use_gradient_reversal=True,
                lambda_factor=float(gcfg.get("grl_lambda", 1.0)),
                hidden_layer_norm=bool(dcfg.get("hidden_layer_norm", True)),
                instance_noise=float(dcfg.get("instance_noise", 0.0)),
            ).to(self.device)
            ccfg = self.da_cfg.get("critic", {}) or {}
            self.optimizer_discriminator = optim.Adam(
                self.discriminator.parameters(),
                lr=float(ccfg.get("lr", 1e-3)),
                weight_decay=float(ccfg.get("weight_decay", 0.0)),
            )
            n_disc = sum(p.numel() for p in self.discriminator.parameters())
            logger.info(
                "DA discriminator: %s parameters, input_dim=%s",
                f"{n_disc:,}", self.model.feature_extractor.feature_dim,
            )
            if ckpt is not None and ckpt.get("domain_discriminator_state_dict"):
                self.discriminator.load_state_dict(
                    ckpt["domain_discriminator_state_dict"]
                )
                if (
                    resume_cfg.get("load_optimizer", True)
                    and ckpt.get("domain_optimizer_state_dict")
                ):
                    self.optimizer_discriminator.load_state_dict(
                        ckpt["domain_optimizer_state_dict"]
                    )
                logger.info("Loaded DA discriminator state from checkpoint")

        if (
            ckpt is not None
            and self.best_epoch >= 0
            and np.isfinite(self.best_metric)
            and (not self.heteroscedastic or self._resume_has_variance)
        ):
            # Make the continuation run self-contained: its incumbent best is the
            # checkpoint we started from.  It is overwritten if a later epoch improves.
            self.save_checkpoint(
                self.best_epoch, is_best=True, history=self.resume_history,
                save_periodic=False,
            )

    # -- loss / metrics -------------------------------------------------

    def _per_event_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_name: Optional[str] = None,
    ) -> torch.Tensor:
        name = str(loss_name or self.loss_name).lower()
        if name == "huber":
            return F.huber_loss(
                pred, target, delta=self.huber_delta, reduction="none"
            )
        if name == "mse":
            return (pred - target) ** 2
        if name == "mae":
            return (pred - target).abs()
        if name in ("gaussian_nll", "nll"):
            raise ValueError("use _per_event_nll for the heteroscedastic loss")
        raise ValueError(f"unknown loss {name}")

    def _nll_raw(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-event NLL and clamped log-variance (z units).

        The variance head predicts ``log_var``, not ``log_sigma``.  The scale is
        therefore ``exp(0.5 * log_var)``.  Keeping this conversion in one place
        matters: mixing ``exp(log_var)`` with a later ``exp(0.5 * log_var)``
        sigma silently optimises a different, much sharper distribution.
        """
        log_var = torch.clamp(log_var, self.log_var_min, self.log_var_max)
        log_sigma = 0.5 * log_var
        sigma = torch.exp(log_sigma)
        if self.distribution == "gaussian":
            nll = self.nll_const + log_sigma + 0.5 * (
                (target - mean) ** 2 / (sigma * sigma)
            )
        else:
            nu = self.student_t_nu
            z = (target - mean) / sigma
            nll = (
                -math.lgamma(0.5 * (nu + 1.0))
                + math.lgamma(0.5 * nu)
                + 0.5 * math.log(nu * math.pi)
                + log_sigma
                + 0.5 * (nu + 1.0) * torch.log1p(z * z / nu)
            )
        return nll, log_var

    def _per_event_nll(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nll, log_var = self._nll_raw(mean, log_var, target)
        if self.beta_nll > 0.0:
            # beta-NLL: the scale prefactor is detached, so the mean is not
            # down-weighted by a scale the model can inflate.
            nll = torch.exp(2.0 * self.beta_nll * log_var).detach() * nll
        return nll, log_var

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
        return out

    def _forward_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(batch)

    def _da_lambda(self, progress: float) -> float:
        """Warmup/ramp schedule for the adversarial term (0 during warmup)."""
        gcfg = self.da_cfg.get("grl", {}) or {}
        warmup = float(gcfg.get("warmup_epochs", 0.0))
        ramp = float(gcfg.get("ramp_epochs", 0.0))
        max_lambda = float(gcfg.get("max_lambda", 0.05))
        if progress < warmup:
            return 0.0
        if ramp > 0.0:
            return max_lambda * min(1.0, (progress - warmup) / ramp)
        return max_lambda

    def _next_domain_batch(self, name: str, loader) -> Optional[Dict[str, Any]]:
        """Cycle a domain loader independently of the task loader."""
        it = self._da_iters.get(name)
        if it is None:
            it = iter(loader)
            self._da_iters[name] = it
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            self._da_iters[name] = it
            try:
                batch = next(it)
            except StopIteration:
                return None
        return self._move_batch(batch)

    def _balanced_domain_batches(
        self,
        source_batch: Dict[str, Any],
        target_batch: Dict[str, Any],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
        """z-flip both domains and align the two halves to one size.

        The target is duplicated original+mirror; a random half of the source is
        mirrored.  Both sides are then subsampled to ``min(n_source, 2*n_target)``
        so vertical orientation is not a domain cue.  Returns the two batches and
        the average domain utilization used to normalise the domain loss.
        """
        n_source = source_batch["features"].shape[0]
        tgt_feat = target_batch["features"]
        tgt_mask = target_batch["mask"]
        tgt_flip = tgt_feat.clone()
        tgt_flip[:, :, 4] = torch.where(
            tgt_mask, -tgt_feat[:, :, 4], tgt_feat[:, :, 4]
        )
        tgt_feat_cat = torch.cat([tgt_feat, tgt_flip], dim=0)
        tgt_lengths_cat = torch.cat(
            [target_batch["lengths"], target_batch["lengths"]], dim=0
        )
        tgt_mask_cat = torch.cat([tgt_mask, tgt_mask], dim=0)
        n_target = tgt_feat_cat.shape[0]

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
        src_feat = source_batch["features"][sel_src].clone()
        src_mask = source_batch["mask"][sel_src]
        n_flip = domain_bs // 2
        flip_idx = torch.randperm(domain_bs, device=self.device)[:n_flip]
        src_feat[flip_idx, :, 4] = torch.where(
            src_mask[flip_idx],
            -src_feat[flip_idx, :, 4],
            src_feat[flip_idx, :, 4],
        )
        src_dom = {
            "features": src_feat,
            "lengths": source_batch["lengths"][sel_src],
            "mask": src_mask,
        }
        tgt_dom = {
            "features": tgt_feat_cat[sel_tgt],
            "lengths": tgt_lengths_cat[sel_tgt],
            "mask": tgt_mask_cat[sel_tgt],
        }
        util = 0.5 * (domain_bs / max(n_source, 1) + domain_bs / max(n_target, 1))
        return src_dom, tgt_dom, util

    def _domain_step(self, da_coef: float) -> Optional[Dict[str, Any]]:
        """Run one DA step and return the adversarial loss.

        Two modes are selected by ``domain_adaptation.critic``:

        * ``steps=0, update_on_adversarial=true`` -- simultaneous DANN: one
          adversarial pass, then both the encoder and the discriminator step
          from gradients computed at the same parameters.
        * ``steps>0, update_on_adversarial=false`` -- detached critic: the
          critic is trained for ``steps`` inner steps on frozen features, then
          the encoder takes one adversarial step; the critic is not updated by
          the adversarial pass.
        * ``steps>0, update_on_adversarial=true`` is the hybrid: inner critic
          steps followed by a separate adversarial update of both.

        The GRL lambda carries ``da_coef``; the adversarial loss itself is not
        multiplied by it, so the discriminator's gradient is independent of the
        schedule (matching the classifier DANN).  At ``da_coef=0`` no
        adversarial pass runs; only the optional critic steps do (lambda=0
        control).
        """
        if self.source_da_loader is None or self.target_da_loader is None:
            return None
        ccfg = self.da_cfg.get("critic", {}) or {}
        gcfg = self.da_cfg.get("grl", {}) or {}
        critic_steps = int(ccfg.get("steps", 1))
        update_on_adv = bool(ccfg.get("update_on_adversarial", False))
        critic_losses: list[float] = []
        critic_accs: list[float] = []

        utils: list[float] = []
        for _ in range(max(0, critic_steps)):
            src_batch = self._next_domain_batch("source", self.source_da_loader)
            tgt_batch = self._next_domain_batch("target", self.target_da_loader)
            if src_batch is None or tgt_batch is None:
                break
            src_dom, tgt_dom, util = self._balanced_domain_batches(
                src_batch, tgt_batch
            )
            utils.append(util)
            with torch.no_grad():
                feat_s = self.model.get_feature_representation(src_dom)
                feat_t = self.model.get_feature_representation(tgt_dom)
            logits = self.discriminator(torch.cat([feat_s, feat_t], dim=0))
            labels = torch.cat([
                torch.zeros(len(feat_s), 1, device=self.device),
                torch.ones(len(feat_t), 1, device=self.device),
            ])
            loss = (
                F.binary_cross_entropy_with_logits(logits, labels)
                / max(util, 1e-6)
            )
            self.optimizer_discriminator.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer_discriminator.step()
            critic_losses.append(float(loss.detach().item()))
            critic_accs.append(
                float(((logits.detach() > 0).float() == labels).float().mean().item())
            )

        if da_coef <= 0.0:
            return {
                "adv_loss": None,
                "critic_loss": float(np.mean(critic_losses)) if critic_losses else None,
                "critic_acc": float(np.mean(critic_accs)) if critic_accs else None,
                "domain_util": float(np.mean(utils)) if utils else 1.0,
                "update_discriminator": False,
            }

        # Adversarial pass: the encoder must see a graph.  The GRL lambda carries
        # the schedule, so the discriminator's own gradient is not scaled by it
        # (matching the classifier DANN).  If the discriminator is also updated
        # here, clear the critic-step grads first so this is a clean adv update.
        self.discriminator.set_lambda(
            float(gcfg.get("grl_lambda", 1.0)) * da_coef
        )
        if update_on_adv:
            self.optimizer_discriminator.zero_grad(set_to_none=True)
        src_batch = self._next_domain_batch("source", self.source_da_loader)
        tgt_batch = self._next_domain_batch("target", self.target_da_loader)
        if src_batch is None or tgt_batch is None:
            return None
        src_dom, tgt_dom, util = self._balanced_domain_batches(
            src_batch, tgt_batch
        )
        feat_s = self.model.get_feature_representation(src_dom)
        feat_t = self.model.get_feature_representation(tgt_dom)
        logits = self.discriminator(torch.cat([feat_s, feat_t], dim=0))
        labels = torch.cat([
            torch.zeros(len(feat_s), 1, device=self.device),
            torch.ones(len(feat_t), 1, device=self.device),
        ])
        adv_loss = (
            F.binary_cross_entropy_with_logits(logits, labels)
            / max(util, 1e-6)
        )
        with torch.no_grad():
            adv_acc = float(
                ((logits.detach() > 0).float() == labels).float().mean().item()
            )
        return {
            "adv_loss": adv_loss,
            "critic_loss": (
                float(np.mean(critic_losses)) if critic_losses
                else float(adv_loss.detach().item())
            ),
            "critic_acc": (
                float(np.mean(critic_accs)) if critic_accs else adv_acc
            ),
            "domain_util": float(np.mean(utils)) if utils else util,
            "update_discriminator": update_on_adv,
        }

    def _collect_calibration_arrays(self, loader) -> Dict[str, np.ndarray]:
        """Collect predictions, targets and sigma on a loader, without gradients."""
        self._set_train_mode(False)
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        sigmas: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch(batch)
                pred, log_var = self.model.forward_with_variance(batch)
                log_var = torch.clamp(log_var, self.log_var_min, self.log_var_max)
                pred_dex = (
                    pred.float() * self.target_std + self.target_mean
                    if self.target_normalization is not None
                    else pred.float()
                )
                sigma_dex = torch.exp(0.5 * log_var.float()) * self.target_std
                preds.append(pred_dex.cpu().numpy())
                targets.append(batch["targets_dex"].detach().cpu().numpy())
                sigmas.append(sigma_dex.cpu().numpy())
                weights.append(batch["weights"].detach().cpu().numpy())
        if not preds:
            return {
                "pred": np.zeros(0, np.float32),
                "target": np.zeros(0, np.float32),
                "sigma": np.zeros(0, np.float32),
                "weight": np.zeros(0, np.float32),
            }
        return {
            "pred": np.concatenate(preds),
            "target": np.concatenate(targets),
            "sigma": np.concatenate(sigmas),
            "weight": np.concatenate(weights),
        }

    @staticmethod
    def _coverage_at_scale(
        pred: np.ndarray, target: np.ndarray, sigma: np.ndarray, scale: float
    ) -> float:
        abs_err = np.abs(pred - target)
        return float(np.mean(abs_err <= sigma * float(scale)))

    def _calibration_scale_summary(self) -> Dict[str, float]:
        """Fit global sigma factors on cal and report them on cal/val.

        The factors are empirical quantiles of |error| / sigma.  They are fitted
        on the calibration split and applied unchanged to the validation split;
        the two sets are not mixed.  This is a global re-scaling, not a
        conditional calibration.
        """
        if self.cal_loader is None:
            return {}
        cal = self._collect_calibration_arrays(self.cal_loader)
        if len(cal["pred"]) == 0:
            return {}
        ratio = np.abs(cal["pred"] - cal["target"]) / np.maximum(
            cal["sigma"], 1e-12
        )
        q68 = float(np.quantile(ratio, 0.68))
        q95 = float(np.quantile(ratio, 0.95))
        out: Dict[str, float] = {
            "scale_factor_68": q68,
            "scale_factor_95": q95,
            "n_cal": float(len(ratio)),
            "raw_cal_coverage_68": self._coverage_at_scale(
                cal["pred"], cal["target"], cal["sigma"], 1.0
            ),
            "raw_cal_coverage_95": self._coverage_at_scale(
                cal["pred"], cal["target"], cal["sigma"], 1.96
            ),
            "calibrated_cal_coverage_68": self._coverage_at_scale(
                cal["pred"], cal["target"], cal["sigma"], q68
            ),
            "calibrated_cal_coverage_95": self._coverage_at_scale(
                cal["pred"], cal["target"], cal["sigma"], q95
            ),
        }
        if self.val_loader is not None:
            val = self._collect_calibration_arrays(self.val_loader)
            if len(val["pred"]) > 0:
                out["n_val"] = float(len(val["pred"]))
                out["raw_val_coverage_68"] = self._coverage_at_scale(
                    val["pred"], val["target"], val["sigma"], 1.0
                )
                out["raw_val_coverage_95"] = self._coverage_at_scale(
                    val["pred"], val["target"], val["sigma"], 1.96
                )
                out["calibrated_val_coverage_68"] = self._coverage_at_scale(
                    val["pred"], val["target"], val["sigma"], q68
                )
                out["calibrated_val_coverage_95"] = self._coverage_at_scale(
                    val["pred"], val["target"], val["sigma"], q95
                )
        return out

    def _run_epoch(
        self, loader, train: bool, max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        self._set_train_mode(train)
        da_active = bool(train and self.da_enabled)
        if da_active:
            self.discriminator.train(True)
            self._da_iters = {}
        acc = _new_accumulator()
        unc_acc = _new_uncertainty_accumulator() if self.heteroscedastic else None
        elapsed_epoch = self.current_epoch - self.start_epoch
        use_nll = bool(
            self.heteroscedastic and elapsed_epoch >= self.warmup_epochs
        )
        da_metrics = {"critic_loss": 0.0, "critic_acc": 0.0,
                      "domain_util": 0.0, "da_coef": 0.0,
                      "da_term": 0.0, "n_da": 0}
        grad_clip = float(self.config["training"].get("grad_clip", 0.0))
        log_every = int(self.config["logging"].get("log_every", 50))
        t0 = time.time()
        n_batches = max(len(loader), 1)
        n_steps = 0

        context = torch.enable_grad if train else torch.no_grad
        with context():
            for step, batch in enumerate(loader):
                if max_batches is not None and step >= max_batches:
                    break
                n_steps += 1
                step_discriminator = False
                batch = self._move_batch(batch)
                target = batch["targets"]          # z units
                target_dex = batch["targets_dex"]  # raw log10(E/GeV)
                weight = batch["weights"]
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.optimizer_sigma is not None:
                        self.optimizer_sigma.zero_grad(set_to_none=True)
                    if da_active:
                        self.optimizer_discriminator.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    sigma_task = None
                    if self.heteroscedastic:
                        pred, log_var = self.model.forward_with_variance(
                            batch,
                            detach_sigma_features=(
                                self.sigma_strategy == "detach"
                            ),
                        )
                        if use_nll:
                            full_per_event, log_var = self._per_event_nll(
                                pred, log_var, target
                            )
                            per_event = full_per_event
                            if self.sigma_strategy == "detach":
                                # Two detached NLL terms: the mean term trains
                                # encoder+mean, the sigma term trains only the
                                # variance head.  Their values decompose the NLL.
                                mean_per_event, _ = self._nll_raw(
                                    pred, log_var.detach(), target
                                )
                                sigma_per_event, _ = self._nll_raw(
                                    pred.detach(), log_var, target
                                )
                                mean_task = (
                                    (mean_per_event * weight).sum()
                                    / weight.sum().clamp_min(1e-12)
                                )
                                sigma_task = (
                                    (sigma_per_event * weight).sum()
                                    / weight.sum().clamp_min(1e-12)
                                )
                                task_loss = mean_task + sigma_task
                                loss_unweighted = full_per_event.mean()
                                main_loss = mean_task
                            else:
                                task_loss = (
                                    (per_event * weight).sum()
                                    / weight.sum().clamp_min(1e-12)
                                )
                                loss_unweighted = per_event.mean()
                                main_loss = task_loss
                        else:
                            per_event = self._per_event_loss(
                                pred, target, loss_name=self.warmup_loss_name
                            )
                            log_var = torch.clamp(
                                log_var, self.log_var_min, self.log_var_max
                            )
                            task_loss = (
                                (per_event * weight).sum()
                                / weight.sum().clamp_min(1e-12)
                            )
                            loss_unweighted = per_event.mean()
                            main_loss = task_loss
                    else:
                        pred = self._forward_batch(batch)
                        per_event = self._per_event_loss(pred, target)
                        task_loss = (
                            (per_event * weight).sum()
                            / weight.sum().clamp_min(1e-12)
                        )
                        loss_unweighted = per_event.mean()
                        main_loss = task_loss
                    loss = main_loss
                    da_coef = 0.0
                    if da_active:
                        da_coef = self._da_lambda(
                            (self.current_epoch - self.start_epoch)
                            + step / n_batches
                        )
                        da_out = self._domain_step(da_coef)
                        if da_out is not None:
                            if da_out["critic_loss"] is not None:
                                da_metrics["critic_loss"] += da_out["critic_loss"]
                                da_metrics["critic_acc"] += da_out["critic_acc"]
                                da_metrics["domain_util"] += da_out.get(
                                    "domain_util", 1.0
                                )
                                da_metrics["n_da"] += 1
                            da_metrics["da_coef"] += da_coef
                            if da_out["adv_loss"] is not None:
                                # GRL carries da_coef; the loss itself is unscaled,
                                # like the classifier DANN.  Log the effective term.
                                loss = main_loss + da_out["adv_loss"]
                                main_loss = loss
                                da_metrics["da_term"] += float(
                                    (da_coef * da_out["adv_loss"]).detach().item()
                                )
                                step_discriminator = bool(
                                    da_out.get("update_discriminator", False)
                                )
                if train:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        if grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), grad_clip
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    elif (
                        self.sigma_strategy == "detach"
                        and sigma_task is not None
                    ):
                        # Two independent graphs: mean/encoder and sigma head.
                        main_loss.backward()
                        self._clip_params(self._main_params, grad_clip)
                        self.optimizer.step()
                        sigma_task.backward()
                        self._clip_params(self._sigma_params, grad_clip)
                        self.optimizer_sigma.step()
                        if da_active and step_discriminator:
                            self.optimizer_discriminator.step()
                    else:
                        loss.backward()
                        if self.sigma_strategy in ("detach", "full_two_opt"):
                            self._clip_params(self._main_params, grad_clip)
                            self._clip_params(self._sigma_params, grad_clip)
                        elif self.sigma_strategy == "frozen":
                            self._clip_params(self._sigma_params, grad_clip)
                        else:
                            self._clip_params(self._main_params, grad_clip)
                        self.optimizer.step()
                        if self.optimizer_sigma is not None:
                            self.optimizer_sigma.step()
                        if da_active and step_discriminator:
                            self.optimizer_discriminator.step()

                pred_dex = (
                    pred.detach().float() * self.target_std + self.target_mean
                    if self.target_normalization is not None
                    else pred.detach().float()
                )
                _update_accumulator(
                    acc,
                    pred_dex,
                    target_dex,
                    weight,
                    batch["particle_types"],
                    loss_sum=float(task_loss.item()) * float(weight.sum().item()),
                    loss_unweighted_sum=float(loss_unweighted.item()) * len(weight),
                )
                if self.heteroscedastic:
                    log_var_d = log_var.detach().float()
                    var_d = torch.exp(log_var_d)
                    nll_raw, _ = self._nll_raw(
                        pred.detach(), log_var.detach(), target.detach()
                    )
                    sigma_dex = torch.exp(0.5 * log_var_d) * self.target_std
                    _update_uncertainty(
                        unc_acc, pred_dex, target_dex, sigma_dex,
                        nll_raw, weight,
                        distribution=self.distribution,
                        student_t_nu=self.student_t_nu,
                    )
                if train and (step + 1) % log_every == 0:
                    m = _finalize(acc)
                    extra = ""
                    if da_active and da_metrics["n_da"]:
                        extra = (
                            f"  domain {da_metrics['critic_loss'] / da_metrics['n_da']:.4f}"
                            f" acc {da_metrics['critic_acc'] / da_metrics['n_da']:.3f}"
                            f" util {da_metrics['domain_util'] / da_metrics['n_da']:.2f}"
                            f" lambda {da_metrics['da_coef'] / max(n_steps, 1):.4f}"
                        )
                    logger.info(
                        "  [%d/%d] loss %.4f (unw %.4f)  mae %.4f dex  "
                        "bias %+.4f%s  (%.1f ev/s)",
                        step + 1, len(loader), m["loss"], m["loss_unweighted"],
                        m["mae_log10"], m["bias_log10"], extra,
                        acc["n"] / max(time.time() - t0, 1e-3),
                    )
        metrics = _finalize(acc)
        if self.heteroscedastic:
            metrics.update(_finalize_uncertainty(unc_acc))
            metrics["unc_phase"] = 1.0 if use_nll else 0.0
        if da_active and da_metrics["n_da"]:
            n = da_metrics["n_da"]
            metrics.update({
                "domain_loss": da_metrics["critic_loss"] / n,
                "domain_acc": da_metrics["critic_acc"] / n,
                "domain_util": da_metrics["domain_util"] / n,
                "da_coef": da_metrics["da_coef"] / max(n_steps, 1),
                "da_loss_term": da_metrics["da_term"] / max(n_steps, 1),
            })
        metrics["epoch_seconds"] = time.time() - t0
        return metrics

    def _refresh_discriminator(self, steps: int) -> None:
        """Re-fit the critic on frozen features (DArepair-style refresh)."""
        if self.source_da_loader is None or self.target_da_loader is None:
            return
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        self.discriminator.train()
        src_iter = iter(self.source_da_loader)
        tgt_iter = iter(self.target_da_loader)
        losses: list[float] = []

        def next_batch(iterator, loader):
            try:
                return next(iterator), iterator
            except StopIteration:
                iterator = iter(loader)
                return next(iterator), iterator

        for _ in range(max(0, int(steps))):
            src_batch, src_iter = next_batch(src_iter, self.source_da_loader)
            tgt_batch, tgt_iter = next_batch(tgt_iter, self.target_da_loader)
            src_batch = self._move_batch(src_batch)
            tgt_batch = self._move_batch(tgt_batch)
            src_dom, tgt_dom, util = self._balanced_domain_batches(
                src_batch, tgt_batch
            )
            with torch.no_grad():
                feat_s = self.model.get_feature_representation(src_dom)
                feat_t = self.model.get_feature_representation(tgt_dom)
            logits = self.discriminator(torch.cat([feat_s, feat_t], dim=0))
            labels = torch.cat([
                torch.zeros(len(feat_s), 1, device=self.device),
                torch.ones(len(feat_t), 1, device=self.device),
            ])
            loss = (
                F.binary_cross_entropy_with_logits(logits, labels)
                / max(util, 1e-6)
            )
            self.optimizer_discriminator.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer_discriminator.step()
            losses.append(float(loss.detach().item()))

        for param in self.model.parameters():
            param.requires_grad_(True)
        self.model.train()
        logger.info(
            "  discriminator refresh: %d steps, last loss %.4f",
            len(losses), losses[-1] if losses else float("nan"),
        )

    def _evaluate_domain_probe(self) -> Dict[str, float]:
        """Linear domain probe on held-out parts (not the adversarial critic)."""
        if not self.da_enabled or self.source_probe_loader is None:
            return {}
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        self.model.eval()
        self.discriminator.eval()

        def collect(loader, label: float) -> tuple[np.ndarray, np.ndarray]:
            feats: list[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    batch = self._move_batch(batch)
                    feats.append(
                        self.model.get_feature_representation(batch)
                        .float().cpu().numpy()
                    )
            if not feats:
                return np.zeros((0, 1), np.float32), np.zeros(0, np.float32)
            return np.concatenate(feats), np.full(
                sum(len(f) for f in feats), label, dtype=np.float32
            )

        src_x, src_y = collect(self.source_probe_loader, 0.0)
        tgt_x, tgt_y = collect(self.target_probe_loader, 1.0)
        self.model.train()
        self.discriminator.train()
        if len(src_x) == 0 or len(tgt_x) == 0:
            return {}
        x = np.concatenate([src_x, tgt_x])
        y = np.concatenate([src_y, tgt_y])
        if len(np.unique(y)) < 2:
            return {}
        seed = int((self.da_cfg.get("probe", {}) or {}).get("seed", 42))
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.5, random_state=seed, stratify=y,
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_tr, y_tr)
        proba = clf.predict_proba(x_te)[:, 1]
        with torch.no_grad():
            critic_logits = self.discriminator.predict_domain(
                torch.from_numpy(x_te).float().to(self.device)
            ).cpu().numpy().ravel()
        out = {
            "domain_probe_auc": float(roc_auc_score(y_te, proba)),
            "domain_probe_acc": float(clf.score(x_te, y_te)),
            "domain_critic_auc": float(roc_auc_score(y_te, critic_logits)),
        }
        probe_cfg = self.da_cfg.get("probe", {}) or {}
        if probe_cfg.get("mlp_enabled", False):
            from sklearn.neural_network import MLPClassifier
            mlp = MLPClassifier(
                hidden_layer_sizes=tuple(probe_cfg.get("mlp_hidden", [128])),
                max_iter=int(probe_cfg.get("mlp_max_iter", 200)),
                early_stopping=True,
                random_state=seed,
            )
            mlp.fit(x_tr, y_tr)
            mlp_proba = mlp.predict_proba(x_te)[:, 1]
            out["domain_probe_mlp_auc"] = float(roc_auc_score(y_te, mlp_proba))
            out["domain_probe_mlp_acc"] = float(mlp.score(x_te, y_te))
        return out

    # -- checkpointing --------------------------------------------------

    def save_checkpoint(
        self, epoch: int, is_best: bool, history: List[Dict], save_periodic: bool
    ) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "optimizer_sigma_state_dict": (
                self.optimizer_sigma.state_dict()
                if self.optimizer_sigma is not None else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "resumed_from": self.resumed_from,
            "experiment_id": self.experiment_id,
            "sigma_strategy": self.sigma_strategy,
            "config": self.config,
            "normalization_config": self.normalization_config,
            "target_normalization": self.target_normalization,
            "training_history": history,
        }
        if self.da_enabled:
            ckpt["domain_adaptation_config"] = self.da_cfg
            ckpt["domain_discriminator_state_dict"] = self.discriminator.state_dict()
            ckpt["domain_optimizer_state_dict"] = (
                self.optimizer_discriminator.state_dict()
            )
        torch.save(ckpt, self.output_dir / "latest_checkpoint.pth")
        if is_best:
            torch.save(ckpt, self.output_dir / "best_model.pth")
            logger.info("New best model at epoch %d (%.4f)", epoch, self.best_metric)
        if save_periodic:
            torch.save(ckpt, self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth")

    # -- training -------------------------------------------------------

    def train(self) -> None:
        self.prepare_data()
        self.prepare_model()
        tcfg = self.config["training"]
        es = tcfg.get("early_stopping", {})
        monitor = str(es.get("monitor", "val_loss"))
        monitor_key = monitor[len("val_"):] if monitor.startswith("val_") else monitor
        es_enabled = bool(es.get("enabled", True))
        patience = int(es.get("patience", 10))
        min_delta = float(es.get("min_delta", 1e-4))
        save_every = int(tcfg.get("save_every", 5))
        validate_every = int(tcfg.get("validate_every", 1))
        debug_cfg = self.config.get("debug", {})
        max_train_batches = (
            int(debug_cfg.get("max_batches_per_epoch", 10))
            if debug_cfg.get("enabled", False) else None
        )
        max_val_batches = (
            int(debug_cfg.get("max_val_batches", 10))
            if debug_cfg.get("enabled", False) else None
        )

        history: List[Dict] = list(self.resume_history)
        total_epochs = int(tcfg["epochs"])
        if not (0 <= self.start_epoch < total_epochs):
            raise ValueError(
                f"start epoch {self.start_epoch} is outside [0, {total_epochs})"
            )
        logger.info(
            "Starting training: epochs %d..%d (total %d)",
            self.start_epoch, total_epochs - 1, total_epochs,
        )
        for epoch in range(self.start_epoch, total_epochs):
            self.current_epoch = epoch
            train_metrics = self._run_epoch(
                self.train_loader, train=True, max_batches=max_train_batches
            )

            val_metrics: Dict[str, float] = {}
            if epoch % validate_every == 0:
                val_metrics = self._run_epoch(
                    self.val_loader, train=False, max_batches=max_val_batches
                )

            if self.da_enabled and val_metrics:
                probe_every = int(
                    (self.da_cfg.get("probe", {}) or {}).get("every_n_epochs", 1)
                )
                if probe_every > 0 and epoch % probe_every == 0:
                    val_metrics.update(self._evaluate_domain_probe())

            scheduler_frozen = bool(
                self.heteroscedastic
                and (self.current_epoch - self.start_epoch) < self.warmup_epochs
            )
            refresh_cfg = self.da_cfg.get("discriminator_refresh", {}) or {}
            refresh_every = int(refresh_cfg.get("every_epochs", 0) or 0)
            if (
                self.da_enabled
                and refresh_every > 0
                and (self.current_epoch + 1) % refresh_every == 0
            ):
                self._refresh_discriminator(
                    int(refresh_cfg.get("steps", 300))
                )

            if self.scheduler is not None and not scheduler_frozen:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            entry = {
                "epoch": epoch,
                "lr": lr,
                "epoch_time": train_metrics["epoch_seconds"]
                + val_metrics.get("epoch_seconds", 0.0),
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(entry)
            self._save_history(history)

            if self.writer is not None:
                self._log_tensorboard(train_metrics, val_metrics, lr, epoch)

            logger.info(
                "Epoch %d/%d [%.1fs] train loss %.4f (unw %.4f) mae %.4f dex | "
                "val loss %.4f (unw %.4f) mae %.4f dex bias %+.4f within x2 %.2f",
                epoch, total_epochs - 1, entry["epoch_time"],
                train_metrics["loss"], train_metrics["loss_unweighted"],
                train_metrics["mae_log10"],
                val_metrics.get("loss", float("nan")),
                val_metrics.get("loss_unweighted", float("nan")),
                val_metrics.get("mae_log10", float("nan")),
                val_metrics.get("bias_log10", float("nan")),
                val_metrics.get("within_factor_2", float("nan")),
            )

            is_best = False
            can_select_best = not (
                self.heteroscedastic
                and (self.current_epoch - self.start_epoch) < self.warmup_epochs
            )
            if val_metrics and monitor_key in val_metrics and can_select_best:
                current = val_metrics[monitor_key]
                if current < self.best_metric - min_delta:
                    self.best_metric = current
                    self.best_epoch = epoch
                    self.early_stopping_counter = 0
                    is_best = True
                else:
                    self.early_stopping_counter += 1

            self.save_checkpoint(
                epoch, is_best, history,
                save_periodic=(epoch % save_every == 0),
            )

            if es_enabled and self.early_stopping_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if self.cal_loader is not None:
            calibration = {
                "latest": self._run_epoch(self.cal_loader, train=False),
            }
            best_path = self.output_dir / "best_model.pth"
            if best_path.exists():
                best_ckpt = torch.load(
                    best_path, map_location=self.device, weights_only=False
                )
                self.model.load_state_dict(
                    best_ckpt["model_state_dict"], strict=True
                )
                calibration["best"] = self._run_epoch(
                    self.cal_loader, train=False
                )
            self.calibration_metrics = calibration
            for tag, m in calibration.items():
                logger.info(
                    "Calibration[%s]: mae %.4f dex, bias %+.4f, "
                    "nll %s, cov68 %s, cov95 %s, pit_ks %s",
                    tag, m.get("mae_log10", float("nan")),
                    m.get("bias_log10", float("nan")),
                    m.get("unc_nll"), m.get("unc_coverage_68"),
                    m.get("unc_coverage_95"), m.get("unc_pit_ks"),
                )
            if self.heteroscedastic:
                self.calibration_metrics["scale_factors"] = (
                    self._calibration_scale_summary()
                )
                logger.info(
                    "Calibration scale: %s",
                    self.calibration_metrics["scale_factors"],
                )

        self._save_summary(history)
        logger.info(
            "Training complete. Best %s %.4f at epoch %d",
            monitor, self.best_metric, self.best_epoch,
        )

    # -- logging --------------------------------------------------------

    def _save_history(self, history: List[Dict]) -> None:
        try:
            import pandas as pd

            pd.DataFrame(history).to_csv(
                self.output_dir / "training_history.csv", index=False
            )
        except Exception as exc:  # pragma: no cover - pandas is present
            logger.error("Failed to save history CSV: %s", exc)

    def _save_summary(self, history: List[Dict]) -> None:
        summary = {
            "experiment_id": self.experiment_id,
            "sigma_strategy": self.sigma_strategy,
            "frozen_base": self.sigma_strategy == "frozen",
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "monitor": self.config["training"].get("early_stopping", {}).get(
                "monitor", "val_loss"
            ),
            "resumed_from": self.resumed_from,
            "start_epoch": self.start_epoch,
            "target_normalization": self.target_normalization,
            "total_epochs": len(history),
            "final_train": {
                k: v for k, v in history[-1].items()
                if not k.startswith("val_") and isinstance(v, (int, float))
            } if history else {},
            "final_val": {
                k[len("val_"):]: v for k, v in history[-1].items()
                if k.startswith("val_")
            } if history else {},
            "model_parameters": self.model.count_parameters(),
            "calibration": self.calibration_metrics,
        }
        with open(self.output_dir / "training_summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

    def _log_tensorboard(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        epoch: int,
    ) -> None:
        assert self.writer is not None
        for name in (
            "loss", "loss_unweighted", "mae_log10", "rmse_log10", "bias_log10",
            "mae_log10_weighted", "mae_factor",
            "within_factor_2", "within_factor_5", "within_factor_10",
        ):
            if name in train_metrics:
                self.writer.add_scalar(f"Train/{name}", train_metrics[name], epoch)
        for name in (
            "da_coef", "domain_loss", "domain_acc", "da_loss_term",
            "unc_nll", "unc_mean_sigma_dex", "unc_coverage_1sigma",
            "unc_coverage_2sigma", "unc_coverage_68", "unc_coverage_95",
            "unc_sigma_error_corr", "unc_pit_ks",
        ):
            if name in train_metrics:
                self.writer.add_scalar(f"Train/{name}", train_metrics[name], epoch)
        for name, value in val_metrics.items():
            if name == "epoch_seconds":
                continue
            self.writer.add_scalar(f"Val/{name}", value, epoch)
        self.writer.add_scalar("LearningRate", lr, epoch)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.debug:
        config.setdefault("debug", {})["enabled"] = True
        config["training"]["epochs"] = 2
    logging.basicConfig(
        level=getattr(logging, config["logging"].get("level", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    with MuEnergyTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main()
