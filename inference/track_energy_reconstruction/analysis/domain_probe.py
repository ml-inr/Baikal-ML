"""Reproducible, converged domain-probe evaluation for energy checkpoints.

The trainer's per-epoch probe uses ``MLPClassifier(max_iter=200,
early_stopping=True)`` on unscaled features.  That is not sufficient for a
certificate of domain invariance: it under-fits and can report an AUC that is too
close to chance.  This script extracts the same held-out DA probe events, fits a
standardised linear probe and a larger MLP (three seeds), and writes a CSV for the
analysis notebook.

The same features support two splits inside the DA-held-out parts: an event-level
split (comparable with the training-history probe) and a stricter part-disjoint
split where entire part keys are assigned to fit or test.  The latter is the
primary check: a probe can otherwise exploit part-specific detector conditions
shared by events in the same part.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference.track_energy_reconstruction.predict_full import load_energy_model
from src.data.mu_energy_dataset import (
    ExpEnergyDomainDataset,
    McEnergyDomainDataset,
    create_mu_energy_dataloader,
    part_split_indices,
)


def _encoder_da_enabled(
    run_dir: Path, _seen: set[Path] | None = None
) -> bool:
    """Whether this checkpoint's encoder was trained with DA.

    Frozen-head runs disable ``domain_adaptation`` while fitting the sigma head,
    but they inherit the encoder from ``resume.from_checkpoint``.  The probe
    measures the encoder features, so the inherited setting is the relevant one.
    """
    _seen = set() if _seen is None else _seen
    run_dir = run_dir.resolve()
    if run_dir in _seen:
        return False
    _seen.add(run_dir)
    cfg_path = run_dir / "train_config.yaml"
    if not cfg_path.exists():
        return False
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    if bool((cfg.get("domain_adaptation") or {}).get("enabled", False)):
        return True
    parent_ckpt = (cfg.get("resume") or {}).get("from_checkpoint")
    if not parent_ckpt:
        return False
    parent = Path(str(parent_ckpt))
    if not parent.is_absolute():
        parent = ROOT / parent
    return _encoder_da_enabled(parent.parent, _seen)


RUN_SPECS = [
    ("BASE", "261027_0935_mu_energy_BASE_cont_partsplit_s42"),
    ("A", "261026_2057_mu_energy_A_lam0p01_detachedcritic_partsplit_scratch_s42"),
    ("D-G", "261027_0935_mu_energy_D-G_partsplit_scratch_s42"),
    ("D-T5", "261027_0935_mu_energy_D-T5_partsplit_scratch_s42"),
    ("C-G", "261027_0935_mu_energy_C-G_partsplit_scratch_s42"),
    ("C-T5", "261027_0935_mu_energy_C-T5_partsplit_scratch_s42"),
    ("B-G", "261027_0935_mu_energy_B-G_partsplit_scratch_s42"),
    ("B-T5", "261027_0935_mu_energy_B-T5_partsplit_scratch_s42"),
    ("AB-G", "261027_0935_mu_energy_AB-G_partsplit_scratch_s42"),
    ("AB-T5", "261027_0935_mu_energy_AB-T5_partsplit_scratch_s42"),
    ("AD-G", "261027_0935_mu_energy_AD-G_partsplit_scratch_s42"),
    ("AD-T5", "261027_0935_mu_energy_AD-T5_partsplit_scratch_s42"),
    ("AC-G", "261027_0935_mu_energy_AC-G_partsplit_scratch_s42"),
    ("AC-T5", "261027_0935_mu_energy_AC-T5_partsplit_scratch_s42"),
    ("H-G", "261027_0935_mu_energy_H-G_d0base_s42"),
    ("H-T5", "261027_0935_mu_energy_H-T5_d0base_s42"),
    ("AH-G", "261027_0935_mu_energy_AH-G_abase_s42"),
    ("AH-T5", "261027_0935_mu_energy_AH-T5_abase_s42"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-events", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(
        ROOT / "inference/track_energy_reconstruction/analysis/reports/da_probe.csv"
    ))
    return ap.parse_args()


def _part_split_masks(
    parts: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split source and target probe events by held-out part key.

    Both domains contribute train and test events.  Within each domain, whole
    part keys go to one side, so no event from a test part is seen while fitting
    the probe.
    """
    rng = np.random.RandomState(seed)
    train = np.zeros(len(y), dtype=bool)
    test = np.zeros(len(y), dtype=bool)
    for domain in (0.0, 1.0):
        domain_mask = y == domain
        unique = np.array(sorted(set(parts[domain_mask].tolist())), dtype=object)
        if len(unique) < 2:
            raise RuntimeError(
                f"domain {domain}: need at least two probe parts, got {len(unique)}"
            )
        perm = rng.permutation(len(unique))
        n_train = max(1, min(len(unique) - 1, len(unique) // 2))
        train_parts = set(unique[perm[:n_train]].tolist())
        in_train = np.array([p in train_parts for p in parts], dtype=bool)
        train |= domain_mask & in_train
        test |= domain_mask & ~in_train
    return np.flatnonzero(train), np.flatnonzero(test)


def _fit_probes(
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    seed: int,
) -> dict:
    """Fit scaled linear and three-seed MLP probes; return test AUCs."""
    scaler = StandardScaler().fit(x[idx_train])
    x_train = scaler.transform(x[idx_train])
    x_test = scaler.transform(x[idx_test])
    lin = LogisticRegression(max_iter=2000)
    lin.fit(x_train, y[idx_train])
    lin_auc = float(roc_auc_score(y[idx_test], lin.predict_proba(x_test)[:, 1]))
    mlp_aucs = []
    n_iters = []
    for mlp_seed in (seed, seed + 1, seed + 2):
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=2000,
            early_stopping=True, n_iter_no_change=30, tol=1e-6,
            random_state=mlp_seed,
        )
        mlp.fit(x_train, y[idx_train])
        mlp_aucs.append(
            float(roc_auc_score(y[idx_test], mlp.predict_proba(x_test)[:, 1]))
        )
        n_iters.append(int(mlp.n_iter_))
    return {
        "linear_auc": lin_auc,
        "linear_n_iter": int(lin.n_iter_[0]),
        "mlp_auc_mean": float(np.mean(mlp_aucs)),
        "mlp_auc_std": float(np.std(mlp_aucs)),
        "mlp_auc_min": float(np.min(mlp_aucs)),
        "mlp_auc_max": float(np.max(mlp_aucs)),
        "mlp_n_iter_mean": float(np.mean(n_iters)),
        "mlp_n_iter_max": int(np.max(n_iters)),
    }


def main() -> None:
    args = parse_args()
    device = args.device
    max_events = args.max_events
    seed = args.seed

    # Domain definitions are common to all energy checkpoints: source=MC muatm,
    # target=exp_full.  Read them from the A checkpoint used for DA training.
    domain_ckpt = (
        ROOT / "trained_models/track_energy_reconstruction/"
        "261026_2057_mu_energy_A_lam0p01_detachedcritic_partsplit_scratch_s42/"
        "best_model.pth"
    )
    _, _, _, domain_ckpt_data = load_energy_model(str(domain_ckpt), device)
    da = domain_ckpt_data["config"]["domain_adaptation"]
    src_cfg = da["source_da"]
    tgt_cfg = da["target_da"]

    src_full = McEnergyDomainDataset(
        str(ROOT / src_cfg["npy_dir"]),
        particle_types=src_cfg.get("particle_types", [0]),
        max_hits=500, max_events=None, seed=seed,
    )
    tgt_full = ExpEnergyDomainDataset(
        str(ROOT / tgt_cfg["npy_dir"]),
        prefix=str(tgt_cfg.get("prefix", "exp")),
        max_hits=500, max_events=None, seed=seed,
    )
    _, src_probe = part_split_indices(
        src_full.part_keys, float(src_cfg.get("probe_fraction", 0.1)), seed
    )
    _, tgt_probe = part_split_indices(
        tgt_full.part_keys, float(tgt_cfg.get("holdout_fraction", 0.2)), seed
    )
    src_probe = np.intersect1d(src_probe, src_full.all_indices)
    tgt_probe = np.intersect1d(tgt_probe, tgt_full.all_indices)
    src_ds = McEnergyDomainDataset(
        str(ROOT / src_cfg["npy_dir"]),
        particle_types=src_cfg.get("particle_types", [0]),
        indices=src_probe, max_hits=500, max_events=max_events, seed=seed,
    )
    tgt_ds = ExpEnergyDomainDataset(
        str(ROOT / tgt_cfg["npy_dir"]),
        prefix=str(tgt_cfg.get("prefix", "exp")),
        indices=tgt_probe, max_hits=500, max_events=max_events, seed=seed,
    )
    y = np.r_[
        np.zeros(len(src_ds), dtype=np.float32),
        np.ones(len(tgt_ds), dtype=np.float32),
    ]
    parts = np.r_[
        np.asarray(src_ds.part_keys[src_ds.all_indices], dtype=object),
        np.asarray(tgt_ds.part_keys[tgt_ds.all_indices], dtype=object),
    ]
    if len(parts) != len(y):
        raise RuntimeError(f"part labels {len(parts)} != probe events {len(y)}")
    all_idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        all_idx, test_size=0.5, random_state=seed, stratify=y
    )
    part_tr, part_te = _part_split_masks(parts, y, seed)
    print(f"probe events: source={len(src_ds):,}, target={len(tgt_ds):,}")
    print(
        f"event split: train={len(idx_tr):,}, test={len(idx_te):,}; "
        f"part split: train={len(part_tr):,}, test={len(part_te):,}"
    )

    rows = []
    for run_id, run_name in RUN_SPECS:
        ckpt_path = (
            ROOT / "trained_models/track_energy_reconstruction" / run_name
            / "best_model.pth"
        )
        if not ckpt_path.exists():
            print(f"skip {run_id}: missing {ckpt_path}")
            continue
        summary_path = ckpt_path.parent / "training_summary.yaml"
        if not summary_path.exists():
            print(f"skip {run_id}: training_summary.yaml missing")
            continue
        model, norm, _, _ = load_energy_model(str(ckpt_path), device)
        src_loader = create_mu_energy_dataloader(
            src_ds, int(src_cfg.get("batch_size", 512)), shuffle=False,
            normalization_config=norm, target_normalization=None,
            augmentation_config=None, num_workers=0,
        )
        tgt_loader = create_mu_energy_dataloader(
            tgt_ds, int(tgt_cfg.get("batch_size", 256)), shuffle=False,
            normalization_config=norm, target_normalization=None,
            augmentation_config=None, num_workers=0,
        )

        def collect(loader):
            out = []
            with torch.no_grad():
                for batch in loader:
                    batch = {
                        k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in batch.items()
                    }
                    out.append(
                        model.get_feature_representation(batch).float().cpu().numpy()
                    )
            return np.concatenate(out, axis=0)

        x = np.concatenate([collect(src_loader), collect(tgt_loader)], axis=0)
        if len(x) != len(y):
            raise RuntimeError(f"features {len(x)} != probe events {len(y)}")
        event_metrics = _fit_probes(x, y, idx_tr, idx_te, seed)
        part_metrics = _fit_probes(x, y, part_tr, part_te, seed)

        # The original per-epoch configuration, for comparison.
        scaler = StandardScaler().fit(x[idx_tr])
        x_tr = scaler.transform(x[idx_tr])
        x_te = scaler.transform(x[idx_te])
        orig = MLPClassifier(
            hidden_layer_sizes=(128,), max_iter=200, early_stopping=True,
            random_state=seed,
        )
        orig.fit(x[idx_tr], y[idx_tr])  # deliberately unscaled, as in trainer
        orig_auc = float(roc_auc_score(y[idx_te], orig.predict_proba(x_te)[:, 1]))

        row = {
            "id": run_id,
            "run": run_name,
            "n_source": len(src_ds),
            "n_target": len(tgt_ds),
            "probe_split": "event-level within DA-held-out parts",
            "original_mlp_auc": orig_auc,
            "original_mlp_n_iter": int(orig.n_iter_),
        }
        row.update(event_metrics)
        row["probe_split_part"] = "part-disjoint within DA-held-out parts"
        row.update({f"{k}_part": v for k, v in part_metrics.items()})
        row["da_enabled"] = bool(_encoder_da_enabled(ckpt_path.parent))
        rows.append(row)
        print(run_id, rows[-1])
        del model
        torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
