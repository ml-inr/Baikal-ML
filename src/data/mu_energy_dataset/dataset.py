"""Memory-mapped NPY dataset for muon-energy regression.

Reads the arrays produced by ``data_manager/mu_energy_ds_builder``.  The task is a
per-event regression: the target is ``log10(E_mu)``, where ``E_mu`` is the muon energy at
entry into the row's cluster volume (companion column 1, r <= 90 m, |z| <= 295 m).

The dataset stores only the flat-selected train+val population: there are no test rows
and no split files.  Per-event flat-selection weights are read from ``flat_weights.npy``
unless explicit ``indices``/``weights`` are supplied.  Train/val is chosen by the training
YAML (``data.train_split``) with :meth:`split`, which slices the row indices and the
aligned weights together, so the loader never filters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MuEnergyNpyDataset(Dataset):
    """Variable-length hit sequences with a scalar ``log10(E)`` regression target."""

    def __init__(
        self,
        npy_dir: str,
        indices: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        max_hits: Optional[int] = None,
        include_probs: bool = False,
    ) -> None:
        d = Path(npy_dir)
        self.features = np.load(d / "features.npy", mmap_mode="r")
        self.offsets = np.load(d / "offsets.npy")
        self.energy = np.load(d / "energy.npy").astype(np.float64)
        self.energy_status = np.load(d / "energy_status.npy")
        self.n_sig_hits = np.load(d / "n_sig_hits.npy")
        self.n_sig_strings = np.load(d / "n_sig_strings.npy")
        self.particle_types = np.load(d / "particle_types.npy")
        self.theta = np.load(d / "theta.npy")
        self.event_fk = np.load(d / "event_fk.npy")
        self.h5_part_keys = np.load(d / "h5_part_keys.npy", allow_pickle=True)
        self.h5_local_event_ids = np.load(d / "h5_local_event_ids.npy")
        self.include_probs = include_probs
        self.probs = (
            np.load(d / "probs.npy", mmap_mode="r") if include_probs else None
        )

        n_total = len(self.energy)
        if indices is None:
            self.indices = np.arange(n_total, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
        if weights is None:
            flat_path = d / "flat_weights.npy"
            if flat_path.exists():
                base = np.load(flat_path).astype(np.float32)
                if len(base) != n_total:
                    raise ValueError(
                        f"{flat_path} has {len(base)} weights but the dataset has "
                        f"{n_total} rows; this is a pre-refactor dataset with split "
                        "files, which the trainer no longer supports"
                    )
                self.weights = base[self.indices]
            else:
                self.weights = np.ones(len(self.indices), dtype=np.float32)
        else:
            w = np.asarray(weights, dtype=np.float32)
            if len(w) != len(self.indices):
                raise ValueError(
                    f"weights length {len(w)} != indices length {len(self.indices)}"
                )
            self.weights = w
        if self.indices.min() < 0 or self.indices.max() >= n_total:
            raise ValueError("indices out of range")
        self.max_hits = max_hits

    def __len__(self) -> int:
        return len(self.indices)

    def split(
        self, train_frac: float, seed: int = 42
    ) -> tuple["MuEnergyNpyDataset", "MuEnergyNpyDataset"]:
        """Return (train, val) views sharing the mmap arrays and aligned weights.

        Mirrors ``BaseNpyDataset.split`` in the nu-classifier: the split is a
        training-config parameter, not a dataset artifact.  The event order is
        reproducible from ``train_frac`` and ``seed`` alone.
        """
        if not 0.0 < float(train_frac) < 1.0:
            raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(self.indices))
        n_train = int(len(perm) * float(train_frac))

        cls = type(self)
        train_ds = cls.__new__(cls)
        val_ds = cls.__new__(cls)
        for ds in (train_ds, val_ds):
            ds.__dict__.update(self.__dict__)

        train_sel = perm[:n_train]
        val_sel = perm[n_train:]
        train_ds.indices = self.indices[train_sel]
        val_ds.indices = self.indices[val_sel]
        train_ds.weights = self.weights[train_sel]
        val_ds.weights = self.weights[val_sel]
        return train_ds, val_ds

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = int(self.indices[idx])
        start = int(self.offsets[real_idx])
        end = int(self.offsets[real_idx + 1])
        n_hits = end - start
        length = n_hits if self.max_hits is None else min(n_hits, self.max_hits)

        hits = np.array(self.features[start:start + length])
        if self.include_probs and self.probs is not None:
            probs = np.array(self.probs[start:start + length]).reshape(-1, 1)
            hits = np.concatenate([hits, probs], axis=1)

        energy = float(self.energy[real_idx])
        return {
            "features": torch.from_numpy(hits),
            "target": torch.tensor(np.log10(energy), dtype=torch.float32),
            "energy": torch.tensor(energy, dtype=torch.float32),
            "energy_status": int(self.energy_status[real_idx]),
            "lengths": torch.tensor(length, dtype=torch.long),
            "original_lengths": torch.tensor(n_hits, dtype=torch.long),
            "weight": torch.tensor(float(self.weights[idx]), dtype=torch.float32),
            "signal_hit_count": int(self.n_sig_hits[real_idx]),
            "signal_string_count": int(self.n_sig_strings[real_idx]),
            "particle_type": int(self.particle_types[real_idx]),
            "theta": torch.tensor(float(self.theta[real_idx]), dtype=torch.float32),
            "event_fk": int(self.event_fk[real_idx]),
            "part_key": str(self.h5_part_keys[real_idx]),
            "local_event_id": int(self.h5_local_event_ids[real_idx]),
        }
