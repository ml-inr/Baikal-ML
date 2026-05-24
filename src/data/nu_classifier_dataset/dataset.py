"""Memory-mapped NPY dataset for nu-classifier training (MC source domain).

Reads flat .npy arrays produced by ``nu_classifier_ds_builder``.
All heavy data lives on disk; the OS pages in only what is needed.

Expected directory layout::

    <npy_dir>/
        features.npy        (total_sig_hits, 5) float32
        probs.npy           (total_sig_hits,)   float32  — optional, sig-noise prob per hit
        offsets.npy          (n_events + 1,) int64
        labels.npy           (n_events,) float32  — 1.0=neutrino, 0.0=muatm
        n_sig_hits.npy       (n_events,) int32
        n_sig_strings.npy    (n_events,) int32
        particle_types.npy   (n_events,) int8
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NuClassifierNpyDataset(Dataset):
    """Zero-copy dataset backed by memory-mapped .npy files.

    Args:
        npy_dir: Directory containing the .npy files.
        max_hits: Truncate events longer than this (None = no truncation).
        max_events: Use at most this many events (None = all). Sampled randomly.
        seed: Random seed for max_events subsampling.
        indices: Optional subset of event indices to expose.
        include_probs: Append raw sig-noise probabilities as a 6th feature column.
    """

    def __init__(
        self,
        npy_dir: str,
        max_hits: Optional[int] = None,
        max_events: Optional[int] = None,
        seed: int = 42,
        indices: Optional[np.ndarray] = None,
        include_probs: bool = False,
    ) -> None:
        npy_dir = Path(npy_dir)

        self.features:       np.ndarray = np.load(npy_dir / "features.npy",       mmap_mode="r")
        self.offsets:        np.ndarray = np.load(npy_dir / "offsets.npy")
        self.labels:         np.ndarray = np.load(npy_dir / "labels.npy")
        self.n_sig_hits:     np.ndarray = np.load(npy_dir / "n_sig_hits.npy")
        self.n_sig_strings:  np.ndarray = np.load(npy_dir / "n_sig_strings.npy")
        self.particle_types: np.ndarray = np.load(npy_dir / "particle_types.npy")

        self.include_probs = include_probs
        self.probs: Optional[np.ndarray] = (
            np.load(npy_dir / "probs.npy", mmap_mode="r") if include_probs else None
        )

        self.max_hits = max_hits

        if indices is not None:
            self._indices = indices.astype(np.int64)
        else:
            self._indices = np.arange(len(self.labels), dtype=np.int64)

        if max_events is not None and max_events < len(self._indices):
            rng = np.random.RandomState(seed)
            sel = rng.choice(len(self._indices), size=max_events, replace=False)
            sel.sort()
            self._indices = self._indices[sel]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self._indices[idx]
        start    = int(self.offsets[real_idx])
        end      = int(self.offsets[real_idx + 1])
        n_hits   = end - start

        if self.max_hits is not None and n_hits > self.max_hits:
            hits   = np.array(self.features[start: start + self.max_hits])
            length = self.max_hits
        else:
            hits   = np.array(self.features[start:end])
            length = n_hits

        if self.include_probs:
            p_end  = start + length
            probs  = np.array(self.probs[start:p_end]).reshape(-1, 1)
            hits   = np.concatenate([hits, probs], axis=1)

        return {
            "features":            torch.from_numpy(hits),
            "labels":              torch.tensor(self.labels[real_idx],         dtype=torch.float32),
            "lengths":             torch.tensor(length,                        dtype=torch.long),
            "original_lengths":    torch.tensor(n_hits,                        dtype=torch.long),
            "signal_hit_count":    int(self.n_sig_hits[real_idx]),
            "signal_string_count": int(self.n_sig_strings[real_idx]),
            "particle_type":       int(self.particle_types[real_idx]),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def split(
        self, train_frac: float, seed: int = 42
    ) -> "tuple[NuClassifierNpyDataset, NuClassifierNpyDataset]":
        """Return (train, val) views sharing the same mmap files."""
        rng    = np.random.RandomState(seed)
        perm   = rng.permutation(len(self._indices))
        n_train = int(len(perm) * train_frac)

        train_ds = NuClassifierNpyDataset.__new__(NuClassifierNpyDataset)
        val_ds   = NuClassifierNpyDataset.__new__(NuClassifierNpyDataset)

        for ds in (train_ds, val_ds):
            ds.features       = self.features
            ds.offsets        = self.offsets
            ds.labels         = self.labels
            ds.n_sig_hits     = self.n_sig_hits
            ds.n_sig_strings  = self.n_sig_strings
            ds.particle_types = self.particle_types
            ds.max_hits       = self.max_hits
            ds.include_probs  = self.include_probs
            ds.probs          = self.probs

        train_ds._indices = self._indices[perm[:n_train]]
        val_ds._indices   = self._indices[perm[n_train:]]
        return train_ds, val_ds

    def get_all_labels(self) -> np.ndarray:
        return self.labels[self._indices]
