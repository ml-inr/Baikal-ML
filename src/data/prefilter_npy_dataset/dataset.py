"""Memory-mapped NPY dataset for prefilter training.

Reads flat .npy arrays produced by ``prefilter_npy_ds_builder``.
All heavy data lives on disk; the OS pages in only what is needed.

Expected directory layout::

    <npy_dir>/
        features.npy        (total_hits, 5) float32
        offsets.npy          (n_events + 1,) int64
        labels.npy           (n_events,) float32  — soft labels
        signal_hits.npy      (n_events,) int32
        signal_strings.npy   (n_events,) int32
        particle_types.npy   (n_events,) int8
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.data.base_npy_dataset import BaseNpyDataset


class PrefilterNpyDataset(BaseNpyDataset):
    """Zero-copy dataset backed by memory-mapped .npy files.

    Args:
        npy_dir: Directory containing the .npy files.
        max_hits: Truncate events longer than this (None = no truncation).
        max_events: Use at most this many events (None = all). Sampled randomly.
        seed: Random seed for max_events subsampling.
        indices: Optional subset of event indices to expose.
    """

    def __init__(
        self,
        npy_dir: str,
        max_hits: Optional[int] = None,
        max_events: Optional[int] = None,
        seed: int = 42,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        npy_dir = Path(npy_dir)

        self.features: np.ndarray = np.load(npy_dir / "features.npy", mmap_mode="r")
        self.offsets: np.ndarray = np.load(npy_dir / "offsets.npy")
        self.labels: np.ndarray = np.load(npy_dir / "labels.npy")
        self.signal_hits: np.ndarray = np.load(npy_dir / "signal_hits.npy")
        self.signal_strings: np.ndarray = np.load(npy_dir / "signal_strings.npy")
        self.particle_types: np.ndarray = np.load(npy_dir / "particle_types.npy")
        self.max_hits = max_hits

        self._init_indices(len(self.labels), max_events, seed, indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self._indices[idx]
        start = int(self.offsets[real_idx])
        end = int(self.offsets[real_idx + 1])
        n_hits = end - start

        if self.max_hits is not None and n_hits > self.max_hits:
            hits = np.array(self.features[start: start + self.max_hits])
            length = self.max_hits
        else:
            hits = np.array(self.features[start:end])
            length = n_hits

        return {
            "features": torch.from_numpy(hits),
            "labels": torch.tensor(self.labels[real_idx], dtype=torch.float32),
            "lengths": torch.tensor(length, dtype=torch.long),
            "original_lengths": torch.tensor(n_hits, dtype=torch.long),
            "signal_hit_count": int(self.signal_hits[real_idx]),
            "signal_string_count": int(self.signal_strings[real_idx]),
            "particle_type": int(self.particle_types[real_idx]),
        }

    def _split_attrs(self) -> List[str]:
        return [
            "features", "offsets", "labels",
            "signal_hits", "signal_strings", "particle_types",
            "max_hits",
        ]
