"""In-memory experimental HDF5 dataset for the target domain.

The exp.h5 file is small (~650K events, ~44M hits) and fits comfortably
in RAM (~900 MB for features + metadata).  Reading everything at init
avoids per-event I/O during training.

All events are treated as background: label=0.0, signal_hits=0, signal_strings=0.

Returns the same ``__getitem__`` dict format as ``PrefilterNpyDataset``
so the same collate function and dataloader factory work for both.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ExpDataset(Dataset):
    """In-memory dataset reading directly from exp.h5.

    Args:
        h5_path: Path to the experimental HDF5 file.
        max_events: Maximum number of events to load (None = all).
        max_hits: Truncate events longer than this (None = no truncation).
        seed: Random seed for event sampling when ``max_events`` is set.
    """

    def __init__(
        self,
        h5_path: str,
        max_events: Optional[int] = None,
        max_hits: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.max_hits = max_hits

        # Read all events into memory
        features_list, lengths_list, event_ids_list = self._load_h5(h5_path)
        n_total = len(lengths_list)

        # Subsample if requested
        if max_events is not None and max_events < n_total:
            rng = np.random.RandomState(seed)
            sel = rng.choice(n_total, size=max_events, replace=False)
            sel.sort()
            features_list = [features_list[i] for i in sel]
            lengths_list = [lengths_list[i] for i in sel]
            event_ids_list = [event_ids_list[i] for i in sel]

        self._features = features_list          # list of (n_hits, 5) float32
        self._lengths = np.array(lengths_list, dtype=np.int32)
        self._event_ids: List[str] = event_ids_list
        self._indices = np.arange(len(self._features), dtype=np.int64)

        logger.info(
            f"ExpDataset: loaded {len(self)} events from {h5_path} "
            f"(total available: {n_total:,})"
        )

    @staticmethod
    def _load_h5(h5_path: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Read all events from exp.h5 sequentially."""
        features_list: List[np.ndarray] = []
        lengths_list: List[int] = []
        event_ids_list: List[str] = []

        with h5py.File(h5_path, "r") as h5:
            grp = h5["exp"]
            parts = sorted(
                k for k in grp["raw"]["data"].keys() if k.startswith("part_")
            )
            for part in parts:
                data = grp["raw"]["data"][part]["data"][:]
                ev_starts = grp["raw"]["ev_starts"][part]["data"][:]
                ev_ids = grp["ev_ids"][part]["data"][:]
                n_events = len(ev_starts) - 1

                for i in range(n_events):
                    s, e = int(ev_starts[i]), int(ev_starts[i + 1])
                    if e > s:
                        features_list.append(data[s:e])
                        lengths_list.append(e - s)
                        event_ids_list.append(ev_ids[i].decode("utf-8"))

        return features_list, lengths_list, event_ids_list

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self._indices[idx]
        hits = self._features[real_idx]
        n_hits = len(hits)

        if self.max_hits is not None and n_hits > self.max_hits:
            hits = hits[: self.max_hits]
            length = self.max_hits
        else:
            length = n_hits

        return {
            "features": torch.from_numpy(hits.copy()),
            "labels": torch.tensor(0.0, dtype=torch.float32),
            "lengths": torch.tensor(length, dtype=torch.long),
            "original_lengths": torch.tensor(n_hits, dtype=torch.long),
            "signal_hit_count": 0,
            "signal_string_count": 0,
            "particle_type": -1,  # sentinel for "exp"
        }

    # ------------------------------------------------------------------
    # Train/val split (same interface as PrefilterNpyDataset)
    # ------------------------------------------------------------------

    def split(
        self, train_frac: float, seed: int = 42
    ) -> "Tuple[ExpDataset, ExpDataset]":
        """Return (train, val) views sharing the same in-memory data."""
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(self._indices))
        n_train = int(len(perm) * train_frac)

        train_ds = ExpDataset.__new__(ExpDataset)
        val_ds = ExpDataset.__new__(ExpDataset)

        for ds in (train_ds, val_ds):
            ds._features = self._features
            ds._lengths = self._lengths
            ds._event_ids = self._event_ids
            ds.max_hits = self.max_hits

        train_ds._indices = self._indices[perm[:n_train]]
        val_ds._indices = self._indices[perm[n_train:]]
        return train_ds, val_ds

    def get_all_labels(self) -> np.ndarray:
        """All labels are 0.0 for exp data."""
        return np.zeros(len(self._indices), dtype=np.float32)

    def get_event_ids(self) -> List[str]:
        """Return event IDs for the current index view (respects split/subsample)."""
        return [self._event_ids[i] for i in self._indices]
