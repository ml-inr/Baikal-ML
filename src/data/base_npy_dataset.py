from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseNpyDataset(Dataset):
    """Shared base for memory-mapped NPY datasets (prefilter and nu-classifier).

    Subclasses must implement:
    - ``__getitem__``
    - ``_split_attrs`` — return the list of attribute names to shallow-copy in ``split``
    """

    def _init_indices(
        self,
        n_total: int,
        max_events: Optional[int],
        seed: int,
        indices: Optional[np.ndarray],
    ) -> None:
        if indices is not None:
            self._indices = indices.astype(np.int64)
        else:
            self._indices = np.arange(n_total, dtype=np.int64)

        if max_events is not None and max_events < len(self._indices):
            rng = np.random.RandomState(seed)
            sel = rng.choice(len(self._indices), size=max_events, replace=False)
            sel.sort()
            self._indices = self._indices[sel]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    def _split_attrs(self) -> List[str]:
        raise NotImplementedError

    def split(self, train_frac: float, seed: int = 42):
        """Return (train, val) views sharing the same mmap arrays."""
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(self._indices))
        n_train = int(len(perm) * train_frac)

        cls = type(self)
        train_ds = cls.__new__(cls)
        val_ds = cls.__new__(cls)

        for ds in (train_ds, val_ds):
            for attr in self._split_attrs():
                setattr(ds, attr, getattr(self, attr))

        train_ds._indices = self._indices[perm[:n_train]]
        val_ds._indices = self._indices[perm[n_train:]]
        return train_ds, val_ds

    def get_all_labels(self) -> np.ndarray:
        return self.labels[self._indices]
