"""Which events a checkpoint saw in training, read from its NPY dataset.

The NPY dataset the trainer consumed records the identity of every event it holds:
`h5_part_keys.npy`, `h5_local_event_ids.npy` and `particle_types.npy` together give
`(data_class, part_key, local_idx)`, which is exactly the address the catalog uses.
That is the first source of this fact, not a reconstruction of it -- the `splits`
table some old databases carry was itself built from these arrays.

Two strictnesses, because "seen in training" is ambiguous:

* **all** -- every event in the dataset.  Conservative, and the right default: the
  trainer's validation half influenced which epoch was kept, so those scores are
  biased too, just less.
* **train** -- only the events that reached the training half, reproducing the two
  seeded steps the trainer applies (`src/data/base_npy_dataset.py`).  For the
  reference checkpoint that is 4,500,000 of the dataset's 5,256,496.

Cost, measured: building the identity frame takes 1.7 s, and anti-joining it against
the 25M scored `mc_merged` events inside DuckDB takes 1.2 s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import PARTICLE_TYPE_NAMES


def _indices_after_max_events(n_total: int, max_events: int | None,
                              seed: int) -> np.ndarray:
    """Reproduces `_init_indices` in src/data/base_npy_dataset.py:26-32."""
    indices = np.arange(n_total, dtype=np.int64)
    if max_events is not None and max_events < len(indices):
        chosen = np.random.RandomState(seed).choice(
            len(indices), size=max_events, replace=False)
        chosen.sort()
        indices = indices[chosen]
    return indices


def _train_half(indices: np.ndarray, train_frac: float, seed: int) -> np.ndarray:
    """Reproduces `split` in src/data/base_npy_dataset.py:43-57."""
    perm = np.random.RandomState(seed).permutation(len(indices))
    return indices[perm[:int(len(perm) * train_frac)]]


def identity(npy_dir: Path, *, domain: str = "source", strictness: str = "all",
             selection: dict | None = None) -> pd.DataFrame:
    """`(data_class, part_key, local_idx)` of the events a training run consumed.

    Parameters
    ----------
    domain:
        "source" reads the labelled MC arrays; "target" reads the `exp_*` ones, which
        carry no class because the domain-adaptation target is unlabelled.
    strictness:
        "all" (every event in the dataset) or "train" (only the training half).
    selection:
        Required for `strictness="train"`: `max_events`, `train_split`, `seed` from
        `Run.training_selection()`.

    `part_key` comes back as a pandas category.  It is 952 distinct strings repeated
    five million times; as Python objects the frame costs 742 MB, as a category, tens
    of megabytes.
    """
    npy_dir = Path(npy_dir)
    prefix = "" if domain == "source" else "exp_"
    part_keys = np.load(npy_dir / f"{prefix}h5_part_keys.npy", allow_pickle=True)
    local_idx = np.load(npy_dir / f"{prefix}h5_local_event_ids.npy")

    if strictness == "train":
        if selection is None:
            raise ValueError("strictness='train' needs `selection` from "
                             "Run.training_selection()")
        indices = _indices_after_max_events(
            len(local_idx), selection["max_events"], selection["seed"])
        indices = np.sort(_train_half(indices, selection["train_split"],
                                      selection["seed"]))
    elif strictness == "all":
        indices = np.arange(len(local_idx), dtype=np.int64)
    else:
        raise ValueError(f"strictness must be 'all' or 'train', got {strictness!r}")

    frame = pd.DataFrame({
        "part_key": pd.Categorical(part_keys[indices].astype(str)),
        "local_idx": local_idx[indices].astype("int64"),
    })
    if domain == "source":
        types = np.load(npy_dir / "particle_types.npy")[indices]
        frame.insert(0, "data_class", pd.Categorical(
            [PARTICLE_TYPE_NAMES[int(code)] for code in types]))
    return frame
