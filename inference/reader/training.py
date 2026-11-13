"""Which events a checkpoint saw in training, read from its NPY dataset.

The NPY dataset the trainer consumed records the identity of every event it holds:
`h5_part_keys.npy`, `h5_local_event_ids.npy` and `particle_types.npy` together give
`(data_class, part_key, local_idx)`, which is exactly the address the catalog uses.
That is the first source of this fact, not a reconstruction of it -- the `splits`
table some old databases carry was itself built from these arrays.

Three strictnesses, because "seen in training" is ambiguous:

* **all** -- every event in the dataset.  Conservative, and the right default: the
  trainer's validation half influenced which epoch was kept, so those scores are
  biased too, just less.
* **train** -- only the events that reached the training half, reproducing the two
  seeded steps the trainer applies (`src/data/base_npy_dataset.py`).  For the
  reference checkpoint that is 4,500,000 of the dataset's 5,256,496.
* **parts** -- every event of every HDF5 part the dataset was *built from*, whether
  or not the builder kept it.  Coarser than the others and strictly safer, and it is
  the only one available for a dataset that does not record per-event back-links --
  which is the case for the prefilter datasets: `prefilter_npy_ds_builder` writes no
  `h5_part_keys.npy`, so the finest handle it leaves is the part list in its
  `build_config.yaml`.  A part the builder never opened cannot have contributed an
  event, so the exclusion is sound; it just also drops events that were available and
  went unused.

`identity` returns per-event addresses for "all" and "train", and part addresses (no
`local_idx`) for "parts"; `NotTrained` anti-joins on whichever columns come back.

Asking for "all" or "train" on a dataset without back-links raises and names "parts",
rather than quietly returning nothing to exclude.  An unfiltered result that looks
filtered is the failure this module exists to prevent.

Cost, measured: building the identity frame takes 1.7 s, and anti-joining it against
the 25M scored `mc_merged` events inside DuckDB takes 1.2 s.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .schema import PARTICLE_TYPE_NAMES

#: `build_config.yaml` keys naming the h5 part list a builder consumed. The prefilter
#: builder records it as `parts_json`; the value is an absolute path from the machine
#: the dataset was built on, so only its file name is used and it is looked up beside
#: the dataset. See `part_identity`.
_PARTS_JSON_KEYS = ("parts_json",)

#: `prefilter_h5_parts.json` keys -> catalog `data_class`. The builder's own
#: `PARTICLE_MAP` in data_manager/prefilter_npy_ds_builder/build.py:32.
_SHORT_TO_CLASS = {"muatm": "muatm_2020", "nuatm": "nuatm_2020",
                   "nue2": "nue2_2020"}


def has_event_backlinks(npy_dir: Path, domain: str = "source") -> bool:
    """Whether this dataset records which h5 event each of its rows came from."""
    prefix = "" if domain == "source" else "exp_"
    return (Path(npy_dir) / f"{prefix}h5_part_keys.npy").exists()


def parts_json_path(npy_dir: Path) -> Path:
    """The h5 part list this dataset was built from, resolved beside the dataset.

    `build_config.yaml` stores the path the builder was given, which was absolute on
    another machine (`/home2/albert/Baikal/datasets/...`) and does not resolve here.
    The file itself travelled with the dataset, so it is looked up by name in the
    dataset directory and its parent.
    """
    npy_dir = Path(npy_dir)
    config_path = npy_dir / "build_config.yaml"
    if not config_path.exists():
        raise LookupError(
            f"{npy_dir} has no build_config.yaml, so the parts it was built from "
            f"cannot be recovered")
    config = yaml.safe_load(config_path.read_text()) or {}
    recorded = next((config[key] for key in _PARTS_JSON_KEYS if config.get(key)),
                    None)
    if not recorded:
        raise LookupError(
            f"{config_path} names no part list (looked for {list(_PARTS_JSON_KEYS)})")
    name = Path(recorded).name
    for candidate in (npy_dir / name, npy_dir.parent / name):
        if candidate.exists():
            return candidate
    raise LookupError(
        f"{config_path} names {recorded!r}, whose file {name!r} is beside neither "
        f"{npy_dir} nor {npy_dir.parent}")


def part_identity(npy_dir: Path) -> pd.DataFrame:
    """`(data_class, part_key)` of every h5 part a dataset was built from.

    Part-level, because that is all the prefilter builder leaves behind. The JSON
    holds bare part numbers per short particle name; the catalog spells them
    `part_<n>` under the full `data_class`.
    """
    listed = json.loads(parts_json_path(npy_dir).read_text())
    rows = [(_SHORT_TO_CLASS.get(short, short), f"part_{number}")
            for short, numbers in listed.items() for number in numbers]
    frame = pd.DataFrame(rows, columns=["data_class", "part_key"])
    frame["data_class"] = pd.Categorical(frame["data_class"])
    frame["part_key"] = pd.Categorical(frame["part_key"])
    return frame


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
        "all" (every event in the dataset), "train" (only the training half), or
        "parts" (every event of every h5 part the dataset was built from, which is
        what a dataset without per-event back-links supports).
    selection:
        Required for `strictness="train"`: `max_events`, `train_split`, `seed` from
        `Run.training_selection()`.

    `part_key` comes back as a pandas category.  It is 952 distinct strings repeated
    five million times; as Python objects the frame costs 742 MB, as a category, tens
    of megabytes.
    """
    if strictness not in ("all", "train", "parts"):
        raise ValueError(
            f"strictness must be 'all', 'train' or 'parts', got {strictness!r}")

    npy_dir = Path(npy_dir)
    prefix = "" if domain == "source" else "exp_"

    if strictness == "parts":
        if domain != "source":
            raise ValueError("strictness='parts' describes the labelled source "
                             "dataset; the DA target has no part list")
        return part_identity(npy_dir)

    if not has_event_backlinks(npy_dir, domain):
        raise LookupError(
            f"{npy_dir} records no per-event back-links "
            f"({prefix}h5_part_keys.npy is absent), so strictness={strictness!r} "
            f"cannot be built. Use strictness='parts' (reader.NOT_TRAINED_PARTS), "
            f"which excludes every event of every part the dataset was built from.")

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
    else:                                        # "all" -- validated above
        indices = np.arange(len(local_idx), dtype=np.int64)

    frame = pd.DataFrame({
        "part_key": pd.Categorical(part_keys[indices].astype(str)),
        "local_idx": local_idx[indices].astype("int64"),
    })
    if domain == "source":
        types = np.load(npy_dir / "particle_types.npy")[indices]
        frame.insert(0, "data_class", pd.Categorical(
            [PARTICLE_TYPE_NAMES[int(code)] for code in types]))
    return frame
