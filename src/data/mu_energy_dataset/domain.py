"""Unlabeled domain datasets for muon-energy domain adaptation.

The energy baseline reads ``MuEnergyNpyDataset`` (labeled MC neutrinos).  DA needs
two additional unlabeled streams:

* MC muons, from the nu-classifier MC NPY layout
  (``features.npy`` / ``offsets.npy`` / ``particle_types.npy`` ...);
* experimental data, from the nu-classifier exp NPY layout
  (``{prefix}_features.npy`` / ``{prefix}_offsets.npy`` ...).

Both classes return the **same item dict as** :class:`MuEnergyNpyDataset`, with
dummy target/energy/weight fields.  That lets the existing energy collate function
(and therefore the same padding, normalization, rotation, z-flip and noise) be
used unchanged for domain batches.  The dummy labels are never used by the DA
loss.

The datasets expose the full-length ``part_keys`` so the trainer can hold out
whole MC parts / exp runs for the domain probe instead of probing on events the
critic was trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def part_split_indices(
    part_keys: np.ndarray,
    holdout_fraction: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices by whole part keys; returns (train, holdout) rows.

    Rows are grouped by the exact part key (e.g. ``part_1234`` or
    ``part_s2020_c02_r0055``).  Entire parts go to one side, so a probe cannot
    see an event from a part the critic was trained on.
    """
    if not 0.0 <= float(holdout_fraction) < 1.0:
        raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}")
    keys = np.asarray(part_keys, dtype=object)
    unique = np.array(sorted(set(keys.tolist())), dtype=object)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique))
    n_hold = int(round(len(unique) * float(holdout_fraction)))
    hold_keys = set(unique[perm[:n_hold]].tolist())
    hold_mask = np.array([k in hold_keys for k in keys], dtype=bool)
    return np.flatnonzero(~hold_mask), np.flatnonzero(hold_mask)


class _BaseDomainDataset(Dataset):
    """Shared indexing/truncation for the two unlabeled layouts."""

    def __init__(
        self,
        offsets: np.ndarray,
        features: np.ndarray,
        n_sig_hits: np.ndarray,
        n_sig_strings: np.ndarray,
        part_keys: np.ndarray,
        local_ids: np.ndarray,
        theta: Optional[np.ndarray],
        particle_types: Optional[np.ndarray],
        *,
        indices: Optional[np.ndarray] = None,
        max_hits: Optional[int] = None,
        max_events: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        n_total = len(offsets) - 1
        for arr, name in (
            (n_sig_hits, "n_sig_hits"),
            (n_sig_strings, "n_sig_strings"),
            (part_keys, "part_keys"),
            (local_ids, "local_ids"),
        ):
            if len(arr) != n_total:
                raise ValueError(f"{name}: {len(arr)} rows, offsets say {n_total}")

        self.features = features
        self.offsets = offsets
        self.n_sig_hits = n_sig_hits
        self.n_sig_strings = n_sig_strings
        self.part_keys = part_keys
        self.local_ids = local_ids
        self.theta = theta
        self.particle_types = particle_types
        self.max_hits = max_hits

        base = np.arange(n_total, dtype=np.int64)
        if indices is not None:
            base = np.intersect1d(base, np.asarray(indices, dtype=np.int64))
        self._indices = base
        if max_events is not None and max_events < len(self._indices):
            rng = np.random.RandomState(seed)
            sel = rng.choice(len(self._indices), size=int(max_events), replace=False)
            self._indices = np.sort(self._indices[sel])

    @property
    def all_indices(self) -> np.ndarray:
        """Filtered row indices before any ``max_events`` subsample."""
        return self._indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = int(self._indices[idx])
        start = int(self.offsets[real_idx])
        end = int(self.offsets[real_idx + 1])
        n_hits = end - start
        length = n_hits if self.max_hits is None else min(n_hits, int(self.max_hits))
        hits = np.array(self.features[start:start + length])

        theta = (
            float(self.theta[real_idx])
            if self.theta is not None else float("nan")
        )
        ptype = (
            int(self.particle_types[real_idx])
            if self.particle_types is not None else -1
        )
        # Dummy fields keep the energy collate function unchanged.  DA never
        # reads the target, energy, weight or event_fk of a domain batch.
        return {
            "features": torch.from_numpy(hits),
            "target": torch.tensor(0.0, dtype=torch.float32),
            "energy": torch.tensor(1.0, dtype=torch.float32),
            "energy_status": -1,
            "lengths": torch.tensor(length, dtype=torch.long),
            "original_lengths": torch.tensor(n_hits, dtype=torch.long),
            "weight": torch.tensor(1.0, dtype=torch.float32),
            "signal_hit_count": int(self.n_sig_hits[real_idx]),
            "signal_string_count": int(self.n_sig_strings[real_idx]),
            "particle_type": ptype,
            "theta": torch.tensor(theta, dtype=torch.float32),
            "event_fk": -1,
            "part_key": str(self.part_keys[real_idx]),
            "local_event_id": int(self.local_ids[real_idx]),
        }


class McEnergyDomainDataset(_BaseDomainDataset):
    """Unlabeled MC domain from the nu-classifier MC NPY layout.

    Args:
        npy_dir: e.g. ``data_manager/datasets/nu_classifier_dataset_h8s3_thr0.8``.
        particle_types: particle codes to keep (0=muatm, 1=nuatm, 2=nue2);
            ``None`` keeps all.
        track_counts_file: optional ``n_tracks.npy`` sidecar aligned with events;
            if given together with ``track_min``, events below the cut are dropped.
    """

    def __init__(
        self,
        npy_dir: str,
        particle_types: Optional[Iterable[int]] = (0,),
        indices: Optional[np.ndarray] = None,
        max_hits: Optional[int] = None,
        max_events: Optional[int] = None,
        seed: int = 42,
        track_counts_file: Optional[str] = None,
        track_min: Optional[int] = None,
    ) -> None:
        d = Path(npy_dir)
        features = np.load(d / "features.npy", mmap_mode="r")
        offsets = np.load(d / "offsets.npy")
        n_sig_hits = np.load(d / "n_sig_hits.npy")
        n_sig_strings = np.load(d / "n_sig_strings.npy")
        ptype = np.load(d / "particle_types.npy")
        part_keys = np.load(d / "h5_part_keys.npy", allow_pickle=True)
        local_ids = np.load(d / "h5_local_event_ids.npy")
        theta_path = d / "theta.npy"
        theta = np.load(theta_path) if theta_path.exists() else None

        rows = np.arange(len(ptype), dtype=np.int64)
        if particle_types is not None:
            rows = rows[np.isin(ptype, np.asarray(list(particle_types), dtype=ptype.dtype))]
        if track_counts_file is not None and track_min is not None:
            n_tracks = np.load(track_counts_file)
            if len(n_tracks) != len(ptype):
                raise ValueError(
                    f"{track_counts_file}: {len(n_tracks)} rows, expected {len(ptype)}"
                )
            rows = rows[n_tracks[rows] >= int(track_min)]
        if indices is not None:
            rows = np.intersect1d(rows, np.asarray(indices, dtype=np.int64))
        super().__init__(
            offsets=offsets,
            features=features,
            n_sig_hits=n_sig_hits,
            n_sig_strings=n_sig_strings,
            part_keys=part_keys,
            local_ids=local_ids,
            theta=theta,
            particle_types=ptype,
            indices=rows,
            max_hits=max_hits,
            max_events=max_events,
            seed=seed,
        )


class ExpEnergyDomainDataset(_BaseDomainDataset):
    """Unlabeled experimental domain from the nu-classifier exp NPY layout."""

    def __init__(
        self,
        npy_dir: str,
        prefix: str = "exp",
        indices: Optional[np.ndarray] = None,
        max_hits: Optional[int] = None,
        max_events: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        d = Path(npy_dir)
        features = np.load(d / f"{prefix}_features.npy", mmap_mode="r")
        offsets = np.load(d / f"{prefix}_offsets.npy")
        n_sig_hits = np.load(d / f"{prefix}_n_sig_hits.npy")
        n_sig_strings = np.load(d / f"{prefix}_n_sig_strings.npy")
        part_keys_path = d / f"{prefix}_h5_part_keys.npy"
        local_ids_path = d / f"{prefix}_h5_local_event_ids.npy"
        if not part_keys_path.exists() or not local_ids_path.exists():
            raise FileNotFoundError(
                f"{d}: exp DA needs {part_keys_path.name} and {local_ids_path.name}"
            )
        part_keys = np.load(part_keys_path, allow_pickle=True)
        local_ids = np.load(local_ids_path)
        super().__init__(
            offsets=offsets,
            features=features,
            n_sig_hits=n_sig_hits,
            n_sig_strings=n_sig_strings,
            part_keys=part_keys,
            local_ids=local_ids,
            theta=None,
            particle_types=None,
            indices=indices,
            max_hits=max_hits,
            max_events=max_events,
            seed=seed,
        )
