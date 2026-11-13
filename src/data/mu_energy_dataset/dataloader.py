"""DataLoader factory for the muon-energy dataset."""

from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional

from torch.utils.data import DataLoader

from .collate import mu_energy_collate_fn
from .dataset import MuEnergyNpyDataset


def create_mu_energy_dataloader(
    dataset: MuEnergyNpyDataset,
    batch_size: int,
    shuffle: bool,
    normalization_config: Optional[Dict[str, List[float]]] = None,
    target_normalization: Optional[Dict[str, float]] = None,
    augmentation_config: Optional[Dict] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_batch: bool = False,
) -> DataLoader:
    collate = partial(
        mu_energy_collate_fn,
        normalization_config=normalization_config,
        target_normalization=target_normalization,
        augmentation_config=augmentation_config,
        shuffle_batch=shuffle_batch,
    )
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=False,
        **kwargs,
    )
