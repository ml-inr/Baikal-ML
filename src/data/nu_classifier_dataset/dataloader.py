"""DataLoader factory for nu_classifier_dataset."""

from functools import partial
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader, Dataset

from .collate import nu_classifier_collate_fn


def create_nu_classifier_dataloader(
    dataset: Dataset,
    batch_size: int = 512,
    shuffle: bool = True,
    normalization_config: Optional[Dict[str, List[float]]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    shuffle_batch: bool = True,
    device: str = "cpu",
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader with the nu_classifier collate function.

    Works with both ``NuClassifierNpyDataset`` and ``NuClassifierExpNpyDataset``.

    Args:
        dataset: A NuClassifierNpyDataset or NuClassifierExpNpyDataset instance.
        batch_size: Batch size.
        shuffle: Reshuffle at every epoch.
        normalization_config: Per-feature means/stds for normalization.
        augmentation_config: Noise + rotation config.
        shuffle_batch: Shuffle events within each batch (in collate).
        device: Target device for collated tensors.
        num_workers: DataLoader workers (0 = main process).
        pin_memory: Pin memory for faster GPU transfer.
    """
    collate_device = "cpu" if num_workers > 0 else device

    collate_wrapper = partial(
        nu_classifier_collate_fn,
        normalization_config=normalization_config,
        augmentation_config=augmentation_config,
        shuffle_batch=shuffle_batch,
        device=collate_device,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
