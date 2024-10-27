from torch.nn.utils.rnn import pad_sequence
from .readers import BaikalDatasetGraph, BaikalDataset, Dataset
from torch.utils.data import Dataset, Subset, DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
import typing as tp


SPLIT_TYPES = ["train", "val", "test"]
VAL_SUBSET_CUT = 100


def create_datasets(
    path_to_data: str,
    use_val_subset: bool = True,
    DatasetType: tp.Type[Dataset] = BaikalDataset,
    **kwargs
):
    datasets = {}
    for split_type in SPLIT_TYPES:
        datasets[split_type] = DatasetType(path_to_data, split_type, **kwargs)
    if use_val_subset:
        datasets["val_subset"] = Subset(
            datasets["val"], list(range(0, len(datasets["val"]), VAL_SUBSET_CUT))
        )
    return datasets


def create_infnite_loader_generator(loader: tp.Iterable):
    loader_iter = iter(loader)
    while True:
        try:
            yield next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            yield next(loader_iter)


def create_dataloaders(
    path_to_data: str,
    batch_size: int = 128,
    is_graph: bool = False,
    use_val_subset: bool = True,
    num_workers: int = 1,
    DatasetType: tp.Type[Dataset] = BaikalDataset,
    is_classification: bool = False,
    is_angle_and_track_cascade: bool = False,
    **kwargs
):
    datasets = create_datasets(path_to_data, use_val_subset, DatasetType, **kwargs)
    if not is_graph:
        # for padding
        def collate_fn(batch):
            x_batch = pad_sequence([x[0] for x in batch], batch_first=True)
            y_batch = pad_sequence(
                [x[1].mT if is_classification else x[1] for x in batch],
                batch_first=True,
            )
            padding_mask = (x_batch > 0).sum(-1)
            padding_mask[padding_mask != 0] = 1
            return x_batch, y_batch.squeeze(-1), padding_mask.bool()

        def collate_fn_angle_track_cascade(batch):
            x_batch = pad_sequence([x[0] for x in batch], batch_first=True)
            y_batch = pad_sequence([x[1][0].mT for x in batch],batch_first=True,
            )
            angles_batch = pad_sequence([x[1][1] for x in batch], batch_first=True)
            padding_mask = (x_batch > 0).sum(-1)
            padding_mask[padding_mask != 0] = 1
            return x_batch, y_batch.squeeze(-1), angles_batch, padding_mask.bool()

        collate_fn = collate_fn if not is_angle_and_track_cascade else collate_fn_angle_track_cascade
        train_loader = create_infnite_loader_generator(
            DataLoader(
                datasets["train"],
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
        )
        if use_val_subset:
            val_loader = DataLoader(
                datasets["val_subset"],
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )
        else:
            val_loader = DataLoader(
                datasets["val"],
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )
        test_loader = DataLoader(
            datasets["test"], batch_size=batch_size, num_workers=num_workers
        )
    else:
        train_loader = create_infnite_loader_generator(
            GraphDataLoader(
                datasets["train"], batch_size=batch_size, num_workers=num_workers
            )
        )
        test_loader = GraphDataLoader(
            datasets["test"], batch_size, num_workers=num_workers
        )
        if use_val_subset:
            val_loader = GraphDataLoader(
                datasets["val_subset"], batch_size, num_workers=num_workers
            )
        else:
            val_loader = GraphDataLoader(
                datasets["val_subset"], batch_size, num_workers=num_workers
            )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
