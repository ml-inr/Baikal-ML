import numpy as np
import h5py as h5
import torch
from torch.utils.data import Dataset
import logging
from .preprocessors import BasePreprocessor, BaseGraphPreprocessor


class BaikalDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        split_type: str,
        preprocessor: BasePreprocessor | BaseGraphPreprocessor,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            data_file (str): path to .h5 file
            split_type (str): train/val/test
        """
        self.path_to_data_file = data_file
        self.hfile = h5.File(data_file, "r")
        self.split_type = split_type
        self.events_amount = (
            self.hfile[self.split_type + "/ev_starts/data"].shape[0] - 1
        )
        self.preprocessor = preprocessor

    def __len__(self):
        return self.events_amount

    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        labels = self.hfile[self.split_type + "/labels/data"][start:end]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(labels, dtype=torch.long)

        data_x, data_y = self.preprocessor(data_x, data_y)

        return data_x, data_y


class BaikalDatasetTres(BaikalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.info(f"counting tres std and var for {self.split_type}...")
        self.tres_data = np.array(self.hfile[self.split_type + "/t_res/data"])
        self.tres_mean = self.tres_data.mean()
        self.tres_std = self.tres_data.std()
        logging.info("finished")

    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        tres = self.tres_data[start:end]

        tres = (tres - self.tres_mean) / (self.tres_std + 1e-8)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(tres, dtype=torch.float32)

        data_x, data_y = self.preprocessor(data_x, data_y)

        return data_x, data_y


class BaikalDatasetTrackCascade(BaikalDataset):
    def __init__(self, *args, tres_cut: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tres_cut = tres_cut

    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        labels = self.hfile[self.split_type + "/labels/data"][start:end]
        tres = self.hfile[self.split_type + "/t_res/data"][start:end]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(labels, dtype=torch.long)
        tres = torch.tensor(tres, dtype=torch.float32)

        data_x, data_y = self.preprocessor(data_x, data_y, tres, self.tres_cut)

        return data_x, data_y


class BaikalDatasetGraph(BaikalDataset):
    def __init__(
        self,
        data_file: str,
        split_type: str,
        preprocessor: BaseGraphPreprocessor,
    ) -> None:
        """
        Args:
            data_file (str): path to .h5 file
            split_type (str): train/val/test
            neighbours (int): amount of nearest neigbours used in graph adj matrix construction
        """
        super().__init__(data_file, split_type, preprocessor)

    def __getitem__(self, idx):
        # data_x, data_y = super().__getitem__(idx)
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        labels = self.hfile[self.split_type + "/labels/data"][start:end]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(labels, dtype=torch.long)

        graph = self.preprocessor(data_x, data_y)

        return graph


class BaikalDatasetTrackCascadeGraph(BaikalDatasetGraph):
    def __init__(self, *args, tres_cut: float = 10.0, **kwargs):
        print(kwargs)
        super().__init__(*args, **kwargs)
        self.tres_cut = tres_cut
        assert self.preprocessor is not None

    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        labels = self.hfile[self.split_type + "/labels/data"][start:end]
        tres = self.hfile[self.split_type + "/t_res/data"][start:end]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(labels, dtype=torch.long)
        tres = torch.tensor(tres, dtype=torch.float32)

        graph = self.preprocessor(data_x, data_y, tres, self.tres_cut)

        return graph


class BaikalDatasetTresGraph(BaikalDataset):
    pass
