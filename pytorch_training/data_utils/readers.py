import numpy as np
import h5py as h5
import torch
from torch.utils.data import Dataset
import logging
from .preprocessors import BasePreprocessor, BaseGraphPreprocessor
import math


class BaikalDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        split_type: str,
        preprocessor: BasePreprocessor | BaseGraphPreprocessor,
        set_tres_stats: bool = False,
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

        # used in inherited classes
        if set_tres_stats:
            logging.info(f"counting tres mean and std for {self.split_type}...")
            self.tres_data = np.array(self.hfile[self.split_type + "/t_res/data"])
            tres_mean = self.tres_data.mean()
            tres_std = self.tres_data.std()
            self.preprocessor.set_stats(tres_mean, tres_std)
            logging.info("finished")

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
    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        tres = self.tres_data[start:end]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(tres, dtype=torch.float32)

        data_x, data_y = self.preprocessor(data_x, data_y)

        return data_x, data_y


class BaikalDatasetTrackCascade(BaikalDataset):
    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        labels = self.hfile[self.split_type + "/labels/data"][start:end]
        tres = self.hfile[self.split_type + "/t_res/data"][start:end]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(labels, dtype=torch.long)
        tres = torch.tensor(tres, dtype=torch.float32)

        data_x, data_y = self.preprocessor(data_x, data_y, tres)

        return data_x, data_y


class BaikalDatasetAngles(BaikalDataset):
    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        thetha, phi = self.hfile[self.split_type + "/prime_prty/data"][idx][:2]
        thetha = float(thetha) * (torch.pi / 180)
        phi = float(phi) * (torch.pi / 180)
        vec = [math.cos(thetha) * math.cos(phi), math.cos(thetha) * math.sin(phi), math.sin(thetha)]
        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(vec, dtype=torch.float32)
        data_y /= data_y.norm()
        return self.preprocessor(data_x, data_y)


class BaikalDatasetAnglesAndTrackCascade(BaikalDataset):
    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        thetha, phi = self.hfile[self.split_type + "/prime_prty/data"][idx][:2]
        tres = self.hfile[self.split_type + "/t_res/data"][start:end]

        a1 = math.sin(float(thetha) * (torch.pi / 180))
        a2 = math.cos(float(phi) * (torch.pi / 180))

        labels = self.hfile[self.split_type + "/labels/data"][start:end]
        track_cascade_labels = torch.tensor(labels, dtype=torch.long)

        tres = torch.tensor(tres, dtype=torch.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        angles = torch.tensor([a1, a2], dtype=torch.float32)

        data_x, data_y = self.preprocessor(data_x, track_cascade_labels, angles, tres)

        return data_x, data_y



class BaikalDatasetGraph(BaikalDataset):
    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        labels = self.hfile[self.split_type + "/labels/data"][start:end]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(labels, dtype=torch.long)

        graph = self.preprocessor(data_x, data_y)

        return graph


class BaikalDatasetTrackCascadeGraph(BaikalDatasetGraph):
    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        labels = self.hfile[self.split_type + "/labels/data"][start:end]
        tres = self.hfile[self.split_type + "/t_res/data"][start:end]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(labels, dtype=torch.long)
        tres = torch.tensor(tres, dtype=torch.float32)

        graph = self.preprocessor(data_x, data_y, tres)

        return graph


class BaikalDatasetTresGraph(BaikalDataset):
    def __getitem__(self, idx):
        start, end = self.hfile[self.split_type + "/ev_starts/data"][idx : idx + 2]
        data = np.array(self.hfile[self.split_type + "/data/data"][start:end])
        tres = self.tres_data[start:end]

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(tres, dtype=torch.float32)

        graph = self.preprocessor(data_x, data_y)

        return graph
