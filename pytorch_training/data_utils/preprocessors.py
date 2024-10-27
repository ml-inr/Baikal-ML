import torch
from abc import ABC, abstractmethod
import logging
import h5py as h5
import numpy as np
from torch_geometric.data import Data as GData
import torch_geometric.nn as gnn
import typing as tp

EPS = 1e-8
# TODO: add noise

# class FilterData:
#     def __init__(
#             self,
#             data_file,
#             split_type,
#             Q_lower_bound: float | None = None,
#             Q_upper_bound: float | None = None,
#             additive_gauss_noise_std: tp.Sequence[float] | None = None = None,
#             mult_gauss_noise_fraction: float | None = None,
#             ),


class BasePreprocessor(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class NoiseSigPreprocessor(BasePreprocessor):
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y[y != 0] = 1
        return x, y.unsqueeze(0)


class TrackCascadePreprocessor(BasePreprocessor):
    def __init__(self, tres_cut):
        self.tres_cut = tres_cut

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, tres: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y[y > 0] = 1  # cascade
        y[y < 0] = 0  # track
        y[torch.abs(tres) < self.tres_cut] = 0
        return x, y.unsqueeze(0)

class TresPreprocessor(BasePreprocessor):
    def __init__(self):
        self.tres_mean = None
        self.tres_std = None

    def set_stats(self, tres_mean: float, tres_std: float):
        self.tres_mean = tres_mean
        self.tres_std = tres_std

    def __call__(
        self, x: torch.Tensor, tres: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tres = (tres - self.tres_mean) / (self.tres_std + EPS)
        return x, tres


class TresAndTrackCascadePreprocessor(TresPreprocessor):
    def __init__(self, tres_cut):
        super().__init__()
        self.tres_cut = tres_cut

    def set_stats(self, tres_mean: float, tres_std: float):
        self.tres_mean = tres_mean
        self.tres_std = tres_std

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, tres: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y[y > 0] = 1
        y[y < 0] = 0
        y[torch.abs(tres) < self.tres_cut] = 0
        tres = (tres - self.tres_mean) / (self.tres_std + EPS)
        labels_and_tres = torch.stack((y, tres), -1)
        return x, labels_and_tres


class AnglePreprocessor(BasePreprocessor):
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, y


class AngleAndTrackCascadePreprocessor(TrackCascadePreprocessor):
    def __call__(
        self,
        x: torch.Tensor,
        track_cascade_labels: torch.Tensor,
        angles: torch.Tensor,
        tres: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        track_cascade_labels[track_cascade_labels > 0] = 1
        track_cascade_labels[track_cascade_labels < 0] = 0
        track_cascade_labels[torch.abs(tres) < self.tres_cut] = 0        
        return x, (track_cascade_labels.unsqueeze(0), angles)


class BaseGraphPreprocessor(ABC):
    def __init__(self, n_neighbours: int):
        self.n_neighbours = n_neighbours

    @abstractmethod
    def __call__(self, *args, **kwargs) -> GData:
        pass


class AngleGraphPreprocessor(BaseGraphPreprocessor):
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> GData:
        edge_index = gnn.knn_graph(x[:, 1], k=self.n_neighbours)
        graph = GData(x=x, edge_index=edge_index, y=y)
        return graph
        

class NoiseSigGraphPreprocessor(BaseGraphPreprocessor):
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> GData:
        y[y != 0] = 1
        edge_index = gnn.knn_graph(x[:, 1], k=self.n_neighbours)
        graph = GData(x=x, edge_index=edge_index, y=y)
        return graph


class TrackCascadeGraphPreprocessor(BaseGraphPreprocessor):
    def __init__(self, n_neighbours: int, tres_cut: float):
        super().__init__(n_neighbours)
        self.tres_cut = tres_cut

    def __call__(self, x: torch.Tensor, y: torch.Tensor, tres: torch.Tensor) -> GData:

        y[y > 0] = 1  # cascade
        y[y < 0] = 0  # track
        y[torch.abs(tres) < self.tres_cut] = 0
        
        edge_index = gnn.knn_graph(x[:, 1], k=self.n_neighbours)
        graph = GData(x=x, edge_index=edge_index, y=y)
        return graph


class TresGraphPreprocessor(BaseGraphPreprocessor):
    def __init__(self, n_neighbours: int):
        super().__init__(n_neighbours)
        self.tres_mean = None
        self.tres_std = None

    def set_stats(self, tres_mean: float, tres_std: float):
        self.tres_mean = tres_mean
        self.tres_std = tres_std

    def __call__(self, x: torch.Tensor, tres: torch.Tensor) -> GData:
        assert (
            self.tres_mean is not None and self.tres_std is not None
        ), "stats for preproccesor weren't not set"
        tres = (tres - self.tres_mean) / (self.tres_std + EPS)
        edge_index = gnn.knn_graph(x[:, 1], k=self.n_neighbours)
        graph = GData(x=x, edge_index=edge_index, y=tres)
        return graph


class TresAndTrackCascadeGraphPreprocessor(TresGraphPreprocessor):
    def __init__(self, n_neighbours: int, tres_cut: float):
        super().__init__(n_neighbours)
        self.tres_cut = tres_cut

    def __call__(self, x: torch.Tensor, y: torch.Tensor, tres: torch.Tensor) -> GData:
        y[y > 0] = 1
        y[y < 0] = 0
        y[torch.abs(tres) < self.tres_cut] = 0
        tres = (tres - self.tres_mean) / (self.tres_std + EPS)
        labels_and_tres = torch.stack((y, tres), dim=-1)
        edge_index = gnn.knn_graph(x[:, 1], k=self.n_neighbours)
        graph = GData(x=x, edge_index=edge_index, y=labels_and_tres)
        return graph
