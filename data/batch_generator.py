import os
import logging
from random import shuffle
from typing import Optional, Generator

import numpy as np
import polars as pl
import torch
from torch import Tensor

try:
    from data.root_manager.chunk_generator import ChunkGenerator
    from data.settings import BatchGeneratorConfig
except ImportError:
    from root_manager.chunk_generator import ChunkGenerator
    from settings import BatchGeneratorConfig


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BatchGenerator:

    def __init__(self, cfg: BatchGeneratorConfig, collect_stats=False):
        self.cfg = cfg
        self.norm_params = cfg.norm_params
        self.aug_params = cfg.augment_parmas

        # Get chunks generator
        self.chunks = ChunkGenerator(cfg.chunk_generator_cfg).get_chunks()

        self.batch_size = cfg.batch_size

        self.name2index = None
        self.aug_stds = None
        self.means, self.stds = None, None

    def _to_padded_batch(self, df: pl.DataFrame) -> tuple[list[list[list]]]:
        """
        Flatten and pad input data to ensure consistent dimensions.
        """
        max_length = df["PulsesTime"].list.len().max()
        logging.debug("Max length determined for padding: %d", max_length)

        result = []
        for row in df.iter_rows():
            row_result = [df_list + [torch.nan] * (max_length - len(df_list)) for df_list in row]
            result.append(row_result)
        # Name-index correspondance
        if self.name2index is None:
            self.name2index = {name: i for i, name in enumerate(df.columns)}
        logging.debug("Data flattened and padded. Returning as list.")
        return result

    def _augment_data(self, data: torch.Tensor):
        if self.aug_stds is None:
            self.aug_stds = [0.0] * data.shape[2]
            for name, std in self.aug_params.to_dict().items():
                self.aug_stds[self.name2index[name]] = std
        aug_stds_as_tensor = (
            torch.tensor(self.aug_stds).reshape(1, 1, len(self.aug_stds)).repeat(data.shape[0], data.shape[1], 1)
        )
        data = data + torch.normal(torch.zeros(data.shape), aug_stds_as_tensor)
        logging.debug(f"Added gauss noise to data: {data.shape=}")

        # Reorder by augmented time
        time_index = self.name2index["PulsesTime"]
        sort_idxs = data[:, :, time_index : time_index + 1].argsort(dim=1).repeat(1, 1, data.shape[2])
        data = data.gather(dim=1, index=sort_idxs)
        assert (data[:, :, time_index].diff(n=1, dim=1).nan_to_num(1.0) > 0).all(), logging.warning(
            f"Wrong time reordering"
        )
        logging.debug(f"Reordered times in data: {data.shape=}")
        return data

    def _norm_data(self, data: torch.Tensor):
        if self.means is None or self.stds is None:
            self.means, self.stds = [0.0] * data.shape[2], [1.0] * data.shape[2]
            for name, (mean, std) in self.norm_params.to_dict().items():
                self.means[self.name2index[name]] = mean
                self.stds[self.name2index[name]] = std
        logging.debug(f"Using norming params: {self.means=}, {self.stds=}")
        means_as_tensor = (
            torch.tensor(self.means).reshape(1, 1, len(self.means)).repeat(data.shape[0], data.shape[1], 1)
        )
        stds_as_tensor = torch.tensor(self.stds).reshape(1, 1, len(self.stds)).repeat(data.shape[0], data.shape[1], 1)
        data = (data - means_as_tensor) / stds_as_tensor
        logging.debug(f"Normilized data: {data.shape=}")
        return data

    def get_batches(self, bs: Optional[int] = None) -> Generator[tuple[Tensor, Tensor], None, None]:
        """
        Generate batches of data for training.
        """

        if bs is None:
            bs = self.batch_size

        batch_index = 0
        for chunk in self.chunks:
            if self.cfg.shuffle:
                chunk = chunk.sample(n=chunk.shape[0], shuffle=True)
                logging.debug("Shuffling data within chunk.")
            for start in range(0, chunk.shape[0], bs):
                stop = start + bs
                pre_batch = chunk[start:stop]
                if pre_batch.shape[0] == 0:
                    logging.warning("Empty batch encountered. Skipping.")
                    break
                features_batch = self._to_padded_batch(pre_batch[self.cfg.features_name])
                features_batch = torch.tensor(features_batch).transpose(1, 2)
                logging.debug(f"Coverting features_batch to tensor of shape: {features_batch.shape}")

                if self.cfg.do_augment:
                    features_batch = self._augment_data(features_batch)
                if self.cfg.do_norm:
                    features_batch = self._norm_data(features_batch)

                labels_batch = pre_batch[self.cfg.labels_name].to_torch()
                logging.debug(
                    "Batch %d generated. features_batch shape: %s, labels_batch shape: %s",
                    batch_index,
                    features_batch.shape,
                    labels_batch.shape,
                )
                batch_index += 1
                yield features_batch, labels_batch


if __name__ == "__main__":
    try:
        from data.root_manager.chunk_generator import ChunkGenerator
        from data.root_manager.settings import ChunkGeneratorConfig
    except:
        from root_manager.chunk_generator import ChunkGenerator
        from root_manager.settings import ChunkGeneratorConfig

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.debug("test")
    logging.info("Starting the training data generator...")
    path_mu = "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/"
    path_nuatm = "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nuatm/root/all/"
    path_nu2 = "/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nue2_100pev/root/all/"

    def explore_paths(p: str, start: int, stop: int):
        files = os.listdir(p)[start:stop]
        return sorted([f"{p}{file}" for file in files])

    mu_paths = explore_paths(path_mu, 0, 2)
    nuatm_paths = explore_paths(path_nuatm, 0, 1)
    nu2_paths = explore_paths(path_nu2, 0, 1)

    chunks_cfg = ChunkGeneratorConfig(mu_paths, nuatm_paths, nu2_paths)
    cfg = BatchGeneratorConfig(chunk_generator=ChunkGenerator(chunks_cfg))

    train_data = BatchGenerator(cfg)
    batches = train_data.get_batches(256)

    for i, (data, labels) in enumerate(batches):
        logging.info(f"Processed batch #{i}")
