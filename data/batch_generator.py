import os
import logging
from typing import Generator

import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

try:
    from data.root_manager.chunk_generator import ChunkGenerator
    from data.config import BatchGeneratorConfig
except ImportError:
    from root_manager.chunk_generator import ChunkGenerator
    from data.config import BatchGeneratorConfig


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BatchGenerator:

    def __init__(self, root_paths, cfg: BatchGeneratorConfig = BatchGeneratorConfig()):
        self.root_paths = root_paths
        self.cfg = cfg
        self.norm_params = self.cfg.norm_params
        self.aug_params = self.cfg.augment_params

        # Get chunks generator
        self.chunks_cfg = self.cfg.chunk_generator_cfg
        self.chunks = ChunkGenerator(root_paths, self.chunks_cfg).get_chunks()

        self.batch_size = self.cfg.batch_size

        self.name2index = None
        self.aug_stds = None
        self.means, self.stds = None, None

    def _df_to_tensors_list(self, df: pl.DataFrame) -> list[Tensor]:
        """
        Converts polars dataframe chunk into list of tensors of shape (L, num_features)
        """
        torch_tensors = list(map(torch.from_numpy, map(lambda x: np.stack(x, axis=-1), list(df.to_numpy()))))
        return torch_tensors

    def _augment_data(self, data: Tensor):
        if self.aug_stds is None:
            self.aug_stds = [0.0] * data.shape[2]
            for name, std in self.aug_params.to_dict().items():
                self.aug_stds[self.name2index[name]] = std
        B, L, num_fetures = data.shape
        aug_stds_as_tensor = torch.tensor(self.aug_stds, device=data.device).reshape(1, 1, num_fetures).repeat(B, L, 1)
        data = data + torch.normal(torch.zeros(data.shape, device=data.device), aug_stds_as_tensor)
        logging.debug(f"Added gauss noise to data: {data.shape=}")

        # Reorder by augmented time
        time_index = self.name2index["PulsesTime"]
        sort_idxs = data[:, :, time_index : time_index + 1].argsort(dim=1).repeat(1, 1, num_fetures)
        data = data.gather(dim=1, index=sort_idxs)
        assert (data[:, :, time_index].diff(n=1, dim=1).nan_to_num(1.0) >= 0).all(), logging.warning(
            f"Wrong time reordering,\n{data[:, :, time_index]=}"
        )
        logging.debug(f"Reordered times in data: {data.shape=}")
        return data

    def _norm_data(self, data: Tensor):
        if self.means is None or self.stds is None:
            self.means, self.stds = [0.0] * data.shape[2], [1.0] * data.shape[2]
            for name, (mean, std) in self.norm_params.to_dict().items():
                self.means[self.name2index[name]] = mean
                self.stds[self.name2index[name]] = std
        logging.debug(f"Using norming params: {self.means=}, {self.stds=}")
        B, L, num_fetures = data.shape
        means_as_tensor = torch.tensor(self.means, device=data.device).reshape(1, 1, num_fetures).repeat(B, L, 1)
        stds_as_tensor = torch.tensor(self.stds, device=data.device).reshape(1, 1, num_fetures).repeat(B, L, 1)
        data = (data - means_as_tensor) / stds_as_tensor
        logging.debug(f"Normilized data: {data.shape=}")
        return data
    
    def _process_data_in_batch(self, batch_list_of_tensors: list[Tensor], device: torch.device) -> Tensor:
        features_batch = pad_sequence(batch_list_of_tensors, batch_first=True, padding_value=float('nan')).to(device)
        if features_batch.shape[0] == 0:
            logging.warning("Empty batch encountered.")
        logging.debug(f"Coverting features_batch to tensor of shape: {features_batch.shape}")
        if self.cfg.do_augment:
            features_batch = self._augment_data(features_batch)
        if self.cfg.do_norm:
            features_batch = self._norm_data(features_batch)
        return features_batch.float()
    
    def _process_labels_in_batch(self, batch_from_chunk: pl.DataFrame, device: torch.device) -> Tensor:
        regime = self.cfg.target_regime.lower()
        if regime == "munusep_2":
            labels_batch = batch_from_chunk['nu_induced'].to_torch().to(device)[:,None]
            labels_batch = torch.concat((~labels_batch, labels_batch), dim=1) # to one-hot
        elif regime == "munusep_3":
            batch_from_chunk = batch_from_chunk.with_columns(
                onehot_encoding=pl.when(~pl.col("enough_info")).then([1., 0., 0.])
                .when(pl.col("enough_info") & ~pl.col("nu_induced")).then([0., 1., 0.])
                .when(pl.col("enough_info") & pl.col("nu_induced")).then([0., 0., 1.])
                .otherwise([0., 0., 0.]) # Fallback in case none of the conditions are met
                )
            labels_batch = torch.tensor(batch_from_chunk["onehot_encoding"].to_list()).to(device)
            ...
        else:
            raise ValueError
        return labels_batch.float()

    def get_batches(self, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> Generator[tuple[Tensor, Tensor, Tensor], None, None]:
        """
        Generate batches of data, mask and targets for training. 
        The data shape is (batch_size, max_length, num_of_feature). Auxillary hits are preplaced with 0.
        The target shape is (batch_size, 2)
        """

        chunk_index = 0
        batch_index = 0
        for chunk in self.chunks:
            logging.info(f"#{chunk_index} chunk loaded.")
            chunk_index += 1
            if self.cfg.shuffle:
                chunk = chunk.sample(n=chunk.shape[0], shuffle=True)
                logging.debug("Shuffling data within chunk.")
            
            data_in_chunk = chunk[self.cfg.features_name]
            if self.name2index is None:
                self.name2index = {c: i for i, c in enumerate(data_in_chunk.columns)}
            list_of_data_as_tensors = self._df_to_tensors_list(data_in_chunk) # compute once here for better usage of GPU after.
            
            # Iterate to get batches
            for start in range(0, chunk.shape[0], self.batch_size):
                stop = start + self.batch_size
                # pad data in batch with nans
                features_batch = self._process_data_in_batch(list_of_data_as_tensors[start:stop], device)
                # get batch of labels
                labels_batch = self._process_labels_in_batch(chunk[start:stop], device)
                logging.debug(
                    "Batch %d generated. features_batch shape: %s, labels_batch shape: %s",
                    batch_index,
                    features_batch.shape,
                    labels_batch.shape,
                )
                batch_index += 1
                mask = ~features_batch.isnan()[:,:, 0:1] # extract mask
                yield features_batch.nan_to_num(0.), mask, labels_batch
        
    def reinit(self):
        self.chunks = ChunkGenerator(self.root_paths, self.chunks_cfg).get_chunks()


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
    all_paths = mu_paths + nuatm_paths + nu2_paths

    # Default settings. To learn more possibilities, examine `data.root_manager.settings` and `root_manager.settings`
    cfg = BatchGeneratorConfig()
    train_data = BatchGenerator(root_paths=all_paths, cfg=cfg)
    batches = train_data.get_batches()

    for i, (data, labels) in enumerate(batches):
        logging.info(f"Processed batch #{i}")
