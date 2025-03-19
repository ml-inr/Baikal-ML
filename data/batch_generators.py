import logging
from typing import List, Generator, Tuple, Optional

from dataclasses import field
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import polars as pl

from data.data_loader import ChunksFromPaths
from data.settings_scheme import ChunksFromPathsConfig


class MCMuNuSepBatchGenerator:
    
    """
    MCMuNuSepBatchGenerator

    This class is a PyTorch-style data generator designed for loading, processing, and batching datasets containing two distinct types of events: muons (mu) and neutrinos (nu). It handles chunked data loading, normalization, augmentation, and batching, providing an efficient mechanism for training deep learning models on large datasets.

    Arguments:
    ----------
    - mu_paths (List[str]): List of file paths containing muon event data.
    - nu_paths (List[str]): List of file paths containing neutrino event data.
    - chunk_gen_cfg (ChunksFromPathsConfig): Configuration for chunk generation.
    - mu_events_per_chunk (int, optional): Number of muon events per chunk. Default is 256 * 100.
    - nu_events_per_chunk (int, optional): Number of neutrino events per chunk. Default is 256 * 100.
    - batch_size (int, optional): Number of samples per batch. Default is 256.
    - features_to_take (Tuple[str], optional): List of feature names to extract. Default is 
    ("PulsesAmpl", "PulsesTime", "Xrel", "Yrel", "Zrel").
    - do_norm (bool, optional): Whether to apply normalization to features. Default is True.
    - norm_params (Tuple[Tuple[float, float]], optional): Normalization parameters (mean, std) for each feature. Default is 
    ((0.0, 238.5), (6.8, 118.7), (0.0, 60.0), (0.0, 60.0), (0.0, 260.0)).
    - do_augment (bool, optional): Whether to apply augmentation (random noise) to features. Default is True.
    - augment_params (Tuple[float], optional): Augmentation noise parameters for each feature. Default is 
    (5.0, 0.1, 2.0, 2.0, 5.0).
    - shuffle (bool, optional): Whether to shuffle the data before batching. Default is True.
    - device (torch.device, optional): Device to load tensors onto (e.g., CPU or GPU). Default is `torch.device("cpu")`.
        
    Attributes:
    -----------
    - mu_chunks (ChunksFromPaths): Chunk loader for muon data.
    - nu_chunks (ChunksFromPaths): Chunk loader for neutrino data.
    - batch_size (int): Number of samples per batch.
    - features_to_take (Tuple[str]): List of feature names to extract from the dataset.
    - device (torch.device): Device to load tensors onto (e.g., CPU or GPU).
    - norm_params (Tuple[Tuple[float, float]]): Normalization parameters for each feature (mean, std).
    - augment_params (Tuple[float]): Augmentation noise parameters for each feature.
    - shuffle (bool): Whether to shuffle data before batching.

    Methods:
    --------
    - __init__: Initializes the generator with dataset paths, configuration, and preprocessing options.
    - _prepare_mu_df: Loads and preprocesses a chunk of muon data.
    - _prepare_nu_df: Loads and preprocesses a chunk of neutrino data.
    - _features_to_list_of_tensors: Converts a Polars DataFrame of features into a list of PyTorch tensors.
    - _augment_features_in_batch: Adds random noise to feature values for data augmentation.
    - _norm_features_in_batch: Normalizes feature values based on the specified parameters.
    - reset: Resets the generator state for iteration.
    - __iter__: Prepares the generator for iteration.
    - __next__: Generates the next batch of data, including features and labels.
    """
    
    def __init__(self,
                 mu_paths: list[str],
                 nu_paths: list[str],
                 chunk_generator_cfg: dict = ChunksFromPathsConfig().to_dict(),
                 mu_events_per_chunk: int = 256*100,
                 nu_events_per_chunk: int = 256*100,
                 batch_size: int = 256,
                 features_to_take: list[str] = [
                         "PulsesAmpl",
                         "PulsesTime",
                         "Xrel",
                         "Yrel",
                         "Zrel"
                     ],
                 do_norm: bool = True,
                 norm_params: list[list[float, float]] = [
                         [0.0, 238.5],  # PulsesAmpl
                         [6.8, 118.7],  # PulsesTime
                         [0.0, 60.0],   # Xrel
                         [0.0, 60.0],   # Yrel
                         [0.0, 260.0]   # Zrel
                     ],
                 do_augment: bool = True,
                 augment_params: list[float] = [
                         5.0,   # PulsesTime (ns)
                         0.1,   # PulsesAmpl (ev)
                         2.0,   # Xrel (m)
                         2.0,   # Yrel (m)
                         5.0    # Zrel (m)
                     ],
                 shuffle: bool = True,
                 device: torch.device = torch.device("cpu")
                 , return_df: bool = False
                 ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        self.chunk_gen_cfg = chunk_generator_cfg
        
        self.mu_chunks = ChunksFromPaths(mu_paths, events_per_chunk=mu_events_per_chunk, **self.chunk_gen_cfg)
        self.nu_chunks = ChunksFromPaths(nu_paths, events_per_chunk=nu_events_per_chunk, **self.chunk_gen_cfg)
        
        self.batch_size = batch_size
        self.features_to_take = features_to_take
        self.device = device
        if do_norm:
            self.norm_params = norm_params
            self.means = torch.tensor([mean for mean, _ in norm_params], device=self.device)
            self.stds = torch.tensor([std for _, std in norm_params], device=self.device)
        else:
            self.norm_params = None
        
        self.augment_params = augment_params if do_augment else None
        
        self.shuffle = shuffle
        
        self.return_df = return_df
        
        self.reset()

    def _prepare_mu_df(self) -> pl.DataFrame:
        try:
            pulses, events, muons = next(self.mu_chunks)
        except StopIteration:
            self._mu_was_over = True
            self.mu_chunks.reset()
            pulses, events, muons = next(self.mu_chunks)
        group_by = ["ev_id", "cluster_id"]
        df = (pulses[group_by + self.features_to_take]
              .sort(by="PulsesTime")
              .group_by(group_by, maintain_order=True)
              .agg(self.features_to_take)
            )
        df = df.with_columns(pl.lit(False).alias("label"))
        return df, events, muons
    
    def _prepare_nu_df(self) -> pl.DataFrame:
        try:
            pulses, events, muons = next(self.nu_chunks)
        except StopIteration:
            self._nu_was_over = True
            self.nu_chunks.reset()
            pulses, events, muons = next(self.nu_chunks)
        group_by = ["ev_id", "cluster_id"]
        df = (pulses[group_by + self.features_to_take]
              .sort(by="PulsesTime")
              .group_by(group_by, maintain_order=True)
              .agg(self.features_to_take)
            )
        df = df.with_columns(pl.lit(True).alias("label"))
        return df, events, muons

    def _features_to_list_of_tensors(self, df: pl.DataFrame) -> List[torch.Tensor]:
        data = list(df.to_numpy())
        list_of_np_arrays = [np.stack(features, axis=-1) for features in data]
        list_of_tensors = [torch.tensor(sequence, device=self.device) for sequence in list_of_np_arrays]
        return list_of_tensors

    def _augment_features_in_batch(self, batch_features: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(batch_features) * torch.tensor(self.augment_params, device=self.device)
        augmented_features = batch_features + noise

        time_index = self.features_to_take.index("PulsesTime")
        sort_idxs = augmented_features[:, :, time_index:time_index + 1].argsort(dim=1).expand(-1, -1, len(self.features_to_take))
        augmented_features = augmented_features.gather(dim=1, index=sort_idxs)
        
        return augmented_features

    def _norm_features_in_batch(self, batch_features: torch.Tensor) -> torch.Tensor:
        return (batch_features - self.means) / self.stds

    def reset(self):
        self._current_chunk_idx = -1
        self._current_global_batch_idx = -1
        self._current_local_batch_idx = -1
        self.chunk_of_data_tensors = None
        self.chunk_of_labels = None
        self._num_batches_to_load = None
        self._mu_was_over, self._nu_was_over = False, False
        self.mu_chunks.reset()
        self.nu_chunks.reset()
        pass
    
    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # You need the cicle to ensure you end the iteration only when returning smth or when StopIteration occurs.
        while True:
            if self.chunk_of_data_tensors is None:
                # loading chunk of data
                self._current_chunk_idx += 1
                logging.debug(f"Start of loading chunk #{self._current_chunk_idx}")
                mu_pulses, mu_events, mu_muons = self._prepare_mu_df()
                nu_pulses, nu_events, nu_muons = self._prepare_nu_df()
                if (self._mu_was_over and self._nu_was_over):
                    logging.info(f"The dataset is over, raising stop iteration")
                    raise StopIteration
                logging.debug(f"Chunks are loaded as dataframes: {mu_pulses.shape=}, {nu_pulses.shape=}")
                self.combined_pulses_df, self.combined_events, self.combined_muons = pl.concat([mu_pulses, nu_pulses]), pl.concat([mu_events, nu_events]), pl.concat([mu_muons, nu_muons])
                logging.debug(f"Chunks are concatenated: {self.combined_pulses_df.shape=}")
                if self.shuffle:
                    self.combined_pulses_df = self.combined_pulses_df.sample(fraction=1.0, shuffle=True)
                    logging.debug(f"DataFrame is shuffled")
                self.chunk_of_data_tensors = self._features_to_list_of_tensors(self.combined_pulses_df[self.features_to_take])
                logging.debug(f"Features are extracted: {len(self.chunk_of_data_tensors)=}, {self.chunk_of_data_tensors[0].shape=}")
                self._num_batches_to_load = int(np.ceil( len(self.chunk_of_data_tensors) / self.batch_size ) )
                self.chunk_of_labels = torch.tensor(self.combined_pulses_df["label"].to_numpy(), device=self.device)
                logging.debug(f"Labels are extracted: {self.chunk_of_labels.shape=}")
                # Clear extra memory
                if not self.return_df:
                    self.combined_pulses_df, self.combined_events, self.combined_muons = None, None, None
            
            logging.debug(f"{self._num_batches_to_load} batches to load from given chunk.")
            if self._current_local_batch_idx+1 < self._num_batches_to_load:
                # generating batches
                self._current_local_batch_idx += 1
                self._current_global_batch_idx += 1
                logging.debug(f"#{self._current_global_batch_idx=}")
                logging.debug(f"#{self._current_local_batch_idx=}")
                i = self._current_local_batch_idx*self.batch_size
                batch_features = pad_sequence(self.chunk_of_data_tensors[i:i + self.batch_size], batch_first=True, padding_value=float('nan'))
                logging.debug(f"Batch for input is selected: {batch_features.shape=}")
                if self.augment_params is not None:
                    batch_features = self._augment_features_in_batch(batch_features)
                    logging.debug(f"Batch for input is augmented: {batch_features.shape=}")
                if self.norm_params is not None:
                    batch_features = self._norm_features_in_batch(batch_features)
                    logging.debug(f"Batch for input is normed: {batch_features.shape=}")
                batch_labels = self.chunk_of_labels[i:i + self.batch_size, None]
                batch_labels = torch.concat((~batch_labels, batch_labels), dim=1) # to one-hot
                logging.debug(f"Batch of labels is selected: {batch_labels.shape=}")
                mask = ~batch_features[:,:, 0:1].isnan() # extract mask
                if self.return_df:
                    df_batch = self.combined_pulses_df[i:i + self.batch_size]
                    df_batch = df_batch.join(
                        self.combined_events
                        , on=["ev_id", "cluster_id"]
                        , how="left"
                        , maintain_order="left"
                        )
                    df_batch = df_batch.join(
                            self.combined_muons
                            , on=["ev_id"]
                            , how="left"
                            , maintain_order="left"
                            ).group_by(df_batch.columns, maintain_order=True).agg([col for col in self.combined_muons.columns if col not in df_batch.columns])
                    return batch_features.nan_to_num(0.).float(), mask, batch_labels.float(), df_batch
                else:
                    return batch_features.nan_to_num(0.).float(), mask, batch_labels.float()
            else:
                self._current_local_batch_idx = -1
                self.chunk_of_data_tensors = None


class ExpBatchGenerator:
    """
    ExpBatchGenerator

    This class is a PyTorch-style data generator designed for loading, processing, and batching a single dataset for machine learning tasks. It handles chunked data loading, optional normalization, augmentation, and batching, providing an efficient mechanism for preparing data in training pipelines.

    Arguments:
    ----------
    - paths (List[str]): List of file paths containing event data.
    - chunk_gen_cfg (ChunksFromPathsConfig): Configuration for chunk generation.
    - events_per_chunk (int, optional): Number of events per chunk. Default is 256 * 200.
    - batch_size (int, optional): Number of samples per batch. Default is 256.
    - features_to_take (List[str], optional): List of feature names to extract. Default is 
    ["PulsesAmpl", "PulsesTime", "Xrel", "Yrel", "Zrel"].
    - do_norm (bool, optional): Whether to apply normalization to features. Default is True.
    - norm_params (List[Tuple[float, float]], optional): Normalization parameters (mean, std) for each feature. Default is 
    [(0.0, 238.5), (6.8, 118.7), (0.0, 60.0), (0.0, 60.0), (0.0, 260.0)].
    - do_augment (bool, optional): Whether to apply augmentation (random noise) to features. Default is False.
    - augment_params (List[float], optional): Augmentation noise parameters for each feature. Default is 
    [5.0, 0.1, 2.0, 2.0, 5.0].
    - shuffle (bool, optional): Whether to shuffle the data before batching. Default is False.
    - device (torch.device, optional): Device to load tensors onto (e.g., CPU or GPU). Default is `torch.device("cpu")`.

    Methods:
    --------
    - __init__: Initializes the generator with dataset paths, configuration, and preprocessing options.
    - _prepare_df: Loads and preprocesses a chunk of data.
    - _features_to_list_of_tensors: Converts a Polars DataFrame of features into a list of PyTorch tensors.
    - _augment_features_in_batch: Adds random noise to feature values for data augmentation.
    - _norm_features_in_batch: Normalizes feature values based on the specified parameters.
    - reset: Resets the generator state for iteration.
    - __iter__: Prepares the generator for iteration.
    - __next__: Generates the next batch of data, including features.
    """
    def __init__(self,
                 paths: List[str],
                 chunk_generator_cfg: dict = ChunksFromPathsConfig(),
                 events_per_chunk: int = 256*200,
                 batch_size: int = 256,
                 features_to_take: List[str] = [
                         "PulsesAmpl",
                         "PulsesTime",
                         "Xrel",
                         "Yrel",
                         "Zrel"
                     ],
                 do_norm: bool = True,
                 norm_params: list[list[float, float]] = [
                         [0.0, 238.5],  # PulsesAmpl
                         [6.8, 118.7],  # PulsesTime
                         [0.0, 60.0],   # Xrel
                         [0.0, 60.0],   # Yrel
                         [0.0, 260.0]   # Zrel
                     ],
                 do_augment: bool = False,
                 augment_params: List[float] = [
                         5.0,   # PulsesTime (ns)
                         0.1,   # PulsesAmpl (ev)
                         2.0,   # Xrel (m)
                         2.0,   # Yrel (m)
                         5.0    # Zrel (m)
                     ],
                 shuffle: bool = False,
                 device: torch.device = torch.device("cpu")
                 , return_df: bool = False
                 ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        self.chunk_gen_cfg = chunk_generator_cfg
        self.chunks = ChunksFromPaths(paths, events_per_chunk=events_per_chunk, **self.chunk_gen_cfg)
        self.batch_size = batch_size
        self.features_to_take = features_to_take
        self.norm_params = norm_params if do_norm else None
        self.augment_params = augment_params if do_augment else None
        self.shuffle = shuffle
        self.device = device
        self.return_df = return_df
        
        self.reset()

    def _prepare_df(self) -> pl.DataFrame:
        try:
            pulses, events, _ = next(self.chunks)
        except StopIteration:
            self._data_was_over = True
            self.chunks.reset()
            pulses, events, _ = next(self.chunks)
        group_by = ["ev_id", "cluster_id"]
        pulses = (pulses[group_by + self.features_to_take]
              .sort(by="PulsesTime")
              .group_by(group_by, maintain_order=True)
              .agg(self.features_to_take)
            )
        return pulses, events

    def _features_to_list_of_tensors(self, df: pl.DataFrame) -> List[torch.Tensor]:
        data = list(df.to_numpy())
        list_of_np_arrays = [np.stack(features, axis=-1) for features in data]
        list_of_tensors = [torch.tensor(sequence, device=self.device) for sequence in list_of_np_arrays]
        return list_of_tensors

    def _augment_features_in_batch(self, batch_features: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(batch_features) * torch.tensor(self.augment_params, device=self.device)
        augmented_features = batch_features + noise

        time_index = self.features_to_take.index("PulsesTime")
        sort_idxs = augmented_features[:, :, time_index:time_index + 1].argsort(dim=1).expand(-1, -1, len(self.features_to_take))
        augmented_features = augmented_features.gather(dim=1, index=sort_idxs)
        
        return augmented_features

    def _norm_features_in_batch(self, batch_features: torch.Tensor) -> torch.Tensor:
        means = torch.tensor([mean for mean, _ in self.norm_params], device=self.device)
        stds = torch.tensor([std for _, std in self.norm_params], device=self.device)
        return (batch_features - means) / stds

    def reset(self):
        self._current_chunk_idx = -1
        self._current_global_batch_idx = -1
        self._current_local_batch_idx = -1
        self.chunk_of_data_tensors = None
        self._num_batches_to_load = None
        self._data_was_over = False
        pass
    
    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[pl.DataFrame]]:
        # You need the cicle to ensure you end the iteration only when returning smth or when StopIteration occurs.
        while True:
            if self._num_batches_to_load is None:
                self._current_chunk_idx += 1
                logging.debug(f"Start of loading chunk #{self._current_chunk_idx}")
                self.pulses_df, self.events_df = self._prepare_df()
                if self._data_was_over:
                    logging.info(f"The dataset is over, raising stop iteration")
                    raise StopIteration
                logging.debug(f"Chunk is loaded as a dataframe: {self.pulses_df.shape=}")
                
                if self.shuffle:
                    self.pulses_df = self.pulses_df.sample(fraction=1.0, shuffle=True)
                    logging.debug(f"DataFrame is shuffled")
                
                self.chunk_of_data_tensors = self._features_to_list_of_tensors(self.pulses_df[self.features_to_take])
                logging.debug(f"Features are extracted: {len(self.chunk_of_data_tensors)=}, {self.chunk_of_data_tensors[0].shape=}")
                self._num_batches_to_load = int(np.ceil( len(self.chunk_of_data_tensors) / self.batch_size ) )
                # Clear extra memory
                if not self.return_df:
                    self.pulses_df, self.events_df = None, None
                    
            if self._current_local_batch_idx+1 < self._num_batches_to_load:
                self._current_local_batch_idx += 1
                self._current_global_batch_idx += 1
                logging.debug(f"#{self._current_global_batch_idx=}")
                logging.debug(f"#{self._current_local_batch_idx=}")
                i = self._current_local_batch_idx*self.batch_size
                batch_features = pad_sequence(self.chunk_of_data_tensors[i:i + self.batch_size], batch_first=True, padding_value=float('nan'))
                logging.debug(f"Batch for input is selected: {batch_features.shape=}")
                if self.augment_params is not None:
                    batch_features = self._augment_features_in_batch(batch_features)
                    logging.debug(f"Batch for input is augmented: {batch_features.shape=}")
                if self.norm_params is not None:
                    batch_features = self._norm_features_in_batch(batch_features)
                    logging.debug(f"Batch for input is normed: {batch_features.shape=}")
                mask = ~batch_features[:,:, 0:1].isnan() # extract mask
                if self.return_df:
                    df_batch = self.pulses_df[i:i + self.batch_size]
                    df_batch = df_batch.join(
                        self.events_df
                        , on=["ev_id", "cluster_id"]
                        , how="left"
                        , maintain_order="left"
                        )
                    return batch_features.nan_to_num(0.).float(), mask, df_batch
                else:
                    return batch_features.nan_to_num(0.), mask
            else:
                self._num_batches_to_load = None
                self._current_local_batch_idx = -1
