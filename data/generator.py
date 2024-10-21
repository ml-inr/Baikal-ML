import os
import logging
from random import shuffle
from typing import Optional, Generator, Tuple

import numpy as np
import polars as pl
import torch
from torch import Tensor

try:
    from data.root_manager.processor import Processor
    from data.settings import GeneratorConfig
except ImportError:
    from root_manager.processor import Processor
    from settings import GeneratorConfig
    
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainGenerator:
    """
    A generator class for preparing training batches from datasets of multiple sources.
    
    Attributes:
        cfg (GeneratorConfig): Configuration settings for the generator.
        collect_stats (bool): Flag to enable or disable collection of dataset statistics.
    """
    
    def __init__(self, cfg: GeneratorConfig, collect_stats=False):
        self.cfg = cfg
        self.norm_params = cfg.norm_params
        self.aug_params = cfg.augment_parmas
        self.proc_params = cfg.processor_params
        
        # Initialize file paths
        self.mu_paths = cfg.mu_paths
        self.nuatm_paths = cfg.nuatm_paths
        self.nu2_paths = cfg.nu2_paths
        
        self.all_paths = self.mu_paths + self.nuatm_paths + self.nu2_paths
        if cfg.shuffle:
            shuffle(self.all_paths)
            logging.info("Shuffled all file paths. Total paths: %d", len(self.all_paths))

        self.df_mu_chunk, self.df_nuatm_chunk, self.df_nu2_chunk = None, None, None

        if collect_stats:
            logging.info("Collecting dataset statistics...")
            self._get_stats()
        
        self.chunk_size = cfg.chunk_size
        self.batch_size = cfg.batch_size

    @staticmethod
    def _calc_sums(df: pl.DataFrame, field_name: str, num_files_local: int, num_files_total: int) -> Tuple[float, float, float]:
        """
        Calculate the number of hits, sum, and squared sum for a given field in the DataFrame.
        
        Args:
            df (pl.DataFrame): Input Polars DataFrame.
            field_name (str): The field to calculate sums for.
            num_files_local (int): Number of local files processed.
            num_files_total (int): Total number of files.
        
        Returns:
            Tuple[float, float, float]: Number of hits, sum, and squared sum.
        """
        num_hits = df[field_name].explode().shape[0] / num_files_local * num_files_total
        S = df[field_name].explode().sum() / num_files_local * num_files_total
        S2 = df[field_name].explode().pow(2).sum() / num_files_local * num_files_total
        logging.debug("Calculated sums for field '%s': num_hits=%.2f, sum=%.2f, squared_sum=%.2f", 
                      field_name, num_hits, S, S2)
        return num_hits, S, S2

    def _estimate_mean_and_std(self, df_mu: pl.DataFrame, df_nuatm: pl.DataFrame, df_nu2: pl.DataFrame, field_name: str, num_files_local: int) -> Tuple[float, float]:
        """
        Estimate mean and standard deviation for a given field across multiple datasets.
        """
        logging.info(f"Estimating mean and std for field: {field_name}")
        mu_hits, S_mu, S2_mu = self._calc_sums(df_mu, field_name, num_files_local, len(self.mu_paths))
        nuatm_hits, S_nuatm, S2_nuatm = self._calc_sums(df_nuatm, field_name, num_files_local, len(self.nuatm_paths))
        nu2_hits, S_nu2, S2_nu2 = self._calc_sums(df_nu2, field_name, num_files_local, len(self.nu2_paths))
        
        total_hits = mu_hits + nuatm_hits + nu2_hits
        mean = (S_mu + S_nuatm + S_nu2) / total_hits
        std = ((S2_mu + S2_nuatm + S2_nu2) / total_hits - mean ** 2) ** 0.5
        
        logging.info("Field '%s': Estimated Mean = %.4f, Std = %.4f", field_name, mean, std)
        return mean, std

    def _get_stats(self, start: int = 0, stop: int = 50) -> None:
        """
        Collect statistics for datasets, including mean and standard deviation estimates.
        """
        assert stop > start, "Stop index must be greater than start index."
        if stop - start > min(len(self.mu_paths), len(self.nuatm_paths), len(self.nu2_paths)):
            logging.warning("Range is too large for the available files. Adjusting to smaller range.")
            start = 0
            stop = min(len(self.mu_paths), len(self.nuatm_paths), len(self.nu2_paths))
        
        # Load processed data
        logging.info("Processing datasets from indices %d to %d.", start, stop)
        df_mu = (mu_proc := Processor(self.mu_paths[start:stop], self.proc_params)).process()
        df_nuatm = (nuatm_proc := Processor(self.nuatm_paths[start:stop], self.proc_params)).process()
        df_nu2 = (nu2_proc := Processor(self.nu2_paths[start:stop], self.proc_params)).process()

        # Estimate statistics
        self.mu_num_estimated = df_mu.shape[0] / (stop - start) * len(self.mu_paths)
        self.nuatm_num_estimated = df_nuatm.shape[0] / (stop - start) * len(self.nuatm_paths)
        self.nu2_num_estimated = df_nu2.shape[0] / (stop - start) * len(self.nu2_paths)

        self.mu_filter_koef = mu_proc.filter_koef
        self.nuatm_filter_koef = nuatm_proc.filter_koef
        self.nu2_filter_koef = nu2_proc.filter_koef
        
        self.mu_nu_ratio = self.mu_num_estimated / (self.nu2_num_estimated + self.nuatm_num_estimated)
        self.nuatm_nu2_ratio = self.nuatm_num_estimated / self.nu2_num_estimated
        
        logging.info("Calculating mean and std for data fields...")
        self.Q_mean, self.Q_std = self._estimate_mean_and_std(df_mu, df_nuatm, df_nu2, 'PulsesAmpl', stop - start)
        self.t_mean, self.t_std = self._estimate_mean_and_std(df_mu, df_nuatm, df_nu2, 'PulsesTime', stop - start)
        
        logging.info("Collected dataset statistics successfully. Ratios - Mu/Nu: %.4f, NuAtm/Nu2: %.4f", self.mu_nu_ratio, self.nuatm_nu2_ratio)

    def get_chunks(self, chunk_size: Optional[int] = None) -> Generator[pl.DataFrame, None, None]:
        """
        Yield data chunks based on the provided chunk size.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        for start in range(0, len(self.all_paths), chunk_size):
            stop = start + chunk_size
            logging.debug("Processing chunk from index %d to %d.", start, stop)
            df = Processor(self.all_paths[start:stop], self.proc_params).process()
            df = df[self.cfg.features + self.cfg.labels]
            
            if self.cfg.do_norm:
                for field_name, stats in self.norm_params.to_dict():
                    mean, std = stats
                    df = df.with_columns(((pl.col(field_name) - mean)/std).alias(field_name))
            
            if self.cfg.do_augment:
                for field_name, std in self.aug_params.to_dict():
                    df = df.with_columns(pl.col(field_name) + np.random.normal(0, std, pl.col(field_name).list.len())).alias(field_name)
            
            if self.cfg.shuffle:
                df = df.sample(n=df.shape[0], shuffle=True)
                logging.debug("Shuffling data within chunk.")
            yield df

    @staticmethod 
    def _to_padded_batch(df: pl.DataFrame[pl.List, ..., pl.List]) -> Tuple[list[list], dict, dict]:
        """
        Flatten and pad input data to ensure consistent dimensions.
        """
        max_length = df['PulsesTime'].list.len().max()
        logging.debug("Max length determined for padding: %d", max_length)

        result = []
        for row in df.iter_rows():
            row_result = [df_list + [torch.nan] * (max_length - len(df_list)) for df_list in row]
            result.append(row_result)
        index2name = {i: name for i, name in enumerate(df.columns)}
        name2index = {name: i for i, name in enumerate(df.columns)}
        logging.debug("Data flattened and padded. Returning as list.")
        return result, index2name, name2index

    def get_batches(self, bs: Optional[int] = None) -> Generator[Tuple[Tensor, Tensor], None, None]:
        """
        Generate batches of data for training.
        """
        
        # TODO: for effective usage of numpy.random make norming and aug here.
        # For clearity add self.featname_pytorchindex_map = dict(...)
        
        if bs is None:
            bs = self.batch_size
        
        chunks = self.get_chunks()
        batch_index = 0
        for chunk in chunks:
            for start in range(0, chunk.shape[0], bs):
                stop = start + bs
                pre_batch = chunk[start:stop]
                if pre_batch.shape[0] == 0:
                    logging.warning("Empty batch encountered. Skipping.")
                    break
                
                data, index2name, name2index = self._to_padded_batch(pre_batch[self.cfg.features])
                data = torch.tensor(data)
                
                if self.cfg.do_augment:
                    stds = [0.] * data.shape[2]
                    for name, std in self.aug_params.to_dict().items():
                        stds[name2index[name]] = std
                    stds = torch.tensor(stds).reshape(1,1,len(stds)).repeat(data.shape[0], data.shape[1], 1)
                    data = data + torch.normal(torch.zeros(data.shape), stds)
                    
                    # Reorder by augmented time
                    time_index = name2index["PulsesTime"]
                    sort_idxs = data[:,:,time_index].argsort(dim=1).repeat(1,1,data.shape[2])
                    data = data.gather(dim=1, index=sort_idxs)
                    
                if self.cfg.do_norm:
                    means, stds = [None]*data.shape[2], [None]*data.shape[2]
                    for i in range(data.shape[2]):
                        mean, std = self.norm_params.to_dict()[index2name[i]]
                        means.append(mean), stds.append(std)
                    
                    
                
                labels = pre_batch[self.cfg.labels].to_torch()
                logging.debug("Batch %d generated. Data shape: %s, Labels shape: %s", batch_index, data.shape, labels.shape)
                batch_index += 1
                yield data, labels

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.debug("test")
    logging.info("Starting the training data generator...")
    path_mu = '/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/'
    path_nuatm = '/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nuatm/root/all/'
    path_nu2 = '/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nue2_100pev/root/all/'

    def explore_paths(p: str, start: int, stop: int):
        files = os.listdir(p)[start:stop]
        return sorted([f"{p}{file}" for file in files])

    mu_paths = explore_paths(path_mu, 0, 2)
    nuatm_paths = explore_paths(path_nuatm, 0, 1)
    nu2_paths = explore_paths(path_nu2, 0, 1)

    gen_sets = GeneratorConfig(mu_paths, nuatm_paths, nu2_paths)
    gen_sets.chunk_size = 2
    train_data = TrainGenerator(gen_sets, collect_stats=False)
    batches = train_data.get_batches(256)

    for i, (data, labels) in enumerate(batches):
        logging.info(f"Processed batch #{i}")