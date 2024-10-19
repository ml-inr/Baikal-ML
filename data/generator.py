from random import shuffle
import typing as tp

import polars as pl
import torch

from data.root_manager.processor import Processor
from data.settings import GeneratorConfig

class TrainGenerator:
    def __init__(self, cfg: GeneratorConfig, collect_stats=False):
        self.cfg = cfg
        self.norm_params = cfg.norm_params
        self.aug_params = cfg.augment_parmas
        self.proc_params = cfg.processor_params
        
        self.mu_paths = cfg.mu_paths
        self.nuatm_paths = cfg.nuatm_paths
        self.nu2_paths = cfg.nu2_paths
        
        self.all_paths = self.mu_paths + self.nuatm_paths + self.nu2_paths
        if cfg.shuffle:
            shuffle(self.all_paths)
        
        self.df_mu_chunk, self.df_nuatm_chunk, self.df_nu2_chunk = None, None, None
        
        if collect_stats:
            self._get_stats()
        
        self.chunk_size = cfg.chunk_size
        self.batch_size = cfg.batch_size
        
        # # Is it right way? No
        # self.mu_in_batch = round(self.batch_size * self.mu_nu_ratio/(1+self.mu_nu_ratio))
        # self.nuatm_in_batch = (self.batch_size - self.mu_in_batch) * self.nuatm_nu2_ratio/(1+self.nuatm_nu2_ratio)
        # self.nu2_in_batch = self.batch_size - self.mu_in_batch - self.nuatm_in_batch

    
    @staticmethod
    def _calc_sums(df: pl.DataFrame, field_name: str, num_files_local: int, num_files_total: int):
        num_hits = df[field_name].explode().shape[0]/num_files_local*num_files_total
        S = df[field_name].explode().sum()/num_files_local*num_files_total
        S2 = df[field_name].explode().pow(2).sum()/num_files_local*num_files_total
        return num_hits, S, S2
    
    def _estimate_mean_and_std(self, df_mu: pl.DataFrame, df_nuatm: pl.DataFrame, df_nu2: pl.DataFrame, field_name: str, num_files_local: int):
        mu_hits, S_mu, S2_mu = self._calc_sums(df_mu, field_name, num_files_local, len(self.mu_paths))
        nuatm_hits, S_nuatm, S2_nuatm = self._calc_sums(df_nuatm, field_name, num_files_local, len(self.nuatm_paths))
        nu2_hits, S_nu2, S2_nu2 = self._calc_sums(df_nu2, field_name, num_files_local, len(self.nu2_paths))
        
        mean = (S_mu+S_nuatm+S_nu2)/(mu_hits+nuatm_hits+nu2_hits)
        std = ((S2_mu+S2_nuatm+S2_nu2)/(mu_hits+nuatm_hits+nu2_hits) - mean**2)**0.5
        
        return mean, std
    
    def _get_stats(self, start: int = 0, stop: int = 50) -> None:
        # checking
        assert stop>start
        if stop-start > min([len(self.mu_paths), len(self.nuatm_paths), len(self.nu2_paths)]):
            print("Range is too big for the files. Set start=0 and stop=min_num_of_files.")
            start = 0
            stop = min([len(self.mu_paths), len(self.nuatm_paths), len(self.nu2_paths)])
            
        # load
        df_mu = (mu_proc:=Processor(self.mu_paths[start:stop], self.proc_params)).process()
        df_nuatm = (nuatm_proc:=Processor(self.nuatm_paths[start:stop], self.proc_params)).process()
        df_nu2 = (nu2_proc:=Processor(self.nu2_paths[start:stop], self.proc_params)).process()
        
        # number of events
        self.mu_num_estimated = df_mu.shape[0]/(stop-start)*len(self.mu_paths)
        self.nuatm_num_estimated = df_nuatm.shape[0]/(stop-start)*len(self.nuatm_paths)
        self.nu2_num_estimated = df_nu2.shape[0]/(stop-start)*len(self.nu2_paths)
        
        # koef of filtering
        self.mu_filter_koef = mu_proc.filter_koef
        self.nuatm_filter_koef = nuatm_proc.filter_koef
        self.nu2_filter_koef = nu2_proc.filter_koef
        
        # mu:nuatm:nu2 ratios
        self.mu_nu_ratio = self.mu_num_estimated / (self.nu2_num_estimated + self.nuatm_num_estimated)
        self.nuatm_nu2_ratio = self.nuatm_num_estimated / self.nu2_num_estimated
        
        # sum and sum^2 of Q and t
        self.Q_mean, self.Q_std = self._estimate_mean_and_std(df_mu, df_nuatm, df_nu2, field_name='PulsesAmpl', num_files_local=stop-start)
        self.t_mean, self.t_std = self._estimate_mean_and_std(df_mu, df_nuatm, df_nu2, field_name='PulsesTime', num_files_local=stop-start)
    
    def generate_chunk(self, chunk_size: tp.Optional[int] = None):
        if chunk_size is None:
            chunk_size = self.chunk_size
        for start in range(0, len(self.all_paths), chunk_size):
            stop = start + chunk_size
            # if start == len(self.all_paths):
            #     break
            #     #raise StopIteration
            print(f"        {start=}, {stop=}")
            df = Processor(self.all_paths[start:stop], self.proc_params).process()
            df = df[self.cfg.features + self.cfg.labels]
            if shuffle:
                df = df.sample(n=df.shape[0], shuffle=True)
            yield df
    
    @staticmethod 
    def _flatten_and_pad(df: pl.DataFrame) -> torch.Tensor:
        # Step 1: Find the maximum length of the df_feat lists
        max_length = df['PulsesTime'].list.len().max()
        max_length

        # Step 2: Create a list of lists to hold the padded df, avoiding manual iteration
        result = []

        # Step 3: Iterate through rows using Polars methods
        for row in df.iter_rows():
            row_result = []
            for df_list in row:
                # Pad each df_list to the maximum length
                padded_df_list = df_list + [0.0] * (max_length - len(df_list))
                row_result.append(padded_df_list)
            result.append(row_result)

        # Convert the result to a Polars DataFrame
        return torch.tensor(result)
     
    def load_batch(self, batch_size: tp.Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size
        
        chunks = self.generate_chunk()
        chunk = next(chunks)
        chunk_stop_iter = False
        
        while not chunk_stop_iter:
            pre_batch = chunk[0:batch_size]
            chunk = chunk[batch_size:]
            if chunk.shape[0]==0:
                try:
                    chunk = next(chunks)
                except StopIteration as e:
                    chunk_stop_iter = True
            
            data = self._flatten_and_pad(pre_batch[self.cfg.features])
            labels = pre_batch[self.cfg.labels].to_torch()
            yield data, labels
            
    def load_batch_v2(self, bs: tp.Optional[int] = None):
        if bs is None:
            bs = self.bs
        
        chunks = self.generate_chunk()
        
        for ch_num, chunk in enumerate(chunks):
            print(f"#{ch_num} chunk shape: {chunk.shape}")
            for start in range(0,chunk.shape[0],bs):
                stop = start + bs
                pre_batch = chunk[start:stop]
                if pre_batch.shape[0]==0:
                    break
                print(f"    #{start=} for batch in {ch_num=}. {pre_batch.shape=}")
                data = self._flatten_and_pad(pre_batch[self.cfg.features])
                labels = pre_batch[self.cfg.labels].to_torch()
                yield data, labels
        
        
        
        
        # if self.cfg.do_norm:
        #         for field_name in self.norm_params.get_fileds():
        #             mean = self.norm_params.to_dict()[field_name][0]
        #             std = self.norm_params.to_dict()[field_name][1]
        #             data = data.with_columns( ((pl.col(field_name)-mean)/std).alias(field_name) )
            
        # if self.cfg.do_augment:
        #     for field_name in self.aug_params.get_fileds():
        #         mean = self.aug_params.to_dict()[field_name][0]
        #         std = self.aug_params.to_dict()[field_name][1]
        #         data = data.with_columns( (pl.col(field_name) +  ).alias(field_name) )
        
        # data, labels = df[self.cfg.features], df[self.cfg.labels]
        
        pass
    
