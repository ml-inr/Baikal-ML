import os
import logging
from random import shuffle
from typing import Optional, Generator

import numpy as np
import polars as pl
import torch
from torch import Tensor

try:
    from data.root_manager.processor import Processor
    from data.root_manager.settings import ChunkGeneratorConfig
except ImportError:
    from root_manager.processor import Processor
    from root_manager.settings import ChunkGeneratorConfig
    
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChunkGenerator:
    
    def __init__(self, cfg: ChunkGeneratorConfig):
        self.cfg = cfg
        self.proc_params = cfg.processor_params
        
        # Initialize file paths
        self.mu_paths = cfg.mu_paths
        self.nuatm_paths = cfg.nuatm_paths
        self.nu2_paths = cfg.nu2_paths
        self.all_paths = self.mu_paths + self.nuatm_paths + self.nu2_paths
        
        if self.cfg.shuffle_paths:
            shuffle(self.all_paths)
            logging.info("Shuffled all file paths. Total paths: %d", len(self.all_paths))
        self.chunk_size = cfg.chunk_size
        
    def get_chunks(self) -> Generator[pl.DataFrame, None, None]:
        """
        Yield data chunks based on the provided chunk size.
        """
        for start in range(0, len(self.all_paths), self.chunk_size):
            stop = start + self.chunk_size
            logging.debug("Processing chunk from index %d to %d.", start, stop)
            df = Processor(self.all_paths[start:stop], self.proc_params).process()
            df = df[self.cfg.fields]
            yield df