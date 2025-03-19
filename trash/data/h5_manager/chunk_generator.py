import os
import logging
from random import shuffle
from typing import Generator

import h5py as h5
import polars as pl

try:
    from data.h5_manager.processor import Processor
    from data.h5_manager.settings import ChunkGeneratorConfig
except ImportError:
    from h5_manager.processor import Processor
    from h5_manager.settings import ChunkGeneratorConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ChunkGenerator:

    def __init__(self, hf: h5.File, root_paths: list[str], cfg: ChunkGeneratorConfig = ChunkGeneratorConfig()):
        self.hf = hf
        self.cfg = cfg
        self.proc_params = cfg.processor_params

        # Initialize file paths
        self.root_paths = root_paths

        if self.cfg.shuffle_paths:
            shuffle(self.root_paths)
            logging.info("Shuffled all file paths. Total paths: %d", len(self.root_paths))
        self.chunk_size = cfg.chunk_size

    def get_chunks(self) -> Generator[pl.DataFrame, None, None]:
        """
        Yield data chunks as a DataFrame, containing self.chunk_size processed h5 subdirs, corresponding to names of root files.
        """
        for start in range(0, len(self.root_paths), self.chunk_size):
            stop = start + self.chunk_size
            logging.debug("Processing chunk from index %d to %d.", start, stop)
            df = Processor(self.hf, self.root_paths[start:stop], self.proc_params).process()
            df = df[self.cfg.fields]
            yield df