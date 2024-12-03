import os
import logging
from random import shuffle
from typing import Generator

import polars as pl

try:
    from data.root_manager.processor import Processor
    from data.root_manager.settings import ChunkGeneratorConfig
except ImportError:
    from root_manager.processor import Processor
    from root_manager.settings import ChunkGeneratorConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ChunkGenerator:

    def __init__(self, root_paths: list[str], cfg: ChunkGeneratorConfig = ChunkGeneratorConfig()):
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
        Yield data chunks as a DataFrame, containing self.chunk_size processed root files.
        """
        for start in range(0, len(self.root_paths), self.chunk_size):
            stop = start + self.chunk_size
            logging.debug("Processing chunk from index %d to %d.", start, stop)
            df = Processor(self.root_paths[start:stop], self.proc_params).process()
            df = df[self.cfg.fields]
            yield df


if __name__ == "__main__":
    try:
        from data.root_manager.chunk_generator import ChunkGenerator
        from data.root_manager.settings import ChunkGeneratorConfig
    except:
        from root_manager.chunk_generator import ChunkGenerator
        from root_manager.settings import ChunkGeneratorConfig

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
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

    chunks_cfg = ChunkGeneratorConfig()

    data = ChunkGenerator(root_paths=all_paths, cfg=chunks_cfg)
    chunks = data.get_chunks()

    for i, df in enumerate(chunks):
        logging.info(f"Processed chunk #{i}: {df.shape=}, {df.columns=}")