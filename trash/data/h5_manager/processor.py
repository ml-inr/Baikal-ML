from time import time

from dataclasses import dataclass
from typing import Tuple, List
import polars as pl
import numpy as np
from joblib import Parallel, delayed
import h5py as h5

try:
    from data.h5_manager.reader.h5_reader import H5Reader
    from data.h5_manager.constants import Constants as Cnst
    from data.h5_manager.settings import ProcessorConfig, FilterParams
except ImportError:
    from h5_manager.reader.h5_reader import H5Reader
    from h5_manager.constants import Constants as Cnst
    from h5_manager.settings import ProcessorConfig, FilterParams

# TODO: enable parallel
class Processor:
    def __init__(self, hf: h5.File, paths_to_roots: List[str], config: ProcessorConfig = ProcessorConfig()):
        """
        Initializes the Processor class, setting up paths and configuration for processing H5 files.

        Args:
            paths_to_roots (List[str]): List of paths to root files, that were used to build H5.
            config (ProcessorSettings): Settings for processing data (filtering and augmentation options).
        """
        self.cfg = config
        self.signatures = self._extract_signatures(paths_to_roots)
        self.filter_koef = "Not estimated untill processing"
        self.hf = hf

    @staticmethod
    def _extract_signatures(paths: List[str]) -> List[str]:
        """
        Extracts list[(ParticleType, FileNum)] based on file paths, used to differentiate between file types (mu, nu, etc.).

        Args:
            paths (List[str]): List of paths to root files.

        Returns:
            list[str, int]: List of extracted signatures for each file.
        """
        signatures = []
        for path in paths:
            # Extracts the particle type from the path
            particle_type = path.split("/")[-4]
            assert particle_type in [
                "mu",
                "muatm",
                "nu",
                "nuatm",
                "nue2",
                "nue2_100pev",
                "nu2",
                "nu2_100pev",
            ]
            if ('nue2' in particle_type) or ('nu2' in particle_type):
                particle_type = 'nue2'
            file_num = int(path.split("/")[-1].split(".")[0])
            signatures.append((particle_type, file_num))
        return signatures
    
    @staticmethod
    def read_cluster_centers(hf: h5.File, particle_type: str, file_num: int) -> pl.DataFrame:
        """
        Reads coordinates of clusters' centers from a HDf5 file using the H5Reader.

        Args:
            path (str): Path to the root file.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the clusters' centers coordinates.
        """
        rr = H5Reader(hf, particle_type, file_num)
        clusters_coords = rr.read_cluster_centers()
        return clusters_coords

    @staticmethod
    def read_file(hf: h5.File, particle_type: str, file_num: int, with_cluster_centers: bool = False) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Reads data from a ROOT file and returns DataFrames for events, pulses, muons, and OM coordinates.

        Args:
            path (str): Path to the root file.
            prefix (str): Prefix used to identify different particle types in the data.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]: DataFrames for events, pulses, muons, and OM coordinates.
        """
        rr = H5Reader(hf, particle_type, file_num)
        results = [
            rr.read_events_as_df(),
            rr.read_pulses_as_df(),
            rr.read_muons_as_df()
        ]
        if with_cluster_centers:
            results.append(rr.read_cluster_centers())
        return tuple(results)

    @staticmethod
    def process_data(filt_cfg: FilterParams, events_df: pl.DataFrame, df_pulses_flat: pl.DataFrame, df_muons_flat: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
        """
        Executes data pipeline: Extract, Transform, Filter.
        Enriches pulses with additional information and filters the data based on the filtering configuration.

        Returns:
            pl.DataFrame: filtered DataFrame.
        """

        # Enrich pulses with new columns: signal flags, cluster and string IDs
        df_pulses_flat = df_pulses_flat.with_columns(
            is_signal=(pl.col("PulsesFlg") != 0),
            cluster_id=(pl.col("PulsesChID") // 288).cast(pl.Int8),
            string_id=(pl.col("PulsesChID") // 36),
            mu_local_id=(pl.col("PulsesFlg") % 1_000_000 - 1).cast(pl.Int16),
        )

        # Apply filtering based on the configuration
        if filt_cfg.only_signal:
            df_pulses_flat = df_pulses_flat.filter(pl.col("is_signal"))
        if (minQ := filt_cfg.min_Q) > 0:
            df_pulses_flat = df_pulses_flat.filter(pl.col("PulsesAmpl") >= minQ)

        # Filter out pulses with bugged large times
        df_pulses_flat = df_pulses_flat.filter(pl.col("PulsesTime") <= filt_cfg.t_threshold)

        # Sort pulses by event ID, cluster ID, and pulse time
        df_pulses_flat = df_pulses_flat.sort(["ev_id", "cluster_id", "PulsesTime"])

        # Group by event ID and cluster ID, performing calculations on grouped data
        # Group pulses
        query = [
            pl.col(c_name) for c_name in df_pulses_flat.columns if c_name not in ["ev_id", "cluster_id", "string_id"]
        ] + [
            pl.col("is_signal").alias("num_signal_hits").sum().cast(pl.Int16),  # Calculate number of signal hits
            pl.col("string_id")
            .filter(pl.col("is_signal"))
            .n_unique()
            .cast(pl.Int8)
            .alias("num_signal_strings"),  # Calculate unique signal strings
        ]
        grouped_pulses = df_pulses_flat.group_by(["ev_id", "cluster_id"], maintain_order=True).agg(query)
        # Group muons
        query = [pl.col(c_name) for c_name in df_muons_flat.columns if c_name not in ["ev_id"]]
        grouped_muons = df_muons_flat.group_by(["ev_id"], maintain_order=True).agg(query)
        
        # Apply filtering based on hit and string counts
        grouped_pulses = grouped_pulses.filter(
            (pl.col("num_signal_hits") >= filt_cfg.min_hits) & (pl.col("num_signal_strings") >= filt_cfg.min_strings)
        )

        # Join filtered pulses and muons with events DataFrame
        final_df = events_df.join(grouped_pulses, on="ev_id", how="inner").join(
            grouped_muons, on="ev_id", how="inner", suffix="_for_mu"
        )

        final_df = final_df.with_columns(nu_induced=pl.col("ev_id").str.starts_with("nu"))
        final_df = final_df.with_columns(enough_info=(pl.col("num_signal_hits")>=5) & (pl.col("num_signal_strings")>=2))

        return final_df, {"NumEventsAfterFilter": final_df.shape[0], "NumEventsBeforeFilter": events_df.shape[0]}

    
    def _read_root2df(self) -> List[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]:
        """
        Loads a chunk of data, that corresponds to named root files, and returns concatenated DataFrames for events, pulses, and muons.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: Concatenated DataFrames for events, pulses, and muons.
        """
        # t0 = time()
        dfs = [self.read_file(self.hf, particle_type, file_num) for particle_type, file_num in self.signatures]
        # t1 = time()
        # print(f"Time for rading files as df: {t1-t0}")
        # events, pulses, muons = zip(*results)
        # t2 = time()
        # print(f"Time for zipping dfs: {t2-t1}")
        
        # The DataFrames
        return dfs

    def process(self) -> pl.DataFrame:
        """
        Executes data pipeline: Extract, Transform, Filter.
        Enriches pulses with additional information and filters the data based on the filtering configuration.

        Returns:
            pl.DataFrame: filtered DataFrame.
        """
        
        dfs = self._read_root2df()
        
        # t0 = time()
        processed_results = Parallel(n_jobs=5)(
            delayed(self.process_data)(self.cfg.filter_cfg, df_events, df_pulses, df_muons) for df_events, df_pulses, df_muons in dfs
        )
        # df_events, df_pulses, df_muons = zip(*dfs)
        # processed_results=self.process_data(self.cfg.filter_cfg, pl.concat(df_events), pl.concat(df_pulses), pl.concat(df_muons))
        # t1 = time()
        # print(f"Time for processing: {t1-t0}")
        processed_dfs, stats_dicts_to_merge = zip(*processed_results)

        self.filter_koef = sum(stats_to_merge["NumEventsAfterFilter"] for stats_to_merge in stats_dicts_to_merge) / \
                        sum(stats_to_merge["NumEventsBeforeFilter"] for stats_to_merge in stats_dicts_to_merge)
             
        t0 = time()
        final_df = pl.concat(processed_dfs)
        t1 = time()
        print(f"Time for concatenation: {t1-t0}")
        return final_df
        
        # dfs_to_merge, stats_to_merge = zip(*[
        #         self.process_data(
        #             self.cfg.filter_cfg, 
        #             *self.read_file(self.hf, particle_type, file_num)
        #         ) 
        #         for particle_type, file_num in self.signatures
        #     ])

        # self.filter_koef = sum(stats["NumEventsAfterFilter"] for stats in stats_to_merge) / \
        #                 sum(stats["NumEventsBeforeFilter"] for stats in stats_to_merge)

        # return pl.concat(dfs_to_merge)
