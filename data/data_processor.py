from dataclasses import dataclass
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import uproot as ur
from data.root_manager.root_reader import RootReader
from typing import Tuple, List


# Parameters for filtering the data, allowing flexible configurations
@dataclass
class FilterParams:
    only_signal: bool = False  # Whether to filter only signal events
    min_hits: int = 5          # Minimum number of hits per cluster to be kept
    min_strings: int = 2       # Minimum number of unique strings in a cluster
    min_Q: float = 0           # Minimum pulse amplitude threshold
    t_threshold: float = 1e5   # Maximum time threshold for pulse filtering


# Settings for processing, including options for augmentation and filtering
@dataclass
class ProcessorSettings:
    center_times: bool = True   # Whether to center the event times
    split_multi: bool = True    # Whether to split multi-cluster events
    filter_cfg: FilterParams = FilterParams()  # Configuration for filtering
    add_augment: bool = False   # Whether to add augmented data


class Processor:
    def __init__(self, paths_to_roots: List[str], config: ProcessorSettings = ProcessorSettings()):
        """
        Initializes the Processor class.
        Args:
            paths_to_roots (List[str]): List of paths to root files.
            config (ProcessorSettings): Settings for processing.
        """
        self.cfg = config
        self.paths = paths_to_roots
        self.prefixes = self._extract_prefixes(paths_to_roots)
        self.OM_coords = self.read_OM_coords(paths_to_roots[0]).set_index("PulsesChID")
        self.dfs = None  # Placeholder for loaded DataFrames

    @staticmethod
    def _extract_prefixes(paths: List[str]) -> List[str]:
        """
        Extracts prefixes based on file paths.
        Args:
            paths (List[str]): List of paths to root files.
        Returns:
            List[str]: List of extracted prefixes.
        """
        prefixes = []
        for path in paths:
            particle_type = path.split("/")[-4]
            assert particle_type in ["mu", "muatm", "nu", "nuatm", "nue2", "nue2_100pev", "nu2", "nu2_100pev"]
            file_name = path.split("/")[-1].split(".")[0]
            prefixes.append(f"{particle_type}_{file_name}_")
        return prefixes

    @staticmethod
    def read_OM_coords(path: str) -> pd.DataFrame:
        """
        Reads OM coordinates from a root file.
        Args:
            path (str): Path to the root file.
        Returns:
            pd.DataFrame: OM coordinates as a DataFrame.
        """
        with ur.open(path) as rf:
            rr = RootReader(rf)
            OM_coords = rr.read_OM_coords()
        return OM_coords

    @staticmethod
    def read_file(path: str, prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Reads data from a root file and returns relevant DataFrames.
        Args:
            path (str): Path to the root file.
            prefix (str): Prefix for the file.
        Returns:
            Tuple[pd.DataFrame]: Tuple of DataFrames for events, pulses, muons, and OM coordinates.
        """
        with ur.open(path) as rf:
            rr = RootReader(rf, prefix=prefix)
            results = (
                rr.read_events_as_df().set_index("ev_id"),
                rr.read_pulses_as_df().set_index(["ev_id", "cluster_id"]),
                rr.read_ind_mu_as_df().set_index(["ev_id", "mu_local_id"]),
                rr.read_OM_coords().set_index("PulsesChID"),
            )
        return results

    def load_chunk_as_dfs(self, chunk_of_paths: List[str], chunk_of_prefixes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads a chunk of root files and returns concatenated DataFrames.
        Args:
            chunk_of_paths (List[str]): List of file paths.
            chunk_of_prefixes (List[str]): Corresponding prefixes.
        Returns:
            Tuple[pd.DataFrame]: Concatenated DataFrames for events, pulses, and muons.
        """
        results = Parallel(n_jobs=-1)(delayed(self.read_file)(path, prefix) for path, prefix in zip(chunk_of_paths, chunk_of_prefixes))
        events, pulses, muons, new_OM_coords = zip(*results)
        for df in new_OM_coords:
            assert self.OM_coords.equals(df)  # Ensure OM coordinates match across files
        events, pulses, muons = pd.concat(events), pd.concat(pulses), pd.concat(muons)
        return events, pulses, muons

    def enrich_and_filter(
        self, 
        events_df: pd.DataFrame, pulses_df: pd.DataFrame, muons_df: pd.DataFrame, 
        filter_settings: FilterParams = FilterParams()
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Filters and enriches the data based on filtering settings.
        Args:
            events_df (pd.DataFrame): DataFrame for events.
            pulses_df (pd.DataFrame): DataFrame for pulses.
            muons_df (pd.DataFrame): DataFrame for muons.
            filter_settings (FilterParams): Settings for filtering.
        Returns:
            Tuple[pd.DataFrame]: Filtered DataFrames for events, pulses, and muons.
        """
        # Filter pulses based on time and amplitude thresholds
        pulses_df = pulses_df[
            (pulses_df["PulsesTime"] <= filter_settings.t_threshold) &
            (pulses_df["PulsesAmpl"] >= filter_settings.min_Q)
        ]
        if filter_settings.only_signal:
            pulses_df = pulses_df[pulses_df["is_signal"]]
        
        # Aggregate data by cluster for filtering
        signal_hits_per_cluster = pulses_df["is_signal"].groupby(["ev_id", "cluster_id"]).sum().rename("signal_hits_num").astype(np.int16)
        string_nums = pulses_df[pulses_df["is_signal"]]["string_id"].groupby(["ev_id", "cluster_id"]).nunique().rename("unique_signal_strings_num").astype(np.float16)
        
        # Join signal hit counts and string counts with the events DataFrame
        events_enrich = events_df.join(signal_hits_per_cluster, how="left").join(string_nums, how="left")
        
        # Apply the filtering criteria
        events_filter = events_enrich[
            (events_enrich["signal_hits_num"] >= filter_settings.min_hits) &
            (events_enrich["unique_signal_strings_num"] >= filter_settings.min_strings)
        ]
        
        # Drop groups of pulses that don't meet the filtering criteria
        bad_idxs = signal_hits_per_cluster.index[signal_hits_per_cluster < filter_settings.min_hits].union(
            string_nums.index[string_nums < filter_settings.min_strings]
        )
        pulses_filter = pulses_df.drop(bad_idxs, errors='ignore')

        # Merge OM coordinates to each pulse after filtering
        pulses_filter = pulses_filter.reset_index().merge(
            self.OM_coords[["Xrel", "Yrel", "Zrel"]],
            on="PulsesChID"
        ).set_index(["ev_id", "cluster_id"])

        # Filter muons based on the filtered events
        muons_filter = muons_df.loc[events_filter.reset_index()["ev_id"]]
        
        return events_filter, pulses_filter, muons_filter

    def save_chunk_to_h5(self, chunk: Tuple[pd.DataFrame], path_to_h5: str, mode: str = "a") -> None:
        """
        Saves a chunk of processed data to an HDF5 file.
        Args:
            chunk (Tuple[pd.DataFrame]): Tuple of DataFrames (events, pulses, muons).
            path_to_h5 (str): Path to the HDF5 file.
            mode (str): Write mode, default is "append".
        """
        events, pulses, muons = chunk
        with pd.HDFStore(path_to_h5, mode=mode) as store:
            store.put('events', events)
            store.put('pulses', pulses)
            store.put('muons', muons)

    def store_all_to_h5(self, path_to_h5: str) -> None:
        """
        Stores all loaded data to an HDF5 file.
        Args:
            path_to_h5 (str): Path to the HDF5 file.
        """
        pass  # Placeholder for storing all data

    def generate_batch(self) -> Tuple[np.ndarray]:
        """
        Generates a batch of data for model training.
        Returns:
            Tuple[np.ndarray]: Tuple of NumPy arrays containing the features and labels.
        """
        pass  # Placeholder for batch generation
