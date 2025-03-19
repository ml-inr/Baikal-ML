import typing as tp
from random import shuffle
import logging
import polars as pl

try:
    from data.root_extractor.main import RootFileReader
    from data.root_extractor.internal_root_paths import MCRootPaths, ExpRootPaths, ExpRecoRootPaths
    from data.settings_scheme import ProcessorConfig
    from data.processor.data_utils import process_data
except ImportError:
    from .root_extractor.main import RootFileReader
    from .root_extractor.internal_root_paths import MCRootPaths, ExpRootPaths, ExpRecoRootPaths
    from .settings_scheme import ProcessorConfig
    from .processor.data_utils import process_data


class ChunksFromPaths:
    """
    Data loader for processing ROOT files. 
    Provides an iterator-based interface to read, preprocess, and return data in chunks.

    Attributes:
        root_paths: List of file paths to ROOT files.
        is_mc_data: Indicates whether the data is Monte Carlo (MC).
        processor_cfg: Configuration for data processing.
        shuffle_paths: Determines whether file paths are shuffled.
        events_per_chunk: Number of events per chunk in the output.
        lookforward: Number of events to prefetch during processing.
        prefix: a string to add at the beginning of the events ids (e.g. 'muatm'). If None, prefixes are extracted from root_paths strings.
    """

    def __init__(
        self,
        root_paths: list[str],
        is_mc_data: bool = True,
        exp_with_reco: bool = False,
        events_per_chunk = 1000,
        lookforward: int = 5000,
        processor_cfg: dict = ProcessorConfig().to_dict(),
        shuffle_paths: bool = False,
        prefix: tp.Optional[str] = None
    ):
        # Initialize file paths
        self.root_paths = root_paths.copy()
        if shuffle_paths:
            shuffle(self.root_paths)
            logging.debug(
                "Shuffled all file paths. Total paths: %d", len(self.root_paths)
            )
        self.is_mc_data = is_mc_data
        self.processor_cfg = processor_cfg
        self.events_per_chunk = events_per_chunk
        self.lookforward = lookforward
        self.prefix = prefix
        # Check if enough events are loaded to form a chunk

        # Initialize paths based on data type (MC or experimental)
        if self.is_mc_data:
            if 'MC_2019' in root_paths[0].split("/")[:-1]:
                self.internal_root_paths = MCRootPaths(coords_header="Events")
                #assert self.internal_root_paths.coords_header=="Events"
            else:
                self.internal_root_paths = MCRootPaths()
        else:
            if exp_with_reco:
                self.internal_root_paths = ExpRecoRootPaths()
            else:
                self.internal_root_paths = ExpRootPaths()
    
        # Validate processor configuration
        if self.processor_cfg is not None:
            assert (self.is_mc_data) or (
                (not self.processor_cfg["has_signal_flg"])
                and (not self.processor_cfg["to_calculate_tres"])
            ), "Unable to process processor data with the given configuration."

        # State tracking for iteration
        self.reset()

    def reset(self):
        """
        Resets the loader's state to its initial configuration.
        This allows reusing the same loader instance for a fresh iteration.
        """
        self._current_path_idx = 0
        self._current_start_idx = 0
        self._current_num_events_loaded = 0
        self.cache_data = []
        logging.info("Loader state has been reset. Ready for a fresh iteration.")

    def __iter__(self):
        # Reset state for iteration
        self.reset()
        return self

    def __next__(
        self,
    ) -> tp.Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, int]]:
        # Iterate over paths and process data
        while self._current_path_idx < len(self.root_paths):
            # Check if enough events are loaded to form a chunk
            if self._current_num_events_loaded >= self.events_per_chunk:
                # Unpack cached data and concatenate into DataFrames
                pulses, events, muons = zip(*self.cache_data)
                pulses = pl.concat(pulses)
                events = pl.concat(events)
                muons = pl.concat(muons) if self.is_mc_data else None
                if self._current_num_events_loaded == self.events_per_chunk:
                    # Reset the counter for the next batch
                    self._current_num_events_loaded = 0
                    self.cache_data = []
                    return pulses, events, muons
                else:
                    # Split events into "to_return" and "to_cache" based on chunk size
                    events_to_return = events[:self.events_per_chunk]
                    events_to_cache = events[self.events_per_chunk:]
                    
                    ids_to_return, ids_to_cache = events_to_return[["ev_id", "cluster_id"]], events_to_cache[["ev_id", "cluster_id"]]

                    # Helper function to split data into "to_return" and "to_cache"
                    def split_data(data, clusters, cache_clusters, join_cols):
                        if data is None:
                            return None, None
                        return (
                            data.join(clusters, on=join_cols, how="inner", maintain_order='left'),  # Data to return
                            data.join(cache_clusters, on=join_cols, how="inner", maintain_order='left'),  # Data to cache
                        )
                    # Split pulses, events, and muons
                    pulses_to_return, pulses_to_cache = split_data(
                        pulses, ids_to_return, ids_to_cache, ["ev_id", "cluster_id"]
                    )
                    if muons is not None:
                        muons_to_return, muons_to_cache = split_data(
                            muons, ids_to_return[["ev_id"]].unique(), ids_to_cache[["ev_id"]].unique(), ["ev_id"]
                        )
                    else:
                        muons_to_return, muons_to_cache = None, None

                    # Cache remaining data and update event count
                    self.cache_data = [(pulses_to_cache, events_to_cache, muons_to_cache)]
                    self._current_num_events_loaded = ids_to_cache.height

                    # Return the processed chunk
                    return pulses_to_return, events_to_return, muons_to_return
            
            path = self.root_paths[self._current_path_idx]
            logging.debug("Processing file: %s", path)
            with RootFileReader(
                path,
                internal_root_paths=self.internal_root_paths,
                prefix=self._extract_prefix(path) if self.prefix is None else self.prefix,
            ) as rr:
                events_in_file = rr.ev_num
                logging.debug(
                    "File contains %d events. Current start index: %d", 
                    events_in_file, 
                    self._current_start_idx
                )
                # Determine the range of events to process
                if self._current_start_idx + self.lookforward >= events_in_file:
                    start, stop = self._current_start_idx, events_in_file
                    self._current_start_idx = 0
                    self._current_path_idx += 1
                else:
                    start, stop = (
                        self._current_start_idx,
                        self._current_start_idx + self.lookforward,
                    )
                    self._current_start_idx += self.lookforward
                logging.info("Reading events from %d to %d in file: %s", start, stop, path)
                # Read data from the ROOT file
                pulses = rr.read_pulses_as_df(start, stop).drop_nans()
                events = rr.read_events_as_df(start, stop).drop_nans()
                coords = rr.read_OM_coords(start, stop).drop_nans()
                if self.is_mc_data:
                    muons = rr.read_muons_as_df(start, stop).drop_nans()
                else:
                    muons = None

            # Process data
            pulses, events, muons = process_data(
                pulses, events, coords, muons, **self.processor_cfg
            )
            logging.debug(
                "Processed data shapes: pulses=%s, events=%s, muons=%s",
                pulses.shape, 
                events.shape, 
                muons.shape if muons is not None else "N/A"
            )
            # Count unique events and clusters
            self._current_num_events_loaded += (
                events[["ev_id", "cluster_id"]]
                .height
            )
            # Cache processed data
            self.cache_data.append((pulses, events, muons))
        
        # Returning final chunk, that < than events_per_chunk
        while len(self.cache_data)>0:
            logging.debug("Some evetns are still in cache: returning them")
            # Unpack cached data and concatenate into DataFrames
            pulses, events, muons = zip(*self.cache_data)
            pulses = pl.concat(pulses)
            events = pl.concat(events)
            muons = pl.concat(muons) if self.is_mc_data else None
            self.cache_data = []
            return pulses, events, muons
        
        logging.info("All files have been processed. Raising StopIteration.")
        raise StopIteration

    def _extract_prefix(self, path):
        """
        Extracts a prefix based on the file path and data type.
        """
        file_name = path.split("/")[-1].split(".")[0]
        if self.is_mc_data:
            # Extracts the particle type from the path
            for name in path.split("/")[::-1]:
                if name in ["mu", "muatm", "nu", "nuatm", "nue2", "nue2_100pev", "nu2", "nu2_100pev"]:
                    particle_type = name
                    break
            else:
                raise ValueError(f"Invalid particle type: {particle_type}")
            prefix = f"{particle_type}_{file_name}_"
        else:
            prefix = f"exp_{file_name}_"
        logging.debug("Extracted prefix: %s", prefix)
        return prefix