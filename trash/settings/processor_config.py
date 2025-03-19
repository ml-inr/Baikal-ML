from dataclasses import dataclass

# Parameters for filtering the data, allowing flexible configurations
@dataclass
class FilterParams:
    only_signal: bool = True  # Whether to filter only signal hits
    min_hits: int = 5  # Minimum number of hits per cluster to be kept
    min_strings: int = 2  # Minimum number of unique single strings in a cluster
    min_Q: float = 0  # Minimum pulse amplitude threshold
    t_threshold: float = 1e5  # Maximum time threshold for pulse filtering

# Settings for processing, including options for filtering
@dataclass
class ProcessorConfig:
    center_times: bool = True  # Whether to center the event times
    calc_tres: bool = False
    filter_cfg: FilterParams = FilterParams()  # Configuration for filtering
    # TODO: add split_multi option to data_processor.py. Now it is True by default.
    #z split_multi: bool = True  # Whether to split multi-cluster events