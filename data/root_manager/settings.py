from dataclasses import dataclass, field, fields, asdict
from typing import List


@dataclass
class BaseConfig:
    def __init__(self):
        pass

    def get_fileds(self):
        return list(fields(self))

    def to_dict(self):
        return asdict(self)


# Parameters for filtering the data, allowing flexible configurations
@dataclass
class FilterParams(BaseConfig):
    """Parameters to filter data from root files"""

    only_signal: bool = True  # Whether to filter only signal hits
    min_hits: int = 5  # Minimum number of hits per cluster to be kept
    min_strings: int = 2  # Minimum number of unique single strings in a cluster
    min_Q: float = 0  # Minimum pulse amplitude threshold
    t_threshold: float = 1e5  # Maximum time threshold for pulse filtering


# Settings for processing, including options for filtering
@dataclass
class ProcessorConfig(BaseConfig):
    """Configuration of root data processor"""

    center_times: bool = True  # Whether to center the event times
    calc_tres: bool = False
    filter_cfg: FilterParams = FilterParams()  # Configuration for filtering
    # TODO: add split_multi option to data_processor.py. Now it is True by default.
    # z split_multi: bool = True  # Whether to split multi-cluster events


@dataclass
class Paths2Root(BaseConfig):
    """Root files to use in generating data"""

    paths_to_muatm: List[str]
    paths_to_nuatm: List[str]
    paths_to_nue2: List[str]


@dataclass
class ChunkGeneratorConfig(BaseConfig):
    """Configuration of data generator"""

    mu_paths: Paths2Root
    nuatm_paths: Paths2Root
    nu2_paths: Paths2Root

    chunk_size: int = 10

    processor_params: ProcessorConfig = ProcessorConfig()

    fields: List[str] = field(
        default_factory=lambda: [
            "PulsesAmpl",
            "PulsesTime",
            "Xrel",
            "Yrel",
            "Zrel",
            "nu_induced",
        ]
    )
    shuffle_paths: bool = True  # if shuffle data (both root files order and events in batches)


if __name__ == "__main__":
    # example of some standard paths for training
    # should be set individually for each experiment
    import os

    path_mu = f"/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/"
    path_nuatm = f"/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nuatm/root/all/"
    path_nu2 = f"/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nue2_100pev/root/all/"

    def explore_paths(p: str, start: int, stop: int):
        files = os.listdir(f"{p}")[start:stop]
        return sorted([f"{p}{file}" for file in files])

    # best trainig dataset for standard settings
    mu_paths = explore_paths(path_mu, 0, 800)
    nuatm_paths = explore_paths(path_nuatm, 0, 1000)
    nu2_paths = explore_paths(path_nu2, 0, 60)

    chunks_cfg = ChunkGeneratorConfig(mu_paths, nuatm_paths, nu2_paths)
