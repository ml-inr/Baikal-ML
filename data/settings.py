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
    """Parameters to filter data from root files
    """
    only_signal: bool = True  # Whether to filter only signal hits
    min_hits: int = 5  # Minimum number of hits per cluster to be kept
    min_strings: int = 2  # Minimum number of unique single strings in a cluster
    min_Q: float = 0  # Minimum pulse amplitude threshold
    t_threshold: float = 1e5  # Maximum time threshold for pulse filtering

# Settings for processing, including options for filtering
@dataclass
class ProcessorConfig(BaseConfig):
    """Configuration of root data processor
    """
    center_times: bool = True  # Whether to center the event times
    calc_tres: bool = False
    filter_cfg: FilterParams = FilterParams()  # Configuration for filtering
    # TODO: add split_multi option to data_processor.py. Now it is True by default.
    #z split_multi: bool = True  # Whether to split multi-cluster events


@dataclass
class Paths2Root(BaseConfig):
    """Root files to use in generating data
    """
    paths_to_muatm: List[str]
    paths_to_nuatm: List[str]
    paths_to_nue2: List[str]

@dataclass
class NormParams(BaseConfig):
    """Parameters to normilize input data
    """
    # times and Amplitudes: [mean, std]. Can differ much from dataset to dataset.
    PulsesTime: List[float] = field(default_factory=lambda: [0., 238.5])
    PulsesAmpl: List[float] = field(default_factory=lambda: [6.8, 118.7])
    
    # geometry: [mean, std]. Made it fixed.
    Xrel: List[float] = field(default_factory=lambda: [0., 60.])
    Yrel: List[float] = field(default_factory=lambda: [0., 60.])
    Zrel: List[float] = field(default_factory=lambda: [0., 260.])
    
@dataclass
class AugmentParams(BaseConfig):
    """
    Parameters to augment input data
    """
    # times and Amplitudes: noise std
    PulsesTime: float = 5. # ns
    PulsesAmpl: float = 0.1 # ev
    
    # geometry: noise std in m
    Xrel: float = 2.
    Yrel: float = 2.
    Zrel: float = 5.
    
    
@dataclass
class GeneratorConfig(BaseConfig):
    """Configuration of data generator
    """
    mu_paths: Paths2Root
    nuatm_paths: Paths2Root
    nu2_paths: Paths2Root
    
    chunk_size: int = 10
    batch_size: int = 256
    
    processor_params: ProcessorConfig = ProcessorConfig()
    
    features: List[str] = field(default_factory=lambda: ['PulsesAmpl', 'PulsesTime', 'Xrel', 'Yrel', 'Zrel'])
    labels: List[str] = field(default_factory=lambda: ['nu_induced']) # for neutrino selection by default
    do_norm: bool = True
    norm_params: NormParams = NormParams()
    do_augment: bool = True
    augment_parmas: AugmentParams = AugmentParams()
    
    shuffle: bool = True # if shuffle data (both root files order and events in batches)
    

if __name__=="__main__":
    # example of some standard paths for training
    # should be set individually for each experiment
    import os
    
    path_mu = f'/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/muatm/root/all/'
    path_nuatm = f'/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nuatm/root/all/'
    path_nu2 = f'/net/62/home3/ivkhar/Baikal/data/initial_data/MC_2020/nue2_100pev/root/all/'

    def explore_paths(p: str, start: int, stop: int):
        files = os.listdir(f'{p}')[start:stop]
        return sorted([f"{p}{file}" for file in files])
    
    # best trainig dataset for standard settings
    mu_paths = explore_paths(path_mu, 0, 800)
    nuatm_paths = explore_paths(path_nuatm, 0, 1000)
    nu2_paths = explore_paths(path_nu2, 0, 60)
    
    train_sets = GeneratorConfig(
        mu_paths,
        nuatm_paths,
        nu2_paths
    )