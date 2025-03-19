from dataclasses import dataclass, field, fields, asdict
from typing import List

from data.settings.config import ProcessorConfig
    
@dataclass
class BaseConfig:
    def __init__(self):
        pass
    def get_fileds(self):
        return list(fields(self))
    def to_dict(self):
        return asdict(self)

@dataclass
class Paths2Root(BaseConfig):
    paths_to_muatm: List[str]
    paths_to_nuatm: List[str]
    paths_to_nue2: List[str]

@dataclass
class NormParams(BaseConfig):
    # times and Amplitudes: [mean, std]. Can differ much from dataset to dataset.
    T: List[float] = field(default_factory=lambda: [0., 240.])
    Q: List[float] = field(default_factory=lambda: [5., 40.])
    
    # geometry: [mean, std]. Made it fixed.
    Xrel: List[float] = field(default_factory=lambda: [0., 60.])
    Yrel: List[float] = field(default_factory=lambda: [0., 60.])
    Zrel: List[float] = field(default_factory=lambda: [0., 264.])
    
@dataclass
class AugmentParams(BaseConfig):
    """
    Parameters to augment input data
    """
    # times and Amplitudes: noise std
    T: float = 5. # ns
    Q: float = 0.1 # ev
    
    # geometry: noise std in m
    Xrel: float = 2.
    Yrel: float = 2.
    Zrel: float = 5.
    
    
@dataclass
class GeneratorConfig(BaseConfig):
    train_paths: Paths2Root
    test_paths: Paths2Root
    val_paths: Paths2Root
    
    processor_params: ProcessorConfig = ProcessorConfig()
    
    filter_koef_muatm: float = None
    filter_koef_nuatm: float = None
    filter_koef_nue2: float = None
    mu_nu_ratio_in_train: float = 1. # desiered ratio of mu and nu events in train when creating batch
    nuatm_nu2_ratio_in_train: float = 10. # desiered ratio of nuatm and nu2 events in train when creating batch
    
    features: List[str] = field(default_factory=lambda: ['PulsesAmpl', 'PulsesTime', 'Xrel', 'Yrel', 'Zrel'])
    labels: List[str] = field(default_factory=lambda: ['nu_induced']) # for neutrino selection by default
    do_norm: bool = True
    norm_params: NormParams = NormParams()
    do_augment: bool = True
    augment_parmas: AugmentParams = AugmentParams()
    
    shuffle: bool = True # if shuffle data (both root files order and events in batches)
    

    