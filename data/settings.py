from dataclasses import dataclass, field, fields, asdict
from typing import List

try:
    from data.root_manager.chunk_generator import ChunkGenerator
except:
    from root_manager.chunk_generator import ChunkGenerator


@dataclass
class BaseConfig:
    def __init__(self):
        pass
    def get_fileds(self):
        return list(fields(self))
    def to_dict(self):
        return asdict(self)


@dataclass
class NormParams(BaseConfig):
    """Parameters to normilize input data
    """
    # times and Amplitudes: [mean, std]. Can differ much from dataset to dataset.
    PulsesTime: tuple[float] = (0., 238.5)
    PulsesAmpl: tuple[float] = (6.8, 118.7)
    
    # geometry: [mean, std]. Made it fixed.
    Xrel: tuple[float] = (0., 60.)
    Yrel: tuple[float] = (0., 60.)
    Zrel: tuple[float] = (0., 260.)
    
    
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
class BatchGeneratorConfig(BaseConfig):
    """Configuration of data generator
    """
    
    chunk_generator: ChunkGenerator
    
    batch_size: int = 256
    features_name: List[str] = field(default_factory=lambda: ['PulsesAmpl', 'PulsesTime', 'Xrel', 'Yrel', 'Zrel'])
    labels_name: List[str] = field(default_factory=lambda: ['nu_induced']) # for neutrino selection by default
    
    do_norm: bool = True
    norm_params: NormParams = NormParams()
    
    do_augment: bool = True
    augment_parmas: AugmentParams = AugmentParams()
    
    shuffle: bool = True # if shuffle data inside chunk
    

if __name__=="__main__":
    # example of some standard paths for training
    # should be set individually for each experiment
    import os
    from data.root_manager.settings import ChunkGeneratorConfig
    
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
    
    chunks_cfg = ChunkGeneratorConfig(
        mu_paths,
        nuatm_paths,
        nu2_paths
    )
    batches_cfg = BatchGeneratorConfig(chunk_generator=ChunkGenerator(chunks_cfg))