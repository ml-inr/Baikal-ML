from dataclasses import dataclass, field, fields, asdict
from typing import List

try:
    from data.root_manager.chunk_generator import ChunkGenerator
    from data.root_manager.settings import ChunkGeneratorConfig
except:
    from root_manager.chunk_generator import ChunkGenerator
    from root_manager.settings import ChunkGeneratorConfig


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
    """Parameters to normilize input data"""

    # times and Amplitudes: [mean, std]. Can differ much from dataset to dataset.
    PulsesTime: tuple[float] = (0.0, 238.5)
    PulsesAmpl: tuple[float] = (6.8, 118.7)

    # geometry: [mean, std]. Made it fixed.
    Xrel: tuple[float] = (0.0, 60.0)
    Yrel: tuple[float] = (0.0, 60.0)
    Zrel: tuple[float] = (0.0, 260.0)


@dataclass
class AugmentParams(BaseConfig):
    """
    Parameters to augment input data
    """

    # times and Amplitudes: noise std
    PulsesTime: float = 5.0  # ns
    PulsesAmpl: float = 0.1  # ev

    # geometry: noise std in m
    Xrel: float = 2.0
    Yrel: float = 2.0
    Zrel: float = 5.0


@dataclass
class BatchGeneratorConfig(BaseConfig):
    """Configuration of data generator"""

    chunk_generator_cfg: ChunkGeneratorConfig = ChunkGeneratorConfig()

    batch_size: int = 256
    features_name: List[str] = field(default_factory=lambda: ["PulsesAmpl", "PulsesTime", "Xrel", "Yrel", "Zrel"])
    labels_name: List[str] = field(default_factory=lambda: ["nu_induced"])  # for neutrino selection by default

    do_norm: bool = True
    norm_params: NormParams = NormParams()

    do_augment: bool = True
    augment_parmas: AugmentParams = AugmentParams()

    shuffle: bool = True  # if shuffle data inside chunk