from dataclasses import dataclass, field, fields, asdict
import typing as tp

@dataclass
class BaseConfig:
    def get_fileds(self):
        return list(fields(self))

    def to_dict(self):
        """Converts the dataclass into deep dict (nested dataclasses are converted into dicts too)
        Returns:
            dict: dict corresponding to dataclass structure.
        """
        return asdict(self)
    
    def to_shallow_dict(self) -> dict:
        """Converts the dataclass into shallow dict (nested dataclasses are not converted)
        Returns:
            dict: dict corresponding to dataclass structure at the first level.
        """
        shallow_dict = {}
        for field in fields(self):
            value = getattr(self, field.name)
            shallow_dict[field.name] = value
        return shallow_dict

# Settings for processing, including options for filtering
@dataclass
class ProcessorConfig(BaseConfig):
    """Configuration of data processor"""
    has_signal_flg: bool  = False
    only_signal: bool = False  # Whether to filter only signal hits. Doesn't work when `has_signal_flg` is False
    min_sig_hits: int = 0  # Minimum number of signal hits per cluster to be kept
    min_sig_strings: int = 0  # Minimum number of unique signal strings in a cluster
    min_Q: float = 0  # Minimum pulse amplitude threshold
    center_times: bool = True  # Whether to center the event times
    relative_coords: bool = True # Whether to add coordinates relatively to the clusters centers
    to_calculate_tres: bool = False # Whether to calculate tres
    t_threshold: float = 1e5  # Maximum time threshold for pulse filtering
    same_coordinates: bool = False # If coords are same for each event or not

@dataclass
class ChunksFromPathsConfig(BaseConfig):
    """Configuration of data generator. All the parameters besides root_paths are described."""
    is_mc_data: bool
    lookforward: int = 50_000
    processor_cfg: tp.Optional[ProcessorConfig] = field(default_factory=ProcessorConfig)
    shuffle_paths: bool = False
    prefix: tp.Optional[str] = None

@dataclass
class MCMuNuSepBatchGeneratorConfig(BaseConfig):
    """Configuration of data generator. All the parameters besides root_paths are described."""
    chunk_generator_cfg: ChunksFromPathsConfig
    mu_events_per_chunk: int = 256*100
    nu_events_per_chunk: int = 256*100
    batch_size: int = 256
    features_to_take: tp.List[str] = field(default_factory=lambda: [
        "PulsesTime",
        "PulsesAmpl",
        "Xrel",
        "Yrel",
        "Zrel",
    ])
    do_norm: bool = True
    norm_params: tp.List[tp.List[float]] = field(default_factory=lambda: [
        [0., 1374.0],
        [1.4, 23.1],
        [0.0, 60.0],
        [0.0, 60.0],
        [0.0, 260.0],
    ])
    do_augment: bool = True
    augment_params: tp.List[float] = field(default_factory=lambda: [
        5.0,  # ns
        0.1,  # ev
        2.0,  # m
        2.0,  # m
        5.0,  # m
    ])
    shuffle: bool = True  # if to shuffle data inside chunk

@dataclass
class ExpBatchGeneratorConfig(BaseConfig):
    """Configuration of data generator. All the parameters besides root_paths are described."""
    chunk_generator_cfg: ChunksFromPathsConfig
    events_per_chunk: int = 256*100
    batch_size: int = 256
    features_to_take: tp.List[str] = field(default_factory=lambda: [
        "PulsesTime",
        "PulsesAmpl",
        "Xrel",
        "Yrel",
        "Zrel",
    ])
    do_norm: bool = True
    norm_params: tp.List[tp.List[float]] = field(default_factory=lambda: [
        [0., 1374.0],
        [1.4, 23.1],
        [0.0, 60.0],
        [0.0, 60.0],
        [0.0, 260.0],
    ])
    do_augment: bool = False
    augment_params: tp.List[float] = field(default_factory=lambda: [
        5.0,  # ns
        0.1,  # ev
        2.0,  # m
        2.0,  # m
        5.0,  # m
    ])
    shuffle: bool = False  # if to shuffle data inside chunk
