from dataclasses import dataclass
from typing import Dict

@dataclass
class GeneralConfig:
    coords_are_same: bool
    take_single_cluster: bool
    take_clust_num: int
    split_multi: bool
    shift_coords_to_cl_center: bool
    center_times: bool
    exclude_big_ts: bool
    t_threshold: float

@dataclass
class InputConfig:
    particle: str
    MC_dir_path: str

@dataclass
class RootPathsConfig:
    data: list
    primary: list
    resp_muons: list
    mu_scalar: list
    geometry: str

@dataclass
class Config:
    general: GeneralConfig
    input: InputConfig
    root_paths: RootPathsConfig

    @staticmethod
    def from_dict(config_dict: Dict) -> 'Config':
        return Config(
            general=GeneralConfig(**config_dict['general']),
            input=InputConfig(**config_dict['input']),
            root_paths=RootPathsConfig(**config_dict['root_paths'])
        )
