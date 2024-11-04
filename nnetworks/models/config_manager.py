from typing import Type

import yaml
from dacite import from_dict
import torch

from ..base_config import BaseConfig
from . import config

def save_model_cfg(cfg: BaseConfig, path: str = "./cfg.yaml", mode: str = 'w') -> None:
    """Saves configuration to path as yaml file.

    Args:
        cfg (BaseConfig): _description_
        path (str): _description_
        mode (str, optional): _description_. Defaults to 'w'.
    """
    
    # Dumper for saving files in easy-to-read format
    class MyDumper(yaml.Dumper):
        def write_line_break(self, data=None):
            super().write_line_break(data)

            if len(self.indents) == 1:
                super().write_line_break()
    
    with open(path, mode) as f:
        yaml.dump(cfg.to_dict(), f, MyDumper, indent=4, width=1000, sort_keys=False)

def yaml2cfg(path_to_yaml: str) -> BaseConfig:
    """Handles loading nested model's configs from nested yaml dict.

    Args:
        path_to_yaml (str): path to .yaml file. Must contain the only 1st level key: name of dataclass to load in.
    Returns:
        BaseConfig: instance of configuration
    """
    with open(path_to_yaml, 'r') as file:
        config_dict = yaml.safe_load(file)
    cfg_class_name = list(config_dict.keys())[0]
    dataclass = getattr(config, cfg_class_name)
    return from_dict(dataclass, config_dict[cfg_class_name])

def model_from_yaml(model_class: Type[torch.nn.Module], path_to_yaml: str) -> torch.nn.Module:
    """Loads model from architecture written in yaml file.

    Args:
        model_class (Type[torch.nn.Module]): class of model to load in
        path_to_yaml (str): path to .yaml configuration file. Must contain the only 1st level key: name of dataclass to load in.

    Returns:
        torch.nn.Module: model with architecture, written in yaml file. Ready to train.
    """
    cfg = yaml2cfg(path_to_yaml)
    return model_class(cfg)