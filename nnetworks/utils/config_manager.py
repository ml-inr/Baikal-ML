from typing import Type

import yaml
from dacite import from_dict
import torch

from ..configurations.base import BaseConfig
from ..models import config

def yaml2modelcfg(path_to_yaml: str) -> BaseConfig:
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
    cfg = yaml2modelcfg(path_to_yaml)
    return model_class(cfg)