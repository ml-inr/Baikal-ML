from dataclasses import dataclass, field
from typing import Any, Dict

from nnetworks.configurations.base import BaseConfig

@dataclass
class MaskedConv1DConfig(BaseConfig):
    in_channels: int = 5
    out_channels: int = 5
    kernel_size: int = 3
    strides: int = 1
    activation: Dict[str, Any] = field(default_factory = lambda: {'LeakyReLU': None})  # Allow detailed configuration
    dropout: float = 0.2
    do_norm: bool = True

@dataclass
class RnnInput(BaseConfig):
    units: int = 32
    return_sequences: bool = False
    activation: str = 'tanh'
    recurrent_activation: str = 'sigmoid'
    dropout: float = 0.1
    recurrent_dropout: float = 0
    merge_mode: str = 'mul'

@dataclass
class ResBlockConfig(BaseConfig):
    id: MaskedConv1DConfig = MaskedConv1DConfig()
    cd: MaskedConv1DConfig = MaskedConv1DConfig()
    skip: MaskedConv1DConfig = MaskedConv1DConfig()

@dataclass
class DenseInput(BaseConfig):
    in_features: int = 5
    units: int = 2
    activation: Dict[str, Any] = field(default_factory = lambda: {'Softmax': None})  # Allow detailed configuration
    dropout: float = 0.2
    do_norm: bool = True
