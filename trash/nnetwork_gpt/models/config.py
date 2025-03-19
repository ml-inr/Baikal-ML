from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ConvLayerConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    activation: str = "ReLU"  # Choose from 'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', etc.
    use_dropout: bool = False
    dropout_rate: Optional[float] = None

@dataclass
class MLPConfig:
    hidden_size: int
    activation: str = "ReLU"
    use_dropout: bool = False
    dropout_rate: Optional[float] = None

@dataclass
class ResNetConfig:
    input_channels: int
    num_classes: int
    conv_layers: List[ConvLayerConfig]
    mlp_layers: List[MLPConfig]
    fc_hidden_size: int = 128
