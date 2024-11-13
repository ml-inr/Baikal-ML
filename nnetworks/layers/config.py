from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    from ..base_config import BaseConfig
except ImportError:
    from nnetworks.base_config import BaseConfig

@dataclass
class MaskedConv1DConfig(BaseConfig):
    in_channels: int = 5
    out_channels: int = 5
    kernel_size: int = 3
    strides: int = 1
    activation: Dict[str, Any] = field(default_factory = lambda: {'LeakyReLU': None})  # Allow detailed configuration
    dropout: float = 0.2
    norm: Optional[Dict[str, Optional[Dict]]] = field(default_factory = lambda: {'MaskedBatchNorm1D': dict(eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, unbiased=True)})

@dataclass
class LstmConfig(BaseConfig):
    input_size: int = 5
    hidden_size: int = 32
    num_layers: int = 1
    return_sequences: bool = False
    dropout: float = 0.1
    bidirectional: bool = True
    return_sequences: bool = True
    norm: Optional[Dict[str, Optional[Dict]]] = field(default_factory = lambda: {'MaskedLayerNorm12': dict(eps=1e-5, affine=True, unbiased=True)})

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
