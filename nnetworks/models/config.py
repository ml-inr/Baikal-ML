from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..layers.config import MaskedConv1DConfig, LstmConfig, ResBlockConfig, DenseInput, TransformerEncodersBlockConfig
from ..base_config import BaseConfig


@dataclass
class MuNuSepLstmConfig(BaseConfig):
    lstm_layers: list[LstmConfig] = field(default_factory = lambda: [LstmConfig()])
    pooling_type: Optional[str] = None # example: field(default_factory = lambda: {"Average"})
    dense_layers: list[DenseInput] = field(default_factory = lambda: [DenseInput()])
    

@dataclass
class MuNuSepResNetConfig(BaseConfig):
    res_blocks: list[ResBlockConfig] = field(default_factory = lambda: [ResBlockConfig()])
    pooling_type: str = "Average"
    dense_layers: list[DenseInput] = field(default_factory = lambda: [DenseInput()])
    
@dataclass
class TransformerClassifierConfig(BaseConfig):
    encoder_config: TransformerEncodersBlockConfig = TransformerEncodersBlockConfig()
    pooling_type: str = "Average"
    dense_layers: list[DenseInput] = field(default_factory = lambda: [DenseInput()])