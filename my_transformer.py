from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class TransformerEncodersBlockConfig(BaseConfig):
    in_features: int = 5
    encoders_number: int = 3
    d_model: int = 512
    nhead: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.
    layer_norm_eps: float = 0.00001
    activation: Dict[str, Any] = field(default_factory = lambda: {'relu': None})