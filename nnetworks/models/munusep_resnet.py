from typing import Optional

import torch
import torch.nn as nn

from .config import MuNuSepResNetConfig
from ..layers.conv import ResBlock
from ..layers.dense import DenseBlock
from ..layers.pooling import GlobalAveragePooling1DMasked, GlobalMaxPooling1DMasked


class MuNuSepResNet(nn.Module):
    def __init__(self, config: MuNuSepResNetConfig = MuNuSepResNetConfig()):
        super(MuNuSepResNet, self).__init__()
        
        # Configuration
        self.config = config

        # Initialize ResBlocks
        self.res_blocks = nn.ModuleList([ResBlock(res_block_config) for res_block_config in self.config.res_blocks])
        
        # Initialize pooling layer
        if self.config.pooling_type == "Average":
            self.pooling = GlobalAveragePooling1DMasked()
        elif self.config.pooling_type == "Max":
            self.pooling = GlobalMaxPooling1DMasked()
        else:
            raise ValueError(f"Unknown pooling type: {self.config.pooling_type}")
        
        # Initialize dense layers
        self.dense_layers = nn.ModuleList([DenseBlock(dense_config) for dense_config in self.config.dense_layers])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Apply ResBlocks with mask
        for res_block in self.res_blocks:
            x, mask = res_block(x, mask)

        # Apply pooling
        x = self.pooling(x, mask)

        # Apply dense layers sequentially
        for dense_block in self.dense_layers:
            x = dense_block(x)

        return x
