from typing import Optional

import torch
import torch.nn as nn

from .config import MuNuSepLstmConfig
from ..layers.recurrent import LstmLayer
from ..layers.dense import DenseBlock
from ..layers.pooling import GlobalAveragePooling1DMasked, GlobalMaxPooling1DMasked


class MuNuSepLstm(nn.Module):
    def __init__(self, config: MuNuSepLstmConfig = MuNuSepLstmConfig()):
        super(MuNuSepLstm, self).__init__()
        
        # Configuration
        self.config = config
        self._check_compatability()

        # Initialize LstmBlocks
        self.lstm_layers = nn.ModuleList([LstmLayer(lstm_block_config) for lstm_block_config in self.config.lstm_layers])
        
        # Initialize pooling layer
        if self.config.pooling_type is not None:
            if self.config.pooling_type == "Average":
                self.pooling = GlobalAveragePooling1DMasked()
            elif self.config.pooling_type == "Max":
                self.pooling = GlobalMaxPooling1DMasked()
            else:
                raise ValueError(f"Unknown pooling type: {self.config.pooling_type}")

        # Initialize dense layers
        self.dense_layers = nn.ModuleList([DenseBlock(dense_config) for dense_config in self.config.dense_layers])

    def _check_compatability(self):
        if not self.config.lstm_layers[-1].return_sequences and self.config.pooling_type is not None:
            raise ValueError("Can't provide pooling on tensor of shape ~ (BatchSize, NumFeatures). Not a sequence!")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Apply LSTMs with mask
        for lstm in self.lstm_layers:
            x, mask = lstm(x, mask)

        # Apply pooling if configured
        if self.config.pooling_type is not None:
            x = self.pooling(x, mask)

        # Apply dense layers sequentially
        for dense_block in self.dense_layers:
            x = dense_block(x)

        return x