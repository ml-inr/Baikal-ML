from typing import Optional

import torch
import torch.nn as nn

from .config import MuNuSepLstmConfig
from ..layers.recurrent import LstmLayer
from ..layers.dense import DenseBlock


class MuNuSepLstm(nn.Module):
    def __init__(self, config: MuNuSepLstmConfig = MuNuSepLstmConfig()):
        super(MuNuSepLstm, self).__init__()
        
        # Configuration
        self.config = config

        # Initialize LstmBlocks
        self.lstm_layers = nn.ModuleList([LstmLayer(lstm_block_config) for lstm_block_config in self.config.lstm_layers])
        
        # Initialize dense layers
        self.dense_layers = nn.ModuleList([DenseBlock(dense_config) for dense_config in self.config.dense_layers])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Apply ResBlocks with mask
        for lstm in self.lstm_layers:
            x, mask = lstm(x, mask)

        # Apply dense layers sequentially
        for dense_block in self.dense_layers:
            x = dense_block(x)

        return x