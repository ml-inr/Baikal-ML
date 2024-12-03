import torch
from torch import nn, Tensor

from .config import TransformerClassifierConfig
from ..layers.attention import TransformerEncodersBlock, PositionalEncoding
from ..layers.dense import DenseBlock
from ..layers.pooling import GlobalAveragePooling1DMasked, GlobalMaxPooling1DMasked


class TransformerClassifier(nn.Module):
    """
    Transformer encoder that stacks multiple encoder layers and applies positional encoding.
    
    Args:
        config (dict): Configuration dictionary with keys:
            - 'num_layers': Number of encoder layers.
            - 'num_heads': Number of attention heads.
            - 'd_model': Model dimension.
            - 'd_ff': Dimension of feedforward network.
            - 'dropout_rate': Dropout rate.
            - 'l2_reg': L2 regularization factor.
            - 'encoder_act_function': Activation function name.
    """
    def __init__(self, config: TransformerClassifierConfig = TransformerClassifierConfig()):
        super().__init__()
        self.config = config
        
        # Initialize ResBlocks
        self.encoder_block = TransformerEncodersBlock(self.config.encoder_config)
        
        # Initialize pooling layer
        if self.config.pooling_type == "Average":
            self.pooling = GlobalAveragePooling1DMasked()
        elif self.config.pooling_type == "Max":
            self.pooling = GlobalMaxPooling1DMasked()
        else:
            raise ValueError(f"Unknown pooling type: {self.config.pooling_type}")
        
        # Initialize dense layers
        self.dense_layers = nn.ModuleList([DenseBlock(dense_config) for dense_config in self.config.dense_layers])

    def forward(self, x, mask):
        # Apply ResBlocks with mask
        x, mask = self.encoder_block(x, mask)
        
        # pooling without cls_token
        cls_token = x[:,0,:]  
        pool = self.pooling(x[:,1:,:], mask[:,1:,:])
        x = torch.cat([cls_token,pool], axis=-1)

        # Apply dense layers sequentially
        for dense_block in self.dense_layers:
            x = dense_block(x)
            
        return x
