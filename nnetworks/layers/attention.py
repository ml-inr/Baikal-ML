import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from .config import TransformerEncodersBlockConfig
from ..utils.cfg_fields_factory import get_activation


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input to retain sequence order.
    
    Args:
        d_model (int): Model dimension.
        max_len (int): Maximum sequence length.
    """
    def __init__(self, d_model, max_len=10_000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Calculate the positional encodings once in advance
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        seq_len = x.size(1)
        # Add positional encoding to the input
        return x + self.pe[None, :seq_len, :]


class TransformerEncodersBlock(nn.Module):
    def __init__(self, config: TransformerEncodersBlockConfig = TransformerEncodersBlockConfig()):
        """Configures Sequence of TransformerEncoders for input x with mask.

        Forward:
            Args:
                x (Tensor): input tensor of shape (B, Seq, Embedd)
                mask (Tensor): mask tensor of shape (B, Seq, 1)
            Returns:
                tuple[Tensor]: encoded x and original mask 
        """
        super(TransformerEncodersBlock, self).__init__()
        self.cfg = config
        assert self.cfg.d_model % self.cfg.nhead == 0
        
        # Embedding
        self.new_features_embedder = nn.Linear(self.cfg.in_features, self.cfg.d_model-self.cfg.in_features, bias=False)
        
        # create block or similar encoders
        torch_enc_layer_kwargs = {k: v for k,v in self.cfg.to_dict().items() if k not in ["encoders_number", "in_features", "activation"]}
        torch_enc_layer_kwargs['activation'] = get_activation(self.cfg.activation)
        torch_enc_layer_kwargs['batch_first'] = True
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(**torch_enc_layer_kwargs) for _ in range(self.cfg.encoders_number)])
       
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(self.cfg.d_model)
        
        # Classification token with specific dimensionality
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.cfg.d_model), requires_grad=True
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Applies TransformerEncoderLayer to input x with mask.

        Args:
            x (Tensor): input tensor of shape (B, Seq, Embedd)
            mask (Tensor): mask tensor of shape (B, Seq, 1)

        Returns:
            tuple[Tensor]: encoded x and original mask 
        """
        # Embed data
        new_features = self.new_features_embedder(x)
        x = torch.cat((x, new_features), dim=-1)
        # Add classification token
        batched_tokens = self.cls_token.repeat(x.shape[0], 1, 1) # Expand token to fit batch size
        x = torch.cat((batched_tokens, x), dim=1)
        cls_mask = torch.ones((x.shape[0], 1, 1), device=x.device) # mask also changes
        mask = torch.cat([cls_mask, mask], dim=1)
        # Introduce positional encodings
        x = self.positional_encoding(x)
        # Convert mask to attention mask format: (N*num_heads,Seq,Seq).
        mask_att = ~((mask * mask.transpose(-1, -2)).to(bool))
        mask_att = torch.cat([mask_att]*self.cfg.nhead, dim=0)
        # Apply layers
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, src_mask=mask_att)
        return x*mask, mask