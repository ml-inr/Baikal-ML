from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .config import MaskedConv1DConfig, ResBlockConfig
from ..utils.cfg_fields_factory import get_activation, get_norm_layer


class MaskedConv1D(nn.Module):
    def __init__(self, config: MaskedConv1DConfig = MaskedConv1DConfig()):
        super(MaskedConv1D, self).__init__()
        self.cfg = config
        self.conv = nn.Conv1d(
            in_channels=self.cfg.in_channels,
            out_channels=self.cfg.out_channels,
            kernel_size=self.cfg.kernel_size,
            stride=self.cfg.strides
        )
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.activation = get_activation(self.cfg.activation)
        if self.cfg.norm is not None:
            self.norm_layer = get_norm_layer(self.cfg.out_channels, norm_kwargs=self.cfg.norm) #MaskedBatchNorm1D(self.cfg.out_channels)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): shape (B, L, num_features)
            mask (Tensor): shape (B, L, 1)

        Returns:
            x, mask: shapes (B, L, num_features) and (B, L, 1)
        """
        x, mask = x.permute((0,2,1)), mask.permute((0,2,1)) # torch convolutions take (B, Features, L) shape as input
        x_local = x * mask.float()

        # Calculate padding for 'same' behavior
        pad_left = (self.cfg.kernel_size - 1) // 2
        pad_right = (self.cfg.kernel_size - 1) // 2 + (self.cfg.kernel_size - 1) % 2

        # Apply padding
        x_local = F.pad(x_local, (pad_left, pad_right))

        # Apply convolution
        x_new = self.conv(x_local)

        x_new = self.dropout(x_new)
        x_new = self.activation(x_new)  # Apply configured activation
        
        # Recalculate the mask so 'True' values correspond to positions of hits as if there wasn't masking in conv.
        new_sizes = (mask.sum(dim=2).int() - 1) // self.cfg.strides + 1
        # Define max length for the mask
        max_len = new_sizes.max().item()
        # Create new mask using broadcasting
        new_mask = torch.arange(max_len, device=mask.device).unsqueeze(0).unsqueeze(0) < new_sizes.unsqueeze(-1)
        x_new = x_new[:,:,:max_len] * new_mask
        x_new, new_mask = x_new.permute((0,2,1)), new_mask.permute((0,2,1)) # restore initial shape
        
        #  Apply norm layer
        if self.cfg.norm is not None:
            if hasattr(self.norm_layer, "requires_mask"):
                x_new = self.norm_layer(x_new, new_mask)
            else:
                x_new = self.norm_layer(x_new)

        return x_new, new_mask


class ResBlock(nn.Module):
    def __init__(self, res_block_input: ResBlockConfig = ResBlockConfig()):
        super(ResBlock, self).__init__()
        self.input_hp = res_block_input
        self.conv_id = MaskedConv1D(self.input_hp.id)
        self.conv_cd = MaskedConv1D(self.input_hp.cd)
        self.conv_skip = MaskedConv1D(self.input_hp.skip)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        # Skip connection
        x_skip = x.clone()
        mask_skip = mask.clone()
        # Identical dimensions
        x, mask = self.conv_id(x, mask)
        # Change dimensions
        x, mask = self.conv_cd(x, mask)
        # Skip convolution
        x_skip, mask_skip = self.conv_skip(x_skip, mask_skip)
        # Concatenate
        # assuming correct convolution, we will always have number_of_signal_hits <= min(x_skip.shape[2], x.shape[2])
        length = min(x_skip.shape[1], x.shape[1])
        x = torch.cat((x[:, :length, :], x_skip[:, :length, :]), dim=2) 
        return x, mask
