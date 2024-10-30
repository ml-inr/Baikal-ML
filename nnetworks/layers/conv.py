from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .config import MaskedConv1DConfig, ResBlockConfig
from .norm import MaskedBatchNorm1D
from ..utils.activation_factory import get_activation


class MaskedConv1D(nn.Module):
    def __init__(self, conv_input: MaskedConv1DConfig = MaskedConv1DConfig()):
        super(MaskedConv1D, self).__init__()
        self.input_hp = conv_input
        self.conv = nn.Conv1d(
            in_channels=self.input_hp.in_channels,
            out_channels=self.input_hp.out_channels,
            kernel_size=self.input_hp.kernel_size,
            stride=self.input_hp.strides
        )
        self.dropout = nn.Dropout(self.input_hp.dropout)
        self.activation = get_activation(self.input_hp.activation)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x_local = x * mask.float()

        # Calculate padding for 'same' behavior
        pad_left = (self.input_hp.kernel_size - 1) // 2
        pad_right = (self.input_hp.kernel_size - 1) // 2 + (self.input_hp.kernel_size - 1) % 2

        # Apply padding
        x_local = F.pad(x_local, (pad_left, pad_right))

        # Apply convolution
        x_new = self.conv(x_local)

        x_new = self.dropout(x_new)
        x_new = self.activation(x_new)  # Apply configured activation
        
        # Recalculate the mask so 'True' values correspond to positions of hits as if there wasn't masking in conv.
        new_sizes = (mask.sum(dim=2).int() - 1) // self.input_hp.strides + 1
        # Define max length for the mask
        max_len = new_sizes.max().item()
        # Create new mask using broadcasting
        new_mask = torch.arange(max_len, device=mask.device).unsqueeze(0).unsqueeze(0) < new_sizes.unsqueeze(-1)
        x_new = x_new[:,:,:max_len] * new_mask
        return x_new, new_mask


class ResBlock(nn.Module):
    def __init__(self, res_block_input: ResBlockConfig = ResBlockConfig()):
        super(ResBlock, self).__init__()
        self.input_hp = res_block_input
        self.conv_id = MaskedConv1D(self.input_hp.id)
        self.conv_cd = MaskedConv1D(self.input_hp.cd)
        self.conv_skip = MaskedConv1D(self.input_hp.skip)
        if self.input_hp.do_norm:
            self.norm_id = MaskedBatchNorm1D(self.input_hp.id.out_channels)
            self.norm_cd = MaskedBatchNorm1D(self.input_hp.cd.out_channels)
            self.norm_skip = MaskedBatchNorm1D(self.input_hp.skip.out_channels)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        # Skip connection
        x_skip = x.clone()
        if mask is None:
            mask = ~x.isnan()[:,0:1,:]
        mask_skip = mask.clone()
        # Identical dimensions
        x, mask = self.conv_id(x, mask)
        if self.input_hp.do_norm:
            x = self.norm_id(x, mask)
        # Change dimensions
        x, mask = self.conv_cd(x, mask)
        if self.input_hp.do_norm:
            x = self.norm_cd(x, mask)
        # Skip convolution
        x_skip, mask_skip = self.conv_skip(x_skip, mask_skip)
        if self.input_hp.do_norm:
            x_skip = self.norm_skip(x_skip, mask_skip)
        # Concatenate
        # assuming correct convolution, we will always have number_of_signal_hits <= min(x_skip.shape[2], x.shape[2])
        length = min(x_skip.shape[2], x.shape[2])
        x = torch.cat((x[:, :, :length], x_skip[:, :, :length]), dim=1) 
        return x, mask
