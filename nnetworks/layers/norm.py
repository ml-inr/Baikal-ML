from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .config import MaskedConv1DConfig, ResBlockConfig
from ..utils.activation_factory import get_activation


class MaskedBatchNorm1D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MaskedBatchNorm1D, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tensor:
        # x shape: (batch_size, num_features, sequence_length)
        # mask shape: (batch_size, 1, sequence_length), 1 for valid positions, 0 for masked
        
        # Calculate the effective mean and variance by ignoring masked values
        masked_x = x * mask  # Zero out the masked positions
        
        # Count valid elements in each sequence position
        valid_count = mask.sum(dim=2, keepdim=True) + self.eps  # Avoid division by zero
                
        # Compute mean and variance across valid positions only
        mean = masked_x.sum((0,2)) / valid_count.sum(0) # mean over length&batch
        mean = mean.view(1, -1, 1) # reshape
        
        variance = ((masked_x - mean) ** 2 * mask).sum((0,2)) / (valid_count.sum(0)-1) # mean over length&batch (with correction)
        variance = variance.view(1, -1, 1) # reshape
        
        if self.track_running_stats:
            # Update running statistics
            exponential_avg_factor = 1.0 - self.momentum
            self.running_mean = exponential_avg_factor * mean + (1 - exponential_avg_factor) * self.running_mean
            self.running_var = exponential_avg_factor * variance + (1 - exponential_avg_factor) * self.running_var
            self.num_batches_tracked += 1

        # Normalize the input
        if self.training or not self.track_running_stats:
            x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        else:
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Apply scale (weight) and shift (bias) if affine
        if self.affine:
            x_normalized = x_normalized * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        
        return x_normalized
