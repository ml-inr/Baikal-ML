from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

    
class MaskedLayerNorm12(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, unbiased=True):
        """Normalizes tensor across 1 and 2 dims. Takes mask into account. 
        Calculated variation with `unbiased` parameter (equivalent to torch.var(..., unbiased))

        Args:
            num_features (int): number of channels/features (dim=1 in this case)
            eps (float, optional): Defaults to 1e-5.
            momentum (float, optional):Defaults to 0.1.
            affine (bool, optional): Defaults to True.
            track_running_stats (bool, optional): Defaults to True.
        """
        super(MaskedLayerNorm12, self).__init__()
        self.eps = eps
        self.affine = affine
        self.unbiased = unbiased
        self.num_features = num_features

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Indicator of masked layer. Helps in configuring models.
        self.requires_mask = True

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tensor:
        """
        Input shape: (batch_size, sequence_length, num_features)
        Mask shape: (batch_size, sequence_length, 1)
        Output shape: (batch_size, num_features, sequence_length)
        """
        if self.affine:
            assert x.shape[2] == self.num_features
        
        # Count valid elements in each sequence position
        valid_count = mask.sum((1,2), keepdim=True)*self.num_features + self.eps  # Avoid division by zero. shape (batch_size, 1, 1)
                
        # Compute mean and variance across valid positions only
        mean = (x * mask).sum((1,2), keepdim=True) / valid_count # mean over length&features. shape (batch_size, 1, 1)
        variance = ((x - mean) ** 2 * mask).sum((1,2), keepdim=True) / (valid_count-(1 if self.unbiased else 0)) # var over length&features (with correction). shape (batch_size, 1, 1)

        # Normalize the input
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # Apply scale (weight) and shift (bias) if affine
        if self.affine:
            x_normalized = x_normalized * self.weight.expand_as(x_normalized) + self.bias.expand_as(x_normalized)
        
        return x_normalized


class MaskedBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, unbiased=True):
        """Normalizes tensor across 1 and 2 dims. Takes mask into account. 
        Calculated variation with `unbiased` parameter (equivalent to torch.var(..., unbiased))

        Args:
            num_features (int): number of channels/features (dim=1 in this case)
            eps (float, optional): Defaults to 1e-5.
            momentum (float, optional):Defaults to 0.1.
            affine (bool, optional): Defaults to True.
            track_running_stats (bool, optional): Defaults to True.
        """
        super(MaskedBatchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.unbiased = unbiased
        
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
        
        # Indicator of masked layer. Helps in configuring models.
        self.requires_mask = True

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tensor:
        """
        Input shape: (batch_size, sequence_length, num_features)
        Mask shape: (batch_size, sequence_length, 1)
        Output shape: (batch_size, sequence_length, num_features)
        """
        
        # Count valid elements in each sequence position
        valid_count = mask.sum((0,1), keepdim=True) + self.eps  # Avoid division by zero. shape (1, num_features, 1)
                
        # Compute mean and variance across valid positions only
        mean = (x * mask).sum((0,1), keepdim=True) / valid_count # mean over length&batch. shape (1, num_features)
        variance = ((x - mean) ** 2 * mask).sum((0,1), keepdim=True) / (valid_count-(1 if self.unbiased else 0)) # var over length&batch (with correction)
        
        if self.track_running_stats:
            # Update running statistics
            exponential_avg_factor = 1.0 - self.momentum
            self.running_mean = exponential_avg_factor * mean[0,0,:] + (1 - exponential_avg_factor) * self.running_mean
            self.running_var = exponential_avg_factor * variance[0,0,:] + (1 - exponential_avg_factor) * self.running_var
            self.num_batches_tracked += 1

        # Normalize the input
        if self.training or not self.track_running_stats:
            x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        else:
            x_normalized = (x - self.running_mean.expand_as(x)) / torch.sqrt(self.running_var.expand_as(x) + self.eps)

        # Apply scale (weight) and shift (bias) if affine
        if self.affine:
            x_normalized = x_normalized * self.weight.expand_as(x_normalized) + self.bias.expand_as(x_normalized)
        
        return x_normalized


if __name__ == "__main__":
    bnorm = MaskedBatchNorm1d(5)
    x = torch.ones((3,5,10))
    mask = torch.ones((3,1,10))
    
    print(bnorm(x,mask))