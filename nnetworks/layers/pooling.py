from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

class GlobalAveragePooling1DMasked(nn.Module):
    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        if mask is not None:
            mask = mask.float()
            return (x * mask).sum(dim=2) / mask.sum(dim=2)
        else:
            return x.mean(dim=2)

class GlobalMaxPooling1DMasked(nn.Module):
    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        if mask is not None:
            x = torch.where(mask.bool(), x, torch.tensor(-float('inf')))
            return x.max(dim=2).values
        else:
            return x.max(dim=2).values
