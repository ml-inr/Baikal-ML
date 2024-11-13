from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

class GlobalAveragePooling1DMasked(nn.Module):
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """_summary_

        Args:
            x (Tensor): of shape (B, L, Features)
            mask (Optional[Tensor], optional): of shape (B, L, 1). Defaults to None.

        Returns:
            Tensor: of shape (B, Features)
        """
        if mask is not None:
            mask = mask.float()
            return (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            return x.mean(dim=1)

class GlobalMaxPooling1DMasked(nn.Module):
    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        """_summary_

        Args:
            x (Tensor): of shape (B, L, Features)
            mask (Optional[Tensor], optional): of shape (B, L, 1). Defaults to None.

        Returns:
            Tensor: of shape (B, Features)
        """
        if mask is not None:
            x = torch.where(mask.bool(), x, torch.tensor(-float('inf')))
            return x.max(dim=1).values
        else:
            return x.max(dim=1).values
