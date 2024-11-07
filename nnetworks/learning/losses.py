import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        
        Parameters:
        - alpha (float): Scaling factor for the focal loss.
        - gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
        - reduction (str): Specifies the reduction to apply to the output ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass for focal loss.
        
        Parameters:
        - inputs (torch.Tensor): Predictions of shape (batch_size, num_classes).
        - targets (torch.Tensor): One-hot encoded targets of shape (batch_size, num_classes).
        
        Returns:
        - torch.Tensor: Computed focal loss.
        """
        
        # Compute the focal loss components
        # Only consider non-zero target elements to calculate the focal loss
        targets = targets.type_as(inputs)  # Ensure targets and inputs have the same dtype
        pt = (inputs * targets).sum(dim=1)  # Extract the probabilities for the true class
        log_pt = pt.log()
        
        # Focal loss calculation
        focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
