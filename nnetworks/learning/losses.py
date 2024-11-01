import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Focal Loss for binary or multi-class classification.

        :param alpha: Weighting factor for class imbalance.
        :param gamma: Focusing parameter to down-weight easy examples.
        :param reduction: Specifies the reduction to apply to the output ('none', 'mean', or 'sum').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """_summary_

        Args:
            inputs (Tensor): one-hot tensors of shape (N, 2)
            targets (Tensor): one-hot tensors of shape (N, 2)

        Returns:
            Tensor: focal loss for binary classification
        """

        # Compute the cross entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate the modulating factor
        pt = torch.exp(-BCE_loss)  # pt = exp(-BCE) is the prediction probability
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Apply reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss