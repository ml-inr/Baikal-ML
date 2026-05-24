from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for domain adaptation."""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    """Wrapper module for gradient reversal."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


# Backward compatibility alias
GradientReversalLayer = GradientReversalFunction
