import torch
import torch.nn as nn

from .config import ConvLayerConfig

def get_activation(activation_name: str) -> nn.Module:
    """Return the activation function based on the name."""
    activations = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(0.01),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh()
    }
    return activations.get(activation_name, nn.ReLU())

class MaskedConv1D(nn.Module):
    def __init__(self, config: ConvLayerConfig):
        super(MaskedConv1D, self).__init__()
        self.conv = nn.Conv1d(config.in_channels, config.out_channels, config.kernel_size, config.stride, config.padding)
        self.activation = get_activation(config.activation)
        self.use_dropout = config.use_dropout
        self.dropout = nn.Dropout(config.dropout_rate) if self.use_dropout and config.dropout_rate else None
        self.batch_norm = nn.BatchNorm1d(config.out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(x)
        masked_input = x.clone()
        masked_input[~mask] = 0  # Replace NaNs with zeros

        convolved = self.conv(masked_input)
        convolved = self.activation(convolved)
        convolved = self.batch_norm(convolved)

        if self.use_dropout and self.dropout:
            convolved = self.dropout(convolved)

        mask = mask.any(dim=1, keepdim=True).float()
        convolved *= mask
        return convolved
