import torch
import torch.nn as nn
from typing import List, Optional

try:
    from config import ResNetConfig, ConvLayerConfig, MLPConfig
    from layers import MaskedConv1D, get_activation
except ImportError:
    try:
        from models.config import ResNetConfig, ConvLayerConfig, MLPConfig
        from models.layers import MaskedConv1D, get_activation
    except ImportError:
        from nnetwork_gpt.models.config import ResNetConfig, ConvLayerConfig, MLPConfig
        from nnetwork_gpt.models.layers import MaskedConv1D, get_activation

class ResNet1D(nn.Module):
    def __init__(self, config: ResNetConfig):
        super(ResNet1D, self).__init__()
        self.conv_layers = self._build_conv_layers(config.conv_layers)
        self.mlp_layers = self._build_mlp_layers(config.mlp_layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            *self.mlp_layers,
            nn.Linear(config.mlp_layers[-1].hidden_size, config.num_classes),
            nn.Softmax(dim=1)
        )
    
    def _build_conv_layers(self, layer_configs: List[ConvLayerConfig]) -> nn.Sequential:
        layers = []
        for cfg in layer_configs:
            layers.append(MaskedConv1D(cfg))
        return nn.Sequential(*layers)
    
    def _build_mlp_layers(self, mlp_configs: List[MLPConfig]) -> nn.Sequential:
        layers = []
        for cfg in mlp_configs:
            layers.append(nn.Linear(cfg.hidden_size, cfg.hidden_size))
            layers.append(get_activation(cfg.activation))
            if cfg.use_dropout and cfg.dropout_rate:
                layers.append(nn.Dropout(cfg.dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc(x)
