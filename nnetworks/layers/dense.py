from torch import Tensor
import torch.nn as nn

from .config import DenseInput
from .norm import MaskedBatchNorm1d
from ..utils.cfg_fields_factory import get_activation


class DenseBlock(nn.Module):
    def __init__(self, dense_block_input: DenseInput = DenseInput()):
        super(DenseBlock, self).__init__()
        self.input_hp = dense_block_input
        self.dropout_layer = nn.Dropout(self.input_hp.dropout)
        self.dense_layer = nn.Linear(in_features=self.input_hp.in_features, out_features=self.input_hp.units)
        self.activation = get_activation(self.input_hp.activation)
        if self.input_hp.do_norm:
            self.norm1d = nn.BatchNorm1d(num_features=self.input_hp.units)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout_layer(x)
        x = self.dense_layer(x)
        x = self.activation(x)
        if self.input_hp.do_norm:
            # fixing bug that batchnorm doesn't work in train with batch of size 1
            if x.shape[0]>1:
                x = self.norm1d(x)
        return x
