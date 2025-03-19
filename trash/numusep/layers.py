import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any

SEED = torch.randint(1, 100, (1,)).item()
torch.manual_seed(SEED)

# Global Average Pooling with mask
class GlobalAveragePooling1DMasked(nn.Module):
    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.float()
            return (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            return x.mean(dim=1)


# Global Max Pooling with mask
class GlobalMaxPooling1DMasked(nn.Module):
    def forward(self, x, mask=None):
        if mask is not None:
            x = torch.where(mask.bool(), x, torch.tensor(-float('inf')))
            return x.max(dim=1).values
        else:
            return x.max(dim=1).values


@dataclass
class RnnInput:
    units: int = 32
    return_sequences: bool = False
    activation: str = 'tanh'
    recurrent_activation: str = 'sigmoid'
    dropout: float = 0.1
    recurrent_dropout: float = 0
    kernel_initializer: Any = None  # Placeholder for initialization
    recurrent_initializer: Any = None  # Placeholder for initialization
    merge_mode: str = 'mul'


# Bidirectional LSTM Layer
class BidirLayer(nn.Module):
    def __init__(self, rnn_input: RnnInput = RnnInput()):
        super(BidirLayer, self).__init__()
        self.input_hp = rnn_input
        self.lstm_layer = nn.LSTM(
            input_size=self.input_hp.units,
            hidden_size=self.input_hp.units,
            batch_first=True,
            dropout=self.input_hp.dropout,
            bidirectional=True,
        )
        self.merge_mode = self.input_hp.merge_mode

    def forward(self, x, seq_mask=None):
        if seq_mask is not None:
            # Assuming the mask is (batch_size, seq_len)
            seq_lengths = seq_mask.sum(dim=1).int()
            packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm_layer(packed_input)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            x, _ = self.lstm_layer(x)
        return x


@dataclass
class ConvInput:
    filters: int = 10
    kernel_size: int = 3
    strides: int = 1
    activation: Any = nn.LeakyReLU(0.3)
    dropout: float = 0.1
    kernel_initializer: Any = None  # Placeholder for initialization
    padding: str = "valid"


# 1D Convolution with mask
class MaskedConv1D(nn.Module):
    def __init__(self, conv_input: ConvInput = ConvInput()):
        super(MaskedConv1D, self).__init__()
        self.input_hp = conv_input
        self.conv = nn.Conv1d(
            in_channels=self.input_hp.filters,
            out_channels=self.input_hp.filters,
            kernel_size=self.input_hp.kernel_size,
            stride=self.input_hp.strides,
            padding=(self.input_hp.kernel_size - 1) // 2 if self.input_hp.padding == "same" else 0
        )
        self.dropout = nn.Dropout(self.input_hp.dropout)
        self.activation = self.input_hp.activation

    def forward(self, x, mask):
        x = x * mask.float()
        x = self.conv(x)
        x = self.dropout(x)
        return x, mask


@dataclass
class ResBlockInput:
    id: ConvInput
    cd: ConvInput
    skip: ConvInput


# Residual Block Layer
class ResBlock(nn.Module):
    def __init__(self, res_block_input: ResBlockInput):
        super(ResBlock, self).__init__()
        self.input_hp = res_block_input

        self.conv_id = MaskedConv1D(self.input_hp.id)
        self.conv_cd = MaskedConv1D(self.input_hp.cd)
        self.conv_skip = MaskedConv1D(self.input_hp.skip)

        self.norm_id = nn.BatchNorm1d(self.input_hp.id.filters)
        self.norm_cd = nn.BatchNorm1d(self.input_hp.cd.filters)
        self.norm_skip = nn.BatchNorm1d(self.input_hp.skip.filters)

    def forward(self, x, mask):
        # Skip connection
        x_skip = x
        mask_skip = mask

        # Identical dimensions
        x, mask = self.conv_id(x, mask)
        x = self.norm_id(x)

        # Change dimensions
        x, mask = self.conv_cd(x, mask)
        x = self.norm_cd(x)

        # Skip convolution
        x_skip, mask_skip = self.conv_skip(x_skip, mask_skip)
        x_skip = self.norm_skip(x_skip)

        # Concatenate
        length = min(x_skip.shape[2], x.shape[2])
        x = torch.cat((x[:, :, :length], x_skip[:, :, :length]), dim=1)
        return x, mask


@dataclass
class DenseInput:
    units: int = 2
    activation: Any = nn.LeakyReLU()
    dropout: float = 0.1


# Dense Block
class DenseBlock(nn.Module):
    def __init__(self, dense_block_input: DenseInput = DenseInput()):
        super(DenseBlock, self).__init__()
        self.input_hp = dense_block_input
        self.dropout_layer = nn.Dropout(self.input_hp.dropout)
        self.dense_layer = nn.Linear(in_features=self.input_hp.units, out_features=self.input_hp.units)

    def forward(self, x):
        x = self.dropout_layer(x)
        x = self.dense_layer(x)
        x = self.input_hp.activation(x)
        return x


# Example usage
if __name__ == "__main__":
    conv_input = ConvInput()
    masked_conv = MaskedConv1D(conv_input)
    print(masked_conv)
