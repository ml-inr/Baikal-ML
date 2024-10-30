from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from .config import RnnInput

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

    def forward(self, x: Tensor, seq_mask: Optional[Tensor] = None):
        if seq_mask is not None:
            seq_lengths = seq_mask.sum(dim=1).int()
            packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
            packed_output, (hn, cn) = self.lstm_layer(packed_input)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            x, (hn, cn) = self.lstm_layer(x)
        
        # Separate forward and backward outputs
        forward_output = x[:, :, :self.input_hp.units]
        backward_output = x[:, :, self.input_hp.units:]
        
        # Apply merge mode
        if self.merge_mode == 'concat':
            output = torch.cat((forward_output, backward_output), dim=-1)
        elif self.merge_mode == 'sum':
            output = forward_output + backward_output
        elif self.merge_mode == 'mul':
            output = forward_output * backward_output
        elif self.merge_mode == 'ave':
            output = (forward_output + backward_output) / 2
        else:
            raise ValueError(f"Unsupported merge mode: {self.merge_mode}")

        return output

