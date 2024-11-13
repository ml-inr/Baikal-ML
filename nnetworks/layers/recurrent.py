from typing import Optional

import torch
from torch import nn, Tensor

try:
    from .config import LstmConfig
    from ..utils.cfg_fields_factory import get_activation, get_norm_layer
except ImportError:
    from nnetworks.layers.config import LstmConfig
    from nnetworks.utils.cfg_fields_factory import get_activation, get_norm_layer


# Define the custom bidirectional LSTM layer in PyTorch
class LstmLayer(nn.Module):
    def __init__(self, config: LstmConfig = LstmConfig()):
        super(LstmLayer, self).__init__()
        self.cfg = config

        # Define the forward and backward LSTM layers separately for custom initialization
        self.lstm_layer = nn.LSTM(
            input_size=self.cfg.input_size,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            batch_first=True,
            dropout=self.cfg.dropout,
            bidirectional=self.cfg.bidirectional # merge mode is concat by torch realization
        )
        if self.cfg.norm is not None:
            self.norm_layer = get_norm_layer(num_features=self.cfg.hidden_size * (2 if self.cfg.bidirectional else 1), norm_kwargs=self.cfg.norm)
        self._check_cfg_compatability()
    
    def _check_cfg_compatability(self):
        if not self.cfg.return_sequences and hasattr(self.norm_layer, "requires_mask"):
            print("Can't provide masked norm for the tensor of shape ~ (BatchSize, Units). There is no sequence!")
            raise ValueError

    def forward(self, x: Tensor, seq_mask: Optional[Tensor] = None):
        """
        Input shape: (batch_index, time_steps, num_features)
        Output shape: 
            (batch_index, time_steps, D*hidden_size) if return_sequences
            (batch_index, D*hidden_size) if not return_sequences
        D = 2 if bidirectional else 1
        """
        # Assuming x has shape (batch_index, time_steps, num_features)
        if seq_mask is None:
            seq_mask = torch.ones((x.shape[0], x.shape[1], 1))
        
        # Handle masking
        seq_lengths = seq_mask.sum((1,2)).to(torch.int64).cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm_layer(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        seq_mask = seq_mask[:, :output.shape[1], :]

        # Return output and mask if provided, similar to TensorFlow
        # Implement return_sequences logic
        if self.cfg.return_sequences:
            if self.cfg.norm is not None:
                if hasattr(self.norm_layer, "requires_mask"):
                    output = self.norm_layer(output, seq_mask)
                else:
                    output = self.norm_layer(output)
        else:
            if self.cfg.bidirectional:
                # If bidirectional, concatenate the last forward and backward hidden states
                # h_n shape is (2*num_of_layers, batch_size, 2*units)
                output = torch.cat((h_n[-2], h_n[-1]), dim=-1)  # Shape: (batch_size, 2 * hidden_size)
            else:
                # h_n shape is (1*num_of_layers, batch_size, units)
                output = h_n[-1]  # Shape: (batch_size, hidden_size)
            if self.cfg.norm is not None:
                if hasattr(self.norm_layer, "requires_mask"):
                    print("Can't provide masked norm for the tensor of shape ~ (BatchSize, Units)")
                    raise ValueError
                output = self.norm_layer(output)
            
        return output, seq_mask
    