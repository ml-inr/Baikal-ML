import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        out_size,
        dropout_p=0.0,
        num_layers=2,
        kernel_size=4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout_p,
            batch_first=True,
        )
        self.conv = nn.Conv1d(hidden_size * 2, out_size, kernel_size, padding="same")

    def forward(self, x):
        lstm_res, _ = self.lstm(x)
        lstm_res = lstm_res.permute(0, 2, 1)
        out = self.conv(lstm_res).permute(0, 2, 1)
        return out
