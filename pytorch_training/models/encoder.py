import torch
import torch.nn as nn


class BatchNorm1dTranspose(nn.BatchNorm1d):
    def forward(self, x):
        # print("inp shape", x.shape)
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class TransformerEncoderLayerBN(nn.TransformerEncoderLayer):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(d_model, *args, **kwargs)
        self.norm1 = BatchNorm1dTranspose(d_model)
        self.norm2 = BatchNorm1dTranspose(d_model)


class Encoder(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        num_layers,
        dim_feedforward_size,
        n_heads,
        out_size,
        dropout_p,
        use_batch_norm=False,
        aggregator=None,
        **kwargs
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.first_layer = nn.Linear(in_features, hidden_size)
        if not use_batch_norm:
            enc_layer = nn.TransformerEncoderLayer(
                hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
            )
        else:
            enc_layer = TransformerEncoderLayerBN(
                hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
            )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(hidden_size, out_size)
        self.aggregator = aggregator

    def forward(self, x, mask):
        print("1", x.min(), x.max(), x.shape)
        x[x > 100] = 100
        x = self.first_layer(x)
        print("2", x.min(), x.max(), x.shape)
        x = self.enc(x, src_key_padding_mask=~mask)
        print("3", x.min(), x.max(), x.shape)
        x = self.head(x)
        print("4", x.min(), x.max(), x.shape)
        if self.aggregator is not None:
            return self.aggregator(x)
        return x


class EncoderTwoHeads(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        num_layers,
        dim_feedforward_size,
        n_heads,
        out_size,
        dropout_p,
        use_batch_norm=False,
        **kwargs
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.first_layer = nn.Linear(in_features, hidden_size)
        if not use_batch_norm:
            enc_layer = nn.TransformerEncoderLayer(
                hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
            )
        else:
            enc_layer = TransformerEncoderLayerBN(
                hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
            )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.head_a = nn.Linear(hidden_size, 2)
        self.head_b = nn.Linear(hidden_size, 1)

    def forward(self, x, mask):
        x = self.first_layer(x)
        x = self.enc(x, src_key_padding_mask=~mask)
        x_a = self.head_a(x)
        x_b = self.head_b(x)
        res = torch.cat((x_a, x_b), dim=-1)
        return res
