import torch
import torch.nn as nn


class BatchNorm1dTranspose(nn.BatchNorm1d):
    def forward(self, x):
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
        second_head_out_size=None,
        use_cls_token=False,
        return_only_cls_token=False,
        **kwargs
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.first_layer = nn.Linear(in_features, hidden_size).to("cuda:0")
        if not use_batch_norm:
            enc_layer = nn.TransformerEncoderLayer(
                hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
            )
        else:
            enc_layer = TransformerEncoderLayerBN(
                hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
            )
        self.enc1 = nn.TransformerEncoder(enc_layer, num_layers).to("cuda:0")
        self.enc2 = nn.TransformerEncoder(enc_layer, num_layers).to("cuda:1")

        self.head = nn.Linear(hidden_size, out_size).to("cuda:1")

        self.class_token = (
            nn.Parameter(
                torch.randn(1, 1, hidden_size),
                requires_grad=True,
            ).to("cuda:1")
            if use_cls_token
            else None
        )
        self.return_only_cls_token = return_only_cls_token
        self.second_head = (
            nn.Linear(hidden_size, second_head_out_size)
            if second_head_out_size is not None
            else None
        )
        self.aggregator = aggregator

    def forward(self, x, mask):
        mask = (~mask).float()  # bool mask makes encoder predict nans sometimes
        x = self.first_layer(x)
        if self.class_token is not None:
            x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
            mask = torch.cat(
                [torch.ones(x.shape[0], 1, dtype=torch.float32).to(mask.device), mask],
                dim=1,
            )
        x = self.enc1(x, src_key_padding_mask=mask)
        x = x.to("cuda:1")
        mask = mask.to("cuda:1")
        # print(next(self.enc2.parameters()).device)
        # a = input()
        x = self.enc2(x, src_key_padding_mask=mask)
        y = self.head(x)
        if self.aggregator is not None:
            return self.aggregator(x)
        if self.second_head is not None:
            z = self.second_head(x)
            return torch.cat([y, z], dim=-1)
        if self.class_token is not None and self.return_only_cls_token:
            return y[:, 0, :]
        return y.to("cuda:0")
