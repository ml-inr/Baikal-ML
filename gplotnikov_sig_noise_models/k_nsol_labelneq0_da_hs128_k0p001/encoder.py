from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .layers import GradientReversal


class BatchNorm1dTranspose(nn.BatchNorm1d):
    """BatchNorm that handles (B, N, C) input by transposing."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class TransformerEncoderLayerBN(nn.TransformerEncoderLayer):
    """Transformer encoder layer with BatchNorm instead of LayerNorm."""

    def __init__(self, d_model: int, *args, **kwargs) -> None:
        super().__init__(d_model, *args, **kwargs)
        self.norm1 = BatchNorm1dTranspose(d_model)
        self.norm2 = BatchNorm1dTranspose(d_model)


class Encoder(nn.Module):
    """Transformer encoder for sequence processing."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int,
        dim_feedforward_size: int,
        n_heads: int,
        out_size: int,
        dropout_p: float = 0.0,
        use_batch_norm: bool = False,
        second_head_out_size: int | None = None,
        use_cls_token: bool = False,
        return_only_cls_token: bool = False,
        return_hidden: bool = False,
        return_hiddens_by_layers: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.dropout_p = dropout_p
        self.return_hidden = return_hidden
        self.return_hiddens_by_layers = return_hiddens_by_layers
        self.return_only_cls_token = return_only_cls_token

        self.first_layer = nn.Linear(in_features, hidden_size)

        encoder_layer_cls = (
            TransformerEncoderLayerBN if use_batch_norm else nn.TransformerEncoderLayer
        )
        enc_layer = encoder_layer_cls(
            hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)

        self.head = nn.Linear(hidden_size, out_size, bias=False)

        self.class_token: nn.Parameter | None = None
        if use_cls_token:
            self.class_token = nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)

        self.second_head: nn.Module | None = None
        if second_head_out_size is not None:
            self.second_head = nn.Linear(hidden_size, second_head_out_size)

    def forward(
        self, x: Tensor, mask: Tensor
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]]:
        mask = (~mask).float()
        x = self.first_layer(x)

        if self.class_token is not None:
            batch_size = x.shape[0]
            cls_tokens = self.class_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.float32, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        hiddens_by_layer: list[Tensor] = []

        if self.return_hiddens_by_layers:
            for layer in self.enc.layers:
                x = layer(x, src_key_padding_mask=mask)
                hiddens_by_layer.append(x)
            y = self.head(x)
        else:
            x = self.enc(x, src_key_padding_mask=mask)
            y = self.head(x)

        if self.second_head is not None:
            z = self.second_head(x)
            res = torch.cat([y.mean(1), z.mean(1)], dim=-1)
        elif self.class_token is not None and self.return_only_cls_token:
            res = y.mean(1)
        else:
            res = y

        if self.return_hiddens_by_layers:
            return res, hiddens_by_layer
        elif self.return_hidden:
            return res, x
        return res


class EncoderDomainAdaptation(nn.Module):
    """Encoder with domain adaptation head using gradient reversal."""

    def __init__(
        self,
        freeze_encoder: bool = False,
        num_domains: int = 2,
        domain_classifier_hidden_size: int = 128,
        domain_classifier_layers: int = 2,
        gradient_reversal_alpha: float = 1.0,
        uncertainty_head_hidden_size: int | None = None,
        uncertainty_head_out_size: int | None = None,
        aggregate_output: bool = True,
        return_hidden: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(**kwargs)
        self.encoder.return_hidden = True

        self.main_head = nn.Linear(self.encoder.hidden_size, self.encoder.out_size)
        self.aggregate_output = aggregate_output
        self.return_hidden = return_hidden

        self.uncertainty_head: nn.Module | None = None
        if uncertainty_head_hidden_size is not None and uncertainty_head_out_size is not None:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(self.encoder.hidden_size, uncertainty_head_hidden_size),
                nn.ReLU(),
                nn.Linear(uncertainty_head_hidden_size, uncertainty_head_out_size),
            )

        self.gradient_reversal = GradientReversal(alpha=gradient_reversal_alpha)
        self.domain_classifier = self._build_domain_classifier(
            self.encoder.hidden_size,
            domain_classifier_hidden_size,
            domain_classifier_layers,
            num_domains,
            self.encoder.dropout_p,
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if freeze_encoder:
            self.freeze_encoder()

    def _build_domain_classifier(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_domains: int,
        dropout_p: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_size = input_size

        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                ]
            )
            current_size = hidden_size

        layers.append(nn.Linear(current_size, num_domains))
        return nn.Sequential(*layers)

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(
        self, x: Tensor, mask: Tensor, return_hidden_states: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        _, hidden_states = self.encoder(x, mask)

        output = self.main_head(hidden_states)
        if self.aggregate_output:
            output = output.mean(1)

        if self.uncertainty_head is not None:
            uncertainty_output = self.uncertainty_head(hidden_states)
            if self.aggregate_output:
                uncertainty_output = uncertainty_output.mean(1)
            output = torch.cat([output, uncertainty_output], dim=-1)

        reversed_features = self.gradient_reversal(hidden_states.mean(1))
        domain_output = self.domain_classifier(reversed_features)

        if return_hidden_states:
            return output, domain_output, hidden_states
        return output, domain_output
