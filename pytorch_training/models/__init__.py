from .encoder import Encoder, EncoderTwoHeads
from .graphnet import GraphnetDynedge
from .lstm import LSTM
from .gat import GAT
from .gincn import GINCN
import torch.nn as nn
import typing as tp


def load_model(model_type: str, model_kwargs: dict[str, tp.Any]) -> nn.Module:
    if model_type == "encoder":
        return Encoder(**model_kwargs)
    elif model_type == "encoder_two_heads":
        return EncoderTwoHeads(**model_kwargs)
    elif model_type == "graphnet":
        return GraphnetDynedge(**model_kwargs)
    elif model_type == "lstm":
        return LSTM(**model_kwargs)
    elif model_type == "gat":
        return GAT(**model_kwargs)
    elif model_type == "gin":
        return GINCN(**model_kwargs)
    else:
        raise NotImplementedError
    # elif model
