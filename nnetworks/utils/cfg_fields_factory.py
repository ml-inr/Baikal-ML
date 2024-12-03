import torch.nn as nn
from nnetworks.layers.norm import MaskedBatchNorm1d, MaskedLayerNorm12

def get_activation(activation_config: dict):
    """
    Create an activation function based on the given configuration.
    Args:
        activation_config (dict): A dictionary where the key is the name of the activation
                                  and the value is a dict of arguments for that activation.
    Returns:
        nn.Module: The corresponding activation function.
    """
    if not activation_config:
        return nn.ReLU()  # Default activation if none provided

    # Example: {"LeakyReLU": {"negative_slope": 0.2}}
    activation_name, activation_args = next(iter(activation_config.items()))
    
    # Supported activations dictionary
    activations = {
        "relu": nn.ReLU,
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "ELU": nn.ELU,
        "Softplus": nn.Softplus,
        "Softmax": nn.Softmax
    }

    if activation_name in activations:
        return activations[activation_name](**activation_args) if activation_args else activations[activation_name]()
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")
    
def get_norm_layer(num_features: int, norm_kwargs: dict):
    """
    Create an norm layer based on the given configuration.
    Args:
        num_features (int): number of channels in input tensor
        norm_kwargs (dict): A dictionary where the key is the name of the norm
                                  and the value is a dict of arguments for that norm.
    Returns:
        nn.Module: The corresponding norm layer.
    """

    # Example: {"LeakyReLU": {"negative_slope": 0.2}}
    norm_name, norm_args = next(iter(norm_kwargs.items()))
    
    # Supported norms dictionary
    norms = {
        "LayerNorm": nn.LayerNorm,
        "BatchNorm1d": nn.BatchNorm1d,
        "MaskedBatchNorm1d": MaskedBatchNorm1d,
        "MaskedLayerNorm12": MaskedLayerNorm12
    }

    if norm_name in norms:
        return norms[norm_name](num_features, **norm_args) if norm_args else norms[norm_name](num_features)
    else:
        raise ValueError(f"Unsupported norm: {norm_name}")
