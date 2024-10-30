import torch.nn as nn

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
