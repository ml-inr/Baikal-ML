import yaml

from ..models.config import MuNuSepResNetConfig
from ..models.munusep_resnet import MuNuSepResNet
from ..layers.config import MaskedConv1DConfig, ResBlockConfig, DenseInput


def load_config_from_yaml(yaml_path: str) -> MuNuSepResNetConfig:
    # Load configuration from YAML
    with open(yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    # Parse ResBlockConfig
    res_blocks = [
        ResBlockConfig(
            id=MaskedConv1DConfig(**block['id']),
            cd=MaskedConv1DConfig(**block['cd']),
            skip=MaskedConv1DConfig(**block['skip'])
        ) for block in config_dict['MuNuSepResNetConfig']['res_blocks']
    ]
    
    # Parse DenseInput layers
    dense_layers = [
        DenseInput(**dense_layer) for dense_layer in config_dict['MuNuSepResNetConfig']['dense_layers']
    ]

    # Assemble the final configuration
    return MuNuSepResNetConfig(
        res_blocks=res_blocks,
        pooling_type=config_dict['MuNuSepResNetConfig']['pooling_type'],
        dense_layers=dense_layers
    )


def munusep_resnet_from_yaml(yaml_path: str) -> MuNuSepResNet:
    # Load the configuration
    model_config = load_config_from_yaml(yaml_path)
    
    # Initialize the model
    model = MuNuSepResNet(config=model_config)
    return model
