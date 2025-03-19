from utils.config_loader import load_config
from models.resnet import ResNet1D
from models.config import ResNetConfig, ConvLayerConfig, MLPConfig

def main():
    config_dict = load_config("settings/model_config.yaml")
    resnet_config = ResNetConfig(
        input_channels=config_dict['resnet']['input_channels'],
        num_classes=config_dict['resnet']['num_classes'],
        conv_layers=[ConvLayerConfig(**layer) for layer in config_dict['resnet']['conv_layers']],
        mlp_layers=[MLPConfig(**mlp) for mlp in config_dict['resnet']['mlp_layers']]
    )

    model = ResNet1D(config=resnet_config)
    print(model)

if __name__ == "__main__":
    main()
