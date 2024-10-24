import yaml
import csv

try:
    from data.root_manager.settings import ProcessorConfig, ChunkGeneratorConfig, FilterParams
    from data.settings import BatchGeneratorConfig, NormParams, AugmentParams
    from data.configurations.base import BaseConfig
except ImportError:
    from root_manager.settings import ProcessorConfig, ChunkGeneratorConfig, FilterParams
    from settings import BatchGeneratorConfig, NormParams, AugmentParams
    from configurations.base import BaseConfig


def save_cfg(cfg: BaseConfig, path: str = "./cfg.yaml", mode: str = 'w') -> None:
    """Saves configuration to path as yaml file.

    Args:
        cfg (BaseConfig): _description_
        path (str): _description_
        mode (str, optional): _description_. Defaults to 'w'.
    """
    
    # Dumper for saving files in easy-to-read format
    class MyDumper(yaml.Dumper):
        def write_line_break(self, data=None):
            super().write_line_break(data)

            if len(self.indents) == 1:
                super().write_line_break()
    
    # Custom representer for lists to force them into flow style
    def represent_list_as_inline(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    yaml.add_representer(list, represent_list_as_inline)
    
    with open(path, mode) as f:
        yaml.dump(cfg.to_dict(), f, MyDumper, indent=4, width=1000, sort_keys=False)
       
        
def load_cfg_as_dict(path: str = "./cfg.yaml") -> dict:
    """Loads configuration from yaml file as dict.

    Args:
        path (str): path to yaml file
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(path, f)
    return cfg


def load_cfg(path: str = "./cfg.yaml", DataClass: BaseConfig = BatchGeneratorConfig) -> BaseConfig:
    """Loads configuration from yaml file as instance of BaseConfig.

    Args:
        path (str): path to yaml file
        DataClass (BaseConfig): type of configuration sheme to load
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['chunk_generator_cfg']['processor_params']['filter_cfg'] = FilterParams(**cfg['chunk_generator_cfg']['processor_params']['filter_cfg'])
    cfg['chunk_generator_cfg']['processor_params'] = ProcessorConfig(**cfg['chunk_generator_cfg']['processor_params'])
    cfg['chunk_generator_cfg'] = ChunkGeneratorConfig(**cfg['chunk_generator_cfg'])
    cfg['norm_params'] = NormParams(**cfg['norm_params'])
    cfg['augment_params'] = AugmentParams(**cfg['augment_params'])
    return DataClass(**cfg)

def save_paths(paths: list[str], where: str = "./paths.csv") -> None:
    with open(where, 'w') as f:
        write = csv.DictWriter(f, fieldnames=['path'])
        for path in paths:
            write.writerow({'path':path})
            
def read_paths(where: str = "./paths.csv") -> list[str]:
    with open(where, 'r') as f:
        csv_reader = csv.reader(f)
        paths = []
        for row in csv_reader:
            paths += row # row is a list with 1 element
    return paths