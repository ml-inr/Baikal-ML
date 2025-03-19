import yaml
import csv

try:
    from .settings_scheme import BaseConfig, ProcessorConfig, ChunksFromPathsConfig, MCMuNuSepBatchGeneratorConfig, ExpBatchGeneratorConfig
except ImportError:
    from data.settings_scheme import BaseConfig, ProcessorConfig, ChunksFromPathsConfig, MCMuNuSepBatchGeneratorConfig, ExpBatchGeneratorConfig

# Saver
def save_datacfg2yaml(cfg: BaseConfig, path: str = "./cfg.yaml", mode: str = 'w') -> None:
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

def save_dict2yaml(cfg: dict, path: str = "./cfg.yaml", mode: str = 'w'):
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
        yaml.dump(cfg, f, MyDumper, indent=4, width=1000, sort_keys=False)
    return cfg

# Loaders     
def load_yaml2dict(path: str = "./cfg.yaml") -> dict:
    """Loads configuration from yaml file as dict.

    Args:
        path (str): path to yaml file
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_batchgen_cfg(path: str = "./cfg.yaml", DataClass: BaseConfig = MCMuNuSepBatchGeneratorConfig) -> BaseConfig:
    """Loads configuration from yaml file as instance of BaseConfig.

    Args:
        path (str): path to yaml file
        DataClass (BaseConfig): type of configuration sheme to load
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['chunk_generator_cfg']['processor_cfg'] = ProcessorConfig(**cfg['chunk_generator_cfg']['processor_cfg'])
    cfg['chunk_generator_cfg'] = ChunksFromPathsConfig(**cfg['chunk_generator_cfg'])
    return DataClass(**cfg)

# Paths
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