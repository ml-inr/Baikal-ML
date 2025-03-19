import yaml
from typing import Dict


class ConfigReader:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self._read_config()

    def _read_config(self) -> Dict:
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")

    def get_config(self) -> Dict:
        return self.config