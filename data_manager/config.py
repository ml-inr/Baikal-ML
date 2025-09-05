"""Configuration management for data processing."""

import logging
from pathlib import Path
from typing import Dict, Any, List
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate YAML configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required sections are missing
        yaml.YAMLError: If YAML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML config: {e}")
    
    # Validate required sections
    _validate_config(config)
    
    # Convert string paths to Path objects
    config = _normalize_paths(config)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration has required sections and fields."""
    required_sections = ['experiment', 'data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate experiment section
    experiment_fields = ['name', 'seed']
    for field in experiment_fields:
        if field not in config['experiment']:
            raise ValueError(f"Missing required field in experiment: {field}")
    
    # Validate data section
    data_fields = ['input_files', 'output_dir', 'splits']
    for field in data_fields:
        if field not in config['data']:
            raise ValueError(f"Missing required field in data: {field}")
    
    # Validate splits sum to 1.0
    splits = config['data']['splits']
    split_sum = sum(splits.values())
    if not (0.99 <= split_sum <= 1.01):  # Allow small floating point errors
        raise ValueError(f"Data splits must sum to 1.0, got {split_sum}")


def _normalize_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string paths to Path objects."""
    # Convert input files to Path objects
    if 'input_files' in config['data']:
        config['data']['input_files'] = [
            Path(f) for f in config['data']['input_files']
        ]
    
    # Convert output directory to Path object
    if 'output_dir' in config['data']:
        config['data']['output_dir'] = Path(config['data']['output_dir'])
    
    return config