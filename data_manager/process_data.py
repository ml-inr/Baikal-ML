"""Main data processing script for converting ROOT files to HDF5."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

from .config import load_config
from .root_reader import ROOTReader
from .h5_writer import HDF5Writer


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    logging.info(f"Set random seed to {seed}")


def merge_data_from_files(reader: ROOTReader, 
                         file_paths: List[Path]) -> Dict[str, np.ndarray]:
    """Merge data from multiple ROOT files.
    
    Args:
        reader: Configured ROOT reader
        file_paths: List of ROOT file paths
        
    Returns:
        Merged data dictionary
    """
    all_data = {}
    
    for file_path in file_paths:
        try:
            logging.info(f"Processing file: {file_path}")
            file_data = reader.read_file(file_path)
            
            if not all_data:
                # Initialize with first file
                all_data = file_data
            else:
                # Merge with existing data
                for branch_name, values in file_data.items():
                    if branch_name in all_data:
                        all_data[branch_name] = np.concatenate([
                            all_data[branch_name], values
                        ])
                    else:
                        logging.warning(f"Branch '{branch_name}' not in previous files")
                        
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")
            continue
    
    if all_data:
        n_events = len(next(iter(all_data.values())))
        logging.info(f"Merged data from {len(file_paths)} files: {n_events} total events")
    
    return all_data


def split_data(data: Dict[str, np.ndarray], 
              splits: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], ...]:
    """Split data into train/validation/test sets.
    
    Args:
        data: Dictionary of branch data
        splits: Dictionary with split ratios
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if not data:
        return {}, {}, {}
    
    n_samples = len(next(iter(data.values())))
    logging.info(f"Splitting {n_samples} events with ratios: {splits}")
    
    # Generate random indices
    indices = np.random.permutation(n_samples)
    
    # Calculate split boundaries
    train_end = int(splits['train'] * n_samples)
    val_end = train_end + int(splits['val'] * n_samples)
    
    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    logging.info(f"Split sizes - Train: {len(train_idx)}, "
                f"Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create split datasets
    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}
    test_data = {k: v[test_idx] for k, v in data.items()}
    
    return train_data, val_data, test_data


def log_data_statistics(data: Dict[str, np.ndarray], 
                       dataset_name: str) -> None:
    """Log statistics about the dataset.
    
    Args:
        data: Dataset dictionary
        dataset_name: Name of dataset for logging
    """
    if not data:
        logging.warning(f"{dataset_name} dataset is empty")
        return
    
    n_events = len(next(iter(data.values())))
    logging.info(f"{dataset_name} dataset statistics:")
    logging.info(f"  Events: {n_events}")
    logging.info(f"  Branches: {len(data)}")
    
    for branch_name, values in data.items():
        stats = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        logging.info(f"  {branch_name}: mean={stats['mean']:.3f}, "
                    f"std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Convert ROOT files to HDF5 format for ML training"
    )
    parser.add_argument(
        '--config', 
        type=Path, 
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logging.info(f"Loaded configuration: {config['experiment']['name']}")
        
        # Set seed for reproducibility
        set_seed(config['experiment']['seed'])
        
        # Initialize components
        reader = ROOTReader(config)
        writer = HDF5Writer(config)
        
        # Merge data from all input files
        all_data = merge_data_from_files(reader, config['data']['input_files'])
        
        if not all_data:
            logging.error("No data extracted from input files")
            return 1
        
        # Split data
        train_data, val_data, test_data = split_data(all_data, config['data']['splits'])
        
        # Log statistics
        log_data_statistics(train_data, "Train")
        log_data_statistics(val_data, "Validation")
        log_data_statistics(test_data, "Test")
        
        # Create output directory
        output_dir = config['data']['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write datasets
        datasets = [
            (train_data, 'train'),
            (val_data, 'val'),
            (test_data, 'test')
        ]
        
        for data, split_name in datasets:
            if data:  # Only write non-empty datasets
                output_path = output_dir / f"{split_name}.h5"
                metadata = {
                    'split': split_name,
                    'config_file': str(args.config),
                    'source_files': [str(f) for f in config['data']['input_files']]
                }
                writer.write_dataset(data, output_path, split_name, metadata)
                logging.info(f"Wrote {split_name} dataset to {output_path}")
        
        logging.info("Data processing completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Data processing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())