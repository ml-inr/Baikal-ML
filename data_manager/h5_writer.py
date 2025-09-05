"""HDF5 file writer for storing processed physics data."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import h5py

logger = logging.getLogger(__name__)


class HDF5Writer:
    """Write physics data to HDF5 format with proper structure and metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HDF5 writer with configuration.
        
        Args:
            config: Configuration dictionary with data processing settings
        """
        self.config = config
        logger.info("Initialized HDF5 writer")
    
    def write_dataset(self, data: Dict[str, np.ndarray], 
                     output_path: Path, dataset_name: str,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write data to HDF5 file with proper structure.
        
        Args:
            data: Dictionary mapping branch names to numpy arrays
            output_path: Path to output HDF5 file
            dataset_name: Name of the dataset (e.g., 'train', 'val', 'test')
            metadata: Optional metadata to store with dataset
            
        Raises:
            ValueError: If data is empty or inconsistent
            OSError: If file cannot be written
        """
        if not data:
            raise ValueError("Cannot write empty dataset")
        
        # Validate data consistency
        self._validate_data(data)
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing dataset '{dataset_name}' to {output_path}")
        
        try:
            with h5py.File(output_path, 'w') as f:
                # Create dataset group
                group = f.create_group(dataset_name)
                
                # Write data arrays
                for branch_name, values in data.items():
                    dataset = group.create_dataset(
                        branch_name, 
                        data=values,
                        compression='gzip',
                        compression_opts=6,
                        shuffle=True
                    )
                    
                    # Add dataset attributes
                    dataset.attrs['dtype'] = str(values.dtype)
                    dataset.attrs['shape'] = values.shape
                    dataset.attrs['min_value'] = float(np.min(values))
                    dataset.attrs['max_value'] = float(np.max(values))
                    dataset.attrs['mean_value'] = float(np.mean(values))
                    dataset.attrs['std_value'] = float(np.std(values))
                
                # Add group metadata
                self._add_metadata(group, data, metadata)
                
                logger.info(f"Successfully wrote {len(data)} branches with "
                          f"{len(next(iter(data.values())))} events")
                
        except Exception as e:
            logger.error(f"Failed to write HDF5 file {output_path}: {e}")
            # Clean up partial file
            if output_path.exists():
                output_path.unlink()
            raise
    
    def append_dataset(self, data: Dict[str, np.ndarray], 
                      output_path: Path, dataset_name: str) -> None:
        """Append data to existing HDF5 dataset.
        
        Args:
            data: Dictionary mapping branch names to numpy arrays
            output_path: Path to existing HDF5 file
            dataset_name: Name of the dataset to append to
            
        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
            ValueError: If data structure doesn't match existing dataset
        """
        if not output_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {output_path}")
        
        logger.info(f"Appending to dataset '{dataset_name}' in {output_path}")
        
        with h5py.File(output_path, 'a') as f:
            if dataset_name not in f:
                raise ValueError(f"Dataset '{dataset_name}' not found in file")
            
            group = f[dataset_name]
            
            for branch_name, new_values in data.items():
                if branch_name not in group:
                    raise ValueError(f"Branch '{branch_name}' not found in existing dataset")
                
                # Get existing dataset
                existing_dataset = group[branch_name]
                old_data = existing_dataset[:]
                
                # Concatenate data
                combined_data = np.concatenate([old_data, new_values])
                
                # Delete old dataset and create new one
                del group[branch_name]
                new_dataset = group.create_dataset(
                    branch_name,
                    data=combined_data,
                    compression='gzip',
                    compression_opts=6,
                    shuffle=True
                )
                
                # Update attributes
                new_dataset.attrs['shape'] = combined_data.shape
                new_dataset.attrs['min_value'] = float(np.min(combined_data))
                new_dataset.attrs['max_value'] = float(np.max(combined_data))
                new_dataset.attrs['mean_value'] = float(np.mean(combined_data))
                new_dataset.attrs['std_value'] = float(np.std(combined_data))
    
    def read_dataset(self, file_path: Path, 
                    dataset_name: str) -> Dict[str, np.ndarray]:
        """Read dataset from HDF5 file.
        
        Args:
            file_path: Path to HDF5 file
            dataset_name: Name of dataset to read
            
        Returns:
            Dictionary mapping branch names to numpy arrays
            
        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
            KeyError: If dataset not found
        """
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        logger.info(f"Reading dataset '{dataset_name}' from {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            if dataset_name not in f:
                available_datasets = list(f.keys())
                raise KeyError(f"Dataset '{dataset_name}' not found. "
                             f"Available: {available_datasets}")
            
            group = f[dataset_name]
            data = {}
            
            for branch_name in group.keys():
                data[branch_name] = group[branch_name][:]
            
            logger.info(f"Read {len(data)} branches with "
                       f"{len(next(iter(data.values())))} events")
            
            return data
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get information about HDF5 file contents.
        
        Args:
            file_path: Path to HDF5 file
            
        Returns:
            Dictionary with file information
        """
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            info = {
                'file_size': file_path.stat().st_size,
                'datasets': {}
            }
            
            def visitor(name, obj):
                if isinstance(obj, h5py.Group):
                    info['datasets'][name] = {
                        'type': 'group',
                        'items': list(obj.keys()),
                        'attrs': dict(obj.attrs)
                    }
                elif isinstance(obj, h5py.Dataset):
                    info['datasets'][name] = {
                        'type': 'dataset',
                        'shape': obj.shape,
                        'dtype': str(obj.dtype),
                        'attrs': dict(obj.attrs)
                    }
            
            f.visititems(visitor)
            return info
    
    def _validate_data(self, data: Dict[str, np.ndarray]) -> None:
        """Validate data consistency before writing.
        
        Args:
            data: Dictionary mapping branch names to numpy arrays
            
        Raises:
            ValueError: If data is inconsistent
        """
        if not data:
            raise ValueError("Data dictionary is empty")
        
        # Check all arrays have same length
        lengths = [len(arr) for arr in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent array lengths: {dict(zip(data.keys(), lengths))}")
        
        # Check for valid data types
        for branch_name, values in data.items():
            if not isinstance(values, np.ndarray):
                raise ValueError(f"Branch '{branch_name}' is not a numpy array")
            
            if values.size == 0:
                raise ValueError(f"Branch '{branch_name}' is empty")
            
            if not np.isfinite(values).all():
                logger.warning(f"Branch '{branch_name}' contains non-finite values")
    
    def _add_metadata(self, group: h5py.Group, data: Dict[str, np.ndarray],
                     metadata: Optional[Dict[str, Any]]) -> None:
        """Add metadata to HDF5 group.
        
        Args:
            group: HDF5 group object
            data: Data being written
            metadata: Optional additional metadata
        """
        # Basic statistics
        n_events = len(next(iter(data.values())))
        group.attrs['n_events'] = n_events
        group.attrs['n_branches'] = len(data)
        group.attrs['branch_names'] = list(data.keys())
        
        # Configuration metadata
        if 'experiment' in self.config:
            group.attrs['experiment_name'] = self.config['experiment']['name']
            group.attrs['seed'] = self.config['experiment']['seed']
        
        # Additional metadata
        if metadata:
            for key, value in metadata.items():
                group.attrs[key] = value
        
        # Processing timestamp
        import time
        group.attrs['created_timestamp'] = time.time()
        group.attrs['created_time'] = time.strftime('%Y-%m-%d %H:%M:%S')