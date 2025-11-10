"""
Nu-Mu Dataset for Neutrino/Muon Binary Classification

This dataset loads both Monte Carlo and experimental data from HDF5 files for the specific task
of neutrino vs muon binary classification:
- True: Neutrino-induced events (nue, numu, nutau types)
- False: Muon-induced events (atmospheric background)

Supports both MC data (muatm, nue2, nuatm, etc.) and experimental data (exp).
Designed specifically for binary Nu-Mu classification task.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

logger = logging.getLogger(__name__)


class NuMuDataset(Dataset):
    """
    PyTorch Dataset for neutrino vs muon binary classification using Baikal detector data.
    
    Features:
    - Variable-length time series (hit sequences per event)
    - 5D hit features: [amplitude, time, x, y, z]
    - Binary classification: neutrino (True) vs muon (False) events
    - Multi-particle type support with configurable sampling
    - Memory-efficient lazy loading with event limits per particle type
    - Advanced sampling strategies (range, random, sequential)
    - Event interleaving for balanced batch creation
    
    Args:
        h5_path: Path to merged HDF5 file containing particle data
        events_per_particle: Dict mapping particle types to max events per type.
            Example: {'muatm_2020': 1000, 'nue2_2020': 500}
        particle_types: List of particle types to load. If None, defaults to:
            ['muatm_2019', 'muatm_2020', 'nuatm_2019', 'nuatm_2020', 'nue2_2019', 'nue2_2020', 'exp']
        neutrino_types: Particle types considered as neutrinos (positive class). If None, defaults to:
            ['nuatm_2019', 'nuatm_2020', 'nue2_2019', 'nue2_2020']
        max_hits: Maximum hits per event (for memory management and truncation). Default: 500
        sampling_config: Dict with sampling parameters per particle type. Example:
            {'muatm_2020': {'mode': 'range', 'start_event': 0, 'end_event': 1000},
             'nue2_2020': {'mode': 'random'}}
            Modes: 'range', 'random', 'sequential'
        device: PyTorch device for tensors ('cpu' or 'cuda'). Default: 'cpu'
        seed: Random seed for reproducible sampling and shuffling. Default: None
        shuffle_events: Whether to shuffle events after loading to remove particle-type ordering bias. Default: True
    
    Raises:
        FileNotFoundError: If h5_path does not exist
        TypeError: If events_per_particle is not a dictionary
        ValueError: If no data found for specified particle types
    
    Example:
        >>> dataset = NuMuDataset(
        ...     h5_path="/path/to/data.h5",
        ...     events_per_particle={'muatm_2020': 1000, 'nue2_2020': 500},
        ...     particle_types=['muatm_2020', 'nue2_2020'],
        ...     neutrino_types=['nue2_2020'],
        ...     max_hits=200,
        ...     device='cpu',
        ...     seed=42
        ... )
        >>> print(f"Dataset has {len(dataset)} events")
        >>> features, label = dataset[0]  # Get first event
        >>> print(f"Event shape: {features.shape}, Label: {label}")
    """
    
    def __init__(
        self,
        h5_path: Union[str, Path],
        events_per_particle: Optional[Dict[str, int]] = None,
        particle_types: Optional[List[str]] = None,
        neutrino_types: Optional[List[str]] = None,
        max_hits: Optional[int] = 500,
        sampling_config: Optional[Dict[str, Dict[str, Any]]] = None,
        device: str = 'cpu',
        seed: Optional[int] = None,
        shuffle_events: bool = True
    ):
        self.h5_path = Path(h5_path)
        self.device = device
        self.max_hits = max_hits
        self.max_events_per_class = events_per_particle
        self.sampling_config = sampling_config or {}
        self.shuffle_events = shuffle_events
        
        # Validate parameters
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        
        if events_per_particle is not None and not isinstance(events_per_particle, dict):
            raise TypeError("events_per_particle must be a dictionary mapping particle types to max event counts")
        
        # Set up reproducible random number generator
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        
        # Default particle types (neutrinos vs atmospheric muons)
        if particle_types is None:
            particle_types = ['muatm_2019', 'muatm_2020', 'nuatm_2019', 'nuatm_2020', 'nue2_2019', 'nue2_2020', 'exp']
        self.particle_types = [name for name in particle_types if name in self.max_events_per_class.keys()] if self.max_events_per_class else particle_types
        
        # Default neutrino types (positive class)
        if neutrino_types is None:
            neutrino_types = ['nuatm_2019', 'nuatm_2020', 'nue2_2019', 'nue2_2020']
        self.neutrino_types = [name for name in neutrino_types if name in self.max_events_per_class.keys()] if self.max_events_per_class else neutrino_types
        
        # Default max num of events per class
        if self.max_events_per_class is None:
            self.max_events_per_class = {name: 10_000 for name in self.particle_types} # Default to 10000 per class if not specified
        
        # Load and prepare data
        self._load_data()
        logger.info(f"Loaded {len(self)} events from {len(self.particle_types)} particle types")
        logger.info(f"Neutrino events: {self.labels.sum()}, Muon events: {(~self.labels).sum()}")

    
    def _load_data(self):
        """Load data from HDF5 file and prepare for training."""
        all_events = []
        all_labels = []
        all_magic_numbers = []
        all_hit_counts = []
        all_event_ids = []
        all_channel_ids = []
        total_loaded = 0
        
        with h5py.File(self.h5_path, 'r') as f:
            for particle_type in self.particle_types:
                if particle_type not in f:
                    logger.warning(f"Particle type '{particle_type}' not found in {self.h5_path}")
                    continue
                
                # # Check if we've reached the max_events limit
                # if self.max_events is not None and total_loaded >= self.max_events:
                #     logger.info(f"Reached max_events limit ({self.max_events}), stopping data loading")
                #     break
                
                # Determine if this particle type is neutrino (positive class)
                is_neutrino = particle_type in self.neutrino_types
                
                # Check per-class limit first, then global limit
                num_events_to_load = self.max_events_per_class[particle_type]
                
                # Load data for this particle type
                events, labels, magic_numbers, hit_counts, event_ids, channel_ids = self._load_particle_data(
                    f, particle_type, is_neutrino, num_events_to_load
                )
                
                all_events.extend(events)
                all_labels.extend(labels)
                all_magic_numbers.extend(magic_numbers)
                all_hit_counts.extend(hit_counts)
                all_event_ids.extend(event_ids)
                all_channel_ids.extend(channel_ids)
                total_loaded += len(events)
                
                logger.info(f"Loaded {len(events)} events from {particle_type} (neutrino={is_neutrino})")
        
        if not all_events:
            raise ValueError(f"No data found for particle types {self.particle_types}")
        
        # Optional shuffling to remove bias from group-by-group loading
        if self.shuffle_events:
            combined_data = list(zip(all_events, all_labels, all_magic_numbers, all_hit_counts, all_event_ids, all_channel_ids))
            self.rng.shuffle(combined_data)
            all_events, all_labels, all_magic_numbers, all_hit_counts, all_event_ids, all_channel_ids = zip(*combined_data)
            all_events, all_labels, all_magic_numbers, all_hit_counts, all_event_ids, all_channel_ids = (
                list(all_events), list(all_labels), list(all_magic_numbers), list(all_hit_counts), list(all_event_ids), list(all_channel_ids)
            )
            logger.info(f"Shuffled {len(combined_data)} events to remove particle-type ordering bias")
        
        # Store as class attributes with optional interleaving for balanced batches
        self.events, self.labels, self.magic_numbers, self.hit_counts, self.event_ids, self.channel_ids = self._organize_events_for_batching(
            all_events, all_labels, all_magic_numbers, all_hit_counts, all_event_ids, all_channel_ids
        )
        
        # Calculate statistics
        self._calculate_stats()
    
    def _organize_events_for_batching(
        self, 
        events: List[torch.Tensor], 
        labels: List[bool], 
        magic_numbers: List[int],
        hit_counts: List[int],
        event_ids: List[str],
        channel_ids: List[int]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[int], torch.Tensor, List[str]]:
        """
        Organize events to ensure good mixing of classes in batches.
        Interleaves neutrino and muon events to prevent single-class batches.
        """
        # Separate events by class
        neutrino_events = []
        neutrino_magic_numbers = []
        neutrino_hit_counts = []
        neutrino_ids = []
        neutrino_channels = []
        muon_events = []
        muon_magic_numbers = []
        muon_hit_counts = []
        muon_ids = []
        muon_channels = []
        
        for i, is_neutrino in enumerate(labels):
            if is_neutrino:
                neutrino_events.append(events[i])
                neutrino_magic_numbers.append(magic_numbers[i])
                neutrino_hit_counts.append(hit_counts[i])
                neutrino_ids.append(event_ids[i])
                neutrino_channels.append(channel_ids[i])
            else:
                muon_events.append(events[i])
                muon_magic_numbers.append(magic_numbers[i])
                muon_hit_counts.append(hit_counts[i])
                muon_ids.append(event_ids[i])
                muon_channels.append(channel_ids[i])
        
        # Interleave neutrino and muon events for better batch mixing
        interleaved_events = []
        interleaved_labels = []
        interleaved_magic_numbers = []
        interleaved_hit_counts = []
        interleaved_ids = []
        interleaved_channels = []
        
        max_len = max(len(neutrino_events), len(muon_events))
        
        for i in range(max_len):
            # Add neutrino event if available
            if i < len(neutrino_events):
                interleaved_events.append(neutrino_events[i])
                interleaved_labels.append(True)
                interleaved_magic_numbers.append(neutrino_magic_numbers[i])
                interleaved_hit_counts.append(neutrino_hit_counts[i])
                interleaved_ids.append(neutrino_ids[i])
                interleaved_channels.append(neutrino_channels[i])
            
            # Add muon event if available
            if i < len(muon_events):
                interleaved_events.append(muon_events[i])
                interleaved_labels.append(False)
                interleaved_magic_numbers.append(muon_magic_numbers[i])
                interleaved_hit_counts.append(muon_hit_counts[i])
                interleaved_ids.append(muon_ids[i])
                interleaved_channels.append(muon_channels[i])
        
        logger.info(f"Interleaved {len(neutrino_events)} neutrino and {len(muon_events)} muon events for balanced batches")
        
        return (
            interleaved_events,
            torch.tensor(interleaved_labels, dtype=torch.bool, device=self.device),
            interleaved_magic_numbers,
            torch.tensor(interleaved_hit_counts, dtype=torch.long, device=self.device),
            interleaved_ids,
            interleaved_channels
        )
    
    def _lazy_load_event_indices(
        self,
        particle_group,
        parts: List[str],
        particle_type: str,
        sampling_cfg: Dict[str, Any],
        max_events: Optional[int] = None
    ) -> List[Tuple[str, int, int, int]]:
        """
        Lazy loading of event indices - only read what we need based on sampling strategy.
        """
        mode = sampling_cfg.get('mode', 'range')
        selected_indices = []
        total_collected = 0
        
        if mode == 'range':
            start_event = sampling_cfg.get('start_event', 0)
            default_end = start_event+max_events if max_events is not None else float('inf')
            end_event = sampling_cfg.get('end_event', default_end)
            
            # Only read parts until we have enough events
            events_seen = 0
            for part in parts:
                if total_collected >= end_event or (max_events and total_collected >= max_events):
                    break
                    
                try:
                    ev_starts = particle_group['raw']['ev_starts'][part]['data'][:]
                    part_events = []
                    
                    for i in range(len(ev_starts) - 1):
                        if events_seen >= end_event or (max_events and total_collected >= max_events):
                            break
                            
                        start_idx = ev_starts[i]
                        end_idx = ev_starts[i + 1]
                        if end_idx > start_idx:  # Valid event with hits
                            if events_seen >= start_event:  # Within our range
                                part_events.append((part, i, start_idx, end_idx))
                                total_collected += 1
                            events_seen += 1
                    
                    selected_indices.extend(part_events)
                    
                except Exception as e:
                    logger.warning(f"Error reading part {part} for {particle_type}: {e}")
                    continue
                    
            logger.info(f"Lazy range loading for {particle_type}: {total_collected} events loaded (range {start_event}-{end_event})")
            
        elif mode == 'random':
            # For random sampling, we need to count total events first, then sample
            # But we'll do it efficiently by reading only ev_starts, not data
            all_event_counts = []
            total_events = 0
            
            # First pass: count events per part (only read ev_starts)
            for part in parts:
                try:
                    ev_starts = particle_group['raw']['ev_starts'][part]['data'][:]
                    valid_events = sum(1 for i in range(len(ev_starts) - 1) 
                                     if ev_starts[i+1] > ev_starts[i])
                    all_event_counts.append((part, valid_events))
                    total_events += valid_events
                except Exception as e:
                    logger.warning(f"Error counting events in part {part} for {particle_type}: {e}")
                    all_event_counts.append((part, 0))
            
            # Determine how many events to sample
            n_to_sample = min(max_events or total_events, total_events)
            if n_to_sample == 0:
                return selected_indices
                
            # Random sampling of event indices
            sampled_global_indices = self.rng.choice(total_events, size=n_to_sample, replace=False)
            sampled_global_indices = sorted(sampled_global_indices)
            
            # Convert global indices to (part, local_index) and load only those
            global_idx = 0
            sample_idx = 0
            
            for part, part_event_count in all_event_counts:
                if sample_idx >= len(sampled_global_indices):
                    break
                    
                part_samples = []
                while (sample_idx < len(sampled_global_indices) and 
                       sampled_global_indices[sample_idx] < global_idx + part_event_count):
                    local_idx = sampled_global_indices[sample_idx] - global_idx
                    part_samples.append(local_idx)
                    sample_idx += 1
                
                if part_samples:
                    try:
                        ev_starts = particle_group['raw']['ev_starts'][part]['data'][:]
                        valid_event_idx = 0
                        for i in range(len(ev_starts) - 1):
                            start_idx = ev_starts[i]
                            end_idx = ev_starts[i + 1]
                            if end_idx > start_idx:  # Valid event
                                if valid_event_idx in part_samples:
                                    selected_indices.append((part, i, start_idx, end_idx))
                                valid_event_idx += 1
                    except Exception as e:
                        logger.warning(f"Error loading sampled events from part {part} for {particle_type}: {e}")
                
                global_idx += part_event_count
            
            logger.info(f"Lazy random sampling for {particle_type}: {len(selected_indices)} events from {total_events}")
            
        else:  # sequential or unknown
            # Fall back to old behavior but with early stopping
            for part in parts:
                if max_events and total_collected >= max_events:
                    break
                    
                try:
                    ev_starts = particle_group['raw']['ev_starts'][part]['data'][:]
                    for i in range(len(ev_starts) - 1):
                        if max_events and total_collected >= max_events:
                            break
                        start_idx = ev_starts[i]
                        end_idx = ev_starts[i + 1]
                        if end_idx > start_idx:  # Valid event
                            selected_indices.append((part, i, start_idx, end_idx))
                            total_collected += 1
                except Exception as e:
                    logger.warning(f"Error reading part {part} for {particle_type}: {e}")
                    continue
                    
            logger.info(f"Sequential loading for {particle_type}: {total_collected} events loaded")
        
        return selected_indices
    
    def _load_particle_data(
        self, 
        h5_file: h5py.File, 
        particle_type: str, 
        is_neutrino: bool,
        max_events_for_type: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[bool], List[int], List[int], List[str]]:
        """Load data for a specific particle type with advanced sampling options."""
        events = []
        labels = []
        magic_numbers = []
        channel_ids = []
        hit_counts = []
        event_ids = []
        
        particle_group = h5_file[particle_type]
        
        # Get all parts for this particle type
        parts = [key for key in particle_group['raw']['data'].keys() if key.startswith('part_')]
        
        # Get sampling config for this particle type
        sampling_cfg = self.sampling_config.get(particle_type, {})
        
        # Lazy loading: collect only the indices we need based on sampling strategy
        selected_indices = self._lazy_load_event_indices(
            particle_group, parts, particle_type, sampling_cfg, max_events_for_type
        )
        
        # Second pass: load selected events
        parts_data_cache = {}  # Cache loaded parts to avoid re-reading
        parts_magic_number_cache = {}
        parts_chanels_id_cache = {}
        for part, original_event_idx, start_idx, end_idx in selected_indices:
            try:
                # Load part data if not cached
                if part not in parts_data_cache:
                    parts_data_cache[part] = particle_group['raw']['data'][part]['data'][:]
                    parts_magic_number_cache[part] = particle_group['raw']['labels'][part]['data'][()]
                    parts_chanels_id_cache[part] = particle_group['raw']['channels'][part]['data'][()]
                
                hit_data = parts_data_cache[part]
                part_magic_numbers = parts_magic_number_cache[part]
                part_channel_id = parts_chanels_id_cache[part]
                event_hits = hit_data[start_idx:end_idx]
                event_magic_numbers = part_magic_numbers[start_idx:end_idx]
                event_channel_id = part_channel_id[start_idx:end_idx]
                
                # Apply max_hits limit if specified
                if self.max_hits is not None and len(event_hits) > self.max_hits:
                    event_hits = event_hits[:self.max_hits]
                    event_magic_numbers = event_magic_numbers[:self.max_hits]
                    event_channel_id = event_channel_id[:self.max_hits]
                
                # Convert to tensor
                event_tensor = torch.tensor(event_hits, dtype=torch.float32, device=self.device)
                #event_magic_numbers_tensor = torch.tensor(event_magic_numbers, dtype=torch.int32, device=self.device)
                
                # Generate unique event ID: particle_type:part_name:event_index
                event_id = f"{particle_type}:{part}:{original_event_idx}"
                
                events.append(event_tensor)
                magic_numbers.append(event_magic_numbers)
                channel_ids.append(event_channel_id)
                labels.append(is_neutrino)
                hit_counts.append(len(event_hits))
                event_ids.append(event_id)
                
            except Exception as e:
                logger.warning(f"Error loading event from part {part} for {particle_type}: {e}")
                continue
        
        logger.info(f"Loaded {len(events)} events for {particle_type}")
        return events, labels, magic_numbers, hit_counts, event_ids, channel_ids
    
    def _calculate_stats(self):
        """Calculate dataset statistics."""
        self.n_neutrino = self.labels.sum().item()
        self.n_muon = (~self.labels).sum().item()
        self.class_balance = self.n_neutrino / len(self) if len(self) > 0 else 0.0
        
        self.min_hits = self.hit_counts.min().item()
        self.max_hits = self.hit_counts.max().item()
        self.mean_hits = self.hit_counts.float().mean().item()
        self.std_hits = self.hit_counts.float().std().item()
    
    def __len__(self) -> int:
        """Return number of events in dataset."""
        return len(self.events)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single event.
        
        Returns:
            Dict with keys:
                'features': Tensor of shape (n_hits, 5) with hit features
                'labels': Boolean tensor indicating neutrino (True) or muon (False)
                'lengths': Integer tensor with actual sequence length
                'event_id': String identifier for the event
        """
        return {
            'features': self.events[idx],
            'labels': self.labels[idx],
            'magic_numbers': self.magic_numbers[idx],
            'lengths': self.hit_counts[idx],
            'event_id': self.event_ids[idx],
            'channels_ids': self.channel_ids[idx]
        }
    
    def _get_stats(self, with_data_stats=True, max_sample_size=1_000_000) -> Dict[str, float]:
        """Get dataset statistics."""
        if with_data_stats:
            logger.info("Calculating detailed feature statistics across dataset...")
            # Calculate feature statistics across all hits in the dataset
            total_hits = 0
            feature_sums = torch.zeros(5, dtype=torch.float64, device=self.device)
            feature_squared_sums = torch.zeros(5, dtype=torch.float64, device=self.device)
            
            for event in self.events[:max_sample_size]:
                n_hits = len(event)
                total_hits += n_hits
                # Sum features across hits for this event
                feature_sums += event.sum(dim=0).double()
                # Sum squared features for variance calculation
                feature_squared_sums += (event ** 2).sum(dim=0).double()
            
            # Calculate means and standard deviations
            means = (feature_sums / total_hits).float()
            variances = (feature_squared_sums / total_hits) - (means.double() ** 2)
            stds = torch.sqrt(torch.clamp(variances, min=0.0)).float()
            
            self.stats = {
                'total_events': len(self),
                'neutrino_events': self.n_neutrino,
                'muon_events': self.n_muon,
                'class_balance': self.class_balance,
                'total_hits': total_hits,
                'min_hits': self.min_hits,
                'max_hits': self.max_hits,
                'mean_hits': self.mean_hits,
                'std_hits': self.std_hits,
                'feature_means': means.cpu().numpy().tolist(),
                'feature_stds': stds.cpu().numpy().tolist(),
                'feature_names': ['amplitude', 'time', 'x', 'y', 'z']
            }
            return self.stats
        else:
            logger.info("Returning precomputed dataset statistics...")
            self.stats = {
                'total_events': len(self),
                'neutrino_events': self.n_neutrino,
                'muon_events': self.n_muon,
                'class_balance': self.class_balance,
                'min_hits': self.min_hits,
                'max_hits': self.max_hits,
                'mean_hits': self.mean_hits,
                'std_hits': self.std_hits
            }
            return self.stats
    
    def collate_fn(
        self, 
        batch: List[Dict[str, torch.Tensor]], 
        normalization_config: Optional[Dict[str, List[float]]] = None,
        shuffle_batch: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
        use_polar_coords: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for variable-length sequences with advanced preprocessing.
        
        Handles batching of variable-length time series with optional data augmentation,
        normalization, coordinate transformations, and memory-safe truncation.
        
        Args:
            batch: List of dicts from dataset __getitem__
                Each dict contains 'features', 'labels', 'lengths', 'event_id'
                Features tensor has shape (n_hits, 5) with [amplitude, time, x, y, z]
            normalization_config: Optional dict for feature normalization with keys:
                - 'means': List of 5 feature means [amp, time, x, y, z]
                - 'stds': List of 5 feature standard deviations
                Example: {'means': [1.0, 0.0, 0.0, 0.0, 50.0], 'stds': [2.0, 1000.0, 40.0, 40.0, 150.0]}
            shuffle_batch: Whether to shuffle events within the batch (default: True)
                Helps prevent overfitting to event ordering
            augmentation_config: Optional dict for data augmentation with keys:
                - 'noise_std': List of 5 Gaussian noise std deviations per feature
                - 'rotation_std': Standard deviation for random rotation angle (radians)
                Example: {'noise_std': [0.1, 10.0, 1.0, 1.0, 5.0], 'rotation_std': 0.1}
            use_polar_coords: If True, add polar coordinates (r, cos_α, sin_α) to existing Cartesian coordinates
                Output features become [amplitude, time, x, y, z, r, cos_α, sin_α] with 8 dimensions
                
        Returns:
            Dict[str, torch.Tensor] with the following keys:
            - 'features': Padded tensor of shape (batch_size, max_seq_len, feature_dim)
                feature_dim is 5 for Cartesian or 8 for polar coordinates
                Features: [amplitude, time, x, y, z] or [amplitude, time, x, y, z, r, cos_α, sin_α]
            - 'labels': Boolean tensor of shape (batch_size,) with neutrino labels
            - 'lengths': Long tensor of actual sequence lengths after truncation
            - 'original_lengths': Long tensor of original sequence lengths before truncation
            - 'mask': Boolean tensor of shape (batch_size, max_seq_len) indicating real data positions
                True for real hits, False for padding
            - 'hits_lost': Long tensor of hits lost per event due to max_hits truncation
                
        Note:
            - Sequences are automatically re-sorted by time after augmentation to maintain temporal order
            - Padding positions are masked out during normalization and augmentation
            - Memory usage is controlled by max_hits truncation before padding
            - Polar coordinate conversion adds r, cos(α), sin(α) as additional features
            
        Example:
            >>> batch = [{'features': torch.randn(50, 5), 'labels': True}, 
            ...          {'features': torch.randn(60, 5), 'labels': False}]
            >>> result = dataset.collate_fn(batch, normalization_config=norm_cfg)
            >>> print(result['features'].shape)  # torch.Size([2, 60, 5])
            >>> print(result['mask'].sum(dim=1))  # tensor([50, 60]) - actual lengths
        """
        # Optional batch shuffling
        if shuffle_batch:
            batch = list(batch)
            self.rng.shuffle(batch)
        
        # Extract features and labels from dict batch
        features = [item['features'] for item in batch]
        labels = [item['labels'] for item in batch]
        magic_numbers = [item['magic_numbers'] for item in batch]
        channels = [item['channels_ids'] for item in batch]
        
        # Get original sequence lengths
        original_lengths = torch.tensor([len(f) for f in features], dtype=torch.long, device=self.device)
        
        # Apply max_hits limit to prevent memory issues
        truncated_features = []
        hits_lost = torch.zeros(len(features), dtype=torch.long, device=self.device)
        
        for i, feat in enumerate(features):
            original_len = len(feat)
            if self.max_hits is not None and original_len > self.max_hits:
                # Truncate to max_hits
                truncated_feat = feat[:self.max_hits]
                hits_lost[i] = original_len - self.max_hits
            else:
                truncated_feat = feat
            truncated_features.append(truncated_feat)
        
        # Get sequence lengths after truncation
        lengths = torch.tensor([len(f) for f in truncated_features], dtype=torch.long, device=self.device)
        max_len = lengths.max().item()
        
        # Determine feature dimension based on coordinate system
        feature_dim = 5 #8 if use_polar_coords else 5
        
        # Pad sequences
        batch_size = len(truncated_features)
        padded_features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32, device=self.device)
        
        for i, feat in enumerate(truncated_features):
            seq_len = len(feat)
            padded_features[i, :seq_len] = feat
        
        # Create mask (True for real data, False for padding)
        mask = torch.arange(max_len, device=self.device)[None, :] < lengths[:, None]
        
        # Apply coordinate transformations and augmentations
        # Feature indices: [amplitude, time, x, y, z] = [0, 1, 2, 3, 4]
        
        # Apply random rotation augmentation if specified
        if augmentation_config is not None and 'rotation_enabled' in augmentation_config:
            rotation_enabled = augmentation_config['rotation_enabled']
            if rotation_enabled:
                # Generate uniform random rotation angles [0, 2π) for each event in the batch
                batch_size = padded_features.shape[0]
                rotation_angles = torch.rand(batch_size, device=self.device) * 2 * torch.pi
                
                # Apply rotation to x, y coordinates for each event
                for b in range(batch_size):
                    angle = rotation_angles[b]
                    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
                    
                    # Extract x, y coordinates for this event (only real hits)
                    event_mask = mask[b]  # Shape: (max_len,)
                    x_coords = padded_features[b, :, 2]  # x coordinates
                    y_coords = padded_features[b, :, 3]  # y coordinates
                    
                    # Apply rotation: [x', y'] = [[cos, -sin], [sin, cos]] * [x, y]
                    x_rotated = cos_a * x_coords - sin_a * y_coords
                    y_rotated = sin_a * x_coords + cos_a * y_coords
                    
                    # Update only real hits (preserve padding)
                    padded_features[b, :, 2] = torch.where(event_mask, x_rotated, x_coords)
                    padded_features[b, :, 3] = torch.where(event_mask, y_rotated, y_coords)
        
        # Apply Gaussian noise augmentation if provided
        if augmentation_config is not None and 'noise_std' in augmentation_config:
            noise_std = torch.tensor(
                augmentation_config['noise_std'],
                dtype=torch.float32,
                device=self.device
            )
            
            # Generate noise with same shape as padded_features
            noise = torch.randn_like(padded_features[:,:,:5]) * noise_std
            
            # Apply noise only to non-padded values
            padded_features = torch.where(
                mask.unsqueeze(-1),  # Real data positions
                padded_features + noise,  # Add noise to real data
                padded_features  # Keep padding unchanged
            )
            
            # Re-sort by time feature (index 1: [amplitude, time, x, y, z])
            time_index = 1
            # Get sorting indices based on time feature
            sort_indices = padded_features[:, :, time_index].argsort(dim=1)
            # Expand indices to all feature dimensions
            sort_indices_expanded = sort_indices.unsqueeze(-1).expand(-1, -1, feature_dim)
            # Apply sorting to maintain temporal order
            padded_features = padded_features.gather(dim=1, index=sort_indices_expanded)
            
            # Also need to re-sort the mask to match the new order
            mask = mask.gather(dim=1, index=sort_indices)
        
        # Add polar coordinates if requested
        if use_polar_coords:
            # Extract x, y coordinates
            x_coords = padded_features[:, :, 2]  # Shape: (batch_size, max_len)
            y_coords = padded_features[:, :, 3]  # Shape: (batch_size, max_len)
            
            # Calculate polar coordinates
            r = torch.sqrt(x_coords**2 + y_coords**2)  # Radius
            cos_a = x_coords/(r+1e-6)
            sin_a = y_coords/(r+1e-6)
            
            # Add r, sin(alpha) and cos(alpha) to batch features
            padded_features = torch.cat([padded_features, torch.zeros(batch_size, max_len, 3, device=self.device)], dim=-1)
            padded_features[:,:,5] = r*mask
            padded_features[:,:,6] = cos_a*mask
            padded_features[:,:,7] = sin_a*mask
        
        # Apply normalization if provided
        if normalization_config is not None:
            means = torch.tensor(
                normalization_config['means'], 
                dtype=torch.float32, 
                device=self.device
            )
            stds = torch.tensor(
                normalization_config['stds'], 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Handle polar coordinates normalization
            if use_polar_coords:
                # For polar coordinates: [amplitude, time, x, y, z, r, cos_α, sin_α]
                means = torch.cat([means, torch.tensor([30, 0.0, 0.0], device=self.device)])
                stds = torch.cat([stds, torch.tensor([30, 1.0, 1.0], device=self.device)])
                # # Create modified means/stds with alpha normalization disabled
                # means_modified = means.clone()
                # stds_modified = stds.clone()
                # means_modified[3] = 0.0  # Don't shift alpha (already centered around 0)
                # stds_modified[3] = 1   # Scale alpha by 2 (divide by 0.5) to normalize [-π, π] range
                
                padded_features = torch.where(
                    mask.unsqueeze(-1),
                    (padded_features - means) / (stds + 1e-8),
                    padded_features
                )
            else:
                # Standard normalization for Cartesian coordinates
                padded_features = torch.where(
                    mask.unsqueeze(-1),
                    (padded_features - means) / (stds + 1e-8),
                    padded_features
                )
        
        return {
            'features': padded_features,
            'labels': torch.stack([label.to(self.device) if hasattr(label, 'to') else torch.tensor(label, device=self.device) for label in labels]),
            'lengths': lengths,
            'original_lengths': original_lengths,
            'mask': mask,
            'hits_lost': hits_lost,
            'magic_numbers': magic_numbers,
            'channels_ids': channels
        }



def create_numu_dataloader(
    h5_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    particle_types: Optional[List[str]] = None,
    neutrino_types: Optional[List[str]] = None,
    max_hits: Optional[int] = 500,
    events_per_particle: Optional[Dict[str, int]] = None,
    sampling_config: Optional[Dict[str, Dict[str, Any]]] = None,
    num_workers: int = 0,
    device: str = 'cpu',
    seed: Optional[int] = None,
    shuffle_events: bool = True,
    normalization_config: Optional[Dict[str, List[float]]] = None,
    shuffle_batch: bool = True,
    augmentation_config: Optional[Dict[str, Any]] = None,
    use_polar_coords: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for neutrino vs muon binary classification with balanced batching.
    
    Args:
        h5_path: Path to HDF5 file (MC or experimental data)
        batch_size: Batch size for training
        shuffle: Whether to  to have the data reshuffled at every epoch
        particle_types: Particle types to include (e.g., ['muatm_2020', 'nue2_2020', 'exp'])
        neutrino_types: Which particles are neutrinos (positive class)
        max_hits: Maximum hits per event
        events_per_particle: Dict mapping particle types to max events
        sampling_config: Dict with sampling parameters per particle type
        num_workers: Number of workers for data loading
        device: PyTorch device
        seed: Random seed for reproducible sampling
        shuffle_events: Whether to shuffle events after loading to remove particle-type ordering bias
        normalization_config: Dict with 'means' and 'stds' lists for feature normalization
        shuffle_batch: Whether to shuffle events within each batch
        augmentation_config: Dict with 'noise_std' list for Gaussian noise augmentation per feature
        
    Returns:
        DataLoader with custom collate function and interleaved events for balanced batches
    """
    dataset = NuMuDataset(
        h5_path=h5_path,
        events_per_particle=events_per_particle,
        particle_types=particle_types,
        neutrino_types=neutrino_types,
        max_hits=max_hits,
        sampling_config=sampling_config,
        device=device,
        seed=seed,
        shuffle_events=shuffle_events
    )
    
    # Create wrapper collate function with normalization, batch shuffling, and augmentation
    def collate_wrapper(batch):
        return dataset.collate_fn(
            batch, 
            normalization_config=normalization_config, 
            shuffle_batch=shuffle_batch,
            augmentation_config=augmentation_config,
            use_polar_coords=use_polar_coords
        )
    
    # Events are already interleaved in dataset for balanced batches
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=(device == 'cuda' and num_workers > 0)
    )
    
    
def create_numu_dataloader_from_ds(
    dataset: NuMuDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    normalization_config: Optional[Dict[str, List[float]]] = None,
    shuffle_batch: bool = True,
    augmentation_config: Optional[Dict[str, Any]] = None,
    use_polar_coords: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from an existing NuMuDataset instance.
    
    Args:
        dataset: Pre-initialized NuMuDataset instance or Subset from train/val split
        batch_size: Batch size for training
        shuffle: Whether to  to have the data reshuffled at every epoch
        num_workers: Number of workers for data loading
        device: PyTorch device
        normalization_config: Dict with 'means' and 'stds' lists for feature normalization
        shuffle_batch: Whether to shuffle events within each batch
        augmentation_config: Dict with 'noise_std' list for Gaussian noise augmentation per feature
        use_polar_coords: If True, concat polar coordinates (r, cos_α, sin_α) to existing Cartesian coordinates
    """
    # Create wrapper collate function with normalization, batch shuffling, and augmentation
    def collate_wrapper(batch):
        # Handle both NuMuDataset and Subset (when train-val splitting) objects
        if hasattr(dataset, 'collate_fn'):
            underlying_dataset = dataset
        else:
            # Handle Subset objects from train/val splits
            underlying_dataset = dataset.dataset
            
        return underlying_dataset.collate_fn(
            batch, 
            normalization_config=normalization_config, 
            shuffle_batch=shuffle_batch,
            augmentation_config=augmentation_config,
            use_polar_coords=use_polar_coords
        )
    
    # Events are already interleaved in dataset for balanced batches
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=pin_memory
    )