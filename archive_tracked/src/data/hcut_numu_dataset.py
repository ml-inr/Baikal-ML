"""
Signal Nu-Mu Dataset for Signal Hit-Based Binary Classification

This dataset loads both Monte Carlo and experimental data from HDF5 files for the specific task
of signal hit-based binary classification:
- Background (0): All muon events + neutrino events with < h_min signal hits
- Signal (1): Neutrino events with >= h_min signal hits

Signal hits are defined as hits where magic_number != 0.
Supports both MC data (muatm, nue2, nuatm, etc.) and experimental data (exp).
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

logger = logging.getLogger(__name__)


MUONS_HGE5_PART = 0.46
SIGNAL_PART_NUATM = 0.325
SIGNAL_PART_NUE2 = 0.787

class HCutNuMuDataset(Dataset):
    """
    PyTorch Dataset for signal hit-based binary classification using Baikal detector data.
    
    Classification logic:
    - Background (0): All muon events (muatm_*) + neutrino events with < h_min signal hits
    - Signal (1): Neutrino events with >= h_min signal hits
    
    Signal hits are defined as hits where magic_number != 0.
    
    Features:
    - Variable-length time series (hit sequences per event)
    - 5D hit features: [amplitude, time, x, y, z]
    - Signal hit counting for neutrino event classification
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
        neutrino_types: Particle types considered as neutrinos (for signal counting). If None, defaults to:
            ['nuatm_2019', 'nuatm_2020', 'nue2_2019', 'nue2_2020']
        h_min: Minimum number of signal hits for neutrino events to be classified as signal (1). Default: 5
        max_hits: Maximum hits per event (for memory management and truncation). Default: 500
        sampling_config: Dict with sampling parameters per particle type. Example:
            {'muatm_2020': {'mode': 'range', 'start_event': 0, 'end_event': 1000},
             'nue2_2020': {'mode': 'random'}}
            Modes: 'range', 'random', 'sequential'
        device: PyTorch device for tensors ('cpu' or 'cuda'). Default: 'cpu'
        seed: Random seed for reproducible sampling and shuffling. Default: None
        shuffle_events: Whether to shuffle events after loading to remove particle-type ordering bias. Default: True
        balance_classes: If True, balance the dataset so that:
            1) signal_nuatm == signal_nue2
            2) total_background == total_signal
            3) muons with hits>=5 == (muons with hits<5 + background neutrinos)
            4) background neutrinos == muons with hits<5
            Preserves maximum signal events. Default: False

    Raises:
        FileNotFoundError: If h5_path does not exist
        TypeError: If events_per_particle is not a dictionary
        ValueError: If no data found for specified particle types
    
    Example:
        >>> dataset = SignalNuMuDataset(
        ...     h5_path="/path/to/data.h5",
        ...     events_per_particle={'muatm_2020': 1000, 'nue2_2020': 500},
        ...     particle_types=['muatm_2020', 'nue2_2020'],
        ...     neutrino_types=['nue2_2020'],
        ...     h_min=5,
        ...     max_hits=200,
        ...     device='cpu',
        ...     seed=42
        ... )
        >>> print(f"Dataset has {len(dataset)} events")
        >>> sample = dataset[0]  # Get first event
        >>> print(f"Event shape: {sample['features'].shape}, Label: {sample['labels']}")
    """
    
    def __init__(
        self,
        h5_path: Union[str, Path],
        events_per_particle: Optional[Dict[str, int]] = None,
        particle_types: Optional[List[str]] = None,
        neutrino_types: Optional[List[str]] = None,
        h_min: int = 5,
        max_hits: Optional[int] = 500,
        sampling_config: Optional[Dict[str, Dict[str, Any]]] = None,
        device: str = 'cpu',
        seed: Optional[int] = None,
        shuffle_events: bool = True,
        balance_classes: bool = False
    ):
        self.h5_path = Path(h5_path)
        self.device = device
        self.h_min = h_min
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
        
        # Default neutrino types (for signal counting)
        if neutrino_types is None:
            neutrino_types = ['nuatm_2019', 'nuatm_2020', 'nue2_2019', 'nue2_2020']
        self.neutrino_types = [name for name in neutrino_types if name in self.max_events_per_class.keys()] if self.max_events_per_class else neutrino_types
        
        # Default max num of events per class
        if self.max_events_per_class is None:
            self.max_events_per_class = {name: 10_000 for name in self.particle_types} # Default to 10000 per class if not specified
        
        # Load and prepare data
        self._load_data()
        logger.info(f"Loaded {len(self)} events from {len(self.particle_types)} particle types")
        logger.info(f"Signal events (1): {self.labels.sum()}, Background events (0): {(~self.labels).sum()}")
        logger.info(f"Signal hit threshold (h_min): {self.h_min}")

        # Optional class balancing
        if balance_classes:
            self.balance_classes()
        

    
    def _load_data(self):
        """Load data from HDF5 file and prepare for training."""
        all_events = []
        all_labels = []
        all_magic_numbers = []
        all_hit_counts = []
        all_event_ids = []
        all_channel_ids = []
        all_signal_hit_counts = []
        total_loaded = 0
        
        with h5py.File(self.h5_path, 'r') as f:
            for particle_type in self.particle_types:
                if particle_type not in f:
                    logger.warning(f"Particle type '{particle_type}' not found in {self.h5_path}")
                    continue
                
                # Determine if this particle type is neutrino (for signal counting)
                is_neutrino_type = particle_type in self.neutrino_types
                
                # Check per-class limit
                num_events_to_load = self.max_events_per_class[particle_type]
                
                # Load data for this particle type
                events, labels, magic_numbers, hit_counts, event_ids, channel_ids, signal_hit_counts = self._load_particle_data(
                    f, particle_type, is_neutrino_type, num_events_to_load
                )
                
                all_events.extend(events)
                all_labels.extend(labels)
                all_magic_numbers.extend(magic_numbers)
                all_hit_counts.extend(hit_counts)
                all_event_ids.extend(event_ids)
                all_channel_ids.extend(channel_ids)
                all_signal_hit_counts.extend(signal_hit_counts)
                total_loaded += len(events)
                
                # Log classification results for this particle type
                signal_events = sum(labels)
                background_events = len(labels) - signal_events
                logger.info(f"Loaded {len(events)} events from {particle_type} (neutrino_type={is_neutrino_type})")
                logger.info(f"  -> Signal (1): {signal_events}, Background (0): {background_events}")
                
                if is_neutrino_type and signal_hit_counts:
                    avg_signal_hits = np.mean(signal_hit_counts)
                    logger.info(f"  -> Average signal hits per event: {avg_signal_hits:.1f}")
        
        if not all_events:
            raise ValueError(f"No data found for particle types {self.particle_types}")
        
        # Optional shuffling to remove bias from group-by-group loading
        if self.shuffle_events:
            combined_data = list(zip(all_events, all_labels, all_magic_numbers, all_hit_counts, all_event_ids, all_channel_ids, all_signal_hit_counts))
            self.rng.shuffle(combined_data)
            all_events, all_labels, all_magic_numbers, all_hit_counts, all_event_ids, all_channel_ids, all_signal_hit_counts = zip(*combined_data)
            all_events, all_labels, all_magic_numbers, all_hit_counts, all_event_ids, all_channel_ids, all_signal_hit_counts = (
                list(all_events), list(all_labels), list(all_magic_numbers), list(all_hit_counts), list(all_event_ids), list(all_channel_ids), list(all_signal_hit_counts)
            )
            logger.info(f"Shuffled {len(combined_data)} events to remove particle-type ordering bias")
        
        # Store as class attributes
        self.events, self.labels, self.magic_numbers, self.hit_counts, self.event_ids, self.channel_ids, self.signal_hit_counts = self._organize_events_for_batching(
            all_events, all_labels, all_magic_numbers, all_hit_counts, all_event_ids, all_channel_ids, all_signal_hit_counts
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
        channel_ids: List[int],
        signal_hit_counts: List[int]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[int], torch.Tensor, List[str], List[int], List[int]]:
        """
        Organize events to ensure good mixing of classes in batches.
        Interleaves signal and background events to prevent single-class batches.
        """
        # Separate events by class
        nuhcut_events = []
        nuhcut_magic_numbers = []
        nuhcut_hit_counts = []
        nuhcut_ids = []
        nuhcut_channels = []
        nuhcut_signal_hit_counts = []
        background_events = []
        background_magic_numbers = []
        background_hit_counts = []
        background_ids = []
        background_channels = []
        background_signal_hit_counts = []
        
        for i, is_signal in enumerate(labels):
            if is_signal:
                nuhcut_events.append(events[i])
                nuhcut_magic_numbers.append(magic_numbers[i])
                nuhcut_hit_counts.append(hit_counts[i])
                nuhcut_ids.append(event_ids[i])
                nuhcut_channels.append(channel_ids[i])
                nuhcut_signal_hit_counts.append(signal_hit_counts[i])
            else:
                background_events.append(events[i])
                background_magic_numbers.append(magic_numbers[i])
                background_hit_counts.append(hit_counts[i])
                background_ids.append(event_ids[i])
                background_channels.append(channel_ids[i])
                background_signal_hit_counts.append(signal_hit_counts[i])
        
        # Interleave signal and background events for better batch mixing
        interleaved_events = []
        interleaved_labels = []
        interleaved_magic_numbers = []
        interleaved_hit_counts = []
        interleaved_ids = []
        interleaved_channels = []
        interleaved_signal_hit_counts = []
        
        max_len = max(len(nuhcut_events), len(background_events))
        
        for i in range(max_len):
            # Add signal event if available
            if i < len(nuhcut_events):
                interleaved_events.append(nuhcut_events[i])
                interleaved_labels.append(True)
                interleaved_magic_numbers.append(nuhcut_magic_numbers[i])
                interleaved_hit_counts.append(nuhcut_hit_counts[i])
                interleaved_ids.append(nuhcut_ids[i])
                interleaved_channels.append(nuhcut_channels[i])
                interleaved_signal_hit_counts.append(nuhcut_signal_hit_counts[i])
            
            # Add background event if available
            if i < len(background_events):
                interleaved_events.append(background_events[i])
                interleaved_labels.append(False)
                interleaved_magic_numbers.append(background_magic_numbers[i])
                interleaved_hit_counts.append(background_hit_counts[i])
                interleaved_ids.append(background_ids[i])
                interleaved_channels.append(background_channels[i])
                interleaved_signal_hit_counts.append(background_signal_hit_counts[i])
        
        logger.info(f"Interleaved {len(nuhcut_events)} signal and {len(background_events)} background events for balanced batches")
        
        return (
            interleaved_events,
            torch.tensor(interleaved_labels, dtype=torch.bool, device=self.device),
            interleaved_magic_numbers,
            torch.tensor(interleaved_hit_counts, dtype=torch.long, device=self.device),
            interleaved_ids,
            interleaved_channels,
            interleaved_signal_hit_counts
        )
    
    def _count_signal_hits(self, magic_numbers: List[int]) -> int:
        """Count signal hits in an event (magic_number != 0)."""
        return sum(1 for magic_num in magic_numbers if magic_num != 0)
    
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
        is_neutrino_type: bool,
        max_events_for_type: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[bool], List[int], List[int], List[str], List[int], List[int]]:
        """Load data for a specific particle type with signal hit-based classification."""
        events = []
        labels = []
        magic_numbers = []
        channel_ids = []
        hit_counts = []
        event_ids = []
        signal_hit_counts = []
        
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
                
                # Count signal hits BEFORE truncation (this is key for the new classification)
                signal_hit_count = self._count_signal_hits(event_magic_numbers)
                
                # Determine label based on signal hit count and particle type
                if is_neutrino_type:
                    # Neutrino events: label = 1 if signal_hit_count >= h_min, else 0
                    is_signal = signal_hit_count >= self.h_min
                else:
                    # Muon events: always background (label = 0)
                    is_signal = False
                
                # Apply max_hits limit if specified (AFTER signal counting)
                if self.max_hits is not None and len(event_hits) > self.max_hits:
                    event_hits = event_hits[:self.max_hits]
                    event_magic_numbers = event_magic_numbers[:self.max_hits]
                    event_channel_id = event_channel_id[:self.max_hits]
                
                # Convert to tensor
                event_tensor = torch.tensor(event_hits, dtype=torch.float32, device=self.device)
                
                # Generate unique event ID: particle_type:part_name:event_index
                event_id = f"{particle_type}:{part}:{original_event_idx}"
                
                events.append(event_tensor)
                magic_numbers.append(event_magic_numbers)
                channel_ids.append(event_channel_id)
                labels.append(is_signal)
                hit_counts.append(len(event_hits))
                event_ids.append(event_id)
                signal_hit_counts.append(signal_hit_count)
                
            except Exception as e:
                logger.warning(f"Error loading event from part {part} for {particle_type}: {e}")
                continue
        
        logger.info(f"Loaded {len(events)} events for {particle_type}")
        return events, labels, magic_numbers, hit_counts, event_ids, channel_ids, signal_hit_counts
    
    def _calculate_stats(self):
        """Calculate dataset statistics."""
        self.n_signal = self.labels.sum().item()
        self.n_background = (~self.labels).sum().item()
        self.class_balance = self.n_signal / len(self) if len(self) > 0 else 0.0
        
        self.min_hits = self.hit_counts.min().item()
        self.max_hits = self.hit_counts.max().item()
        self.mean_hits = self.hit_counts.float().mean().item()
        self.std_hits = self.hit_counts.float().std().item()
        
        # Signal hit statistics
        if self.signal_hit_counts:
            self.min_signal_hits = min(self.signal_hit_counts)
            self.max_signal_hits = max(self.signal_hit_counts)
            self.mean_signal_hits = np.mean(self.signal_hit_counts)
            self.std_signal_hits = np.std(self.signal_hit_counts)
        else:
            self.min_signal_hits = self.max_signal_hits = self.mean_signal_hits = self.std_signal_hits = 0

    def balance_classes(self) -> Dict[str, int]:
        """
        Balance the dataset to achieve:
        1. Equal number of signal 'nuatm' and signal 'nue2' events
        2. Total background events = Total signal events
        3. Within background:
           a) muons with signal_hits >= 5 = (muons with hits < 5 + background neutrinos)
           b) background neutrinos (nuatm + nue2) = muons with signal_hits < 5
        4. Preserve as many signal events as possible

        This results in:
        - muons_high = n_signal_per_type
        - muons_low = n_signal_per_type // 2
        - background_nu = n_signal_per_type // 2 (split equally between nuatm and nue2)
        - total_background = 2 * n_signal_per_type

        Returns:
            Dict with statistics about the balancing operation
        """
        logger.info("Balancing dataset classes...")

        # Group events by label, particle type, and signal hits
        signal_nuatm_indices = []
        signal_nue2_indices = []
        bg_muon_high_indices = []  # muons with signal_hits >= 5
        bg_muon_low_indices = []   # muons with signal_hits < 5
        bg_nuatm_indices = []      # background nuatm (signal_hits < h_min)
        bg_nue2_indices = []       # background nue2 (signal_hits < h_min)

        for i in range(len(self.events)):
            event_id = self.event_ids[i]
            particle_type = event_id.split(':')[0]  # Format: "particle_type:part:event_index"
            is_signal = self.labels[i].item()
            signal_hits = self.signal_hit_counts[i]

            if is_signal:
                # Signal events (neutrinos with signal_hits >= h_min)
                if 'nuatm' in particle_type:
                    signal_nuatm_indices.append(i)
                elif 'nue2' in particle_type:
                    signal_nue2_indices.append(i)
                else:
                    signal_nuatm_indices.append(i)
            else:
                # Background events
                if 'muatm' in particle_type or 'mu' in particle_type:
                    # Muon events - split by signal hits
                    if signal_hits >= 5:
                        bg_muon_high_indices.append(i)
                    else:
                        bg_muon_low_indices.append(i)
                elif 'nuatm' in particle_type:
                    bg_nuatm_indices.append(i)
                elif 'nue2' in particle_type:
                    bg_nue2_indices.append(i)
                else:
                    # Unknown background - treat as muon_low
                    bg_muon_low_indices.append(i)

        # Statistics before balancing
        n_signal_nuatm_before = len(signal_nuatm_indices)
        n_signal_nue2_before = len(signal_nue2_indices)
        n_bg_muon_high_before = len(bg_muon_high_indices)
        n_bg_muon_low_before = len(bg_muon_low_indices)
        n_bg_nuatm_before = len(bg_nuatm_indices)
        n_bg_nue2_before = len(bg_nue2_indices)
        n_background_before = n_bg_muon_high_before + n_bg_muon_low_before + n_bg_nuatm_before + n_bg_nue2_before

        logger.info(f"Before balancing:")
        logger.info(f"  Signal: nuatm={n_signal_nuatm_before}, nue2={n_signal_nue2_before}")
        logger.info(f"  Background: muon_high={n_bg_muon_high_before}, muon_low={n_bg_muon_low_before}, "
                   f"nuatm={n_bg_nuatm_before}, nue2={n_bg_nue2_before}")

        # Step 1: Equalize signal types (keep min of both)
        n_signal_per_type = min(n_signal_nuatm_before, n_signal_nue2_before)

        if n_signal_per_type == 0:
            logger.warning("No signal events of one type - cannot balance")
            return {'balanced': False}

        # Step 2: Calculate background quotas based on constraints
        # muons_high = n_signal_per_type
        # muons_low = n_signal_per_type // 2
        # background_nu = n_signal_per_type // 2 (split between nuatm and nue2)
        n_muon_high_target = n_signal_per_type
        n_muon_low_target = n_signal_per_type // 2
        n_bg_nu_target = n_signal_per_type // 2  # total for both nuatm and nue2
        n_bg_nuatm_target = n_bg_nu_target // 2
        n_bg_nue2_target = n_bg_nu_target - n_bg_nuatm_target  # handle odd numbers

        # Check if we have enough events in each category
        n_muon_high_final = min(n_muon_high_target, n_bg_muon_high_before)
        n_muon_low_final = min(n_muon_low_target, n_bg_muon_low_before)
        n_bg_nuatm_final = min(n_bg_nuatm_target, n_bg_nuatm_before)
        n_bg_nue2_final = min(n_bg_nue2_target, n_bg_nue2_before)

        # Adjust signal if background is limiting
        limiting_factor = min(
            n_muon_high_final / n_muon_high_target if n_muon_high_target > 0 else 1.0,
            n_muon_low_final / n_muon_low_target if n_muon_low_target > 0 else 1.0,
            n_bg_nuatm_final / n_bg_nuatm_target if n_bg_nuatm_target > 0 else 1.0,
            n_bg_nue2_final / n_bg_nue2_target if n_bg_nue2_target > 0 else 1.0
        )

        if limiting_factor < 1.0:
            logger.warning(f"Insufficient background events, scaling by {limiting_factor:.3f}")
            n_signal_per_type = int(n_signal_per_type * limiting_factor)
            n_muon_high_final = n_signal_per_type
            n_muon_low_final = n_signal_per_type // 2
            n_bg_nu_final = n_signal_per_type // 2
            n_bg_nuatm_final = n_bg_nu_final // 2
            n_bg_nue2_final = n_bg_nu_final - n_bg_nuatm_final

        # Randomly select from each category
        self.rng.shuffle(signal_nuatm_indices)
        self.rng.shuffle(signal_nue2_indices)
        self.rng.shuffle(bg_muon_high_indices)
        self.rng.shuffle(bg_muon_low_indices)
        self.rng.shuffle(bg_nuatm_indices)
        self.rng.shuffle(bg_nue2_indices)

        selected_signal_nuatm = signal_nuatm_indices[:n_signal_per_type]
        selected_signal_nue2 = signal_nue2_indices[:n_signal_per_type]
        selected_bg_muon_high = bg_muon_high_indices[:n_muon_high_final]
        selected_bg_muon_low = bg_muon_low_indices[:n_muon_low_final]
        selected_bg_nuatm = bg_nuatm_indices[:n_bg_nuatm_final]
        selected_bg_nue2 = bg_nue2_indices[:n_bg_nue2_final]

        # Combine all selected indices
        selected_indices = (selected_signal_nuatm + selected_signal_nue2 +
                          selected_bg_muon_high + selected_bg_muon_low +
                          selected_bg_nuatm + selected_bg_nue2)

        # Extract selected events
        new_events = [self.events[i] for i in selected_indices]
        new_labels = [self.labels[i].item() for i in selected_indices]
        new_magic_numbers = [self.magic_numbers[i] for i in selected_indices]
        new_hit_counts = [self.hit_counts[i].item() for i in selected_indices]
        new_event_ids = [self.event_ids[i] for i in selected_indices]
        new_channel_ids = [self.channel_ids[i] for i in selected_indices]
        new_signal_hit_counts = [self.signal_hit_counts[i] for i in selected_indices]

        # Re-organize for balanced batching (interleave signal and background)
        self.events, self.labels, self.magic_numbers, self.hit_counts, self.event_ids, self.channel_ids, self.signal_hit_counts = self._organize_events_for_batching(
            new_events, new_labels, new_magic_numbers, new_hit_counts, new_event_ids, new_channel_ids, new_signal_hit_counts
        )

        # Recalculate statistics
        self._calculate_stats()

        # Calculate final counts
        total_signal = 2 * n_signal_per_type
        total_background = n_muon_high_final + n_muon_low_final + n_bg_nuatm_final + n_bg_nue2_final

        # Statistics after balancing
        stats = {
            'signal_nuatm_before': n_signal_nuatm_before,
            'signal_nue2_before': n_signal_nue2_before,
            'bg_muon_high_before': n_bg_muon_high_before,
            'bg_muon_low_before': n_bg_muon_low_before,
            'bg_nuatm_before': n_bg_nuatm_before,
            'bg_nue2_before': n_bg_nue2_before,
            'signal_nuatm_after': n_signal_per_type,
            'signal_nue2_after': n_signal_per_type,
            'bg_muon_high_after': n_muon_high_final,
            'bg_muon_low_after': n_muon_low_final,
            'bg_nuatm_after': n_bg_nuatm_final,
            'bg_nue2_after': n_bg_nue2_final,
            'total_signal_after': total_signal,
            'total_background_after': total_background,
            'total_events_after': len(self.events),
            'balanced': True
        }

        logger.info(f"After balancing:")
        logger.info(f"  Signal: nuatm={n_signal_per_type}, nue2={n_signal_per_type} (total={total_signal})")
        logger.info(f"  Background: muon_high={n_muon_high_final}, muon_low={n_muon_low_final}, "
                   f"nuatm={n_bg_nuatm_final}, nue2={n_bg_nue2_final} (total={total_background})")
        logger.info(f"Total events: {len(self.events)}")

        # Verify constraints
        rest_of_bg = n_muon_low_final + n_bg_nuatm_final + n_bg_nue2_final
        logger.info(f"Constraint check: muon_high({n_muon_high_final}) ~= rest_of_bg({rest_of_bg}): {abs(n_muon_high_final - rest_of_bg) < 0.01 * max(rest_of_bg, rest_of_bg)}")
        logger.info(f"Constraint check: bg_nu({n_bg_nuatm_final + n_bg_nue2_final}) ~= muon_low({n_muon_low_final}): {abs(n_bg_nuatm_final + n_bg_nue2_final - n_muon_low_final) < 0.01 * max(n_bg_nuatm_final + n_bg_nue2_final, n_muon_low_final)}")

        return stats

    def __len__(self) -> int:
        """Return number of events in dataset."""
        return len(self.events)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single event.
        
        Returns:
            Dict with keys:
                'features': Tensor of shape (n_hits, 5) with hit features
                'labels': Boolean tensor indicating signal (True) or background (False)
                'lengths': Integer tensor with actual sequence length
                'event_id': String identifier for the event
                'signal_hit_count': Number of signal hits (magic_number != 0) in original event
        """
        return {
            'features': self.events[idx],
            'labels': self.labels[idx],
            'magic_numbers': self.magic_numbers[idx],
            'lengths': self.hit_counts[idx],
            'event_id': self.event_ids[idx],
            'channels_ids': self.channel_ids[idx],
            'signal_hit_count': self.signal_hit_counts[idx]
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
                'signal_events': self.n_signal,
                'background_events': self.n_background,
                'class_balance': self.class_balance,
                'h_min_threshold': self.h_min,
                'total_hits': total_hits,
                'min_hits': self.min_hits,
                'max_hits': self.max_hits,
                'mean_hits': self.mean_hits,
                'std_hits': self.std_hits,
                'min_signal_hits': self.min_signal_hits,
                'max_signal_hits': self.max_signal_hits,
                'mean_signal_hits': self.mean_signal_hits,
                'std_signal_hits': self.std_signal_hits,
                'feature_means': means.cpu().numpy().tolist(),
                'feature_stds': stds.cpu().numpy().tolist(),
                'feature_names': ['amplitude', 'time', 'x', 'y', 'z']
            }
            return self.stats
        else:
            logger.info("Returning precomputed dataset statistics...")
            self.stats = {
                'total_events': len(self),
                'signal_events': self.n_signal,
                'background_events': self.n_background,
                'class_balance': self.class_balance,
                'h_min_threshold': self.h_min,
                'min_hits': self.min_hits,
                'max_hits': self.max_hits,
                'mean_hits': self.mean_hits,
                'std_hits': self.std_hits,
                'min_signal_hits': self.min_signal_hits,
                'max_signal_hits': self.max_signal_hits,
                'mean_signal_hits': self.mean_signal_hits,
                'std_signal_hits': self.std_signal_hits
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
        
        Same functionality as NuMuDataset.collate_fn but with additional signal hit tracking.
        
        Returns:
            Dict[str, torch.Tensor] with the same keys as NuMuDataset plus:
            - 'signal_hit_counts': Long tensor of original signal hit counts per event
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
        signal_hit_counts = [item['signal_hit_count'] for item in batch]
        
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
        
        # Determine feature dimension
        feature_dim = 5
        
        # Pad sequences
        batch_size = len(truncated_features)
        padded_features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32, device=self.device)
        
        for i, feat in enumerate(truncated_features):
            seq_len = len(feat)
            padded_features[i, :seq_len] = feat
        
        # Create mask (True for real data, False for padding)
        mask = torch.arange(max_len, device=self.device)[None, :] < lengths[:, None]
        
        # Apply coordinate transformations and augmentations (same as NuMuDataset)
        # [Rest of augmentation logic identical to NuMuDataset.collate_fn...]
        
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
            feature_dim = 8
        
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
            'channels_ids': channels,
            'signal_hit_counts': torch.tensor(signal_hit_counts, dtype=torch.long, device=self.device)
        }


def create_hcut_numu_dataloader(
    h5_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    particle_types: Optional[List[str]] = None,
    neutrino_types: Optional[List[str]] = None,
    h_min: int = 5,
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
    Create a DataLoader for signal hit-based binary classification with balanced batching.
    
    Classification logic:
    - Background (0): All muon events + neutrino events with < h_min signal hits
    - Signal (1): Neutrino events with >= h_min signal hits
    
    Args:
        h5_path: Path to HDF5 file (MC or experimental data)
        batch_size: Batch size for training
        shuffle: Whether to have the data reshuffled at every epoch
        particle_types: Particle types to include (e.g., ['muatm_2020', 'nue2_2020', 'exp'])
        neutrino_types: Which particles are neutrinos (for signal counting)
        h_min: Minimum signal hits for neutrino events to be classified as signal
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
        use_polar_coords: If True, add polar coordinates (r, cos_α, sin_α) to existing Cartesian coordinates
        
    Returns:
        DataLoader with custom collate function and interleaved events for balanced batches
    """
    dataset = HCutNuMuDataset(
        h5_path=h5_path,
        events_per_particle=events_per_particle,
        particle_types=particle_types,
        neutrino_types=neutrino_types,
        h_min=h_min,
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
    
    
def create_hcut_numu_dataloader_from_ds(
    dataset: HCutNuMuDataset,
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
    Create a DataLoader from an existing SignalNuMuDataset instance.
    
    Args:
        dataset: Pre-initialized SignalNuMuDataset instance or Subset from train/val split
        batch_size: Batch size for training
        shuffle: Whether to have the data reshuffled at every epoch
        num_workers: Number of workers for data loading
        normalization_config: Dict with 'means' and 'stds' lists for feature normalization
        shuffle_batch: Whether to shuffle events within each batch
        augmentation_config: Dict with 'noise_std' list for Gaussian noise augmentation per feature
        use_polar_coords: If True, concat polar coordinates (r, cos_α, sin_α) to existing Cartesian coordinates
        pin_memory: Whether to use pinned memory for faster GPU transfer
    """
    # Create wrapper collate function with normalization, batch shuffling, and augmentation
    def collate_wrapper(batch):
        # Handle both SignalNuMuDataset and Subset (when train-val splitting) objects
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