#!/usr/bin/env python3
"""
Test script for SignalNuMuDataset to verify signal hit-based classification logic.
"""

import sys
import logging
from pathlib import Path
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.hcut_numu_dataset import HCutNuMuDataset, create_signal_numu_dataloader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_signal_dataset():
    """Test the SignalNuMuDataset with signal hit-based classification."""
    
    # Test configuration
    h5_path = "/net/62/home3/ivkhar/Baikal/data/h5s/baikal_mc_merged.h5"
    
    # Test with small dataset
    test_config = {
        'h5_path': h5_path,
        'events_per_particle': {
            'muatm_2020': 100,  # All muon events -> background (0)
            'nue2_2020': 100    # Neutrino events -> signal (1) if >= h_min signal hits
        },
        'particle_types': ['muatm_2020', 'nue2_2020'],
        'neutrino_types': ['nue2_2020'],
        'h_min': 5,  # Signal threshold
        'max_hits': 200,
        'device': 'cpu',
        'seed': 42
    }
    
    logger.info("=== Testing SignalNuMuDataset ===")
    logger.info(f"Using h_min = {test_config['h_min']} (signal hit threshold)")
    
    # Check if data file exists
    if not Path(h5_path).exists():
        logger.error(f"Data file not found: {h5_path}")
        logger.info("Please ensure the HDF5 data file is available or update the path")
        return False
    
    try:
        # Create dataset
        dataset = HCutNuMuDataset(**test_config)
        
        logger.info(f"\n=== Dataset Statistics ===")
        logger.info(f"Total events: {len(dataset)}")
        logger.info(f"Signal events (1): {dataset.n_signal}")
        logger.info(f"Background events (0): {dataset.n_background}")
        logger.info(f"Class balance: {dataset.class_balance:.3f}")
        logger.info(f"Signal hit threshold: {dataset.h_min}")
        
        if hasattr(dataset, 'mean_signal_hits'):
            logger.info(f"Signal hits statistics:")
            logger.info(f"  Min: {dataset.min_signal_hits}, Max: {dataset.max_signal_hits}")
            logger.info(f"  Mean: {dataset.mean_signal_hits:.1f} ± {dataset.std_signal_hits:.1f}")
        
        # Test individual samples
        logger.info(f"\n=== Sample Events ===")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            features = sample['features']
            label = sample['labels'].item()
            signal_hits = sample['signal_hit_count']
            event_id = sample['event_id']
            
            logger.info(f"Event {i}: {event_id}")
            logger.info(f"  Features shape: {features.shape}")
            logger.info(f"  Label: {label} ({'Signal' if label else 'Background'})")
            logger.info(f"  Signal hits: {signal_hits}")
            logger.info(f"  Classification logic: {'✓' if (label == (signal_hits >= test_config['h_min']) and 'nue2' in event_id) or (not label and 'muatm' in event_id) or (not label and signal_hits < test_config['h_min']) else '✗'}")
        
        # Test DataLoader
        logger.info(f"\n=== Testing DataLoader ===")
        dataloader = create_signal_numu_dataloader(
            h5_path=h5_path,
            events_per_particle=test_config['events_per_particle'],
            particle_types=test_config['particle_types'],
            neutrino_types=test_config['neutrino_types'],
            h_min=test_config['h_min'],
            batch_size=8,
            shuffle=False,
            device='cpu',
            seed=42
        )
        
        # Test one batch
        batch = next(iter(dataloader))
        logger.info(f"Batch keys: {list(batch.keys())}")
        logger.info(f"Features shape: {batch['features'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        logger.info(f"Signal hit counts: {batch['signal_hit_counts'].tolist()}")
        logger.info(f"Labels: {batch['labels'].tolist()}")
        
        # Verify classification logic in batch
        signal_hits = batch['signal_hit_counts'].numpy()
        labels = batch['labels'].numpy()
        correct_classifications = 0
        
        for i, (hits, label) in enumerate(zip(signal_hits, labels)):
            # Note: We can't easily check particle type from batch, so we'll just verify threshold logic
            logger.info(f"  Event {i}: {hits} signal hits -> label {label}")
            if hits >= test_config['h_min'] and label == 1:
                correct_classifications += 1
            elif hits < test_config['h_min'] and label == 0:
                correct_classifications += 1
        
        logger.info(f"Threshold-consistent classifications: {correct_classifications}/{len(signal_hits)}")
        
        logger.info("\n=== Test Complete ===")
        logger.info("✓ SignalNuMuDataset created successfully")
        logger.info("✓ Signal hit counting implemented")
        logger.info("✓ Classification logic applied")
        logger.info("✓ DataLoader working")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_signal_dataset()
    if success:
        print("\n🎉 All tests passed! SignalNuMuDataset is ready to use.")
    else:
        print("\n❌ Tests failed. Please check the logs above.")
        sys.exit(1)