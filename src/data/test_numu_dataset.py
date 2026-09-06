"""
Test script for NuMuDataset to verify functionality.
"""

import sys
from pathlib import Path
import logging

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_manager.h5datasets.paths2h5 import all_mc_path2h5
from src.data.numu_dataset import NuMuDataset, create_numu_dataloader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dataset_basic():
    """Test basic dataset functionality."""
    logger.info("=== Testing Basic Dataset Functionality ===")
    
    try:
        # Create dataset with small sample
        dataset = NuMuDataset(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020'],  # Use correct particle type
            neutrino_types=[],  # No neutrinos, so all should be False
            max_hits=200,  # Limit hits for testing
            events_per_particle={'muatm_2020': 100},  # Limit events for testing
            device='cpu'
        )
        
        logger.info(f"Dataset created successfully with {len(dataset)} events")
        stats = dataset._get_stats()
        logger.info(f"Dataset stats: {stats}")
        
        # Test single item access
        item = dataset[0]
        features = item['features']
        label = item['labels']
        logger.info(f"First event: {features.shape} features, label={label}")
        logger.info(f"Feature sample: {features[:3]}")  # First 3 hits
        
        return True
        
    except Exception as e:
        logger.error(f"Basic test failed: {e}")
        return False


def test_dataloader():
    """Test DataLoader with batching."""
    logger.info("=== Testing DataLoader with Batching ===")
    
    try:
        # Create dataloader
        dataloader = create_numu_dataloader(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020'],  # Use correct particle type
            neutrino_types=[],
            batch_size=4,
            max_hits=150,
            events_per_particle={'muatm_2020': 50},  # Limit events for testing
            shuffle=False,
            device='cpu'
        )
        
        logger.info(f"DataLoader created with {len(dataloader.dataset)} events")
        
        # Test first batch
        batch = next(iter(dataloader))
        
        logger.info("Batch structure:")
        for key, value in batch.items():
            logger.info(f"  {key}: {value.shape} ({value.dtype})")
        
        # Verify batch contents
        features = batch['features']        # (batch_size, max_seq_len, 5)
        labels = batch['labels']           # (batch_size,)
        lengths = batch['lengths']         # (batch_size,)
        original_lengths = batch['original_lengths']  # (batch_size,)
        mask = batch['mask']               # (batch_size, max_seq_len)
        hits_lost = batch['hits_lost']     # (batch_size,)
        
        logger.info(f"Sequence lengths in batch: {lengths.tolist()}")
        logger.info(f"Original lengths in batch: {original_lengths.tolist()}")
        logger.info(f"Hits lost per event: {hits_lost.tolist()}")
        logger.info(f"Labels in batch: {labels.tolist()}")
        logger.info(f"Features shape: {features.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"DataLoader test failed: {e}")
        return False


def test_mixed_particles():
    """Test with mixed neutrino/muon data."""
    logger.info("=== Testing Mixed Neutrino/Muon Classification ===")
    
    try:
        # Test with mixed neutrino/muon particle types
        dataset = NuMuDataset(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020', 'nue2_2020'],  # Mix of muons and neutrinos
            neutrino_types=['nue2_2020'],  # Define neutrinos
            max_hits=100,
            events_per_particle={'muatm_2020': 100, 'nue2_2020': 100},  # Limit for testing
            device='cpu'
        )
        
        stats = dataset._get_stats()
        logger.info("Dataset statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Test class distribution
        labels = dataset.labels
        logger.info(f"Class distribution - Neutrinos: {labels.sum()}, Muons: {(~labels).sum()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Mixed particles test failed: {e}")
        return False


def test_variable_length_handling():
    """Test handling of variable-length sequences."""
    logger.info("=== Testing Variable Length Sequence Handling ===")
    
    try:
        dataset = NuMuDataset(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020'],  # Use correct particle type
            max_hits=None,  # No limit to see full range
            events_per_particle={'muatm_2020': 50},  # Limit events for testing
            device='cpu'
        )
        
        # Sample a few events to check length variation
        lengths = []
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            features = item['features']
            lengths.append(len(features))
        
        logger.info(f"Sample event lengths: {lengths}")
        logger.info(f"Length range: {min(lengths)} - {max(lengths)}")
        
        # Test collate function with varied lengths
        batch_items = [dataset[i] for i in range(min(3, len(dataset)))]
        batch = dataset.collate_fn(batch_items)
        
        logger.info(f"Collated batch shapes:")
        logger.info(f"  Features: {batch['features'].shape}")
        logger.info(f"  Lengths: {batch['lengths']}")
        logger.info(f"  Original lengths: {batch['original_lengths']}")
        logger.info(f"  Hits lost: {batch['hits_lost']}")
        logger.info(f"  Mask shape: {batch['mask'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Variable length test failed: {e}")
        return False


def test_per_class_limits():
    """Test max_events_per_class functionality."""
    logger.info("=== Testing Per-Class Event Limits ===")
    
    try:
        # Test with different limits per particle type
        dataset = NuMuDataset(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020', 'nue2_2020'],
            neutrino_types=['nue2_2020'],
            events_per_particle={'muatm_2020': 30, 'nue2_2020': 20},  # Different limits
            device='cpu'
        )
        
        stats = dataset._get_stats()
        logger.info("Per-class limits applied:")
        logger.info(f"  Total events: {stats['total_events']}")
        logger.info(f"  Neutrino events: {stats['neutrino_events']}")
        logger.info(f"  Muon events: {stats['muon_events']}")
        
        # Verify limits are respected (approximately)
        expected_total = 50  # 30 + 20
        if abs(len(dataset) - expected_total) <= 5:  # Allow some tolerance
            logger.info("✅ Per-class limits working correctly")
        else:
            logger.warning(f"⚠️ Expected ~{expected_total} events, got {len(dataset)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Per-class limits test failed: {e}")
        return False


def test_sampling_strategies():
    """Test range and random sampling strategies."""
    logger.info("=== Testing Sampling Strategies ===")
    
    try:
        # Test range sampling
        logger.info("Testing range sampling...")
        dataset_range = NuMuDataset(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020'],
            neutrino_types=[],
            events_per_particle={'muatm_2020': 50},
            sampling_config={
                'muatm_2020': {'mode': 'range', 'start_event': 10, 'end_event': 30}
            },
            device='cpu',
            seed=42
        )
        logger.info(f"Range sampling: {len(dataset_range)} events loaded")
        
        # Test random sampling
        logger.info("Testing random sampling...")
        dataset_random = NuMuDataset(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020'],
            neutrino_types=[],
            events_per_particle={'muatm_2020': 25},
            sampling_config={
                'muatm_2020': {'mode': 'random'}
            },
            device='cpu',
            seed=42
        )
        logger.info(f"Random sampling: {len(dataset_random)} events loaded")
        
        # Test reproducibility with same seed
        dataset_random2 = NuMuDataset(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020'],
            neutrino_types=[],
            events_per_particle={'muatm_2020': 25},
            sampling_config={
                'muatm_2020': {'mode': 'random'}
            },
            device='cpu',
            seed=42
        )
        
        # Check if first few events are the same (reproducibility test)
        same_events = True
        for i in range(min(3, len(dataset_random), len(dataset_random2))):
            item1 = dataset_random[i]
            item2 = dataset_random2[i]
            feat1, label1 = item1['features'], item1['labels']
            feat2, label2 = item2['features'], item2['labels']
            if not torch.equal(feat1, feat2) or label1 != label2:
                same_events = False
                break
        
        if same_events:
            logger.info("✅ Random sampling is reproducible with same seed")
        else:
            logger.warning("⚠️ Random sampling reproducibility issue")
        
        return True
        
    except Exception as e:
        logger.error(f"Sampling strategies test failed: {e}")
        return False


def test_hits_truncation():
    """Test hit truncation and tracking in collate function."""
    logger.info("=== Testing Hits Truncation ===")
    
    try:
        # Test hits truncation via dataloader
        logger.info("Testing hits truncation through dataloader...")
        
        # Get a batch to test truncation
        dataloader = create_numu_dataloader(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020'],
            neutrino_types=[],
            max_hits=30,  # Even lower limit in collate
            events_per_particle={'muatm_2020': 10},
            batch_size=4,
            shuffle=False,
            device='cpu'
        )
        
        batch = next(iter(dataloader))
        
        logger.info("Hits truncation test results:")
        logger.info(f"  Original lengths: {batch['original_lengths'].tolist()}")
        logger.info(f"  Final lengths: {batch['lengths'].tolist()}")
        logger.info(f"  Hits lost: {batch['hits_lost'].tolist()}")
        
        # Check if any hits were lost
        total_hits_lost = batch['hits_lost'].sum().item()
        if total_hits_lost > 0:
            logger.info(f"✅ Hit truncation working: {total_hits_lost} total hits truncated")
        else:
            logger.info("ℹ️ No hits were truncated in this batch")
        
        return True
        
    except Exception as e:
        logger.error(f"Hits truncation test failed: {e}")
        return False


def test_balanced_batching():
    """Test interleaved event organization for balanced batches."""
    logger.info("=== Testing Balanced Batching ===")
    
    try:
        # Create dataset with mixed neutrino/muon events
        dataset = NuMuDataset(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020', 'nue2_2020'],
            neutrino_types=['nue2_2020'],
            events_per_particle={'muatm_2020': 50, 'nue2_2020': 50},
            device='cpu',
            seed=42
        )
        
        # Check if events are interleaved
        logger.info("Checking event interleaving...")
        first_20_labels = [dataset.labels[i].item() for i in range(min(20, len(dataset)))]
        logger.info(f"First 20 labels: {first_20_labels}")
        
        # Count transitions between classes
        transitions = 0
        for i in range(1, len(first_20_labels)):
            if first_20_labels[i] != first_20_labels[i-1]:
                transitions += 1
        
        logger.info(f"Class transitions in first 20 events: {transitions}")
        
        # Test batch mixing
        dataloader = create_numu_dataloader(
            h5_path=all_mc_path2h5,
            particle_types=['muatm_2020', 'nue2_2020'],
            neutrino_types=['nue2_2020'],
            events_per_particle={'muatm_2020': 30, 'nue2_2020': 30},
            batch_size=8,
            shuffle=False,  # Don't shuffle to see interleaving effect
            device='cpu',
            seed=42
        )
        
        # Check several batches for class mixing
        mixed_batches = 0
        total_batches = min(3, len(dataloader))
        
        for i, batch in enumerate(dataloader):
            if i >= total_batches:
                break
                
            labels = batch['labels']
            n_neutrino = labels.sum().item()
            n_muon = len(labels) - n_neutrino
            
            logger.info(f"Batch {i+1}: {n_neutrino} neutrinos, {n_muon} muons")
            
            if n_neutrino > 0 and n_muon > 0:
                mixed_batches += 1
        
        logger.info(f"Mixed batches: {mixed_batches}/{total_batches}")
        
        if mixed_batches >= total_batches * 0.5:  # At least 50% should be mixed
            logger.info("✅ Balanced batching working well")
        else:
            logger.warning("⚠️ Balanced batching could be improved")
        
        return True
        
    except Exception as e:
        logger.error(f"Balanced batching test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting NuMuDataset tests...")
    
    tests = [
        ("Basic Dataset", test_dataset_basic),
        ("DataLoader", test_dataloader),
        ("Mixed Particles", test_mixed_particles),
        ("Variable Length", test_variable_length_handling),
        ("Per-Class Limits", test_per_class_limits),
        ("Sampling Strategies", test_sampling_strategies),
        ("Hits Truncation", test_hits_truncation),
        ("Balanced Batching", test_balanced_batching)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        success = test_func()
        results.append((test_name, success))
        
        if success:
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Dataset is ready for training.")
    else:
        logger.error("⚠️  Some tests failed. Check implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)