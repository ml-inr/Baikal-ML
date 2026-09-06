#!/usr/bin/env python3
"""
Test script for Domain Adaptation training pipeline.

Quick validation of the DA trainer implementation before full training.
Tests data loading, model initialization, and training loop execution.
"""

import sys
from pathlib import Path
import yaml
import torch
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.archive.da_numu_trainer import DomainAdaptationTrainer
from src.models.domain_discriminator import create_da_model
from src.models.base_models import create_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_da_model_creation():
    """Test domain adaptation model creation."""
    logger.info("Testing DA model creation...")
    
    # Load config
    config_path = project_root / "experiments" / "da_neutrino_baseline.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create base model
    base_model = create_model(config['model'])
    logger.info(f"Base model created with {base_model.count_parameters():,} parameters")
    
    # Create DA model
    da_model = create_da_model(base_model, config)
    param_counts = da_model.count_parameters()
    
    logger.info(f"DA model created:")
    logger.info(f"  Base model: {param_counts['base_model']:,} parameters")
    logger.info(f"  Domain discriminator: {param_counts['domain_discriminator']:,} parameters")
    logger.info(f"  Total: {param_counts['total']:,} parameters")
    
    # Test forward pass with dummy data
    batch_size = 4
    seq_len = 50
    dummy_batch = {
        'features': torch.randn(batch_size, seq_len, 5),
        'lengths': torch.tensor([50, 45, 40, 35]),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    # Test inference mode
    da_model.eval()
    with torch.no_grad():
        class_logits = da_model(dummy_batch)
        logger.info(f"Classification output shape: {class_logits.shape}")
        
        # Test training mode with features
        da_model.train()
        class_logits, features, domain_logits = da_model(dummy_batch, return_features=True)
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Domain logits shape: {domain_logits.shape}")
    
    logger.info("✅ DA model creation test passed!")
    return True


def test_da_trainer_initialization():
    """Test DA trainer initialization without full training."""
    logger.info("Testing DA trainer initialization...")
    
    # Load config and enable debug mode
    config_path = project_root / "experiments" / "da_neutrino_baseline.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable debug mode for quick testing
    config['debug']['enabled'] = True
    config['training']['epochs'] = 2
    config['logging']['clearml']['enabled'] = False  # Disable ClearML for testing
    
    # Reduce data size for quick testing
    for particle in config['data']['source_domain']['events_per_particle']:
        config['data']['source_domain']['events_per_particle'][particle] = 100
    for particle in config['data']['target_domain']['events_per_particle']:
        config['data']['target_domain']['events_per_particle'][particle] = 100
    
    try:
        # Create trainer
        trainer = DomainAdaptationTrainer(config)
        logger.info(f"Trainer initialized with device: {trainer.device}")
        
        # Test data preparation
        trainer.prepare_data()
        logger.info(f"Source train batches: {len(trainer.source_train_loader)}")
        logger.info(f"Source val batches: {len(trainer.source_val_loader)}")
        logger.info(f"Target train batches: {len(trainer.target_train_loader)}")
        logger.info(f"Target val batches: {len(trainer.target_val_loader)}")
        
        # Test model preparation
        trainer.prepare_model()
        param_counts = trainer.da_model.count_parameters()
        logger.info(f"Model prepared with {param_counts['total']:,} total parameters")
        
        # Test one training step
        logger.info("Testing one training epoch...")
        trainer.current_epoch = 0
        train_metrics = trainer.train_epoch()
        
        logger.info("Train metrics:")
        for key, value in train_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Test validation
        logger.info("Testing validation...")
        val_metrics = trainer.validate_epoch()
        
        logger.info("Validation metrics:")
        for key, value in val_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info("✅ DA trainer initialization test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ DA trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lambda_scheduling():
    """Test lambda scheduling for gradient reversal."""
    logger.info("Testing lambda scheduling...")
    
    config_path = project_root / "experiments" / "da_neutrino_baseline.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['debug']['enabled'] = True
    config['logging']['clearml']['enabled'] = False
    
    trainer = DomainAdaptationTrainer(config)
    
    # Test different lambda scheduling strategies
    total_epochs = 100
    
    logger.info("Progressive lambda scheduling:")
    for epoch in [0, 10, 25, 50, 75, 99]:
        lambda_val = trainer._get_lambda_factor(epoch, total_epochs)
        logger.info(f"  Epoch {epoch}: λ = {lambda_val:.4f}")
    
    # Test constant scheduling
    trainer.lambda_scheduler_config = {'type': 'constant', 'lambda': 0.5}
    logger.info(f"Constant lambda: {trainer._get_lambda_factor(50, total_epochs):.4f}")
    
    # Test linear scheduling
    trainer.lambda_scheduler_config = {'type': 'linear', 'start_lambda': 0.0, 'end_lambda': 1.0}
    logger.info("Linear lambda scheduling:")
    for epoch in [0, 25, 50, 75, 99]:
        lambda_val = trainer._get_lambda_factor(epoch, total_epochs)
        logger.info(f"  Epoch {epoch}: λ = {lambda_val:.4f}")
    
    logger.info("✅ Lambda scheduling test passed!")
    return True


def main():
    """Run all DA training tests."""
    logger.info("🧪 Starting Domain Adaptation training tests...")
    
    tests = [
        ("DA Model Creation", test_da_model_creation),
        ("Lambda Scheduling", test_lambda_scheduling),
        ("DA Trainer Initialization", test_da_trainer_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All Domain Adaptation tests passed! Ready for training.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)