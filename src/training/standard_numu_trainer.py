"""
Standard training loop for neutrino detection without domain adaptation.

Implements comprehensive training pipeline for binary classification with
attention-based models, using preset normalization statistics and proper
data handling following project conventions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import numpy as np

# ClearML integration
try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    Task = None
    Logger = None

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_models import create_model
from src.utils.training import set_reproducible_seeds
from src.data.numu_dataset import NuMuDataset, create_numu_dataloader_from_ds
from src.training.metrics import MetricsTracker, calculate_class_weights, binary_cross_entropy_with_logits_weighted

logger = logging.getLogger(__name__)




class StandardTrainer:
    """
    Standard training pipeline for neutrino detection binary classification.
    
    Uses preset normalization statistics from data_manager/stats_dict/default_mc.yaml
    and handles complete training workflow including validation, early stopping,
    and model checkpointing.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['data'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set up reproducibility
        seed = config['experiment'].get('seed', 42)
        set_reproducible_seeds(seed)
        
        if config['reproducibility'].get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Load preset normalization statistics
        if self.config['data'].get('normalization', {}).get('enabled', True):
            self.normalization_config = self._load_normalization_config()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf') if self.config['training']['early_stopping']['mode'] == 'max' else float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker(device=self.device)
        
        # Output directory
        self.output_dir = Path(config['logging']['output_dir']) / Path(config['experiment']['name'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ClearML initialization
        self.clearml_task = None
        self.clearml_logger = None
        self._init_clearml()
        
        logger.info(f"Initialized StandardTrainer with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Using preset normalization: means={self.normalization_config['means']}")
    
    def _load_normalization_config(self) -> Dict:
        """Load preset normalization statistics from default_mc.yaml."""
        stats_path = Path(project_root) / 'data_manager' / 'stats_dict' / 'default_mc.yaml'
        
        if not stats_path.exists():
            logger.warning(f"Normalization stats file not found: {stats_path}")
            # Fallback to no normalization
            return {'means': [0.0] * 5, 'stds': [1.0] * 5}
        
        with open(stats_path, 'r') as f:
            normalization_config = yaml.safe_load(f)
        
        logger.info(f"Loaded normalization config from {stats_path}")
        return normalization_config
    
    def _init_clearml(self):
        """Initialize ClearML experiment tracking."""
        if not CLEARML_AVAILABLE:
            logger.warning("ClearML not available. Install with: pip install clearml")
            return
        
        clearml_config = self.config['logging'].get('clearml', {})
        if not clearml_config.get('enabled', False):
            logger.info("ClearML tracking disabled in config")
            return
        
        try:
            # Initialize ClearML task
            project_name = clearml_config.get('project_name', 'neutrino_detection')
            task_name = clearml_config.get('task_name') or self.config['experiment']['name']
            
            self.clearml_task = Task.init(
                project_name=project_name,
                task_name=task_name,
                tags=self.config['experiment'].get('tags', [])
            )
            
            # Connect configuration
            self.clearml_task.connect(self.config)
            
            # Get logger
            self.clearml_logger = self.clearml_task.get_logger()
            
            logger.info(f"ClearML initialized - Project: {project_name}, Task: {task_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ClearML: {e}")
            self.clearml_task = None
            self.clearml_logger = None
    
    def _save_metrics_history(self, training_history: list, epoch: int):
        """Save metrics history up to current epoch."""
        import pandas as pd
        
        if not training_history:
            return
        
        # Save incremental CSV
        df = pd.DataFrame(training_history)
        #z history_path = self.output_dir / f'training_history_epoch_{epoch:03d}.csv'
        #z df.to_csv(history_path, index=False)
        # Also update the main history file
        main_history_path = self.output_dir / 'training_history.csv'
        df.to_csv(main_history_path, index=False)
        
        logger.debug(f"Saved metrics history up to epoch {epoch}")
    
    def _log_metrics_to_clearml(self, epoch: int, train_metrics: Dict, val_metrics: Dict, epoch_metrics: Dict):
        """Log metrics to ClearML for visualization."""
        if not self.clearml_logger:
            return
        
        try:
            # Log training metrics
            for metric_name, value in train_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self.clearml_logger.report_scalar(
                        title="Training Metrics",
                        series=metric_name,
                        value=value,
                        iteration=epoch
                    )
            
            # Log validation metrics
            for metric_name, value in val_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self.clearml_logger.report_scalar(
                        title="Validation Metrics", 
                        series=metric_name,
                        value=value,
                        iteration=epoch
                    )
            
            # Log learning rate and timing
            if 'learning_rate' in epoch_metrics:
                self.clearml_logger.report_scalar(
                    title="Training Info",
                    series="learning_rate",
                    value=epoch_metrics['learning_rate'],
                    iteration=epoch
                )
            
            if 'epoch_time' in epoch_metrics:
                self.clearml_logger.report_scalar(
                    title="Training Info",
                    series="epoch_time",
                    value=epoch_metrics['epoch_time'],
                    iteration=epoch
                )
                
        except Exception as e:
            logger.warning(f"Failed to log metrics to ClearML: {e}")
    
    def _log_model_to_clearml(self, epoch: int, is_best: bool = False):
        """Log model artifacts to ClearML."""
        if not self.clearml_task:
            return
        
        try:
            # Log best model
            if is_best:
                best_model_path = self.output_dir / 'best_model.pth'
                self.clearml_task.upload_artifact(
                    name=f"best_model",
                    artifact_object=str(best_model_path)
                )
                logger.info(f"Uploaded best model to ClearML")
            
            # Log checkpoint periodically
            if epoch % 10 == 0:  # Every 10 epochs
                checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
                if checkpoint_path.exists():
                    self.clearml_task.upload_artifact(
                        name=f"checkpoint_epoch_{epoch:03d}",
                        artifact_object=str(checkpoint_path)
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to log model to ClearML: {e}")
    
    def _log_model_info_to_clearml(self):
        """Log model architecture and parameters to ClearML."""
        if not self.clearml_logger:
            return
        
        try:
            # Log model architecture as text
            model_summary = str(self.model)
            param_count = self.model.count_parameters()
            
            architecture_info = f"""
Model Architecture:
{model_summary}

Total Parameters: {param_count:,}

Model Configuration:
{yaml.dump(self.config['model'], default_flow_style=False)}

Training Configuration:
{yaml.dump(self.config['training'], default_flow_style=False)}
"""
            
            self.clearml_logger.report_text(
                title="Model Architecture",
                series="Architecture",
                text=architecture_info,
                iteration=0
            )
            
            # Log parameter count as a scalar
            self.clearml_logger.report_scalar(
                title="Model Info",
                series="parameter_count",
                value=param_count,
                iteration=0
            )
            
            logger.info(f"Logged model architecture to ClearML ({param_count:,} parameters)")
            
        except Exception as e:
            logger.warning(f"Failed to log model info to ClearML: {e}")
    
    def prepare_data(self):
        """Prepare training, validation, and test data loaders."""
        logger.info("Preparing data loaders...")
        
        data_config = self.config['data']
        
        # Create full dataset
        full_dataset = NuMuDataset(
            h5_path=data_config['h5_path'],
            particle_types=data_config['particle_types'],
            neutrino_types=data_config['neutrino_types'],
            max_hits=data_config['max_hits'],
            events_per_particle=data_config['events_per_particle'],
            sampling_config=data_config.get('sampling_config', None),
            device=self.device,
            seed=self.config['experiment']['seed'],
            shuffle_events=False
        )
        
        # create_numu_dataloader(
        #     h5_path=data_config['h5_path'],
        #     batch_size=1,  # Temporary for splitting
        #     shuffle=False,
        #     particle_types=data_config['particle_types'],
        #     neutrino_types=data_config['neutrino_types'],
        #     max_hits=data_config['max_hits'],
        #     events_per_particle=data_config['events_per_particle'],
        #     sampling_config=data_config.get('sampling_config', None),
        #     device=self.device,
        #     seed=self.config['experiment']['seed'],
        #     shuffle_events=False #data_config.get('shuffle_events_init', True)
        # ).dataset
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(data_config['train_split'] * total_size)
        val_size = total_size-train_size # int(data_config['val_split'] * total_size)
        
        logger.info(f"Dataset split - Train: {train_size}, Val: {val_size}")
        
        # Split dataset
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config['experiment']['seed'])
        )
        
        # Calculate class weights from training data labels
        class_weights = self._calculate_class_weights(train_dataset)
        self.class_weights = class_weights
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        
        self.train_loader = create_numu_dataloader_from_ds(
            train_dataset, batch_size,
            shuffle=self.config['dataloader'].get('reshuffle_train', True),
            normalization_config = self.normalization_config,
            shuffle_batch=True,
            augmentation_config=self.config['dataloader'].get('augmentation', None),
            use_polar_coords=self.config['dataloader'].get('use_polar_coords', False),
            num_workers=self.config['dataloader'].get('num_workers', 4),
            pin_memory=self.config['dataloader'].get('pin_memory', True)
            )
        
        self.val_loader = create_numu_dataloader_from_ds(
            val_dataset, batch_size,
            shuffle=False,
            normalization_config = self.normalization_config,
            shuffle_batch=False,
            augmentation_config=None,
            use_polar_coords=self.config['dataloader'].get('use_polar_coords', False),
            num_workers=self.config['dataloader'].get('num_workers', 4),
            pin_memory=self.config['dataloader'].get('pin_memory', True)
            )

        # self.train_loader = self._create_dataloader(
        #     train_dataset, batch_size, 
        #     shuffle_batch=True,
        #     reshaffle_epoch=self.config['dataloader'].get('reshuffle_train', True),
        #     aug_config=self.config['dataloader'].get('augmentation', None)
        #     )
        # self.val_loader = self._create_dataloader(
        #     val_dataset, batch_size,
        #     shuffle_batch=False,
        #     reshaffle_epoch=self.config['dataloader'].get('reshuffle_val', False),
        #     aug_config=None
        #     )

        logger.info(f"Data preparation complete using preset normalization statistics")
    
    def _calculate_class_weights(self, train_dataset) -> torch.Tensor:
        """Calculate class weights from training dataset labels."""
        logger.info("Calculating class weights from training data...")
        
        # Extract labels directly from dataset to avoid collate function issues
        all_labels = []
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            if isinstance(item, dict):
                label = item['labels']
            else:
                # Fallback for tuple format
                _, label = item
            all_labels.append(label.item() if hasattr(label, 'item') else label)
        
        all_labels = torch.tensor(all_labels, dtype=torch.float)
        class_weights = calculate_class_weights(all_labels)
        
        return class_weights
    
    # def _create_dataloader(self, dataset, batch_size: int, shuffle_batch: bool, reshaffle_epoch: bool, aug_config=None) -> DataLoader:
    #     """Create a DataLoader with proper collate function and preset normalization."""
        
    #     def collate_fn(batch):
    #         """Custom collate function with preset normalization and augmentation."""
    #         return dataset.dataset.collate_fn(
    #             batch,
    #             normalization_config=self.normalization_config,
    #             shuffle_batch=shuffle_batch,
    #             augmentation_config=aug_config
    #         )
        
    #     return DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=reshaffle_epoch,
    #         collate_fn=collate_fn,
    #         num_workers=self.config['dataloader'].get('num_workers', 4),
    #         pin_memory=self.config['dataloader'].get('pin_memory', True)
    #     )
    
    def prepare_model(self):
        """Initialize model, optimizer, and scheduler."""
        logger.info("Preparing model...")
        
        # Create model
        self.model = create_model(self.config['model'])
        self.model.to(self.device)
        
        logger.info(f"Model created with {self.model.count_parameters()} parameters")
        
        # Save model summary
        if self.config['logging'].get('save_model_summary', True):
            summary_path = self.output_dir / 'model_summary.txt'
            with open(summary_path, 'w') as f:
                f.write(str(self.model))
                f.write(f"\n\nTotal parameters: {self.model.count_parameters()}")
        
        # Create optimizer
        optimizer_config = self.config['training']
        if optimizer_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        elif optimizer_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['optimizer']}")
        
        # Create scheduler
        if optimizer_config.get('scheduler') == 'cosine':
            scheduler_params = optimizer_config.get('scheduler_params', {})
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get('T_max', optimizer_config['epochs']),
                eta_min=scheduler_params.get('eta_min', 1e-6)
            )
        elif optimizer_config.get('scheduler') == 'step':
            scheduler_params = optimizer_config.get('scheduler_params', {})
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 30),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        
        logger.info(f"Optimizer: {optimizer_config['optimizer']}")
        logger.info(f"Scheduler: {optimizer_config.get('scheduler', 'None')}")
        
        # Log model info to ClearML
        self._log_model_info_to_clearml()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics_tracker.reset_epoch()
        
        total_batches = len(self.train_loader)
        log_every = self.config['logging'].get('log_every', 10)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            logits = self.model(batch)
            labels = batch['labels'].float()
            
            # Calculate loss
            loss = self._calculate_loss(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            self.metrics_tracker.update_train(logits, labels, loss)
            
            # Logging
            if batch_idx % log_every == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{total_batches}, "
                          f"Loss: {loss.item():.4f}")
            
            # Debug mode: limit batches
            if self.config['debug'].get('enabled', False):
                max_batches = self.config['debug'].get('max_batches_per_epoch', 10)
                if batch_idx >= max_batches:
                    break
        
        return self.metrics_tracker.train_metrics.compute()
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                logits = self.model(batch)
                labels = batch['labels'].float()
                
                # Calculate loss
                loss = self._calculate_loss(logits, labels)
                
                # Update metrics
                self.metrics_tracker.update_val(logits, labels, loss)
        
        return self.metrics_tracker.val_metrics.compute()
    
    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate training loss with class weighting."""
        loss_config = self.config['training']
        
        if loss_config['loss'] == 'bce_with_logits':
            # Use calculated class weights
            pos_weight = None
            if hasattr(self, 'class_weights') and loss_config.get('class_weights') != 'auto':
                # Use config weights if specified, otherwise use calculated weights
                if loss_config.get('class_weights') is not None:
                    weights = torch.tensor(loss_config['class_weights'], device=self.device)
                else:
                    weights = self.class_weights.to(self.device)
                pos_weight = weights[1] / weights[0]  # positive_weight / negative_weight
            
            loss = binary_cross_entropy_with_logits_weighted(logits, labels, pos_weight)
        else:
            raise ValueError(f"Unknown loss function: {loss_config['loss']}")
        
        return loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, training_history: list = None):
        """Save model checkpoint with metrics history."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'normalization_config': self.normalization_config,
            'training_history': training_history[:epoch+1] if training_history else []
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save metrics history at this checkpoint
        if training_history:
            self._save_metrics_history(training_history[:epoch+1], epoch)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved at epoch {epoch}")
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
    
    def check_early_stopping(self, current_metric: float) -> bool:
        """Check early stopping criteria."""
        early_stopping_config = self.config['training']['early_stopping']
        
        if not early_stopping_config.get('enabled', True):
            return False
        
        monitor = early_stopping_config.get('monitor', 'val_f1')
        mode = early_stopping_config.get('mode', 'max')
        patience = early_stopping_config.get('patience', 10)
        min_delta = early_stopping_config.get('min_delta', 0.001)
        
        if mode == 'max':
            improved = current_metric > (self.best_metric + min_delta)
        else:
            improved = current_metric < (self.best_metric - min_delta)
        
        if improved:
            self.best_metric = current_metric
            self.best_epoch = self.current_epoch
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                return True
            return False
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Prepare data and model
        self.prepare_data()
        self.prepare_model()
        
        # Save configuration and normalization stats
        if self.config['logging'].get('save_config', True):
            config_path = self.output_dir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            norm_path = self.output_dir / 'normalization_config.yaml'
            with open(norm_path, 'w') as f:
                yaml.dump(self.normalization_config, f, default_flow_style=False)
        
        # Training loop
        num_epochs = self.config['training']['epochs']
        validate_every = self.config['training'].get('validate_every', 1)
        save_every = self.config['training'].get('save_every', 5)
        
        training_history = []
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if epoch % validate_every == 0:
                val_metrics = self.validate_epoch()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save epoch metrics
            self.metrics_tracker.save_epoch_metrics(epoch)
            
            # Combine metrics
            epoch_metrics = {} #{**train_metrics, **{k+"_val": v for k,v in val_metrics.items()}}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            epoch_metrics['epoch_time'] = time.time() - start_time
            
            training_history.append({**epoch_metrics, **train_metrics, **{"val_"+k: v for k,v in val_metrics.items()}})
            
            # ClearML logging
            self._log_metrics_to_clearml(epoch, train_metrics, val_metrics, epoch_metrics)
            
            # Logging
            logger.info(f"Epoch {epoch}/{num_epochs-1} completed in {epoch_metrics['epoch_time']:.2f}s")
            logger.info(f"Train - Loss: {train_metrics.get('loss', 0):.4f}, "
                       f"Acc: {train_metrics.get('accuracy', 0):.4f}, "
                       f"F1: {train_metrics.get('f1', 0):.4f}")
            
            if val_metrics:
                logger.info(f"Val - Loss: {val_metrics.get('loss', 0):.4f}, "
                           f"Acc: {val_metrics.get('accuracy', 0):.4f}, "
                           f"F1: {val_metrics.get('f1', 0):.4f}")
            
            # Check for best model
            monitor_metric = self.config['training']['early_stopping'].get('monitor', 'val_f1')
            # ???
            current_metric = val_metrics.get(monitor_metric.replace('val_', ''), train_metrics.get(monitor_metric.replace('train_', ''), 0))
            
            is_best = False
            if self.config['training']['early_stopping'].get('mode', 'max') == 'max':
                is_best = current_metric > self.best_metric
            else:
                is_best = current_metric < self.best_metric
            
            # Save checkpoint with training history
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best, training_history)
            # # Log model to ClearML
            # self._log_model_to_clearml(epoch, is_best)
            
            # Early stopping
            if val_metrics and self.check_early_stopping(current_metric):
                logger.info(f"Training stopped early at epoch {epoch}")
                break
            
            # Update best metric and epoch AFTER early stopping check
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch    
            
        # Save final results
        self._save_training_results(training_history)
        
        logger.info(f"Training completed. Best {monitor_metric}: {self.best_metric:.4f} at epoch {self.best_epoch}")
    
    def _save_training_results(self, training_history: list):
        """Save training history and final results."""
        import pandas as pd
        
        # Save training history as CSV
        df = pd.DataFrame(training_history)
        history_path = self.output_dir / 'training_history.csv'
        df.to_csv(history_path, index=False)
        
        # Save final metrics summary
        summary = {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'total_epochs': self.current_epoch + 1,
            'final_train_metrics': self.metrics_tracker.train_metrics.compute(),
            'final_val_metrics': self.metrics_tracker.val_metrics.compute(),
            'normalization_config': self.normalization_config
        }
        
        summary_path = self.output_dir / 'training_summary.yaml'
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Training results saved to {self.output_dir}")


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train neutrino detection model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with limited batches')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override debug mode
    if args.debug:
        config['debug']['enabled'] = True
        config['training']['epochs'] = 3
        logger.info("Debug mode enabled - limited epochs and batches")
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config['logging'].get('level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trainer and start training
    trainer = StandardTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()