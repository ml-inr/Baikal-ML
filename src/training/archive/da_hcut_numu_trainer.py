"""
Domain Adaptation trainer for neutrino detection MC→Exp transfer learning.

Implements Domain-Adversarial Neural Networks (DANN) approach to adapt
Monte Carlo trained models to experimental data. Uses adversarial training
with gradient reversal to learn domain-invariant features.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_models import create_model
from src.utils.training import set_reproducible_seeds
from src.models.domain_discriminator import create_da_model
from src.data.hcut_numu_dataset import HCutNuMuDataset, create_hcut_numu_dataloader_from_ds
from src.training.metrics import MetricsTracker, calculate_class_weights, binary_cross_entropy_with_logits_weighted

logger = logging.getLogger(__name__)




class DomainAdaptationTrainer:
    """
    Domain Adaptation trainer for MC→Exp neutrino detection transfer learning.

    Implements DANN (Domain-Adversarial Neural Networks) approach:
    - Shared feature extractor for both domains
    - Classification head for signal/background prediction (MC only)
    - Domain discriminator for MC/Exp prediction (background events only)
    - Adversarial training with gradient reversal

    CRITICAL DESIGN: Uses only background events for domain discrimination to avoid
    the feature extractor learning that "MC domain contains signal neutrinos while
    Exp domain contains only background", which would confound neutrino detection.

    Background events = muon events + neutrino events with < h_min signal hits
    Signal events = neutrino events with >= h_min signal hits

    Loss = λ_cls * L_classification + λ_domain * L_domain_confusion
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
        self.base_model = None
        self.da_model = None
        self.optimizer_feature = None
        self.optimizer_classifier = None
        self.optimizer_discriminator = None
        self.scheduler_feature = None
        self.scheduler_classifier = None
        self.scheduler_discriminator = None
        
        # Data loaders
        self.source_train_loader = None
        self.source_val_loader = None
        self.target_train_loader = None
        self.target_val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('-inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker(device=self.device)
        
        # Output directory
        self.output_dir = Path(config['logging']['output_dir']) / Path(config['experiment']['name'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard logging
        if config['logging'].get('tensorboard', True):
            tensorboard_dir = self.output_dir / 'tensorboard'
            tensorboard_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")
        else:
            self.writer = None
        
        # Lambda scheduling for domain adaptation
        self.lambda_scheduler_config = config['domain_adaptation'].get('lambda_scheduler', {})
        
        logger.info(f"Initialized DomainAdaptationTrainer with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure TensorBoard writer is closed."""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed via context manager")
    
    def close(self):
        """Manually close TensorBoard writer."""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer manually closed")
    
    def _load_normalization_config(self) -> Dict:
        """Load preset normalization statistics from default_mc.yaml."""
        stats_path = Path(project_root) / 'data_manager' / 'stats_dict' / 'default_mc.yaml'
        
        if not stats_path.exists():
            logger.warning(f"Normalization stats file not found: {stats_path}")
            return {'means': [0.0] * 5, 'stds': [1.0] * 5}
        
        with open(stats_path, 'r') as f:
            normalization_config = yaml.safe_load(f)
        
        logger.info(f"Loaded normalization config from {stats_path}")
        return normalization_config
    
    def prepare_data(self):
        """Prepare source (MC) and target (Exp) data loaders."""
        logger.info("Preparing domain adaptation data loaders...")
        
        data_config = self.config['data']
        source_config = data_config['source_domain']  # MC data
        target_config = data_config['target_domain']  # Exp data
        
        # Create source domain dataset (MC)
        source_dataset = HCutNuMuDataset(
            h5_path=source_config['h5_path'],
            events_per_particle=source_config['events_per_particle'],
            particle_types=source_config['particle_types'],
            neutrino_types=source_config['neutrino_types'],
            h_min=source_config['h_min_cut'],
            max_hits=data_config['max_hits'],
            device=self.device,
            seed=self.config['experiment']['seed'],
            shuffle_events=source_config.get('shuffle_events', True),
            balance_classes=source_config.get('balance_classes', False)
        )
        
        # Create target domain dataset (Exp)
        target_dataset = HCutNuMuDataset(
            h5_path=target_config['h5_path'],
            events_per_particle=target_config['events_per_particle'],
            particle_types=target_config['particle_types'],
            neutrino_types=target_config.get('neutrino_types', []),  # May be empty for unlabeled
            h_min=source_config['h_min_cut'],
            max_hits=data_config['max_hits'],
            device=self.device,
            seed=self.config['experiment']['seed'],
            shuffle_events=target_config.get('shuffle_events', True)
        )
        
        # Split source dataset
        source_total = len(source_dataset)
        source_train_size = int(source_config['train_split'] * source_total)
        source_val_size = source_total - source_train_size
        
        source_train_dataset, source_val_dataset = random_split(
            source_dataset, [source_train_size, source_val_size],
            generator=torch.Generator().manual_seed(self.config['experiment']['seed'])
        )
        
        # Split target dataset using train_split parameter
        target_total = len(target_dataset)
        target_train_size = int(target_config['train_split'] * target_total)
        target_val_size = target_total - target_train_size
        
        target_train_dataset, target_val_dataset = random_split(
            target_dataset, [target_train_size, target_val_size],
            generator=torch.Generator().manual_seed(self.config['experiment']['seed'])
        )
        
        logger.info(f"Source domain split - Train: {source_train_size}, Val: {source_val_size}")
        logger.info(f"Target domain split - Train: {target_train_size}, Val: {target_val_size}")
        
        # Calculate class weights from source training data only
        self.class_weights = self._calculate_class_weights(source_train_dataset)
        
        # Create data loaders with separate batch sizes
        source_batch_size = self.config['training'].get('source_batch_size', 
                                                        self.config['training'].get('batch_size', 32))
        target_batch_size = self.config['training'].get('target_batch_size', 
                                                        self.config['training'].get('batch_size', 32))
        
        logger.info(f"Using source batch size: {source_batch_size}, target batch size: {target_batch_size}")
        
        self.source_train_loader = create_hcut_numu_dataloader_from_ds(
            source_train_dataset, source_batch_size,
            shuffle=True,
            normalization_config=self.normalization_config,
            shuffle_batch=True,
            augmentation_config=self.config['dataloader'].get('augmentation', None),
            num_workers=self.config['dataloader'].get('num_workers', 4),
            pin_memory=self.config['dataloader'].get('pin_memory', True)
        )
        
        self.source_val_loader = create_hcut_numu_dataloader_from_ds(
            source_val_dataset, source_batch_size,
            shuffle=False,
            normalization_config=self.normalization_config,
            shuffle_batch=False,
            augmentation_config=None, # No augmentation for validation
            num_workers=self.config['dataloader'].get('num_workers', 4),
            pin_memory=self.config['dataloader'].get('pin_memory', True)
        )
        
        self.target_train_loader = create_hcut_numu_dataloader_from_ds(
            target_train_dataset, target_batch_size,
            shuffle=True,
            normalization_config=self.normalization_config,
            shuffle_batch=True,
            augmentation_config=self.config['dataloader'].get('augmentation', None),
            num_workers=self.config['dataloader'].get('num_workers', 4),
            pin_memory=self.config['dataloader'].get('pin_memory', True)
        )
        
        self.target_val_loader = create_hcut_numu_dataloader_from_ds(
            target_val_dataset, target_batch_size,
            shuffle=False,
            normalization_config=self.normalization_config,
            shuffle_batch=False,
            augmentation_config=None,  # No augmentation for validation
            num_workers=self.config['dataloader'].get('num_workers', 4),
            pin_memory=self.config['dataloader'].get('pin_memory', True)
        )
        
        logger.info("Domain adaptation data preparation complete")
    
    def _calculate_class_weights(self, source_train_dataset) -> torch.Tensor:
        """Calculate class weights from source training dataset labels."""
        logger.info("Calculating class weights from source training data...")
        
        all_labels = []
        for i in range(len(source_train_dataset)):
            item = source_train_dataset[i]
            if isinstance(item, dict):
                label = item['labels']
            else:
                _, label = item
            all_labels.append(label.item() if hasattr(label, 'item') else label)
        
        all_labels = torch.tensor(all_labels, dtype=torch.float, device=self.device)
        class_weights = calculate_class_weights(all_labels)
        
        return class_weights
    
    def prepare_model(self):
        """Initialize base model, domain adaptation model, and optimizers."""
        logger.info("Preparing domain adaptation model...")
        
        # Create base model
        self.base_model = create_model(self.config['model'])
        self.base_model.to(self.device)
        
        # Create domain adaptation model
        self.da_model = create_da_model(self.base_model, self.config['domain_adaptation'])
        self.da_model.to(self.device)
        
        param_counts = self.da_model.count_parameters()
        logger.info(f"Domain adaptation model created:")
        logger.info(f"  Base model: {param_counts['base_model']:,} parameters")
        logger.info(f"  Domain discriminator: {param_counts['domain_discriminator']:,} parameters")
        logger.info(f"  Total: {param_counts['total']:,} parameters")
        
        # Save model summary
        if self.config['logging'].get('save_model_summary', True):
            summary_path = self.output_dir / 'model_summary.txt'
            with open(summary_path, 'w') as f:
                f.write("=== Domain Adaptation Model Summary ===\n\n")
                f.write("Base Model:\n")
                f.write(str(self.base_model))
                f.write(f"\n\nDomain Discriminator:\n")
                f.write(str(self.da_model.domain_discriminator))
                f.write(f"\n\nParameter Counts:\n")
                for component, count in param_counts.items():
                    f.write(f"  {component}: {count:,}\n")
        
        # Create optimizers (separate for different components)
        training_config = self.config['training']
        
        # Feature extractor optimizer
        self.optimizer_feature = self._create_optimizer(
            self.da_model.base_model.feature_extractor.parameters(),
            training_config.get('feature_optimizer', training_config)
        )
        
        # Classifier optimizer  
        self.optimizer_classifier = self._create_optimizer(
            self.da_model.base_model.classifier.parameters(),
            training_config.get('classifier_optimizer', training_config)
        )
        
        # Domain discriminator optimizer
        self.optimizer_discriminator = self._create_optimizer(
            self.da_model.domain_discriminator.parameters(),
            training_config.get('discriminator_optimizer', training_config)
        )
        
        # Create schedulers
        self._create_schedulers(training_config)
        
        logger.info("Domain adaptation model preparation complete")
    
    def _create_optimizer(self, parameters, config: Dict) -> optim.Optimizer:
        """Create optimizer for given parameters."""
        if config['optimizer'] == 'adamw':
            return optim.AdamW(
                parameters,
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0.01)
            )
        elif config['optimizer'] == 'adam':
            return optim.Adam(
                parameters,
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    def _create_schedulers(self, training_config: Dict):
        """Create learning rate schedulers."""
        scheduler_type = training_config.get('scheduler')
        
        if scheduler_type == 'cosine':
            scheduler_params = training_config.get('scheduler_params', {})
            T_max = int(scheduler_params.get('T_max', training_config['epochs']))
            eta_min = float(scheduler_params.get('eta_min', 1e-6))
            
            self.scheduler_feature = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_feature, T_max=T_max, eta_min=eta_min)
            self.scheduler_classifier = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_classifier, T_max=T_max, eta_min=eta_min)
            self.scheduler_discriminator = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_discriminator, T_max=T_max, eta_min=eta_min)
                
        elif scheduler_type == 'step':
            scheduler_params = training_config.get('scheduler_params', {})
            step_size = int(scheduler_params.get('step_size', 30))
            gamma = float(scheduler_params.get('gamma', 0.1))
            
            self.scheduler_feature = optim.lr_scheduler.StepLR(
                self.optimizer_feature, step_size=step_size, gamma=gamma)
            self.scheduler_classifier = optim.lr_scheduler.StepLR(
                self.optimizer_classifier, step_size=step_size, gamma=gamma)
            self.scheduler_discriminator = optim.lr_scheduler.StepLR(
                self.optimizer_discriminator, step_size=step_size, gamma=gamma)
    
    def _get_lambda_factor(self, step: int = None, epoch: int = None, total_epochs: int = None, total_steps: int = None) -> float:
        """Calculate lambda factor for gradient reversal based on training progress.
        
        Can use either step-based or epoch-based scheduling depending on configuration.
        """
        scheduler_config = self.lambda_scheduler_config
        schedule_by = scheduler_config.get('schedule_by', 'epoch')  # 'step' or 'epoch'
        
        # Determine progress based on scheduling method
        if schedule_by == 'step' and step is not None and total_steps is not None:
            progress = step / total_steps
        elif schedule_by == 'epoch' and epoch is not None and total_epochs is not None:
            progress = epoch / total_epochs
        else:
            logger.warning(f"Invalid lambda scheduling configuration: {schedule_by}. Using epoch-based.")
            progress = (epoch or 0) / (total_epochs or 1)
        
        if scheduler_config.get('type') == 'progressive':
            # Progressive lambda: start from 0, increase to max_lambda
            max_lambda = scheduler_config.get('max_lambda', 1.0)
            gamma = scheduler_config.get('gamma', 10.0)
            # Formula from DANN paper: λ = 2 / (1 + exp(-γ * p)) - 1
            # where p = progress
            lambda_factor = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
            return lambda_factor * max_lambda
            
        elif scheduler_config.get('type') == 'constant':
            return scheduler_config.get('lambda', 1.0)
            
        elif scheduler_config.get('type') == 'linear':
            # Linear increase from start_lambda to end_lambda
            start_lambda = scheduler_config.get('start_lambda', 0.0)
            end_lambda = scheduler_config.get('end_lambda', 1.0)
            return start_lambda + (end_lambda - start_lambda) * progress
            
        else:
            # Default constant lambda
            return 1.0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with clean domain adaptation.
        
        Uses separate forward passes to avoid gradient contamination:
        1. Classification forward: source_batch → features + classification (no domain)
        2. Domain forward: balanced domain batches → features + domain (with GRL)
        
        This prevents unwanted gradient flow from discarded domain predictions.
        """
        self.da_model.train()
        self.metrics_tracker.reset_epoch()
        
        # Calculate total steps for lambda scheduling
        total_epochs = self.config['training']['epochs']
        total_steps = total_epochs * min(len(self.source_train_loader), len(self.target_train_loader))
        
        # Initialize lambda factor for this epoch (will be updated per batch if step-based)
        initial_lambda_factor = self._get_lambda_factor(
            step=self.current_step, epoch=self.current_epoch, 
            total_steps=total_steps, total_epochs=total_epochs
        )
        
        # Create iterators for both domains
        source_iter = iter(self.source_train_loader)
        target_iter = iter(self.target_train_loader)
        
        total_batches = min(len(self.source_train_loader), len(self.target_train_loader))
        log_every = self.config['logging'].get('log_every', 10)
        
        epoch_metrics = {
            'classification_loss': 0.0,
            'domain_loss': 0.0,
            'total_loss': 0.0,
            'domain_accuracy': 0.0,
            'lambda_factor': initial_lambda_factor,
            'domain_utilization': 0.0,
            'source_background_avg': 0.0,
            'target_events_avg': 0.0,
            'domain_batch_size_avg': 0.0
        }
        
        # Collect domain predictions for AUC calculation
        domain_predictions = []
        domain_labels_true = []
        
        # Collect data for signal hits subset metrics
        all_source_logits = []
        all_source_labels = []
        all_source_magic_numbers = []
        
        # Collect domain predictions for signal hits subsets
        domain_predictions_8hits = []
        domain_labels_8hits = []
        domain_predictions_16hits = []
        domain_labels_16hits = []
        
        for batch_idx in range(total_batches):
            # Get batches from both domains
            try:
                source_batch = next(source_iter)
                target_batch = next(target_iter)
            except StopIteration:
                break
            
            # Move to device
            for key in source_batch:
                if isinstance(source_batch[key], torch.Tensor):
                    source_batch[key] = source_batch[key].to(self.device)
                    
            for key in target_batch:
                if isinstance(target_batch[key], torch.Tensor):
                    target_batch[key] = target_batch[key].to(self.device)
            
            # Update lambda factor for gradient reversal (step-based if configured)
            lambda_factor = self._get_lambda_factor(
                step=self.current_step, epoch=self.current_epoch, 
                total_steps=total_steps, total_epochs=total_epochs
            )
            self.da_model.set_lambda(lambda_factor)
            
            # === CLEAN ADVERSARIAL TRAINING ===
            # Train classification on source domain (all events) - NO domain computation
            source_class_logits, source_features = self.da_model.classification_forward(source_batch)
            source_labels = source_batch['labels'].float()
            
            # Collect data for signal hits subset metrics
            all_source_logits.append(source_class_logits.detach().cpu())
            all_source_labels.append(source_labels.detach().cpu())
            all_source_magic_numbers.extend(source_batch['magic_numbers'])
            
            # Classification loss (source domain only)
            classification_loss = self._calculate_classification_loss(source_class_logits, source_labels)
            
            # Zero gradients for all optimizers at the start
            self.optimizer_feature.zero_grad()
            self.optimizer_classifier.zero_grad()
            self.optimizer_discriminator.zero_grad()
            
            # Backward pass for classification loss
            classification_loss.backward()
            
            # === DOMAIN ADAPTATION TRAINING ===
            # CRITICAL: Use only background events for domain discrimination
            # Background = muon events + neutrino events with < h_min signal hits (all have label=0)
            # This prevents the feature extractor from learning "MC has signal neutrinos, Exp has only background"
            source_background_mask = (source_labels == 0).squeeze()
            n_source_background = source_background_mask.sum().item()
            n_target_events = target_batch['features'].shape[0]

            # Skip domain training if no background events in source batch (very rare with large batches)
            if n_source_background == 0:
                logger.warning("No background events in source batch - skipping domain training step")
                epoch_metrics['classification_loss'] += classification_loss.item()
                epoch_metrics['domain_loss'] += 0.0
                epoch_metrics['total_loss'] += classification_loss.item()
                epoch_metrics['domain_accuracy'] += 0.5  # Random chance
                continue
            
            # MINIMUM BATCH STRATEGY: Use min(source_background, target_events) from each domain
            domain_batch_size = min(n_source_background, n_target_events)

            # Sample domain_batch_size events from each domain
            # Source domain: randomly select from available background events (muons + low-signal neutrinos)
            source_background_indices = torch.where(source_background_mask)[0]
            if n_source_background > domain_batch_size:
                selected_source_indices = source_background_indices[torch.randperm(n_source_background, device=self.device)[:domain_batch_size]]
            else:
                selected_source_indices = source_background_indices

            # Target domain: randomly select from all events (all are background in experimental data)
            if n_target_events > domain_batch_size:
                selected_target_indices = torch.randperm(n_target_events, device=self.device)[:domain_batch_size]
            else:
                selected_target_indices = torch.arange(n_target_events, device=self.device)
            
            # Create balanced domain batches
            source_domain_features = source_batch['features'][selected_source_indices]
            source_domain_lengths = source_batch['lengths'][selected_source_indices]
            source_domain_mask = source_batch['mask'][selected_source_indices]
            source_domain_batch = {
                'features': source_domain_features,
                'lengths': source_domain_lengths,
                'mask': source_domain_mask
            }
            
            target_domain_features = target_batch['features'][selected_target_indices]
            target_domain_lengths = target_batch['lengths'][selected_target_indices]
            target_domain_mask = target_batch['mask'][selected_target_indices]
            target_domain_batch = {
                'features': target_domain_features,
                'lengths': target_domain_lengths,
                'mask': target_domain_mask
            }
            
            # Get domain predictions for balanced batches (WITH gradient reversal)
            source_domain_logits, _ = self.da_model.domain_forward(source_domain_batch)
            target_domain_logits, _ = self.da_model.domain_forward(target_domain_batch)
            
            # Create domain labels (0=source/MC background, 1=target/Exp background)
            source_domain_labels = torch.zeros(domain_batch_size, 1, dtype=torch.float, device=self.device)
            target_domain_labels = torch.ones(domain_batch_size, 1, dtype=torch.float, device=self.device)
            
            # Calculate domain losses on balanced batches
            source_domain_loss = nn.functional.binary_cross_entropy_with_logits(
                source_domain_logits, source_domain_labels)
            target_domain_loss = nn.functional.binary_cross_entropy_with_logits(
                target_domain_logits, target_domain_labels)
            
            # Average domain loss (no weighting needed since batches are equal)
            domain_loss = (source_domain_loss + target_domain_loss) / 2
            
            # LOSS NORMALIZATION: Scale to represent full batch impact
            # Normalize by the effective utilization rate
            source_utilization = domain_batch_size / n_source_background
            target_utilization = domain_batch_size / n_target_events
            avg_utilization = (source_utilization + target_utilization) / 2
            
            # Scale loss to represent what it would be with full batches
            normalized_domain_loss = domain_loss / avg_utilization
            
            # Backward pass for domain loss (accumulates gradients with classification gradients)
            # The gradient reversal layer will automatically reverse gradients to the feature encoder
            normalized_domain_loss.backward()
            
            # === SINGLE COMBINED OPTIMIZER STEP ===
            # Feature encoder gets: ∇L_classification + ∇(-λ*L_domain) [reversed by GRL]
            # Classifier gets: ∇L_classification only 
            # Discriminator gets: ∇L_domain only
            self.optimizer_feature.step()      # Combined adversarial update
            self.optimizer_classifier.step()   # Classification update only
            self.optimizer_discriminator.step()# Domain discrimination update only
            
            # Calculate domain accuracy on balanced batches
            with torch.no_grad():
                source_domain_preds = torch.sigmoid(source_domain_logits) < 0.5  # Should be 0 (MC background)
                target_domain_preds = torch.sigmoid(target_domain_logits) >= 0.5  # Should be 1 (Exp background)
                domain_accuracy = (source_domain_preds.float().mean() + target_domain_preds.float().mean()) / 2
                
                # Collect predictions for AUC calculation
                source_probs = torch.sigmoid(source_domain_logits).cpu()
                target_probs = torch.sigmoid(target_domain_logits).cpu()
                
                domain_predictions.extend(source_probs.flatten().tolist())
                domain_predictions.extend(target_probs.flatten().tolist())
                
                domain_labels_true.extend([0.0] * domain_batch_size)  # Source = 0
                domain_labels_true.extend([1.0] * domain_batch_size)  # Target = 1
                
                # Collect domain predictions for signal hits subsets
                # Filter source domain features by signal hits
                if len(selected_source_indices) > 0:
                    source_magic_batch = [source_batch['magic_numbers'][i] for i in selected_source_indices]
                    source_signal_hits = np.array([np.sum(np.array(numbers) != 0) for numbers in source_magic_batch])
                    
                    # >8 signal hits
                    mask_8 = source_signal_hits > 8
                    if np.sum(mask_8) > 0:
                        domain_predictions_8hits.extend(source_probs[mask_8].flatten().tolist())
                        domain_labels_8hits.extend([0.0] * np.sum(mask_8))
                        # Target domain (assume all have enough complexity for >8 hits)
                        domain_predictions_8hits.extend(target_probs[:np.sum(mask_8)].flatten().tolist())
                        domain_labels_8hits.extend([1.0] * np.sum(mask_8))
                    
                    # >16 signal hits
                    mask_16 = source_signal_hits > 16
                    if np.sum(mask_16) > 0:
                        domain_predictions_16hits.extend(source_probs[mask_16].flatten().tolist())
                        domain_labels_16hits.extend([0.0] * np.sum(mask_16))
                        # Target domain (assume proportional complexity)
                        domain_predictions_16hits.extend(target_probs[:np.sum(mask_16)].flatten().tolist())
                        domain_labels_16hits.extend([1.0] * np.sum(mask_16))
            
            # Update metrics (classification on all source events)
            self.metrics_tracker.update_train(source_class_logits, source_labels, classification_loss)
            
            # Track epoch metrics
            epoch_metrics['classification_loss'] += classification_loss.item()
            epoch_metrics['domain_loss'] += normalized_domain_loss.item()  # Use normalized loss for tracking
            epoch_metrics['total_loss'] += (classification_loss.item() + normalized_domain_loss.item())
            epoch_metrics['domain_accuracy'] += domain_accuracy.item()
            epoch_metrics['lambda_factor'] = lambda_factor  # Update with current lambda (last batch value)
            
            # Track utilization metrics for monitoring
            epoch_metrics['domain_utilization'] += avg_utilization
            epoch_metrics['source_background_avg'] += n_source_background
            epoch_metrics['target_events_avg'] += n_target_events
            epoch_metrics['domain_batch_size_avg'] += domain_batch_size
            
            # Logging with domain batch information
            if batch_idx % log_every == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{total_batches}, "
                          f"Class Loss: {classification_loss.item():.4f}, "
                          f"Domain Loss: {normalized_domain_loss.item():.4f}, "
                          f"Domain Acc: {domain_accuracy.item():.4f}, "
                          f"Lambda: {lambda_factor:.4f}")
                logger.info(f"  Domain Batches: {domain_batch_size} (from {n_source_background} source background, "
                          f"{n_target_events} target events), Util: {avg_utilization:.3f}")
                logger.debug(f"  Clean forward pass: Classification only on {source_batch['features'].shape[0]} events, "
                           f"Domain only on {domain_batch_size*2} balanced events")
            
            # Increment step counter
            self.current_step += 1
            
            # Debug mode: limit batches
            if self.config['debug'].get('enabled', False):
                max_batches = self.config['debug'].get('max_batches_per_epoch', 10)
                if batch_idx >= max_batches:
                    break
        
        # Average epoch metrics
        metrics_to_average = ['classification_loss', 'domain_loss', 'total_loss', 'domain_accuracy',
                             'domain_utilization', 'source_background_avg', 'target_events_avg', 'domain_batch_size_avg']
        
        for key in metrics_to_average:
            if key in epoch_metrics:
                epoch_metrics[key] /= total_batches
        
        # Calculate domain AUC
        domain_auc = 0.0
        if len(domain_predictions) > 0 and len(set(domain_labels_true)) > 1:  # Need both classes
            try:
                from sklearn.metrics import roc_auc_score
                domain_auc = roc_auc_score(domain_labels_true, domain_predictions)
            except Exception as e:
                logger.warning(f"Failed to calculate domain AUC: {e}")
                domain_auc = 0.0
        
        epoch_metrics['domain_auc'] = domain_auc
        
        # Calculate domain AUC for signal hits subsets
        domain_auc_8hits = 0.0
        if len(domain_predictions_8hits) > 0 and len(set(domain_labels_8hits)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                domain_auc_8hits = roc_auc_score(domain_labels_8hits, domain_predictions_8hits)
            except Exception as e:
                logger.warning(f"Failed to calculate domain AUC for >8 hits: {e}")
        
        domain_auc_16hits = 0.0
        if len(domain_predictions_16hits) > 0 and len(set(domain_labels_16hits)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                domain_auc_16hits = roc_auc_score(domain_labels_16hits, domain_predictions_16hits)
            except Exception as e:
                logger.warning(f"Failed to calculate domain AUC for >16 hits: {e}")
        
        epoch_metrics['domain_auc_8hits'] = domain_auc_8hits
        epoch_metrics['domain_auc_16hits'] = domain_auc_16hits
        
        # Calculate metrics for signal hits subsets (MC events only)
        signal_hits_metrics = self._calculate_signal_hits_metrics(
            all_source_logits, all_source_labels, all_source_magic_numbers
        )
        train_metrics = self.metrics_tracker.train_metrics.compute()
        train_metrics.update(epoch_metrics)
        train_metrics.update(signal_hits_metrics)
        
        # Log to TensorBoard
        if self.writer is not None:
            self._log_tensorboard_train(train_metrics, self.current_epoch)
        
        return train_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate both classification and domain discrimination performance.
        
        Validates:
        1. Classification performance on source domain (has labels)
        2. Domain discrimination performance on both domains (muon events only)
        """
        self.da_model.eval()
        
        val_domain_loss = 0.0
        val_domain_accuracy = 0.0
        val_domain_batches = 0
        
        # Collect domain predictions for validation AUC calculation
        val_domain_predictions = []
        val_domain_labels_true = []
        
        # Collect data for validation signal hits subset metrics
        all_val_source_logits = []
        all_val_source_labels = []
        all_val_source_magic_numbers = []
        
        with torch.no_grad():
            # === CLASSIFICATION VALIDATION ===
            # Validate classification on source domain (has labels)
            for batch in self.source_val_loader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass for classification only (no domain computation)
                logits, _ = self.da_model.classification_forward(batch)
                labels = batch['labels'].float()
                
                # Calculate classification loss
                loss = self._calculate_classification_loss(logits, labels)
                
                # Update classification metrics
                self.metrics_tracker.update_val(logits, labels, loss)
                
                # Collect data for signal hits subset metrics
                all_val_source_logits.append(logits.detach().cpu())
                all_val_source_labels.append(labels.detach().cpu())
                all_val_source_magic_numbers.extend(batch['magic_numbers'])
            
            # === DOMAIN DISCRIMINATION VALIDATION ===
            # Validate domain discrimination on muon events from both validation sets
            source_val_iter = iter(self.source_val_loader)
            target_val_iter = iter(self.target_val_loader)  # Use proper target validation data
            
            val_batches = min(len(self.source_val_loader), len(self.target_val_loader))
            
            for _ in range(val_batches):
                try:
                    source_batch = next(source_val_iter)
                    target_batch = next(target_val_iter)
                except StopIteration:
                    break
                
                # Move to device
                for key in source_batch:
                    if isinstance(source_batch[key], torch.Tensor):
                        source_batch[key] = source_batch[key].to(self.device)
                        
                for key in target_batch:
                    if isinstance(target_batch[key], torch.Tensor):
                        target_batch[key] = target_batch[key].to(self.device)
                
                # Filter source batch to keep only background events (label=0)
                # Background = muons + neutrino events with < h_min signal hits
                source_labels = source_batch['labels'].float()
                source_background_mask = (source_labels == 0).squeeze()
                n_source_background = source_background_mask.sum().item()
                n_target_events = target_batch['features'].shape[0]

                # Skip if no background events in source validation batch
                if n_source_background == 0:
                    continue

                # Use minimum batch strategy for balanced validation
                domain_batch_size = min(n_source_background, n_target_events)

                # Sample domain_batch_size events from each domain
                source_background_indices = torch.where(source_background_mask)[0]
                if n_source_background > domain_batch_size:
                    selected_source_indices = source_background_indices[:domain_batch_size]
                else:
                    selected_source_indices = source_background_indices
                
                if n_target_events > domain_batch_size:
                    selected_target_indices = torch.arange(domain_batch_size, device=self.device)
                else:
                    selected_target_indices = torch.arange(n_target_events, device=self.device)
                
                # Create balanced domain validation batches
                source_domain_batch = {
                    'features': source_batch['features'][selected_source_indices],
                    'lengths': source_batch['lengths'][selected_source_indices],
                    'mask': source_batch['mask'][selected_source_indices]
                }
                
                target_domain_batch = {
                    'features': target_batch['features'][selected_target_indices],
                    'lengths': target_batch['lengths'][selected_target_indices],
                    'mask': target_batch['mask'][selected_target_indices]
                }
                
                # Get domain predictions (WITH gradient reversal)
                source_domain_logits, _ = self.da_model.domain_forward(source_domain_batch)
                target_domain_logits, _ = self.da_model.domain_forward(target_domain_batch)
                
                # Create domain labels (0=source/MC background, 1=target/Exp background)
                source_domain_labels = torch.zeros(domain_batch_size, 1, dtype=torch.float, device=self.device)
                target_domain_labels = torch.ones(domain_batch_size, 1, dtype=torch.float, device=self.device)
                
                # Calculate domain losses
                source_domain_loss = nn.functional.binary_cross_entropy_with_logits(
                    source_domain_logits, source_domain_labels)
                target_domain_loss = nn.functional.binary_cross_entropy_with_logits(
                    target_domain_logits, target_domain_labels)
                
                domain_loss = (source_domain_loss + target_domain_loss) / 2
                val_domain_loss += domain_loss.item()
                
                # Calculate domain accuracy
                source_domain_preds = torch.sigmoid(source_domain_logits) < 0.5  # Should be 0
                target_domain_preds = torch.sigmoid(target_domain_logits) >= 0.5  # Should be 1
                domain_accuracy = (source_domain_preds.float().mean() + target_domain_preds.float().mean()) / 2
                val_domain_accuracy += domain_accuracy.item()
                
                # Collect predictions for validation AUC calculation
                source_probs = torch.sigmoid(source_domain_logits).cpu()
                target_probs = torch.sigmoid(target_domain_logits).cpu()
                
                val_domain_predictions.extend(source_probs.flatten().tolist())
                val_domain_predictions.extend(target_probs.flatten().tolist())
                
                val_domain_labels_true.extend([0.0] * domain_batch_size)  # Source = 0
                val_domain_labels_true.extend([1.0] * domain_batch_size)  # Target = 1
                
                val_domain_batches += 1
        
        # Get classification metrics
        val_metrics = self.metrics_tracker.val_metrics.compute()
        
        # Add domain validation metrics
        if val_domain_batches > 0:
            val_metrics['domain_loss'] = val_domain_loss / val_domain_batches
            val_metrics['domain_accuracy'] = val_domain_accuracy / val_domain_batches
        else:
            val_metrics['domain_loss'] = 0.0
            val_metrics['domain_accuracy'] = 0.5  # Random chance
        
        # Calculate validation domain AUC
        val_domain_auc = 0.0
        if len(val_domain_predictions) > 0 and len(set(val_domain_labels_true)) > 1:  # Need both classes
            try:
                from sklearn.metrics import roc_auc_score
                val_domain_auc = roc_auc_score(val_domain_labels_true, val_domain_predictions)
            except Exception as e:
                logger.warning(f"Failed to calculate validation domain AUC: {e}")
                val_domain_auc = 0.0
        
        val_metrics['domain_auc'] = val_domain_auc
        
        # Calculate validation metrics for signal hits subsets
        val_signal_hits_metrics = self._calculate_signal_hits_metrics(
            all_val_source_logits, all_val_source_labels, all_val_source_magic_numbers
        )
        # Add 'val_' prefix to signal hits metrics
        for key, value in val_signal_hits_metrics.items():
            val_metrics[f'val_{key}'] = value
        
        # Log to TensorBoard
        if self.writer is not None:
            self._log_tensorboard_val(val_metrics, self.current_epoch)
        
        return val_metrics
    
    def _calculate_classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate classification loss with class weighting."""
        loss_config = self.config['training']
        
        if loss_config['loss'] == 'bce_with_logits':
            pos_weight = None
            if hasattr(self, 'class_weights') and loss_config.get('class_weights') != 'auto':
                if loss_config.get('class_weights') is not None:
                    weights = torch.tensor(loss_config['class_weights'], device=self.device)
                else:
                    weights = self.class_weights.to(self.device)
                pos_weight = weights[1] / weights[0]
            
            loss = binary_cross_entropy_with_logits_weighted(logits, labels, pos_weight)
        else:
            raise ValueError(f"Unknown loss function: {loss_config['loss']}")
        
        return loss
    
    def _calculate_signal_hits_metrics(
        self, 
        all_logits: List[torch.Tensor], 
        all_labels: List[torch.Tensor], 
        all_magic_numbers: List[List]
    ) -> Dict[str, float]:
        """Calculate classification metrics for events with >8 and >16 signal hits."""
        metrics = {}
        
        if not all_logits:
            return metrics
        
        try:
            # Concatenate all predictions and labels
            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0).numpy()
            probs = torch.sigmoid(logits).numpy().flatten()
            
            # Calculate signal hits for each event
            signal_hits = np.array([np.sum(np.array(numbers) != 0) for numbers in all_magic_numbers])
            
            # Calculate metrics for >8 signal hits
            mask_8_hits = signal_hits > 8
            if np.sum(mask_8_hits) > 0:
                labels_8 = labels[mask_8_hits]
                probs_8 = probs[mask_8_hits]
                
                if len(np.unique(labels_8)) > 1:  # Need both classes for AUC
                    from sklearn.metrics import roc_auc_score
                    metrics['cls_auc_8hits'] = roc_auc_score(labels_8, probs_8)
                else:
                    metrics['cls_auc_8hits'] = 0.0
                    
                metrics['cls_events_8hits'] = np.sum(mask_8_hits)
            else:
                metrics['cls_auc_8hits'] = 0.0
                metrics['cls_events_8hits'] = 0
                
            # Calculate metrics for >16 signal hits
            mask_16_hits = signal_hits > 16
            if np.sum(mask_16_hits) > 0:
                labels_16 = labels[mask_16_hits]
                probs_16 = probs[mask_16_hits]
                
                if len(np.unique(labels_16)) > 1:  # Need both classes for AUC
                    from sklearn.metrics import roc_auc_score
                    metrics['cls_auc_16hits'] = roc_auc_score(labels_16, probs_16)
                else:
                    metrics['cls_auc_16hits'] = 0.0
                    
                metrics['cls_events_16hits'] = np.sum(mask_16_hits)
            else:
                metrics['cls_auc_16hits'] = 0.0
                metrics['cls_events_16hits'] = 0
                
            logger.debug(f"Signal hits metrics - >8 hits: {metrics['cls_events_8hits']} events, "
                        f"AUC: {metrics['cls_auc_8hits']:.4f} | >16 hits: {metrics['cls_events_16hits']} events, "
                        f"AUC: {metrics['cls_auc_16hits']:.4f}")
                        
        except Exception as e:
            logger.warning(f"Failed to calculate signal hits metrics: {e}")
            metrics.update({
                'cls_auc_8hits': 0.0,
                'cls_events_8hits': 0,
                'cls_auc_16hits': 0.0,
                'cls_events_16hits': 0
            })
        
        return metrics
    
    def _log_tensorboard_train(self, metrics: Dict[str, float], epoch: int):
        """Log training metrics to TensorBoard."""
        # Classification metrics
        if 'loss' in metrics:
            self.writer.add_scalar('Train/Classification_Loss', metrics['loss'], epoch)
        if 'accuracy' in metrics:
            self.writer.add_scalar('Train/Classification_Accuracy', metrics['accuracy'], epoch)
        if 'f1' in metrics:
            self.writer.add_scalar('Train/Classification_F1', metrics['f1'], epoch)
        if 'precision' in metrics:
            self.writer.add_scalar('Train/Classification_Precision', metrics['precision'], epoch)
        if 'recall' in metrics:
            self.writer.add_scalar('Train/Classification_Recall', metrics['recall'], epoch)
        if 'auc' in metrics:
            self.writer.add_scalar('Train/Classification_AUC', metrics['auc'], epoch)
        
        # Domain adaptation metrics
        if 'classification_loss' in metrics:
            self.writer.add_scalar('Train/DA_Classification_Loss', metrics['classification_loss'], epoch)
        if 'domain_loss' in metrics:
            self.writer.add_scalar('Train/DA_Domain_Loss', metrics['domain_loss'], epoch)
        if 'total_loss' in metrics:
            self.writer.add_scalar('Train/DA_Total_Loss', metrics['total_loss'], epoch)
        if 'domain_accuracy' in metrics:
            self.writer.add_scalar('Train/DA_Domain_Accuracy', metrics['domain_accuracy'], epoch)
        if 'domain_auc' in metrics:
            self.writer.add_scalar('Train/DA_Domain_AUC', metrics['domain_auc'], epoch)
        if 'lambda_factor' in metrics:
            self.writer.add_scalar('Train/DA_Lambda_Factor', metrics['lambda_factor'], epoch)
        
        # Domain utilization metrics
        if 'domain_utilization' in metrics:
            self.writer.add_scalar('Train/Domain_Utilization', metrics['domain_utilization'], epoch)
        if 'domain_batch_size_avg' in metrics:
            self.writer.add_scalar('Train/Domain_Batch_Size', metrics['domain_batch_size_avg'], epoch)
        
        # Signal hits subset metrics
        if 'cls_auc_8hits' in metrics:
            self.writer.add_scalar('Train/Classification_AUC_8hits', metrics['cls_auc_8hits'], epoch)
        if 'cls_events_8hits' in metrics:
            self.writer.add_scalar('Train/Classification_Events_8hits', metrics['cls_events_8hits'], epoch)
        if 'cls_auc_16hits' in metrics:
            self.writer.add_scalar('Train/Classification_AUC_16hits', metrics['cls_auc_16hits'], epoch)
        if 'cls_events_16hits' in metrics:
            self.writer.add_scalar('Train/Classification_Events_16hits', metrics['cls_events_16hits'], epoch)
        
        # Domain discriminator metrics for signal hits subsets
        if 'domain_auc_8hits' in metrics:
            self.writer.add_scalar('Train/DA_Domain_AUC_8hits', metrics['domain_auc_8hits'], epoch)
        if 'domain_auc_16hits' in metrics:
            self.writer.add_scalar('Train/DA_Domain_AUC_16hits', metrics['domain_auc_16hits'], epoch)
    
    def _log_tensorboard_val(self, metrics: Dict[str, float], epoch: int):
        """Log validation metrics to TensorBoard."""
        # Classification metrics
        if 'loss' in metrics:
            self.writer.add_scalar('Val/Classification_Loss', metrics['loss'], epoch)
        if 'accuracy' in metrics:
            self.writer.add_scalar('Val/Classification_Accuracy', metrics['accuracy'], epoch)
        if 'f1' in metrics:
            self.writer.add_scalar('Val/Classification_F1', metrics['f1'], epoch)
        if 'precision' in metrics:
            self.writer.add_scalar('Val/Classification_Precision', metrics['precision'], epoch)
        if 'recall' in metrics:
            self.writer.add_scalar('Val/Classification_Recall', metrics['recall'], epoch)
        if 'auc' in metrics:
            self.writer.add_scalar('Val/Classification_AUC', metrics['auc'], epoch)
        
        # Domain metrics
        if 'domain_loss' in metrics:
            self.writer.add_scalar('Val/Domain_Loss', metrics['domain_loss'], epoch)
        if 'domain_accuracy' in metrics:
            self.writer.add_scalar('Val/Domain_Accuracy', metrics['domain_accuracy'], epoch)
        if 'domain_auc' in metrics:
            self.writer.add_scalar('Val/Domain_AUC', metrics['domain_auc'], epoch)
        
        # Validation signal hits subset metrics
        if 'val_cls_auc_8hits' in metrics:
            self.writer.add_scalar('Val/Classification_AUC_8hits', metrics['val_cls_auc_8hits'], epoch)
        if 'val_cls_events_8hits' in metrics:
            self.writer.add_scalar('Val/Classification_Events_8hits', metrics['val_cls_events_8hits'], epoch)
        if 'val_cls_auc_16hits' in metrics:
            self.writer.add_scalar('Val/Classification_AUC_16hits', metrics['val_cls_auc_16hits'], epoch)
        if 'val_cls_events_16hits' in metrics:
            self.writer.add_scalar('Val/Classification_Events_16hits', metrics['val_cls_events_16hits'], epoch)
    
    def _log_tensorboard_lr(self, epoch: int):
        """Log learning rates to TensorBoard."""
        if hasattr(self, 'optimizer_feature'):
            self.writer.add_scalar('LearningRate/Feature_Extractor', 
                                 self.optimizer_feature.param_groups[0]['lr'], epoch)
        if hasattr(self, 'optimizer_classifier'):
            self.writer.add_scalar('LearningRate/Classifier', 
                                 self.optimizer_classifier.param_groups[0]['lr'], epoch)
        if hasattr(self, 'optimizer_discriminator'):
            self.writer.add_scalar('LearningRate/Discriminator', 
                                 self.optimizer_discriminator.param_groups[0]['lr'], epoch)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, training_history: list = None):
        """Save model checkpoint with metrics history."""
        checkpoint = {
            'epoch': epoch,
            'base_model_state_dict': self.da_model.base_model.state_dict(),
            'domain_discriminator_state_dict': self.da_model.domain_discriminator.state_dict(),
            'optimizer_feature_state_dict': self.optimizer_feature.state_dict(),
            'optimizer_classifier_state_dict': self.optimizer_classifier.state_dict(),
            'optimizer_discriminator_state_dict': self.optimizer_discriminator.state_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'normalization_config': self.normalization_config,
            'training_history': training_history[:epoch+1] if training_history else []
        }
        
        # Add scheduler states if they exist
        if hasattr(self, 'scheduler_feature') and self.scheduler_feature is not None:
            checkpoint['scheduler_feature_state_dict'] = self.scheduler_feature.state_dict()
            checkpoint['scheduler_classifier_state_dict'] = self.scheduler_classifier.state_dict()
            checkpoint['scheduler_discriminator_state_dict'] = self.scheduler_discriminator.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'da_checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_da_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best DA model saved at epoch {epoch}")
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'latest_da_checkpoint.pth'
        torch.save(checkpoint, latest_path)
    
    def check_early_stopping(self, current_metric: float) -> bool:
        """Check early stopping criteria."""
        early_stopping_config = self.config['training']['early_stopping']
        
        if not early_stopping_config.get('enabled', True):
            return False
        
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
        """Main domain adaptation training loop."""
        logger.info("Starting domain adaptation training...")
        
        # Prepare data and model
        self.prepare_data()
        self.prepare_model()
        
        # Save configuration
        if self.config['logging'].get('save_config', True):
            config_path = self.output_dir / 'da_config.yaml'
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
            if hasattr(self, 'scheduler_feature') and self.scheduler_feature is not None:
                self.scheduler_feature.step()
                self.scheduler_classifier.step()
                self.scheduler_discriminator.step()
            
            # Log learning rates to TensorBoard
            if self.writer is not None:
                self._log_tensorboard_lr(epoch)
            
            # Save epoch metrics
            self.metrics_tracker.save_epoch_metrics(epoch)
            
            # Combine metrics
            epoch_metrics = {
                'epoch': epoch,
                'learning_rate_feature': self.optimizer_feature.param_groups[0]['lr'],
                'learning_rate_classifier': self.optimizer_classifier.param_groups[0]['lr'],
                'learning_rate_discriminator': self.optimizer_discriminator.param_groups[0]['lr'],
                'epoch_time': time.time() - start_time
            }
            
            training_history.append({**epoch_metrics, **train_metrics, 
                                   **{"val_"+k: v for k,v in val_metrics.items()}})
            
            # Save training history as CSV after each epoch
            self._save_training_history_csv(training_history)
            
            # Logging
            logger.info(f"Epoch {epoch}/{num_epochs-1} completed in {epoch_metrics['epoch_time']:.2f}s")
            logger.info(f"Train - Class Loss: {train_metrics.get('classification_loss', 0):.4f}, "
                       f"Domain Loss: {train_metrics.get('domain_loss', 0):.4f}, "
                       f"Acc: {train_metrics.get('accuracy', 0):.4f}, "
                       f"F1: {train_metrics.get('f1', 0):.4f}")
            logger.info(f"Domain - Acc: {train_metrics.get('domain_accuracy', 0):.4f}, "
                       f"AUC: {train_metrics.get('domain_auc', 0):.4f}, "
                       f"Lambda: {train_metrics.get('lambda_factor', 0):.4f}")
            logger.info(f"Domain Subsets - >8 hits AUC: {train_metrics.get('domain_auc_8hits', 0):.4f}, "
                       f">16 hits AUC: {train_metrics.get('domain_auc_16hits', 0):.4f}")
            logger.info(f"Signal Hits - >8 hits: {train_metrics.get('cls_events_8hits', 0)} events, "
                       f"AUC: {train_metrics.get('cls_auc_8hits', 0):.4f} | "
                       f">16 hits: {train_metrics.get('cls_events_16hits', 0)} events, "
                       f"AUC: {train_metrics.get('cls_auc_16hits', 0):.4f}")
            
            if val_metrics:
                logger.info(f"Val - Class Loss: {val_metrics.get('loss', 0):.4f}, "
                           f"Acc: {val_metrics.get('accuracy', 0):.4f}, "
                           f"F1: {val_metrics.get('f1', 0):.4f}")
                logger.info(f"Val Domain - Loss: {val_metrics.get('domain_loss', 0):.4f}, "
                           f"Acc: {val_metrics.get('domain_accuracy', 0):.4f}, "
                           f"AUC: {val_metrics.get('domain_auc', 0):.4f}")
                logger.info(f"Val Signal Hits - >8 hits: {val_metrics.get('val_cls_events_8hits', 0)} events, "
                           f"AUC: {val_metrics.get('val_cls_auc_8hits', 0):.4f} | "
                           f">16 hits: {val_metrics.get('val_cls_events_16hits', 0)} events, "
                           f"AUC: {val_metrics.get('val_cls_auc_16hits', 0):.4f}")
            
            # Check for best model
            monitor_metric = self.config['training']['early_stopping'].get('monitor', 'val_f1')
            current_metric = val_metrics.get(monitor_metric.replace('val_', ''), 
                                           train_metrics.get(monitor_metric.replace('train_', ''), 0))
            
            is_best = False
            if self.config['training']['early_stopping'].get('mode', 'max') == 'max':
                is_best = current_metric > self.best_metric
            else:
                is_best = current_metric < self.best_metric
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best, training_history)
            
            # Early stopping
            if val_metrics and self.check_early_stopping(current_metric):
                logger.info(f"Domain adaptation training stopped early at epoch {epoch}")
                break
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
        
        # Save final results
        self._save_training_results(training_history)
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            logger.info(f"TensorBoard logs saved to: {self.output_dir / 'tensorboard'}")
        
        logger.info(f"Domain adaptation training completed. Best {monitor_metric}: {self.best_metric:.4f} at epoch {self.best_epoch}")
    
    def _save_training_history_csv(self, training_history: list):
        """Save training history as CSV file (called at each epoch)."""
        try:
            import pandas as pd
            
            if not training_history:
                logger.warning("Empty training history - skipping CSV save")
                return
            
            df = pd.DataFrame(training_history)
            history_path = self.output_dir / 'da_training_history.csv'
            df.to_csv(history_path, index=False)
            
            logger.debug(f"Training history saved to {history_path} (epochs: {len(training_history)})")
            
        except Exception as e:
            logger.error(f"Failed to save training history CSV: {e}")
    
    def _save_training_results(self, training_history: list):
        """Save training history and final results."""
        import pandas as pd
        
        # Save training history as CSV (final save)
        df = pd.DataFrame(training_history)
        history_path = self.output_dir / 'da_training_history.csv'
        df.to_csv(history_path, index=False)
        
        # Calculate final domain discriminator metrics from last epoch
        final_domain_metrics = {}
        if training_history:
            last_epoch = training_history[-1]
            final_domain_metrics = {
                'final_domain_accuracy': last_epoch.get('domain_accuracy', 0.0),
                'final_domain_loss': last_epoch.get('domain_loss', 0.0),
                'final_lambda_factor': last_epoch.get('lambda_factor', 0.0),
                'final_val_domain_accuracy': last_epoch.get('val_domain_accuracy', 0.0),
                'final_val_domain_loss': last_epoch.get('val_domain_loss', 0.0)
            }
        
        # Calculate domain adaptation effectiveness metrics
        domain_adaptation_metrics = {}
        if len(training_history) > 5:  # Only if we have enough epochs
            # Compare early vs late domain accuracy to see adaptation progress
            early_epochs = training_history[:5]
            late_epochs = training_history[-5:]
            
            early_domain_acc = sum(epoch.get('domain_accuracy', 0.5) for epoch in early_epochs) / len(early_epochs)
            late_domain_acc = sum(epoch.get('domain_accuracy', 0.5) for epoch in late_epochs) / len(late_epochs)
            
            domain_adaptation_metrics = {
                'early_domain_accuracy': early_domain_acc,
                'late_domain_accuracy': late_domain_acc,
                'domain_adaptation_progress': late_domain_acc - early_domain_acc,
                'domain_confusion_achieved': abs(late_domain_acc - 0.5) < 0.1  # Close to random = good confusion
            }
        
        # Save final metrics summary
        summary = {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'total_epochs': self.current_epoch + 1,
            'final_train_metrics': self.metrics_tracker.train_metrics.compute(),
            'final_val_metrics': self.metrics_tracker.val_metrics.compute(),
            'domain_discriminator_metrics': final_domain_metrics,
            'domain_adaptation_analysis': domain_adaptation_metrics,
            'normalization_config': self.normalization_config,
            'model_parameters': self.da_model.count_parameters()
        }
        
        summary_path = self.output_dir / 'da_training_summary.yaml'
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Domain adaptation training results saved to {self.output_dir}")
        
        # Log domain adaptation effectiveness summary
        if domain_adaptation_metrics:
            logger.info(f"Domain Adaptation Analysis:")
            logger.info(f"  Early domain accuracy: {domain_adaptation_metrics['early_domain_accuracy']:.4f}")
            logger.info(f"  Late domain accuracy: {domain_adaptation_metrics['late_domain_accuracy']:.4f}")
            logger.info(f"  Adaptation progress: {domain_adaptation_metrics['domain_adaptation_progress']:.4f}")
            logger.info(f"  Domain confusion achieved: {domain_adaptation_metrics['domain_confusion_achieved']}")


def main():
    """Main entry point for domain adaptation training.
    
    Usage:
        python src/training/da_numu_trainer.py --config experiments/da_config.yaml
        
    TensorBoard Visualization:
        # Start TensorBoard in separate terminal (after training starts):
        tensorboard --logdir=experiments/your_experiment_name/tensorboard
        
        # Open browser to: http://localhost:6006
        
        # Available metrics:
        # - Train/Val Classification: Loss, Accuracy, F1, Precision, Recall, AUC
        # - Domain Adaptation: Domain Loss, Domain Accuracy, Lambda Factor
        # - Learning Rates: Feature Extractor, Classifier, Discriminator
        # - Utilization: Domain Batch Size, Domain Utilization Rate
        
    Configuration:
        logging:
          tensorboard: true  # Enable TensorBoard logging (default: true)
          output_dir: "experiments"
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train domain adaptation model for neutrino detection')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to domain adaptation configuration YAML file')
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
    
    # Create trainer and start training (with proper resource cleanup)
    with DomainAdaptationTrainer(config) as trainer:
        trainer.train()


if __name__ == '__main__':
    main()