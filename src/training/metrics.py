"""
Metrics calculation for binary neutrino classification.

Implements comprehensive metrics for evaluating neutrino vs muon binary classification
performance, including class-balanced metrics and confusion matrix analysis.
"""

from typing import Dict, List, Tuple, Optional
import logging

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class BinaryClassificationMetrics:
    """
    Comprehensive binary classification metrics for neutrino detection.
    
    Handles both batch-wise updates during training and epoch-level aggregation.
    Designed specifically for neutrino (positive) vs muon (negative) classification.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.labels = []
        self.logits = []
        self.losses = []
    
    def update(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        loss: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with batch predictions.
        
        Args:
            logits: Model logits [batch_size, 1]
            labels: True labels [batch_size] (bool or 0/1)
            loss: Batch loss value (optional)
        """
        # Convert to numpy for sklearn metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            true_labels = labels.cpu().numpy().astype(int).flatten()
            
            self.predictions.extend(preds.tolist())
            self.labels.extend(true_labels.tolist())
            self.logits.extend(logits.cpu().numpy().flatten().tolist())
            
            if loss is not None:
                self.losses.append(loss.item())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions.
        
        Returns:
            Dict with metric names and values
        """
        if not self.predictions:
            logger.warning("No predictions accumulated for metrics computation")
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        logits = np.array(self.logits)
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['precision'] = precision_score(labels, predictions, zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, zero_division=0)
        metrics['f1'] = f1_score(labels, predictions, zero_division=0)
        
        # AUC (requires probabilities)
        try:
            metrics['auc'] = roc_auc_score(labels, probs)
        except ValueError as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics['auc'] = 0.0
        
        # Class-specific metrics
        cm = confusion_matrix(labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Sensitivity (True Positive Rate) - important for neutrino detection
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Specificity (True Negative Rate) - important for background rejection
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Precision for each class
            metrics['precision_neutrino'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['precision_muon'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            # Class counts
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            
            # Total samples per class
            metrics['total_neutrinos'] = int(tp + fn)
            metrics['total_muons'] = int(tn + fp)
        
        # Loss (if available)
        if self.losses:
            metrics['loss'] = np.mean(self.losses)
        
        # Class balance information
        positive_ratio = np.mean(labels)
        metrics['positive_class_ratio'] = positive_ratio
        metrics['class_balance'] = min(positive_ratio, 1 - positive_ratio) / max(positive_ratio, 1 - positive_ratio)
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix as numpy array."""
        if not self.predictions:
            return np.array([])
        
        return confusion_matrix(self.labels, self.predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if not self.predictions:
            return "No predictions available"
        
        target_names = ['Muon (Background)', 'Neutrino (Signal)']
        return classification_report(
            self.labels, 
            self.predictions, 
            target_names=target_names,
            zero_division=0
        )
    
    def summary(self) -> str:
        """Get human-readable summary of metrics."""
        metrics = self.compute()
        
        if not metrics:
            return "No metrics available"
        
        summary_lines = [
            f"Binary Classification Metrics Summary:",
            f"{'='*40}",
            f"Accuracy:     {metrics.get('accuracy', 0):.4f}",
            f"Precision:    {metrics.get('precision', 0):.4f}",
            f"Recall:       {metrics.get('recall', 0):.4f}",
            f"F1 Score:     {metrics.get('f1', 0):.4f}",
            f"AUC:          {metrics.get('auc', 0):.4f}",
            f"",
            f"Class-Specific Metrics:",
            f"Sensitivity (Neutrino Recall): {metrics.get('sensitivity', 0):.4f}",
            f"Specificity (Muon Recall):     {metrics.get('specificity', 0):.4f}",
            f"",
            f"Sample Counts:",
            f"Total Neutrinos: {metrics.get('total_neutrinos', 0)}",
            f"Total Muons:     {metrics.get('total_muons', 0)}",
            f"True Positives:  {metrics.get('true_positives', 0)}",
            f"False Positives: {metrics.get('false_positives', 0)}",
            f"True Negatives:  {metrics.get('true_negatives', 0)}",
            f"False Negatives: {metrics.get('false_negatives', 0)}"
        ]
        
        if 'loss' in metrics:
            summary_lines.insert(7, f"Loss:         {metrics['loss']:.4f}")
        
        return "\n".join(summary_lines)


def calculate_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate balanced class weights for binary classification.
    
    Args:
        labels: Binary labels tensor [batch_size]
        
    Returns:
        Class weights tensor [2] for [negative_class, positive_class]
    """
    labels_np = labels.cpu().numpy().astype(int)
    positive_count = np.sum(labels_np)
    negative_count = len(labels_np) - positive_count
    total_count = len(labels_np)
    
    if positive_count == 0 or negative_count == 0:
        logger.warning("One class has zero samples - using equal weights")
        return torch.tensor([1.0, 1.0])
    
    # Inverse frequency weighting
    weight_negative = total_count / (2 * negative_count)
    weight_positive = total_count / (2 * positive_count)
    
    weights = torch.tensor([weight_negative, weight_positive], dtype=torch.float32)
    
    logger.info(f"Calculated class weights - Negative: {weight_negative:.3f}, Positive: {weight_positive:.3f}")
    logger.info(f"Class distribution - Negative: {negative_count}, Positive: {positive_count}")
    
    return weights


def binary_cross_entropy_with_logits_weighted(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    pos_weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Weighted binary cross-entropy loss with logits.
    
    Args:
        logits: Model logits [batch_size, 1]
        targets: Binary targets [batch_size] (0 or 1)
        pos_weight: Weight for positive class (optional)
        
    Returns:
        Loss value
    """
    targets = targets.float().view(-1, 1)
    logits = logits.view(-1, 1)
    
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='mean'
    )
    
    return loss


class MetricsTracker:
    """
    Track metrics across training and validation phases.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.train_metrics = BinaryClassificationMetrics(device)
        self.val_metrics = BinaryClassificationMetrics(device)
        self.history = {'train': {}, 'val': {}}
    
    def reset_epoch(self):
        """Reset metrics for new epoch."""
        self.train_metrics.reset()
        self.val_metrics.reset()
    
    def update_train(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        """Update training metrics."""
        self.train_metrics.update(logits, labels, loss)
    
    def update_val(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        """Update validation metrics."""
        self.val_metrics.update(logits, labels, loss)
    
    def compute_epoch_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics for current epoch."""
        train_metrics = self.train_metrics.compute()
        val_metrics = self.val_metrics.compute()
        
        # Add prefixes
        train_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
        val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
        
        return {'train': train_metrics, 'val': val_metrics}
    
    def save_epoch_metrics(self, epoch: int):
        """Save current epoch metrics to history."""
        metrics = self.compute_epoch_metrics()
        
        for phase in ['train', 'val']:
            for metric_name, value in metrics[phase].items():
                if metric_name not in self.history[phase]:
                    self.history[phase][metric_name] = []
                self.history[phase][metric_name].append(value)
    
    def get_best_metric(self, metric_name: str, mode: str = 'max') -> Tuple[float, int]:
        """
        Get best value of a metric and the epoch it occurred.
        
        Args:
            metric_name: Name of metric (e.g., 'val_f1')
            mode: 'max' or 'min'
            
        Returns:
            (best_value, best_epoch)
        """
        phase, metric = metric_name.split('_', 1)
        
        if phase not in self.history or metric not in self.history[phase]:
            return 0.0, 0
        
        values = self.history[phase][metric]
        if not values:
            return 0.0, 0
        
        if mode == 'max':
            best_value = max(values)
            best_epoch = values.index(best_value)
        else:
            best_value = min(values)
            best_epoch = values.index(best_value)
        
        return best_value, best_epoch