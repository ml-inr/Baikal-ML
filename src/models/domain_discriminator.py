"""
Domain discriminator and gradient reversal layer for domain adaptation.

Implements Domain-Adversarial Neural Networks (DANN) components for MC→Exp
domain adaptation in neutrino detection. The gradient reversal layer enables
adversarial training to learn domain-invariant features.
"""

from typing import Dict
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for adversarial domain adaptation.
    
    During forward pass, acts as identity function.
    During backward pass, reverses gradients by multiplying with -lambda.
    This enables adversarial training where the feature extractor learns
    to confuse the domain discriminator.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_factor: float) -> torch.Tensor:
        """Forward pass - identity function."""
        ctx.lambda_factor = lambda_factor
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass - reverse gradients."""
        return grad_output.neg() * ctx.lambda_factor, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer wrapper for easy integration.
    
    Args:
        lambda_factor: Scaling factor for gradient reversal (default: 1.0)
    """
    
    def __init__(self, lambda_factor: float = 1.0):
        super().__init__()
        self.lambda_factor = lambda_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal to input tensor."""
        return GradientReversalFunction.apply(x, self.lambda_factor)
    
    def set_lambda(self, lambda_factor: float):
        """Update lambda factor during training."""
        self.lambda_factor = lambda_factor


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator network for distinguishing MC vs Exp data.
    
    Takes feature representations and predicts domain (0=MC, 1=Exp).
    Used in adversarial training to learn domain-invariant features.
    
    Args:
        input_dim: Dimension of input features from feature extractor
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
        use_gradient_reversal: Whether to include gradient reversal layer
        lambda_factor: Initial lambda for gradient reversal
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        use_gradient_reversal: bool = True,
        lambda_factor: float = 1.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_gradient_reversal = use_gradient_reversal
        
        # Gradient reversal layer
        if use_gradient_reversal:
            self.gradient_reversal = GradientReversalLayer(lambda_factor)
        
        # Build discriminator layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final domain classification layer (binary: MC vs Exp)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.discriminator = nn.Sequential(*layers)
        
        logger.info(f"Created DomainDiscriminator: {input_dim}→{hidden_dims}→1, "
                   f"dropout={dropout}, batch_norm={use_batch_norm}, "
                   f"gradient_reversal={use_gradient_reversal}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through domain discriminator.
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            domain_logits: Domain classification logits [batch_size, 1]
                          (0=MC, 1=Exp after sigmoid)
        """
        # Apply gradient reversal if enabled
        if self.use_gradient_reversal:
            features = self.gradient_reversal(features)
        
        # Domain classification
        domain_logits = self.discriminator(features)
        return domain_logits
    
    def set_lambda(self, lambda_factor: float):
        """Update lambda factor for gradient reversal."""
        if self.use_gradient_reversal:
            self.gradient_reversal.set_lambda(lambda_factor)
    
    def predict_domain(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict domain probabilities.
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            domain_probs: Domain probabilities [batch_size, 1] (0=MC, 1=Exp)
        """
        with torch.no_grad():
            domain_logits = self.forward(features)
            domain_probs = torch.sigmoid(domain_logits)
        return domain_probs


class DomainAdaptationModel(nn.Module):
    """
    Complete domain adaptation model combining feature extractor and domain discriminator.
    
    Wraps the existing NuMuClassifierModel with an additional domain discriminator
    for adversarial domain adaptation training.
    
    Args:
        base_model: Pre-initialized NuMuClassifierModel
        discriminator_config: Configuration dict for domain discriminator
    """
    
    def __init__(self, base_model: nn.Module, discriminator_config: Dict):
        super().__init__()
        
        self.base_model = base_model
        
        # Extract feature dimension from base model
        feature_dim = base_model.feature_extractor.feature_dim
        
        # Create domain discriminator
        self.domain_discriminator = DomainDiscriminator(
            input_dim=feature_dim,
            hidden_dims=discriminator_config.get('hidden_dims', [64, 32]),
            dropout=discriminator_config.get('dropout', 0.2),
            use_batch_norm=discriminator_config.get('use_batch_norm', True),
            use_gradient_reversal=discriminator_config.get('use_gradient_reversal', True),
            lambda_factor=discriminator_config.get('lambda_factor', 1.0)
        )
        
        logger.info(f"Created DomainAdaptationModel with feature_dim={feature_dim}")
    
    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False):
        """
        Forward pass through domain adaptation model.
        
        Args:
            batch: Batch dict containing 'features', 'lengths', 'mask'
            return_features: Whether to return intermediate features
            
        Returns:
            If return_features=False:
                class_logits: Classification logits [batch_size, 1]
            If return_features=True:
                (class_logits, features, domain_logits)
        """
        # Extract features using base model
        features = self.base_model.get_feature_representation(batch)
        
        # Classification prediction
        class_logits = self.base_model.classifier(features)
        
        if return_features:
            # Domain prediction (for training)
            domain_logits = self.domain_discriminator(features)
            return class_logits, features, domain_logits
        else:
            # Inference mode
            return class_logits
    
    def classification_forward(self, batch: Dict[str, torch.Tensor]):
        """
        Forward pass for classification only (no domain computation).
        
        This method extracts features and computes classification logits without
        applying the gradient reversal layer or computing domain predictions.
        Used to avoid unwanted gradient contamination during domain adaptation training.
        
        Args:
            batch: Batch dict containing 'features', 'lengths', 'mask'
            
        Returns:
            class_logits: Classification logits [batch_size, 1]
            features: Feature representations [batch_size, feature_dim]
        """
        # Extract features using base model (no GRL applied)
        features = self.base_model.get_feature_representation(batch)
        
        # Classification logits
        class_logits = self.base_model.classifier(features)
        
        return class_logits, features
    
    def domain_forward(self, batch: Dict[str, torch.Tensor]):
        """
        Forward pass for domain discrimination only.
        
        This method extracts features and computes domain predictions with
        proper gradient reversal applied. Used for balanced domain adaptation
        training on muon-only batches.
        
        Args:
            batch: Batch dict containing 'features', 'lengths', 'mask'
            
        Returns:
            domain_logits: Domain prediction logits [batch_size, 1]  
            features: Feature representations [batch_size, feature_dim]
        """
        # Extract features using base model
        features = self.base_model.get_feature_representation(batch)
        
        # Domain prediction with gradient reversal
        domain_logits = self.domain_discriminator(features)
        
        return domain_logits, features
    
    def get_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features without gradient reversal."""
        return self.base_model.get_feature_representation(batch)
    
    def predict_domain(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict domain without training (no gradient reversal)."""
        features = self.get_features(batch)
        # Temporarily disable gradient reversal for domain prediction
        original_use_grl = self.domain_discriminator.use_gradient_reversal
        self.domain_discriminator.use_gradient_reversal = False
        
        domain_probs = self.domain_discriminator.predict_domain(features)
        
        # Restore gradient reversal setting
        self.domain_discriminator.use_gradient_reversal = original_use_grl
        return domain_probs
    
    def set_lambda(self, lambda_factor: float):
        """Update lambda factor for gradient reversal."""
        self.domain_discriminator.set_lambda(lambda_factor)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        base_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        domain_params = sum(p.numel() for p in self.domain_discriminator.parameters() if p.requires_grad)
        total_params = base_params + domain_params
        
        return {
            'base_model': base_params,
            'domain_discriminator': domain_params,
            'total': total_params
        }


def create_da_model(base_model: nn.Module, config: Dict) -> DomainAdaptationModel:
    """
    Factory function to create domain adaptation model.
    
    Args:
        base_model: Pre-trained or initialized NuMuClassifierModel
        config: Domain adaptation configuration dict
        
    Returns:
        DomainAdaptationModel instance
    """
    discriminator_config = config.get('domain_discriminator', {})
    return DomainAdaptationModel(base_model, discriminator_config)