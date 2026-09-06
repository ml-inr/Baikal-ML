"""
Base neural network models for neutrino detection using attention-based architectures.

This module implements attention-based feature extractors and classifiers optimized
for neutrino detection from variable-length hit sequences. Uses Transformer encoders
to capture complex spatial-temporal relationships in detector data.
"""

from typing import Dict, Tuple, Optional
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Normalized amplitude clip threshold corresponding to Q=100 PE.
# Computed as (Q_max - amp_mean) / amp_std = (100 - 2) / 60 ≈ 1.633.
_AMP_CLIP_Q100: float = 98.0 / 60.0


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence position information in Transformer.
    
    Since hit sequences have temporal ordering, we add positional information
    to help the attention mechanism understand sequence structure.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AttentionFeatureExtractor(nn.Module):
    """
    Attention-based feature extractor for variable-length hit sequences.
    
    Uses Transformer encoder architecture to process sequences of detector hits
    with 5D features [amplitude, time, x, y, z]. Captures both temporal and
    spatial relationships through self-attention mechanisms.
    
    Args:
        input_dim: Dimension of input features (default: 5 for [amp, time, x, y, z])
        d_model: Transformer model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Feedforward network dimension
        dropout: Dropout probability
        pooling: Pooling strategy ('cls', 'mean', 'max', 'attention')
        use_positional_encoding: Whether to add positional encoding
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pooling: str = 'cls',
        use_positional_encoding: bool = True,
        max_seq_len: int = 500
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pooling = pooling
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # CLS token for classification (if using cls pooling)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output feature dimension
        self.feature_dim = d_model
        
        # Attention pooling layer (if needed)
        if pooling == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Created AttentionFeatureExtractor: {input_dim}→{d_model}, "
                   f"{num_layers} layers, {num_heads} heads, pooling={pooling}")
    
    def forward(
        self, 
        sequences: torch.Tensor, 
        lengths: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through attention-based feature extractor.
        
        Args:
            sequences: Input sequences [batch_size, max_len, input_dim]
            lengths: Actual sequence lengths [batch_size]
            mask: Mask for padded positions [batch_size, max_len]
            
        Returns:
            features: Fixed-size feature representation [batch_size, feature_dim]
        """
        batch_size, max_len, _ = sequences.shape
        
        # Project input to model dimension
        embedded = self.input_projection(sequences)  # [batch_size, max_len, d_model]
        
        # Add positional encoding
        if self.use_positional_encoding:
            # Transpose for positional encoding (expects seq_len first)
            embedded = embedded.transpose(0, 1)  # [seq_len, batch_size, d_model]
            embedded = self.pos_encoding(embedded)
            embedded = embedded.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Add CLS token if using cls pooling
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
            embedded = torch.cat([cls_tokens, embedded], dim=1)  # [batch_size, max_len+1, d_model]
            
            # Update mask to include CLS token
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # [batch_size, max_len+1]
        
        # Create attention mask for transformer (inverted: True = ignore)
        attention_mask = ~mask  # [batch_size, seq_len]
        
        # Transformer encoder forward pass
        encoded = self.transformer_encoder(
            embedded, 
            src_key_padding_mask=attention_mask
        )  # [batch_size, seq_len, d_model]
        
        # Apply pooling strategy
        if self.pooling == 'cls':
            # Use CLS token representation
            features = encoded[:, 0, :]  # [batch_size, d_model]
            
        elif self.pooling == 'mean':
            # Mean pooling over valid positions
            features = self._mean_pooling(encoded, mask)
            
        elif self.pooling == 'max':
            # Max pooling over valid positions
            features = self._max_pooling(encoded, mask)
            
        elif self.pooling == 'attention':
            # Attention-based pooling
            features = self._attention_pooling(encoded, mask)
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Layer normalization and dropout
        features = self.layer_norm(features)
        features = self.dropout(features)
        
        return features
    
    def _mean_pooling(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over valid (non-padded) positions."""
        if self.pooling == 'cls':
            # Skip CLS token for mean pooling
            encoded = encoded[:, 1:, :]
            mask = mask[:, 1:]
        
        # Apply mask and compute mean
        masked_encoded = encoded * mask.unsqueeze(-1).float()
        lengths = mask.sum(dim=1, keepdim=True).float()
        features = masked_encoded.sum(dim=1) / (lengths + 1e-8)
        return features
    
    def _max_pooling(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Max pooling over valid positions."""
        if self.pooling == 'cls':
            # Skip CLS token for max pooling
            encoded = encoded[:, 1:, :]
            mask = mask[:, 1:]
        
        # Set padded positions to large negative values
        masked_encoded = encoded.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        features = masked_encoded.max(dim=1)[0]
        return features
    
    def _attention_pooling(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attention-based pooling over valid positions."""
        if self.pooling == 'cls':
            # Skip CLS token for attention pooling
            encoded = encoded[:, 1:, :]
            mask = mask[:, 1:]
        
        # Compute attention weights
        attention_scores = self.attention_pooling(encoded).squeeze(-1)  # [batch_size, seq_len]
        
        # Mask attention scores for padded positions
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Weighted sum of encoded representations
        features = (encoded * attention_weights.unsqueeze(-1)).sum(dim=1)
        return features


class BinaryClassifier(nn.Module):
    """
    Binary classifier head for neutrino vs muon classification.
    
    Takes feature representations from the attention-based feature extractor 
    and outputs binary classification logits.
    
    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build classifier layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
        logger.info(f"Created BinaryClassifier: {input_dim}→{hidden_dims}→1, "
                   f"dropout={dropout}, batch_norm={use_batch_norm}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through binary classifier.
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            logits: Binary classification logits [batch_size, 1]
        """
        logits = self.classifier(features)
        return logits


class NuMuClassifierModel(nn.Module):
    """
    Complete attention-based model for standard neutrino detection without domain adaptation.
    
    Combines attention-based feature extractor and binary classifier for end-to-end training.
    Uses Transformer encoder to capture complex spatial-temporal relationships in hit data.
    """
    
    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.amp_clip: Optional[float] = config.get('amp_clip', _AMP_CLIP_Q100)

        # Feature extractor
        feature_config = config.get('feature_extractor', {})
        self.feature_extractor = AttentionFeatureExtractor(
            input_dim=feature_config.get('input_dim', 5),
            d_model=feature_config.get('d_model', 128),
            num_heads=feature_config.get('num_heads', 8),
            num_layers=feature_config.get('num_layers', 4),
            dim_feedforward=feature_config.get('dim_feedforward', 512),
            dropout=feature_config.get('dropout', 0.1),
            pooling=feature_config.get('pooling', 'cls'),
            use_positional_encoding=feature_config.get('use_positional_encoding', True),
            max_seq_len=feature_config.get('max_seq_len', 500)
        )
        
        # Binary classifier
        classifier_config = config.get('classifier', {})
        self.classifier = BinaryClassifier(
            input_dim=self.feature_extractor.feature_dim,
            hidden_dims=classifier_config.get('hidden_dims', [64, 32]),
            dropout=classifier_config.get('dropout', 0.2),
            use_batch_norm=classifier_config.get('use_batch_norm', True)
        )

        # Spectral normalization of the encoder (fights feature collapse / OOD).
        # Applied here so both training and inference (create_model from the
        # stored config) register the same parametrization before load_state_dict.
        if config.get('spectral_norm', {}).get('enabled', False):
            apply_spectral_norm_encoder(self.feature_extractor)

        logger.info(f"Created StandardNeutrinoModel with {self.count_parameters()} parameters")
    
    def _clip_amplitude(
        self,
        batch: Dict[str, torch.Tensor],
        amp_clip: Optional[float],
    ) -> Dict[str, torch.Tensor]:
        """Clip the amplitude channel (index 0) on real hits only.

        Args:
            batch: Batch dict with 'features' (B, L, 5) and 'mask' (B, L).
            amp_clip: Upper bound in normalized space. None = no clipping.

        Returns:
            Batch dict with clipped features (new tensor, original unchanged).
        """
        if amp_clip is None:
            return batch
        features = batch['features'].clone()
        amp = features[:, :, 0]
        features[:, :, 0] = torch.where(batch['mask'], amp.clamp(max=amp_clip), amp)
        return {**batch, 'features': features}

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through complete model.

        Args:
            batch: Batch dict containing 'features', 'lengths', 'mask'.

        Returns:
            logits: Binary classification logits [batch_size, 1]
        """
        batch = self._clip_amplitude(batch, self.amp_clip)
        features = self.feature_extractor(
            sequences=batch['features'],
            lengths=batch['lengths'],
            mask=batch['mask']
        )
        logits = self.classifier(features)
        return logits

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_feature_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get feature representation without classification (useful for DA).

        Args:
            batch: Batch dict containing 'features', 'lengths', 'mask'.

        Returns:
            features: Feature representation [batch_size, feature_dim]
        """
        batch = self._clip_amplitude(batch, self.amp_clip)
        return self.feature_extractor(
            sequences=batch['features'],
            lengths=batch['lengths'],
            mask=batch['mask']
        )


def apply_spectral_norm_encoder(fe: "AttentionFeatureExtractor") -> int:
    """Apply spectral normalization to every linear weight of the transformer
    encoder — input projection, each layer's attention (combined QKV in_proj +
    output projection) and feed-forward (linear1, linear2). This bounds the
    encoder's Lipschitz constant, which fights feature collapse / over-confident
    mapping of OOD (exp) inputs. Encoder only — the classifier head and domain
    discriminator are deliberately left unconstrained.

    Returns the number of parametrized weights.
    """
    from torch.nn.utils.parametrizations import spectral_norm

    n = 0
    spectral_norm(fe.input_projection, name="weight"); n += 1
    for layer in fe.transformer_encoder.layers:
        attn = layer.self_attn
        if getattr(attn, "in_proj_weight", None) is not None:
            spectral_norm(attn, name="in_proj_weight"); n += 1
        else:  # separate q/k/v projections (kdim/vdim != embed_dim)
            for pn in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
                if getattr(attn, pn, None) is not None:
                    spectral_norm(attn, name=pn); n += 1
        spectral_norm(attn.out_proj, name="weight"); n += 1
        spectral_norm(layer.linear1, name="weight"); n += 1
        spectral_norm(layer.linear2, name="weight"); n += 1
    logger.info(f"Spectral norm applied to {n} encoder weights")
    return n


class RandomFeatureGPHead(nn.Module):
    """SNGP output head: a Random-Fourier-Feature approximation of a GP with a
    Laplace covariance and mean-field logit adjustment (Liu et al. 2020).

    Distance-aware: inputs far from the training manifold (large embedding
    distance, enabled by the spectral-normed encoder) get large predictive
    variance → the mean-field logit ``logit/√(1+λ·var)`` is shrunk toward 0
    (score → 0.5). This is the principled counterpart to the empirical
    Mahalanobis OOD score, without its high-dim covariance-inversion instability.

    Interface mirrors ``BinaryClassifier``: ``forward(features) -> logits`` so
    the inference tooling (feature_extractor -> classifier) works unchanged. In
    eval mode the returned logits are already mean-field adjusted. The trainer
    additionally calls ``reset_precision`` / ``update_precision`` /
    ``update_covariance`` to build the Laplace covariance.

    Args:
        input_dim: encoder feature dim (d_model).
        num_rff: number of random Fourier features (GP approximation rank).
        length_scale: RBF length scale for the random features.
        ridge: precision-matrix ridge (prior precision), also numerical floor.
        mean_field_factor: λ in the mean-field logit (π/8 for the probit approx).
        normalize_input: L2-normalize the encoder features before the RFF map
            (makes the GP distance cosine-based, robust to feature norm drift).
    """

    def __init__(
        self,
        input_dim: int,
        num_rff: int = 1024,
        length_scale: float = 1.0,
        ridge: float = 1e-3,
        mean_field_factor: float = math.pi / 8.0,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_rff = num_rff
        self.length_scale = length_scale
        self.ridge = ridge
        self.mean_field_factor = mean_field_factor
        self.normalize_input = normalize_input

        # Fixed random Fourier features: phi(h) = sqrt(2/D) cos(W h + b).
        self.register_buffer("rff_weight", torch.randn(num_rff, input_dim))
        self.register_buffer("rff_bias", torch.rand(num_rff) * 2.0 * math.pi)

        # Trainable GP output weights (binary -> 1 output).
        self.beta = nn.Linear(num_rff, 1, bias=False)

        # Laplace covariance state (buffers -> saved in state_dict).
        self.register_buffer("precision", ridge * torch.eye(num_rff))
        self.register_buffer("covariance", torch.eye(num_rff))
        self.register_buffer("cov_valid", torch.zeros(1))  # 0/1 flag

        logger.info(
            f"Created RandomFeatureGPHead: {input_dim}->{num_rff} RFF, "
            f"length_scale={length_scale}, ridge={ridge}, normalize_input={normalize_input}"
        )

    def _phi(self, h: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        proj = F.linear(h / self.length_scale, self.rff_weight, self.rff_bias)
        return math.sqrt(2.0 / self.num_rff) * torch.cos(proj)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits; mean-field adjusted in eval when covariance is valid."""
        phi = self._phi(features)
        logits = self.beta(phi)
        if (not self.training) and bool(self.cov_valid.item()):
            var = torch.einsum("bi,ij,bj->b", phi, self.covariance, phi).clamp_min(0.0)
            logits = logits / torch.sqrt(1.0 + self.mean_field_factor * var).unsqueeze(-1)
        return logits

    @torch.no_grad()
    def reset_precision(self) -> None:
        self.precision.copy_(self.ridge * torch.eye(self.num_rff, device=self.precision.device))
        self.cov_valid.zero_()

    @torch.no_grad()
    def update_precision(self, features: torch.Tensor) -> None:
        """Accumulate the Laplace precision: Σ_i p_i(1-p_i) φ_i φ_iᵀ (+ ridge init)."""
        phi = self._phi(features)
        p = torch.sigmoid(self.beta(phi)).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
        w = p * (1.0 - p)
        self.precision.add_(torch.einsum("b,bi,bj->ij", w, phi, phi))

    @torch.no_grad()
    def update_covariance(self) -> None:
        """Invert the accumulated precision to get the predictive covariance."""
        eye = torch.eye(self.num_rff, device=self.precision.device)
        self.covariance.copy_(torch.linalg.solve(self.precision, eye))
        self.cov_valid.fill_(1.0)


class NuMuSNGPModel(nn.Module):
    """SNGP nu-classifier: spectral-normed attention encoder + RFF-GP head.

    Same encoder as ``NuMuClassifierModel`` (so weights are transferable), but
    spectral normalization is ON by default (SNGP needs the bi-Lipschitz encoder
    for distance-awareness) and the head is a ``RandomFeatureGPHead``. Exposes
    ``feature_extractor`` and ``classifier`` so the existing inference utilities
    work; ``encode`` returns the encoder features for the covariance update.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.amp_clip: Optional[float] = config.get("amp_clip", _AMP_CLIP_Q100)

        fe = config.get("feature_extractor", {})
        self.feature_extractor = AttentionFeatureExtractor(
            input_dim=fe.get("input_dim", 5),
            d_model=fe.get("d_model", 128),
            num_heads=fe.get("num_heads", 8),
            num_layers=fe.get("num_layers", 4),
            dim_feedforward=fe.get("dim_feedforward", 512),
            dropout=fe.get("dropout", 0.1),
            pooling=fe.get("pooling", "cls"),
            use_positional_encoding=fe.get("use_positional_encoding", True),
            max_seq_len=fe.get("max_seq_len", 500),
        )

        # SNGP requires the spectral-normed (bi-Lipschitz) encoder; default ON.
        if config.get("spectral_norm", {"enabled": True}).get("enabled", True):
            apply_spectral_norm_encoder(self.feature_extractor)

        gp = config.get("gp_head", {})
        self.classifier = RandomFeatureGPHead(
            input_dim=self.feature_extractor.feature_dim,
            num_rff=gp.get("num_rff", 1024),
            length_scale=gp.get("length_scale", 1.0),
            ridge=gp.get("ridge", 1e-3),
            mean_field_factor=gp.get("mean_field_factor", math.pi / 8.0),
            normalize_input=gp.get("normalize_input", True),
        )
        logger.info(f"Created NuMuSNGPModel with {self.count_parameters()} parameters")

    # amplitude clip shared with NuMuClassifierModel
    _clip_amplitude = NuMuClassifierModel._clip_amplitude

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self._clip_amplitude(batch, self.amp_clip)
        return self.feature_extractor(
            sequences=batch["features"], lengths=batch["lengths"], mask=batch["mask"],
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(batch))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: Dict) -> nn.Module:
    """
    Factory function to create models based on configuration.

    Args:
        config: Model configuration dict

    Returns:
        model: Initialized PyTorch model
    """
    model_type = config.get('type', 'numu')

    if model_type == 'numu':
        model = NuMuClassifierModel(config)
    elif model_type == 'numu_sngp':
        model = NuMuSNGPModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model