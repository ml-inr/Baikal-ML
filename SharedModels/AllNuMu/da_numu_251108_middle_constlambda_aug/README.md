# Baikal Neutrino Detection Model

## Model Overview

This is a **domain-adapted binary classification model** for detecting neutrino events in the Baikal neutrino telescope data. The model has been trained using domain adaptation techniques to transfer knowledge from Monte Carlo simulations to experimental data.

IMPORTANT
The model is not for production usage yet: unexpected behaviour found on experimental data.

### Model Details
- **Task**: Binary neutrino detection (neutrino vs muon classification)
- **Architecture**: Transformer encoder + Binary classifier  
- **Parameters**: 541,633
- **Input**: Variable-length sequences of detector hits
- **Output**: Binary logits (probability scores for neutrino events)

### Training Information
- **Source Domain**: Monte Carlo data (muatm_2020, nue2_2020, nuatm_2020)
- **Target Domain**: Experimental data from Baikal detector
- **Method**: Domain-Adversarial Neural Networks (DANN)
- **Experiment**: `da_numu_251108_middle_constlambda_aug`

## Files Included

- `base_model.pth` - PyTorch model weights
- `model_config.yaml` - Model architecture configuration  
- `normalization_config.yaml` - Data preprocessing parameters
- `README.md` - This documentation

## How to Use

### 1. Setup Environment

```python
import torch
import yaml
import numpy as np
from src.base_models import create_model
```

### 2. How to Load the Model

```python
# Load model configuration
with open('model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model instance
model = create_model(config['model'])

# Load trained weights
model.load_state_dict(torch.load('base_model.pth'))
model.eval()

print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### 3. How to Prepare Your Data

Your input data should be formatted as detector hits with 5 features:

```python
# Input format: [amplitude, time, x, y, z]
# - amplitude: Signal strength (normalized)
# - time: Hit timing in nanoseconds (centered per event) 
# - x, y, z: 3D coordinates in meters (cluster-centered)

# Example for a single event with N hits:
event_features = torch.tensor([
    [0.695, -2336.3, 145.0, 109.8, -172.6],  # Hit 1
    [1.234, -1890.1, 120.5, 95.2, -180.1],   # Hit 2
    # ... more hits
])  # Shape: [N_hits, 5]

# Prepare batch format
batch_data = {
    'features': event_features.unsqueeze(0),  # Add batch dimension: [1, N_hits, 5]
    'lengths': torch.tensor([len(event_features)]),  # Actual sequence length
    'mask': torch.ones(1, len(event_features), dtype=bool) # (True for real data, False for padding)
}
```

### 4. How to Make Predictions

```python
# Forward pass
with torch.no_grad():
    logits = model(batch_data)
    probabilities = torch.sigmoid(logits)

# Interpret results
neutrino_prob = probabilities.item()
print(f"Neutrino probability: {neutrino_prob:.3f}")

if neutrino_prob > 0.5: # The treshold is to be adjusted!
    print("Prediction: NEUTRINO EVENT")
else:
    print("Prediction: MUON EVENT") 
```

### 5. Data Normalization (Important!)

Apply the same normalization used during training:

```python
# Load normalization parameters
with open('normalization_config.yaml', 'r') as f:
    norm_config = yaml.safe_load(f)

means = torch.tensor(norm_config['means'])  # [5] features
stds = torch.tensor(norm_config['stds'])    # [5] features

# Normalize your features
normalized_features = (event_features - means) / stds

# Then use normalized_features in your batch_data
```

## Input Data Requirements

### Expected Input Format
- **Batch format**: Dictionary with `'features'`, `'lengths'` and `'mask'` keys
- **Features shape**: `[batch_size, max_sequence_length, 5]`
- **Lengths shape**: `[batch_size]` - actual sequence lengths
- **Mask shape**: `[batch_size, max_sequence_length]` - batch padding mask
- **Feature order**: `[amplitude, time, x, y, z]`

### Data Preprocessing
1. **Coordinate centering**: Shift coordinates to cluster center
2. **Time centering**: Center hit times to zero mean per event
3. **Normalization**: Apply mean/std normalization from `normalization_config.yaml`
4. **Sequence handling**: Variable length sequences up to 500 hits (remove hits after 500)

## Example Complete Usage

```python
import torch
import yaml
from src.models.base_models import create_model

# Load model
with open('model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
model = create_model(config['model'])
model.load_state_dict(torch.load('base_model.pth'))
model.eval()

# Load normalization
with open('normalization_config.yaml', 'r') as f:
    norm_config = yaml.safe_load(f)
means = torch.tensor(norm_config['means'])
stds = torch.tensor(norm_config['stds'])

# Your detector hit data (example of 1 event)
raw_hits = torch.tensor([
    [0.695, -2336.3, 145.0, 109.8, -172.6],
    [1.234, -1890.1, 120.5, 95.2, -180.1],
    # ... more hits
])

# Normalize
normalized_hits = (raw_hits - means) / stds
# Add 'batch' dimension as axis 0
input_hits = normalized_hits.unsqueeze(0)
# Prepare batch
batch = {
    'features': input_hits,
    'lengths': torch.tensor([input_hits.shape[1]]),
    'mask': torch.ones(input_hits.shape[0], input_hits.shape[1], dtype=bool)
}

# Predict
with torch.no_grad():
    logits = model(batch)
    prob = torch.sigmoid(logits).item()
    
print(f"Neutrino probability: {prob:.3f}")
```

## Model Performance Notes

- **Domain adapted**: Trained on MC data, adapted for experimental data
- **Robust**: Handles variable sequence lengths (up to 500 hits)