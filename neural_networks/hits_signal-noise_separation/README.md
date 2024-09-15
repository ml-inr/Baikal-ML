# Neural network architectures and related code

## Architectures
There are two types of neural networks with very similar metrics: UNet-like and Transormer-based. There architectures are present in `sig_noise_nn_arch_unet.py` and `sig_noise_nn_arch_encoder.py` accordingly. Use `sig_noise_train_<nn_type>.py` to train them. The realization is given in TensorFlow.

### UNet-like
Uses additional skip connections. Data is pre- and post-processed with bidirectioal LSTM. Because of that, neural networks is slow (~0.5 speed of transformer). However, these LSTMs are important - without them metrics are considerably worse. 

### Encoder
Uses BatchNorm instead of LayerNorm. Significantly improves the metrics.

## Generators
Datasets are managed in `sig_noise_generators.py`. For each batch, the output is padded to the maximal number of hits in the batch. Special values for paddings are used.

### Options
- Set hits with `*Qs* > threshold` to the threshold value. 
- Excluded hits `*Qs* < threshold`.
- Relabel signal hits as noise if their time residual exceeds threshold value.

### *Note*
Generators always yild tuple (data, labels, weights). Importantly, labels[:,:2] are one-hot encodings of signal-noise hits, while labels[:,3] are time residuals. This is do to avoid writing custom training loop.

## Loss functions 
Custom realization of 3 loss functions - Entropy, Focal, and TResPenalty. TResPenalty penalizes neural network for predicting noise hits with big time residuals as signal ones. 

## Metrics

### AccuracyExtr
Evaluates accuracy taking into account padding mask.

### TResExtr
Evaluates average time residual for hits identified as signal ones.