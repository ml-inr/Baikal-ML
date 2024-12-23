# hits_signal-noise_separation/sig_noise_nn_arch_encoder.py
import sys
import tensorflow as tf

import tensorflow as tf
from tensorflow import keras

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model)
        ])

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, training=False):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerEncoderModel(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_shape, rate=0.1):
        super(TransformerEncoderModel, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = keras.layers.Dense(d_model, input_shape=input_shape)

        self.enc_layers = [TransformerEncoder(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = keras.layers.Dropout(rate)
        self.out_linear = keras.layers.Dense(1)
        self.softmax = keras.layers.Softmax()

    def call(self, x, training=False):
        x = self.embedding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        x = self.out_linear(x)
        x = self.softmax(x)
        return x
