import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EncoderUNetBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv_downsample = layers.Conv1D(filters, kernel_size, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.concat = layers.Concatenate(axis=-1)
        self.add = layers.Add()

    def call(self, inputs, training=False):
        x, mask = inputs
        x1 = self.conv1(x)
        x1 = tf.nn.gelu(self.bn1(x1, training=training))
        x1 = x1 * mask
        
        x2 = self.conv2(x1)
        x2 = tf.nn.gelu(self.bn2(x2, training=training))
        x2 = x2 * mask
        
        x3 = self.add([x2, x1])
        x3 = self.conv_downsample(x3)
        x3 = tf.nn.gelu(self.bn3(x3, training=training))
        
        mask_downsampled = layers.MaxPooling1D(pool_size=2, padding='same')(mask)
        encoding = x3 * mask_downsampled
        
        return encoding, mask_downsampled

class Encoder(tf.keras.layers.Layer):

    def __init__(self, filters, kernels):
        super().__init__()
        assert len(filters) == len(kernels)
        self.blocks = [EncoderUNetBlock(f, k) for f, k in zip(filters, kernels)]

    def call(self, inputs, training=False):
        x, mask = inputs
        encodings = [x]
        masks = [mask]
        for block in self.blocks:
            x, mask = block((x, mask), training=training)
            encodings.append(x)
            masks.append(mask)
        return encodings, masks

class DecoderUNetBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv_upsample = layers.Conv1DTranspose(filters, kernel_size, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.concat = layers.Concatenate(axis=-1)
        self.add = layers.Add()

    def call(self, inputs, training=False):
        x, skip_x, skip_mask, next_mask = inputs
        x1 = self.conv1(x)
        x1 = tf.nn.gelu(self.bn1(x1, training=training))
        x1 = x1 * skip_mask
        
        x2 = self.conv2(x1)
        x2 = tf.nn.gelu(self.bn2(x2, training=training))
        x2 = x2 * skip_mask
        
        x3 = self.add([x2, x1])
        x3 = self.conv_upsample(x3)
        x3 = tf.nn.gelu(self.bn3(x3, training=training))
        
        x3 = x3[:, :tf.shape(skip_x)[1], :]
        x3 = x3 * next_mask
        output = self.concat([x3, skip_x])
        
        return output

class Decoder(tf.keras.layers.Layer):

    def __init__(self, filters, kernels):
        super().__init__()
        assert len(filters) == len(kernels)
        self.blocks = [DecoderUNetBlock(f, k) for f, k in zip(filters, kernels)]

    def call(self, inputs, training=False):
        encodings, masks = inputs
        x = encodings[0]
        for block, skip_x, skip_mask, next_mask in zip(self.blocks, encodings[1:], masks[0:], masks[1:]):
            x = block((x, skip_x, skip_mask, next_mask), training=training)
        return x

class UNetModel(tf.keras.Model):

    def __init__(self, pre_lstm_units, post_lstm_units, enc_filters, enc_kernels, dec_filters, dec_kernels, last_kernel):
        super().__init__()
        self.pre_lstm = layers.Bidirectional(layers.LSTM(pre_lstm_units, activation='tanh', recurrent_activation='sigmoid', return_sequences=True), merge_mode='mul')
        self.encoder = Encoder(enc_filters, enc_kernels)
        self.decoder = Decoder(dec_filters, dec_kernels)
        self.post_lstm = layers.Bidirectional(layers.LSTM(post_lstm_units, activation='tanh', recurrent_activation='sigmoid', return_sequences=True), merge_mode='mul')
        self.final_conv = layers.Conv1D(2, last_kernel, padding='same')
        self.concat = layers.Concatenate(axis=-1)

    def call(self, inputs, training=False):
        mask = inputs[:, :, -1:]
        x = self.pre_lstm(inputs)
        x = x * mask
        
        encodings, masks = self.encoder((x, mask), training=training)
        x = self.decoder([list(reversed(encodings)), list(reversed(masks))], training=training)
        
        x = self.post_lstm(x)
        x = x * mask
        x = self.final_conv(x)
        
        preds = tf.where(tf.cast(mask, bool), tf.nn.softmax(x, axis=-1), tf.constant([0., 1.]))
        return preds

    def get_config(self):
        config = super().get_config()
        config.update({
            "pre_lstm_units": self.pre_lstm.forward_layer.units,
            "post_lstm_units": self.post_lstm.forward_layer.units,
            "enc_filters": [block.conv1.filters for block in self.encoder.blocks],
            "enc_kernels": [block.conv1.kernel_size[0] for block in self.encoder.blocks],
            "dec_filters": [block.conv1.filters for block in self.decoder.blocks],
            "dec_kernels": [block.conv1.kernel_size[0] for block in self.decoder.blocks],
            "last_kernel": self.final_conv.kernel_size[0]
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)