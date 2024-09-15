import tensorflow as tf

class QKVProjector(tf.keras.layers.Layer):

    def __init__(self, m_dim):
        super().__init__()
        self.m_dim = m_dim

    def build(self, input_shape):
        num_fs = input_shape[-1]
        self.proj_matrix_Q = self.add_weight("proj_matrix_Q", shape=(num_fs, self.m_dim), initializer="glorot_uniform")
        self.proj_matrix_K = self.add_weight("proj_matrix_K", shape=(num_fs, self.m_dim), initializer="glorot_uniform")
        self.proj_matrix_V = self.add_weight("proj_matrix_V", shape=(num_fs, self.m_dim), initializer="glorot_uniform")
        self.bias_Q = self.add_weight("bias_Q", shape=(self.m_dim,), initializer="zeros")
        self.bias_K = self.add_weight("bias_K", shape=(self.m_dim,), initializer="zeros")
        self.bias_V = self.add_weight("bias_V", shape=(self.m_dim,), initializer="zeros")

    def call(self, node):
        qs = tf.matmul(node, self.proj_matrix_Q) + self.bias_Q
        ks = tf.matmul(node, self.proj_matrix_K) + self.bias_K
        vs = tf.matmul(node, self.proj_matrix_V) + self.bias_V
        return qs, ks, vs

class NLPAttention(tf.keras.layers.Layer):

    def call(self, inputs):
        qs, ks, vs, mask = inputs
        norm_softmax = tf.math.rsqrt(tf.cast(tf.shape(qs)[-1], tf.float32))
        attention_logits = tf.matmul(qs, ks, transpose_b=True)
        attention_logits = attention_logits * norm_softmax
        attention_logits += (1.0 - mask) * -1e9
        attention_weights = tf.nn.softmax(attention_logits)
        attention_weights *= mask
        return tf.matmul(attention_weights, vs)

class MultiheadAttentionNLP(tf.keras.layers.Layer):

    def __init__(self, num_heads, m_dim, out_dim):
        super().__init__()
        self.m_dim = m_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.proj_layer = QKVProjector(num_heads * m_dim)
        self.att_layer = NLPAttention()

    def build(self, input_shape):
        self.matrix_out = self.add_weight("matrix_out", shape=(self.num_heads * self.m_dim, self.out_dim), initializer="glorot_uniform")
        self.bias = self.add_weight("bias", shape=(self.out_dim,), initializer="zeros")

    def call(self, inputs):
        x, mask = inputs
        qs, ks, vs = self.proj_layer(x)
        qs = tf.stack(tf.split(qs, self.num_heads, axis=-1), axis=0)
        ks = tf.stack(tf.split(ks, self.num_heads, axis=-1), axis=0)
        vs = tf.stack(tf.split(vs, self.num_heads, axis=-1), axis=0)
        msgs = self.att_layer((qs, ks, vs, mask))
        msgs = tf.concat(tf.unstack(msgs, axis=0), axis=-1)
        return tf.matmul(msgs, self.matrix_out) + self.bias

class FeedForward(tf.keras.layers.Layer):

    def __init__(self, d_ff, d_model):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(d_ff, activation='gelu')
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        return self.dense2(self.dense1(x))

class PredictLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, x):
        return self.dense(x)

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, d_ff):
        super().__init__()
        self.mha = MultiheadAttentionNLP(num_heads, d_model // num_heads, d_model)
        self.ffn = FeedForward(d_ff, d_model)
        self.layernorm1 = tf.keras.layers.BatchNormalization()
        self.layernorm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x, mask = inputs
        attn_output = self.mha((x, mask))
        out1 = self.layernorm1(x + attn_output, training=training)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output, training=training) * mask

class Encoder(tf.keras.Model):
    
    def __init__(self, num_layers, num_heads, d_model, d_ff):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.enc_layers = [EncoderLayer(num_heads, d_model, d_ff) 
                           for _ in range(num_layers)]
        self.predict_layer = PredictLayer()

    def build(self, input_shape):
        if input_shape[-1]<self.d_model:
            self.to_pad = True
            self.num_to_pad = self.d_model-input_shape[-1]
        else:
            self.to_pad = False

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, training=False):
        x, mask = inputs[:, :, :-1], inputs[:, :, -1:]
        if self.to_pad:
            x = tf.concat( [x, tf.zeros( (tf.shape(x)[0],tf.shape(inputs)[1],self.num_to_pad+1) ) ], axis=-1 )
        for i in range(self.num_layers):
            x = self.enc_layers[i]((x, mask), training=training)
        preds = self.predict_layer(x)
        preds = tf.where(tf.cast(mask,bool), preds, tf.constant([0.,1.]))
        return preds