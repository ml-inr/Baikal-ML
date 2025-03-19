import tensorflow as tf
from custom_norms import MaskedMovingBatchNormalization, MaskedLayerNormalization

class QKVProjector(tf.keras.layers.Layer):

    def __init__(self, m_dim, l2_reg):
        super().__init__()
        self.m_dim = m_dim
        self.Q_proj = tf.keras.layers.Dense(m_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.K_proj = tf.keras.layers.Dense(m_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.V_proj = tf.keras.layers.Dense(m_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, node):
        return self.Q_proj(node), self.K_proj(node), self.V_proj(node)

class NLPAttention(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)
        self.norm_softmax = 1. / tf.math.sqrt(tf.cast(input_shape[1][-1], tf.float32))

    def call(self, inputs):
        qs, ks, vs, mask = inputs
        attention_logits = tf.matmul(qs, ks, transpose_b=True) * mask
        attention_logits = attention_logits * self.norm_softmax
        attention_logits += (1.0 - mask) * -1e9
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)
        attention_weights *= mask
        return tf.matmul(attention_weights, vs)

class MultiheadAttentionNLP(tf.keras.layers.Layer):

    def __init__(self, num_heads, m_dim, out_dim, dropout_rate, l2_reg):
        super().__init__()
        self.m_dim = m_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.proj_layer = QKVProjector(num_heads * m_dim, l2_reg)
        self.att_layer = NLPAttention()
        self.out_dense = tf.keras.layers.Dense(out_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x, mask = inputs
        mask_att = mask*tf.transpose(mask, perm=[0,2,1])
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]

        x = self.dropout(x, training=training)
        qs, ks, vs = self.proj_layer(x)
        qkvs = tf.stack([qs, ks, vs], axis=0)
        qkvs = tf.reshape(qkvs, (3, batch_size, seq_length, self.num_heads, self.m_dim))    
        # Transpose to [3, num_heads, batch_size, seq_len, m_dim]
        qkvs = tf.transpose(qkvs, perm=[0, 3, 1, 2, 4])
        msgs = self.att_layer([qkvs[0], qkvs[1], qkvs[2], mask_att])
        # Reshape back
        msgs = tf.transpose(msgs, perm=[1, 2, 0, 3])
        msgs = tf.reshape(msgs, (batch_size, seq_length, self.num_heads * self.m_dim))

        return self.out_dense(msgs)

class FeedForward(tf.keras.layers.Layer):

    def __init__(self, d_ff, d_model, dropout_rate, l2_reg, act_function):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(d_ff, activation=act_function, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.dense2 = tf.keras.layers.Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, d_ff, dropout_rate, l2_reg, act_function):
        super().__init__()
        self.mha = MultiheadAttentionNLP(num_heads, d_model // num_heads, d_model, dropout_rate, l2_reg)
        self.ffn = FeedForward(d_ff, d_model, dropout_rate, l2_reg, act_function)
        self.layernorm1 = MaskedLayerNormalization()
        self.layernorm2 = MaskedLayerNormalization()

    def call(self, inputs, training=False):
        x, mask = inputs
        bn_mask = tf.cast(mask[:,:,0], tf.bool)
        attn_output = self.mha([x, mask], training=training)
        out1 = self.layernorm1(x + attn_output, training=training, mask=bn_mask)
        ffn_output = self.ffn(out1, training=training)
        return self.layernorm2(out1 + ffn_output, training=training, mask=bn_mask) * mask

# positional encodings
class PositionalEncoding(tf.keras.layers.Layer):
    
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
    def build(self, input_shape):
        position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.d_model))
        
        # Calculate sin and cos separately
        angles = position * div_term
        sines = tf.math.sin(angles)
        cosines = tf.math.cos(angles)
        
        # Interleave sines and cosines
        pe = tf.stack([sines, cosines], axis=2)
        pe = tf.reshape(pe, [self.max_len, self.d_model])
        
        self.pe = tf.Variable(pe, trainable=False)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]

class TransformerEncoder(tf.keras.layers.Layer):
    
    def __init__(self, encoder_config):
        super().__init__()
        self.init_config = encoder_config
        self.num_layers = encoder_config['num_layers']
        num_heads = encoder_config['num_heads']
        self.d_model = encoder_config['d_model']
        d_ff = encoder_config['d_ff']
        dropout_rate = encoder_config['dropout_rate']
        l2_reg = encoder_config['l2_reg']
        act_function = encoder_config['encoder_act_function']
        self.positional_encoding = PositionalEncoding(self.d_model)  
        self.enc_layers = [EncoderLayer(num_heads, self.d_model, d_ff, dropout_rate, l2_reg, act_function) 
                           for _ in range(self.num_layers)]
        # should be non-zero to break symmetry - token specialized for certain task
        self.cls_token = self.add_weight(
            name="cls_token", 
            shape=(1, 1, self.d_model-6), # 6 will be occupied by reconstructed parameters
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            regularizer=tf.keras.regularizers.l2(encoder_config['l2_reg'])
        ) 

    def get_config(self):
        config = super().get_config()
        config.update(self.init_config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        super().build(input_shape)
        f_dim = input_shape[0][-1]
        if f_dim<self.d_model:
            ident_matrix = tf.keras.initializers.Identity()(shape=(f_dim,f_dim))
            rand_matrix = tf.keras.initializers.GlorotNormal()(shape=(f_dim,self.d_model-f_dim))
            matrix = tf.concat([ident_matrix,rand_matrix], axis=-1)
        else:
            matrix = tf.keras.initializers.GlorotNormal()(shape=(f_dim,self.d_model))
        self.embedding_matrix = self.add_weight("embedding_matrix", shape=(f_dim, self.d_model), initializer=lambda *args, **kwargs: matrix, trainable=True)
    
    def call(self, inputs, training=False):
        x, mask, gp_to_cls_token = inputs
        batch_size = tf.shape(x)[0]

        # Use reconstructed parameters in classification token
        gp_to_cls_token = gp_to_cls_token[:,tf.newaxis,:]
        rand_cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        cls_tokens = tf.concat([gp_to_cls_token,rand_cls_tokens], axis=-1)

        # Embed
        x = tf.linalg.matmul(x,self.embedding_matrix)
        # Add CLS token to the beginning of each sequence
        x = tf.concat([cls_tokens, x], axis=1)
        # Introduce positional encodings
        x = self.positional_encoding(x)
        # Add mask
        cls_mask = tf.ones((batch_size, 1, 1))
        mask = tf.concat([cls_mask, mask], axis=1)

        for i in range(self.num_layers):
            x = self.enc_layers[i]([x, mask], training=training)
        # Use classificatin token and pooled average 
        pool = tf.math.reduce_sum(x[:,1:]*mask[:,1:], axis=1) / tf.math.reduce_sum(mask[:,1:], axis=1)
        return tf.concat([x[:,0,:],pool], axis=-1)