import torch
import torch.nn as nn
import torch.nn.functional as F

class QKVProjector(nn.Module):
    def __init__(self, m_dim):
        super(QKVProjector, self).__init__()
        self.Q_proj = nn.Linear(m_dim, m_dim)
        self.K_proj = nn.Linear(m_dim, m_dim)
        self.V_proj = nn.Linear(m_dim, m_dim)
        nn.init.xavier_uniform_(self.Q_proj.weight)
        nn.init.xavier_uniform_(self.K_proj.weight)
        nn.init.xavier_uniform_(self.V_proj.weight)

    def forward(self, node):
        return self.Q_proj(node), self.K_proj(node), self.V_proj(node)

class NLPAttention(nn.Module):
    def __init__(self):
        super(NLPAttention, self).__init__()

    def forward(self, qs, ks, vs, mask):
        attention_logits = torch.matmul(qs, ks.transpose(-1, -2)) * mask
        attention_logits = attention_logits / (qs.size(-1))**0.5
        attention_logits += (1.0 - mask) * -1e9
        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_weights *= mask
        return torch.matmul(attention_weights, vs)

class MultiheadAttentionNLP(nn.Module):
    def __init__(self, num_heads, m_dim, out_dim, dropout_rate=0.1):
        super(MultiheadAttentionNLP, self).__init__()
        self.num_heads = num_heads
        self.m_dim = m_dim
        self.proj_layer = QKVProjector(num_heads * m_dim)
        print(m_dim, num_heads * m_dim)
        self.att_layer = NLPAttention()
        self.out_dense = nn.Linear(num_heads * m_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        batch_size, seq_length = x.size(0), x.size(1)
        mask_att = mask * mask.transpose(-1, -2)

        x = self.dropout(x)
        print(x.shape)
        qs, ks, vs = self.proj_layer(x)
        qkvs = torch.stack([qs, ks, vs], dim=0).reshape(3, batch_size, seq_length, self.num_heads, self.m_dim)
        qkvs = qkvs.permute(0, 3, 1, 2, 4)
        msgs = self.att_layer(qkvs[0], qkvs[1], qkvs[2], mask_att)
        msgs = msgs.permute(1, 2, 0, 3).reshape(batch_size, seq_length, self.num_heads * self.m_dim)

        return self.out_dense(msgs)

class FeedForward(nn.Module):
    def __init__(self, d_ff, d_model, dropout_rate, act_function):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(d_model, d_ff)
        self.dense2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = getattr(F, act_function)

    def forward(self, x):
        x = self.activation(self.dense1(x))
        x = self.dropout(x)
        return self.dense2(x)

class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, dropout_rate: float = 0.1, act_function: str = 'relu'):
        """_summary_

        Args:
            num_heads (_type_): _description_
            d_model (int): ... Must be equal last dim size in input x
            d_ff (_type_): _description_
            dropout_rate (_type_): _description_
            act_function (str): _description_
        """
        super(EncoderLayer, self).__init__()
        assert d_model % num_heads == 0
        self.mha = MultiheadAttentionNLP(num_heads, d_model // num_heads, d_model, dropout_rate)
        self.ffn = FeedForward(d_ff, d_model, dropout_rate, act_function)
        self.layernorm1 = nn.LayerNorm(2*d_model // num_heads)
        self.layernorm2 = nn.LayerNorm(2*d_model)

    def forward(self, x, mask):
        bn_mask = mask[:, :, 0].bool()
        attn_output = self.mha(x, mask)
        out1 = self.layernorm1(x + attn_output, mask=bn_mask)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output, mask=bn_mask) * mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_config):
        super(TransformerEncoder, self).__init__()
        self.d_model = encoder_config['d_model']
        self.num_layers = encoder_config['num_layers']
        num_heads = encoder_config['num_heads']
        d_ff = encoder_config['d_ff']
        dropout_rate = encoder_config['dropout_rate']
        l2_reg = encoder_config['l2_reg']
        act_function = encoder_config['encoder_act_function']
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(num_heads, self.d_model, d_ff, dropout_rate, l2_reg, act_function)
            for _ in range(self.num_layers)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model-6))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, mask, gp_to_cls_token):
        batch_size = x.size(0)

        gp_to_cls_token = gp_to_cls_token.unsqueeze(1)
        cls_tokens = torch.cat([gp_to_cls_token, self.cls_token.repeat(batch_size, 1, 1)], dim=-1)

        x = torch.matmul(x, self.embedding_matrix)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.positional_encoding(x)

        cls_mask = torch.ones((batch_size, 1, 1), device=x.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        for layer in self.enc_layers:
            x = layer(x, mask)

        pool = (x[:, 1:] * mask[:, 1:]).sum(1) / mask[:, 1:].sum(1)
        return torch.cat([x[:, 0, :], pool], dim=-1)
