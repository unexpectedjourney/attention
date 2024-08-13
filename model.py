import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)


class PositionalEncodding(nn.Module):
    def __init__(self, d_model, seq_len, dropout_value):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout_value)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(seq_len, d_model)
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        x = self.dropout(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - x_mean) / (x_std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout_value):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_value)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout_value):
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        assert d_model % heads == 0, "d_model % heads != 0"

        self.d_head = d_model // heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_value)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        attention_result = attention_scores @ value

        return attention_result, attention_scores

    def forward(self, q, k, v, mask):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(q.shape[0], q.shape[1], self.heads, self.d_head).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.heads, self.d_head).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.heads, self.d_head).transpose(1, 2)

        x, attention_scores = MultiHeadAttention.attention(q, k, v, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_head)
        x = self.w_o(x)

        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout_value):
        super().__init__()
        self.dropout = nn.Dropout(dropout_value)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        x = x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block,
        feed_forward_block,
        dropout_value,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_value) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block,
        cross_attention_block,
        feed_forward_block,
        dropout_value
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_value) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, target_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        x = self.norm(x)
        return x


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.proj(x)
        x = torch.log_softmax(x, dim=-1)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        src_pos,
        tgt_pos,
        projection_layer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.src_pos(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, src, src_mask, tgt, tgt_mask):
        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)
        x = self.decoder(tgt, x, src_mask, tgt_mask)
        return x

    def project(self, x):
        x = self.projection_layer(x)
        return x

    @staticmethod
    def initialize(transformer):
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return transformer

    @staticmethod
    def build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        src_seq_len,
        tgt_seq_len,
        d_model=512,
        n=6,
        heads=8,
        dropout_value=0.1,
        d_ff=2048
    ):
        src_embed = InputEmbedding(d_model, src_vocab_size)
        tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
        src_pos = PositionalEncodding(d_model, src_seq_len, dropout_value)
        tgt_pos = PositionalEncodding(d_model, tgt_seq_len, dropout_value)

        encoder_layers = []
        for _ in range(n):
            encoder_self_attention = MultiHeadAttention(d_model, heads, dropout_value)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_value)
            encoder_block = EncoderBlock(
                encoder_self_attention,
                feed_forward_block,
                dropout_value
            )
            encoder_layers.append(encoder_block)

        decoder_layers = []
        for _ in range(n):
            decoder_self_attention = MultiHeadAttention(d_model, heads, dropout_value)
            decoder_cross_attention = MultiHeadAttention(d_model, heads, dropout_value)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_value)
            decoder_block = DecoderBlock(
                decoder_self_attention,
                decoder_cross_attention,
                feed_forward_block,
                dropout_value
            )
            decoder_layers.append(decoder_block)

        encoder = Encoder(nn.ModuleList(encoder_layers))
        decoder = Decoder(nn.ModuleList(decoder_layers))

        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        transformer = Transformer(
            encoder,
            decoder,
            src_embed,
            tgt_embed,
            src_pos,
            tgt_pos,
            projection_layer,
        )
        transformer = Transformer.initialize(transformer)
        return transformer


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = Transformer.build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config['seq_len'],
        d_model=config['d_model']
    )
    return model
