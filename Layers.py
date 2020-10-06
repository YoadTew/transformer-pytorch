import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, dim_k, dim_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(n_head, d_model, dim_k, dim_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, X, slf_attn_mask=None):
        encoder_self_output, X_self_attn = self.self_attention(X, X, X, slf_attn_mask)
        encoder_self_output = self.pos_ffn(encoder_self_output)

        return encoder_self_output, X_self_attn

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, dim_k, dim_v, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(n_head, d_model, dim_k, dim_v, dropout=dropout)
        self.encoder_attention = MultiHeadAttention(n_head, d_model, dim_k, dim_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, decoder_input, encoder_output, slf_attn_mask=None, ec_enc_attn_mask=None):
        decoder_self_output, decoder_self_attn = self.self_attention(
            decoder_input, decoder_input, decoder_input, slf_attn_mask)

        decoder_encoder_output, decoder_encoder_attn = self.encoder_attention(
            decoder_self_output, encoder_output, encoder_output)

        decoder_encoder_output = self.pos_ffn(decoder_encoder_output)

        return decoder_encoder_output, decoder_self_attn, decoder_encoder_attn
