import torch
import torch.nn as nn
import numpy as np
from Layers import EncoderLayer, DecoderLayer

def get_pad_mask(seq, pad_idx):
    """
    Mask to not pay attention to padding tokens
    :param seq:
    :param pad_idx:
    :return:
    """
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the decoder future tokens. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):

    def __init__(self, dim_word_embed, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, dim_word_embed))

    def _get_sinusoid_encoding_table(self, n_position, dim_word_embed):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / dim_word_embed) for hid_j in range(dim_word_embed)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        :param x: tensor of word embeddings (batch x num_tokens x dim_word_embed)
        :return: tensor of word embedding + positional embeddings (batch x num_tokens x dim_word_embed)
        """
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):

    def __init__(self, source_vocabulary_size, dim_word_embed, n_layers, n_head, dim_k, dim_v,
                 d_model, d_inner, pad_idx, dropout=0.1, n_position=200):
        """
        :param source_vocabulary_size: size of words vocabulary
        :param dim_word_embed: dimension of word embedding
        :param n_layers: number of encoder layers on top of each other
        :param n_head: number of heads in multi-head attention
        :param dim_k: dimension of key and queries vectors
        :param dim_v: dimension of value vectors
        :param d_model: dimension of encoded word, d_model = dim_word_embed
        :param d_inner: dimension of hidden layer inside encoder MLP
        :param pad_idx: index of word to pad to zero in embedding layer
               (will turn the embedding of this word to be a tensor of zeros)
        :param dropout: dropout percent
        :param n_position: numbers of positions for position encoding (maximal number of token, can be raised as needed)
        """
        super(Encoder, self).__init__()

        self.src_word_emb = nn.Embedding(source_vocabulary_size, dim_word_embed, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(dim_word_embed, n_position=n_position)

        self.dropout = nn.Dropout(p=dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, dim_k, dim_v, dropout)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attention=False):

        encoder_self_attention_list = []

        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.encoder_layers:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            encoder_self_attention_list += [enc_slf_attn] if return_attention else []

        if return_attention:
            return enc_output, encoder_self_attention_list
        return enc_output

class Decoder(nn.Module):

    def __init__(self, target_vocabulary_size, dim_word_embed, n_layers, n_head, dim_k, dim_v,
                 d_model, d_inner, pad_idx, dropout=0.1, n_position=200):
        super().__init__()

        self.trg_word_emb = nn.Embedding(target_vocabulary_size, dim_word_embed, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(dim_word_embed, n_position=n_position)

        self.dropout = nn.Dropout(p=dropout)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, dim_k, dim_v, dropout)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, encoder_output, src_mask, return_attention=False):

        decoder_slf_attn_list, decoder_encoder_attn_list = [], []

        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.decoder_layers:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, encoder_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            decoder_slf_attn_list += [dec_slf_attn] if return_attention else []
            decoder_encoder_attn_list += [dec_enc_attn] if return_attention else []

        if return_attention:
            return dec_output, decoder_slf_attn_list, decoder_encoder_attn_list

        return dec_output

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, source_vocabulary_size, target_vocabulary_size, src_pad_idx, trg_pad_idx,
            dim_word_embed=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, dim_k=64, dim_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.encoder = Encoder(
            source_vocabulary_size=source_vocabulary_size, n_position=n_position,
            dim_word_embed=dim_word_embed, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, dim_k=dim_k, dim_v=dim_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            target_vocabulary_size=target_vocabulary_size, n_position=n_position,
            dim_word_embed=dim_word_embed, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, dim_k=dim_k, dim_v=dim_v,
            pad_idx=trg_pad_idx, dropout=dropout)

        self.target_word_classification = nn.Linear(d_model, target_vocabulary_size, bias=False)

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

        assert d_model == dim_word_embed, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.target_word_classification.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            # Share the weight between target word embedding & source word embedding
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.target_word_classification(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
