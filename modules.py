import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):

    def __init__(self, scale_factor):
        super(ScaledDotProductAttention, self).__init__()

        self.scale_factor = scale_factor

    def forward(self, q, k, v, mask=None):
        # q shape is batch x n_head x len_q x dim_k
        # k shape is batch x n_head x len_k x dim_k
        # energy shape is batch x n_head x len_q x len_k
        energy = torch.matmul(q / self.scale_factor, k.transpose(2,3))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        attention = F.softmax(energy, dim=-1)

        # attention shape is batch x n_head x len_q x len_k
        # v shape is batch x n_head x len_v x dim_v  (len_v = len_k)
        output = torch.matmul(attention, v)

        # output shape is batch x n_head x len_q x dim_v
        return output, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, dim_k, dim_v, dropout=0.1):
        """
        :param n_head: number of heads
        :param d_model: embedding dimension of tokens
        :param d_k: keys and queries dimension after projection
        :param d_v: values dimension after projection. d_v * n_head = d_model
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0, f'Embedding dimension ({d_model}) should be divisible by nr. of heads ({n_head})'

        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_v = dim_v

        # Matrices Wq, Wk and Wv transforms the embeddings of size d_model to n_heads embeddings of size d_k, d_k and d_v
        # We are using 1 matrix to create all n_head embeddings
        self.W_q = nn.Linear(d_model, dim_k * n_head, bias=False)
        self.W_k = nn.Linear(d_model, dim_k * n_head, bias=False)
        self.W_v = nn.Linear(d_model, dim_v * n_head, bias=False)

        # Linear layer at the end of multi-head attention
        self.unifyheads = nn.Linear(dim_v * n_head, d_model, bias=False)

        self.attention = ScaledDotProductAttention(scale_factor=dim_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        q, k and v are of the size (batch, length(number of tokens), d_model)
        generally length of keys and values should be equals, but length of queries can be different
        """
        dim_k, dim_v, n_head = self.dim_k, self.dim_v, self.n_head
        batch, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # X=q is the input to the multi-head attention, so we save it for residual connection
        residual = q

        # Pass through the pre-attention projection: batch x len_q x (n_head * dim_k/dim_v)
        # Separate different heads: batch x len_(q/k/v) x n_head x dim_k/dim_v
        q = self.W_q(q).view(batch, len_q, n_head, dim_k)
        k = self.W_k(k).view(batch, len_k, n_head, dim_k)
        v = self.W_v(v).view(batch, len_v, n_head, dim_v)

        # Transpose for attention dot product: batch x n_head x len_q x dim_k/dim_v
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # after attention q shape is batch x n_head x len_q x dim_v
        q, attention = self.attention(q, k, v, mask=mask)

        # concat all the heads together
        # q shape is batch x len_q x dim_v * n_head
        q = q.transpose(1, 2).reshape(batch, len_q, n_head*dim_v)
        q = self.dropout(self.unifyheads(q))
        q += residual

        q = self.layer_norm(q)

        return q, attention

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x