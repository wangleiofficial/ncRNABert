import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

class MultiheadAttention(nn.Module):
    r"""
    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, 
                 kdim=None, vdim=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"


        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, self.kdim, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, self.vdim, bias=False, **factory_kwargs)


        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.rotary_emb = RotaryEmbedding(dim=int(self.head_dim/2))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Optional[Tensor] = None,
            ) -> Tuple[Tensor, Optional[Tensor]]:

        q, k, v = self.q_proj(query),  self.k_proj(key), self.v_proj(value)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_dim)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        attn_weight = torch.einsum("...ij,...kj->...ik", q, k)/self.scaling

        if attn_mask is not None:
            attn_weight = attn_weight.masked_fill(attn_mask==0, -1e9)
        attn_weight = F.dropout(torch.softmax(attn_weight, dim=-2), p=self.dropout)

        attn = torch.einsum("...ij,...jd->...id", attn_weight, v)

        attn = rearrange(attn, "b h l d -> b l (h d)")
        attn = self.out_proj(attn)

        return attn, attn_weight

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x
    

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(
            activation_dropout,
        )
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x
    

class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        att_dropout=0.,
        ffn_dropout=0.,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.att_dropout = att_dropout

        self.mha_layer = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            dropout=self.att_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.ffn = FeedForwardNetwork(embed_dim, ffn_embed_dim, ffn_dropout)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, self_attn_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn_weight = self.mha_layer(
            query=x,
            key=x,
            value=x,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_weight

        