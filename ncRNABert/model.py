""" 
"""

import torch.nn as nn
import torch
from einops import rearrange

from .modules import RobertaLMHead, TransformerLayer
from .utils import length_to_mask


class RNABertModel(nn.Module):
    def __init__(
        self,
        num_token=14,
        num_layers: int = 24,
        embed_dim: int = 1024,
        attention_heads: int = 16,
        dropout: float = 0.,
        padding_idx:int = 13
    ):
        super().__init__()
        self.num_token = num_token
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout

        self._init_submodules()

    def _init_submodules(self):
        self.embed_tokens = nn.Embedding(
            self.num_token,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    self.dropout,
                    self.dropout
                )
                for _ in range(self.num_layers)
            ]
        )

        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.num_token,
            weight=self.embed_tokens.weight
        )

    def forward(self, tokens, lengths=None, repr_layers=[], need_head_weights=False):

        assert tokens.ndim == 2
        x = self.embed_tokens(tokens)

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        attn_mask = length_to_mask(lengths)
        attn_mask = rearrange(attn_mask, 'b l -> b () () l') * rearrange(attn_mask, 'b l -> b () l ()')

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(x, attn_mask)
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn)
                
        x = self.emb_layer_norm_after(x)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        
        if need_head_weights:
            result["attentions"] = attn_weights

        return result