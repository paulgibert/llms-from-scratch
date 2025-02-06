import torch
from torch import nn
from llama3.normalization import RMSNorm
from llama3.attention import MultiHeadGroupedQueryAttention
from llama3.feedforward import SWIGLUFeedForward

class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, head_dim: int, n_heads: int, n_kv_heads: int,
                 ff_dim: int, max_bsz: int, max_seqlen: int):
        super().__init__()
        self.attention_norm = RMSNorm(model_dim)
        self.attention = MultiHeadGroupedQueryAttention(model_dim, head_dim, n_heads, n_kv_heads,
                                                   max_bsz, max_seqlen)
        self.ffn_norm = RMSNorm(model_dim)
        self.ffn = SWIGLUFeedForward(model_dim, ff_dim)
    
    def forward(self, x, pos: int, mask: torch.Tensor):
        x += self.attention(self.attention_norm(x), pos, mask)
        x += self.ffn(self.ffn_norm(x))
        return x
