import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seqlen: int, freq: int=500000):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen
        self.freq = freq
        
        self.cos_cached = None
        self.sin_cached = None
        self._build_cache()
    
    def _build_cache(self):
        D = self.dim
        exponent = -2 * torch.arange(D // 2).repeat_interleave(2) / D
        w = torch.pow(self.freq, exponent) # (D)
        
        pos = torch.arange(self.max_seqlen).unsqueeze(1) # (N, 1)
        theta = pos * w.unsqueeze(0) # (N, D)
        
        self.cos_cached = torch.cos(theta)
        self.sin_cached = torch.sin(theta)
    
    def forward(self, x, start_pos: int):
        _, seqlen, _, _ = x.shape

        x2 = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape_as(x)

        cos = self.cos_cached[start_pos : start_pos + seqlen, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[start_pos : start_pos + seqlen, :].unsqueeze(0).unsqueeze(2)
        return x * cos + x2 * sin
