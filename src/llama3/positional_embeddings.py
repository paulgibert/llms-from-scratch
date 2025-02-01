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
        exponent = -2 * torch.arange(1, D/2 + 1).repeat_interleave(2) / D
        w = torch.pow(self.freq, exponent) # (D)
        
        pos = torch.arange(1, self.max_seqlen + 1).unsqueeze(1) # (N, 1)
        theta = w.unsqueeze(0) * pos # (N, D)
        
        self.cos_cached = torch.cos(theta)
        self.sin_cached = torch.sin(theta)
    
    def forward(self, x, start_pos: int):
        seqlen = x.shape[0]

        x2 = x.view(*x.shape[:-1], int(self.dim/2), 2).clone()
        x2[..., 1] = -x2[..., 1]
        x2 = x2.flip(dims=[-1]).view(x.shape)

        cos = x * self.cos_cached[start_pos : start_pos + seqlen, :].unsqueeze(0).unsqueeze(2)
        sin = x2 * self.sin_cached[start_pos : start_pos + seqlen, :].unsqueeze(0).unsqueeze(2)
        return cos + sin
