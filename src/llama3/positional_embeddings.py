import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, base: int):
        self.base = base
        self.cos_cached = None
        self.sin_cached = None
    
    def _build_cache(self, x):
        _, seqlen, d = x.shape
        
        if self.cos_cached is not None and seqlen <= self.cos_cached.shape[0]:
            return

        exp = torch.arange(0, d, 2).float() / d
        theta = 1.0 / torch.power(self.base, exp).to(x.device)
        seq_idx = torch.arange(seqlen, device=x.device).float().to(x.device)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]
    
    def forward(self, x):
        split = x.shape[-1] // 2
        neg_half_x = torch.cat([-x[..., split:], x[..., :split]], dim=-1)
        return (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
