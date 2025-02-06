import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, model_dim: int, eps: float=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(model_dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.gain
        return x
