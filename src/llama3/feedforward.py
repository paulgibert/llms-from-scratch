from torch import nn
from torch.nn import functional as F


class SWIGLUFeedForward(nn.Module):
    """
    Implements the SWIGLU FeedForward NN used in Llama3.
    
    y = w3(silu(w1(x) * w2(x))
    
    Bias terms are omitted for parameter efficiency due
    to the use of layer norms elsewhere in the architecture.
    """
    def __init__(self, model_dim: int, ff_dim: int):
        """
        Args:
            model_dim: The model dimension.
            ff_dim: The internal dimension of the feedforward network.
        """
        super().__init__()
        
        self.w1 = nn.Linear(model_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(model_dim, ff_dim, bias=False)
        self.w3 = nn.Linear(ff_dim, model_dim, bias=False)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
