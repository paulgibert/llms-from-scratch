import torch
from llama3.normalization import RMSNorm


def test_RMSNorm():
    """
    Check that RMSNorm outputs correct shape.
    """
    bsz = 4
    seqlen = 40
    model_dim = 16
    
    norm = RMSNorm(model_dim)
    x = torch.rand(bsz, seqlen, model_dim)
    output = norm(x)
    assert(output.shape == (bsz, seqlen, model_dim))
