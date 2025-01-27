import torch
from llama3.attention import MultiHeadGroupedQueryAttention



def test_MultiHeadGroupedAttention():
    """
    Check that MultiHeadGroupedAttention outputs the correct shape.
    """
    model_dim = 16
    head_dim = 8
    n_heads = 10
    n_kv_heads = 5
    max_bsz = 32
    max_seqlen = 64
    
    model = MultiHeadGroupedQueryAttention(model_dim, head_dim, n_heads,
                                         n_kv_heads, max_bsz, max_seqlen)
    x = torch.rand(12, 48, model_dim)
    freqs_cis = torch.randn(48, 4)
    mask = torch.rand(48, 48)
    output = model(x, 0, freqs_cis, mask) # TODO: Test positional encoding
    assert(output.shape == (12, 48, model_dim))
