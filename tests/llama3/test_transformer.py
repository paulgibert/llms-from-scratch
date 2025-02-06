import torch
from llama3.transformer import TransformerBlock


def test_TransformerBlock():
    """
    Check that TransformerBlock outputs the correct shape.
    """
    model_dim = 16
    head_dim = 8
    n_heads = 10
    n_kv_heads = 5
    ff_dim = 128
    bsz = 16
    max_bsz = 32
    seqlen = 42
    max_seqlen = 64
    
    
    block = TransformerBlock(model_dim, head_dim, n_heads, n_kv_heads,
                             ff_dim, max_bsz, max_seqlen)
    x = torch.rand(bsz, seqlen, model_dim)
    pos = 0 # TODO: What about pos != 0?
    mask = torch.rand(seqlen, seqlen)
    output = block(x, pos, mask)
    assert(output.shape == (bsz, seqlen, model_dim))