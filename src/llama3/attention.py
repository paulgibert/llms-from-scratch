import torch
from torch import nn
from torch.nn import functional as F
from llama3.positional_embeddings import RotaryPositionalEmbeddings


def precompute_freqs_cis(dim, end, theta = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MultiHeadGroupedQueryAttention(nn.Module):
    """
    Implements multi head grouped query attention used in Llama3.
    
    Grouped query attention is essentially standard attention performed
    on non-overlapping sections of the input. The attention-weighted values
    of each section are concatenated at the end.
    
    Divide the N x d_model token sequence into G groups.
    
    Define Q, K, V weights for each group. For example, the Q weights for
    group g would have the shape (n/g, d_k).
    
    Compte attention scores for each group:
    
    A_g = softmax(Q_g * K_g.T / sqrt(d_k))
    
    Weight the values of each group by the attention scores and
    concatenate:
    
    concat([A_g * V_g for g in range(G)])
    
    Llama3 also has an optimization where fewer KV heads are stored and duplicated to match
    the number of Q heads. 
    """
    def __init__(self, model_dim: int, head_dim: int, n_heads: int, n_kv_heads: int,
                 max_bsz: int, max_seqlen: int):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        
        self.wq = nn.Linear(model_dim, head_dim * n_heads, bias=False)
        self.wk = nn.Linear(model_dim, head_dim * n_kv_heads, bias=False)
        self.wv = nn.Linear(model_dim, head_dim * n_kv_heads, bias=False)
        self.wo = nn.Linear(head_dim * n_heads, model_dim, bias=False)
                
        self.k_cache = torch.zeros(max_bsz, max_seqlen, n_kv_heads, head_dim)
        self.v_cache = torch.zeros(max_bsz, max_seqlen, n_kv_heads, head_dim)
        
    def forward(self, x, pos: int, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x) # (bsz, seqlen, head_dim * n_heads), (bsz, seqlen, head_dim * n_kv_heads), (bsz, seqlen, head_dim * n_kv_heads)
        
        queries = queries.view(bsz, seqlen, self.n_heads, self.head_dim)    # (..., ..., n_heads * head_dim) -> (..., ..., n_heads, head_dim)
        keys = keys.view(bsz, seqlen, self.n_kv_heads, self.head_dim)       # (..., ..., n_kv_heads * head_dim) -> (..., ..., n_kv_heads, head_dim)
        values = values.view(bsz, seqlen, self.n_kv_heads, self.head_dim)   # (..., ..., n_kv_heads * head_dim) -> (..., ..., n_kv_heads, head_dim)
        
        # Encode position with RoPE
        queries, keys = apply_rotary_emb(queries, keys, freqs_cis=freqs_cis)
        
        # On the first call, the k and v caches need to be moved to the correct device
        self.k_cache = self.k_cache.to(keys.device)
        self.v_cache = self.v_cache.to(values.device)
        
        # Update the k and v caches
        self.k_cache[:bsz, pos : pos + seqlen] = keys
        self.v_cache[:bsz, pos : pos + seqlen] = values
        
        # Retrieve all keys and values from the caches
        keys = self.k_cache[:bsz, : pos + seqlen]
        values = self.v_cache[:bsz, : pos + seqlen]
        
        # Repeat key and value heads to match number of query heads
        repeats = int(self.n_heads / self.n_kv_heads)
        keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)        # (..., ..., n_kv_heads, ...) -> (..., ..., n_heads, ...)
        values = torch.repeat_interleave(values, repeats=repeats, dim=2)    # (..., ..., n_kv_heads, ...) -> (..., ..., n_heads, ...)
        
        # Need to transpose to match expected input for `scaled_dot_product_attention`
        queries = queries.transpose(1, 2)   # (..., seqlen, n_heads, ...) -> (..., n_heads, seqlen, ...)
        keys = keys.transpose(1, 2)         # (..., seqlen, n_heads, ...) -> (..., n_heads, seqlen, ...)
        values = values.transpose(1, 2)     # (..., seqlen, n_heads, ...) -> (..., n_heads, seqlen, ...)

        out = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask) # (bsz, n_heads, seqlen, head_dim)
        
        # Torch might complain if we don't use `contiguous`
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1) # (..., n_heads, seqlen, head_dim) -> (..., seqlen, n_heads * head_dim)
        return self.wo(out) # (..., ..., n_heads * head_dim) -> (..., ..., model_dim)
