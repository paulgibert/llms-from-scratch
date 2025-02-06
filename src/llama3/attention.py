import torch
from torch import nn
from torch.nn import functional as F
from llama3.positional_embeddings import RotaryPositionalEmbeddings


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
        
        self.rope = RotaryPositionalEmbeddings(head_dim, max_seqlen)
                
        self.k_cache = torch.zeros(max_bsz, max_seqlen, n_kv_heads, head_dim)
        self.v_cache = torch.zeros(max_bsz, max_seqlen, n_kv_heads, head_dim)
        
    def forward(self, x, pos: int, mask):
        bsz, seqlen, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x) # (bsz, seqlen, head_dim * n_heads), (bsz, seqlen, head_dim * n_kv_heads), (bsz, seqlen, head_dim * n_kv_heads)
        
        queries = queries.view(bsz, seqlen, self.n_heads, self.head_dim)    # (..., ..., n_heads * head_dim) -> (..., ..., n_heads, head_dim)
        keys = keys.view(bsz, seqlen, self.n_kv_heads, self.head_dim)       # (..., ..., n_kv_heads * head_dim) -> (..., ..., n_kv_heads, head_dim)
        values = values.view(bsz, seqlen, self.n_kv_heads, self.head_dim)   # (..., ..., n_kv_heads * head_dim) -> (..., ..., n_kv_heads, head_dim)
 
        # Encode position with RoPE
        queries = self.rope(queries, pos)
        keys = self.rope(keys, pos)

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

        print(queries.shape)
        print(keys.shape)
        print(values.shape)
        out = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask) # (bsz, n_heads, seqlen, head_dim)
        
        # Torch might complain if we don't use `contiguous`
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1) # (..., n_heads, seqlen, head_dim) -> (..., seqlen, n_heads * head_dim)
        return self.wo(out) # (..., ..., n_heads * head_dim) -> (..., ..., model_dim)
