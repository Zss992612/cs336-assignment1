import torch
from torch import nn
from cs336_basics.modules.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.rope import RoPE
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None, use_rope=False, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0
        self.d_head = d_model // num_heads
        self.use_rope = use_rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if use_rope is True:
            self.ro_pe = RoPE(theta, self.d_head, max_seq_len, device=device)
        else:
            self.ro_pe = None
    def forward(self, in_features, token_positions=None,):
        q = self.q_proj(in_features)
        k = self.k_proj(in_features)
        v = self.v_proj(in_features)
        
        q = q.reshape(*q.shape[:-1], self.num_heads, self.d_head).transpose(-2, -3)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.d_head).transpose(-2, -3)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.d_head).transpose(-2, -3)
        if self.use_rope is True:
            q = self.ro_pe(q, token_positions)
            k = self.ro_pe(k, token_positions)
        seq_len = in_features.shape[-2]
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
        )
        attn_out = scaled_dot_product_attention(q, k, v, mask)
        attn_out = attn_out.transpose(-2, -3)
        attn_out = attn_out.reshape(*attn_out.shape[:-2], self.d_model)
        out = self.o_proj(attn_out)
        return out
