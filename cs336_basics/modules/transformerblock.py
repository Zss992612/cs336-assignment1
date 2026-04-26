import torch
from torch import nn
from cs336_basics.modules.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.modules.swiglu import SwiGlu
from cs336_basics.modules.softmax import softmax
from cs336_basics.modules.rmsnorm import RMSNorm

class TransFormerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, use_rope=True, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, use_rope, device=device, dtype=dtype)

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGlu(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.expand(*x.shape[:-2], seq_len)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))

        return x

