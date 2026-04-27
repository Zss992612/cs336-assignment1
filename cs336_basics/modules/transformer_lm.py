import torch
from torch import nn
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.embedding import Embedding
from cs336_basics.modules.rmsnorm import RMSNorm
from cs336_basics.modules.transformerblock import TransFormerBlock
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, repo_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransFormerBlock(
                d_model = d_model,
                num_heads = num_heads,
                d_ff = d_ff,
                theta = repo_theta,
                max_seq_len = context_length,
                device = device,
                dtype = dtype,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits