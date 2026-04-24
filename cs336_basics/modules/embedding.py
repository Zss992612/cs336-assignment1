import torch
import math
from torch import nn
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty((vocab_size, d_model),device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)
    def forward(self, token_ids):
        return self.weight[token_ids]
