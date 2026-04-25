import torch
from cs336_basics.modules.silu import SiLU
from cs336_basics.modules.linear import Linear
from torch import nn

class SwiGlu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=None)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=None)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=None)
        self.si_lu = SiLU()
    def forward(self, in_features):
        return self.w2(self.si_lu(self.w1(in_features)) * self.w3(in_features))

