import torch

from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
    def forward(self, x):
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        mean_square = torch.mean(x_float * x_float, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        result = x_float / rms
        result = result * self.weight
        return result.to(in_dtype)