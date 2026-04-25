import torch
from torch import nn

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_features):
        return in_features * torch.sigmoid(in_features)