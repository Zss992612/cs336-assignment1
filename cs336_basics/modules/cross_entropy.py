import torch
from torch import nn
import math
from cs336_basics.modules.rmsnorm import RMSNorm
def cross_entropy(inputs, targets):
    max_logits = inputs.max(dim=1, keepdim=True).values
    shifted = inputs - max_logits
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.exp(shifted).sum(dim=1))
    target_logits = inputs.gather(
        dim = -1,
        index = targets.unsqueeze(-1)
    ).squeeze(-1)
    
    loss = log_sum_exp - target_logits
    return loss.mean()