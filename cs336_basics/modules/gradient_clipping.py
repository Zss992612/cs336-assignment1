import math
import torch
def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    
    grads = []

    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad)

    total_norm = torch.sqrt(sum((grad ** 2).sum() for grad in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(scale)