import torch
from torch.optim import Optimizer 
import math

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")

        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]

                with torch.no_grad():
                    p.data -= lr * weight_decay * p.data
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                    lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                    p.data -= lr_t * m / (torch.sqrt(v) + eps)
        return loss
