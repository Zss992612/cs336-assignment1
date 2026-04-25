import torch

def softmax(in_features, dim):
    max_value = torch.max(in_features, dim=dim, keepdim=True).values
    shifted = in_features - max_value
    exp_values = torch.exp(shifted)
    denom = torch.sum(exp_values, dim=dim, keepdim=True)
    output = exp_values / denom
    return output