import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = np.stack([dataset[i : i + context_length] for i in starts])
    targets = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets