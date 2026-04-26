import torch
from cs336_basics.modules.transformerblock import TransFormerBlock

d_model = 64
num_heads = 4
d_ff = 128
max_seq_len = 16
theta = 10000.0

x = torch.randn(2, 12, d_model)

TFB = TransFormerBlock(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    theta=theta,
    max_seq_len=max_seq_len,
    device=x.device,
    dtype=x.dtype,
)

print(TFB)
print()
print("state_dict keys:")
for k in TFB.state_dict().keys():
    print(k)