import math
from cs336_basics.modules.softmax import softmax
def scaled_dot_product_attention(Q, K, V, mask=None):
    K_t = K.transpose(-2, -1)
    scores = Q @ K_t / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    output = attn_weights @ V
    return output