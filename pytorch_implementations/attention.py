import torch
import math
from torch import nn
import torch.nn.functional as F

def self_attention(X: torch.Tensor, params: dict, scaled: bool) -> torch.Tensor:
    w_v, w_q, w_k = params["Omega_v"], params["Omega_q"], params["Omega_k"]
    b_v, b_q, b_k = params["omega_v"], params["omega_q"], params["omega_k"]
    
    V = (w_v @ X) + b_v
    Q = (w_q @ X) + b_q
    K = (w_k @ X) + b_k

    if scaled:
        return V @ F.softmax(((K.T @ Q)/math.sqrt(w_q.shape[0])), dim=0)
    else:
        return V @ F.softmax((K.T @ Q), dim=0)
    
def multihead_self_attention(X: torch.Tensor, params: dict, H: int, mask: bool) -> torch.Tensor:
    W_v, W_q, W_k, W_c  = params["Omega_v"], params["Omega_q"], params["Omega_k"], params["Omega_c"]
    b_v, b_q, b_k = params["omega_v"], params["omega_q"], params["omega_k"]

    N = X.shape[1]
    D = W_q.shape[2]
    D_h = W_q.shape[1]

    V = (W_v @ X[None, :, :]) + b_v  ### shape: (H, D_h, N) = ((H, D_h, D) @ (1, D, N)) + (H, D_h, 1)
    Q = (W_q @ X[None, :, :]) + b_q
    K = (W_k @ X[None, :, :]) + b_k

    raw_attention = (K.mT @ Q)
    if mask:
        bool_mask = torch.tril(torch.ones(N, N, device=raw_attention.device), diagonal=-1).bool()
        raw_attention.masked_fill_(bool_mask, float('-inf'))

    head_outputs = V @ F.softmax((raw_attention/math.sqrt(W_q.shape[1])), dim=1)  ### shape: (H, D_h, N) = (H, D_h, N) @ (H, N, N)
    outputs_stacked = head_outputs.reshape(-1, head_outputs.shape[-1])  ### shape: (D, N) = (H * D_h, N)

    final_output = W_c @ outputs_stacked  ### shape: (D, N) = (D, D) @ (D, N)

    return final_output

import torch
import torch.nn as nn

torch.manual_seed(1)
D, N, H = 16, 6, 4
Dh = D // H
X = torch.randn(D, N)
params = dict(
    Omega_q=torch.randn(H, Dh, D), omega_q=torch.randn(H, Dh, 1),
    Omega_k=torch.randn(H, Dh, D), omega_k=torch.randn(H, Dh, 1),
    Omega_v=torch.randn(H, Dh, D), omega_v=torch.randn(H, Dh, 1),
    Omega_c=torch.randn(D, D),
)

out = multihead_self_attention(X, params, H, mask=False)
assert out.shape == (D, N)

# Independent reference: copy your parameters into nn.MultiheadAttention
mha = nn.MultiheadAttention(embed_dim=D, num_heads=H, bias=True, batch_first=False)
mha.eval()
with torch.no_grad():
    mha.in_proj_weight.copy_(torch.cat([params["Omega_q"].reshape(D, D),
                                        params["Omega_k"].reshape(D, D),
                                        params["Omega_v"].reshape(D, D)], dim=0))
    mha.in_proj_bias.copy_(torch.cat([params["omega_q"].reshape(D),
                                      params["omega_k"].reshape(D),
                                      params["omega_v"].reshape(D)], dim=0))
    mha.out_proj.weight.copy_(params["Omega_c"])
    mha.out_proj.bias.zero_()

x_seq = X.T.unsqueeze(1)                       # (N, batch=1, D)
with torch.no_grad():
    ref, _ = mha(x_seq, x_seq, x_seq, need_weights=False)
assert torch.allclose(out, ref[:, 0, :].T, atol=1e-5), "MhSa (eq 12.12) mismatch"

# Masked version vs. reference causal mask
causal = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
out_m = multihead_self_attention(X, params, H, mask=True)
with torch.no_grad():
    ref_m, _ = mha(x_seq, x_seq, x_seq, attn_mask=causal, need_weights=False)
assert torch.allclose(out_m, ref_m[:, 0, :].T, atol=1e-5), "masked self-attention mismatch"
print("P2: all checks passed")
