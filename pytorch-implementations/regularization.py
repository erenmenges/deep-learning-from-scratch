import torch
from torch import nn
import torch.nn.functional as F

def l2_penalty(model: nn.Module) -> torch.Tensor:
    weights_flat = torch.cat([parameter.flatten() for name,parameter in model.named_parameters() if name.endswith(".weight")])
    l2_element = (weights_flat ** 2).sum()
    return l2_element

def label_smoothed_ce(logits: torch.Tensor, targets: torch.Tensor, rho: float) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=1)

    q = torch.full(logits.shape, rho/(logits.shape[1]-1))
    q[torch.arange(q.shape[0]), targets] = 1 - rho

    return -1 * (1/logits.shape[0]) * (q * log_p).sum()
