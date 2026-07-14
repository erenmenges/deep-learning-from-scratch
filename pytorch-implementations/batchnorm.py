import torch
from torch import nn

class BatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((num_features,)))
        self.delta = nn.Parameter(torch.zeros((num_features,)))
        self.capture = False
        self.register_buffer("m_frozen", torch.zeros(num_features))
        self.register_buffer("s_frozen", torch.ones(num_features))

    def _batch_statistics(self, h: torch.Tensor, dims: tuple) -> tuple:
        m_h = h.mean(dim=dims, keepdim=True)
        s_h = torch.sqrt(torch.mean((h - m_h) ** 2, dim=dims, keepdim=True))
        return m_h, s_h


    def forward(self, h: torch.Tensor) -> torch.Tensor:
        dims = (0,) if h.ndim == 2 else (0,2,3)

        if self.training:
            m_h, s_h = self._batch_statistics(h, dims)
        else:
            shape_frozen = [1] * h.ndim
            shape_frozen[1] = -1
            m_h = self.m_frozen.view(shape_frozen)
            s_h = self.s_frozen.view(shape_frozen)
        
        h_forward = (h - m_h) / (s_h + self.eps)

        shape_dg = [1] * h.ndim
        shape_dg[1] = -1
        g = self.gamma.view(shape_dg)
        d = self.delta.view(shape_dg)
        h_forward = h_forward * g + d

        if self.capture:
            self.freeze_statistics(h)
        
        return h_forward 
        
    def freeze_statistics(self, h_all: torch.Tensor) -> None:
        with torch.no_grad():    
            dims = (0,) if h_all.ndim == 2 else (0,2,3)
            m_h, s_h = self._batch_statistics(h_all, dims)

            self.m_frozen.copy_(m_h.reshape(-1))
            self.s_frozen.copy_(s_h.reshape(-1))