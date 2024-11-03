import torch
from torch import nn

class PoissonEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        outspike = torch.rand_like(x).le(x).to(x)
        return outspike