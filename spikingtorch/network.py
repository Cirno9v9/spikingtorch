import torch
from torch import nn
from . import layer, neuron, surrogate

class SingleFeedForward(nn.Module):
    def __init__(self, tau) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )
    
    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
    def reset(self):
        for layer in self.layers.children():
            layer.reset()
