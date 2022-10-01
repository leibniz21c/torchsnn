import torch.nn as nn
from torchsnn.neurons import BaseNeurons


class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()

    def reset(self):
        for component in self.children():
            if isinstance(component, BaseNeurons):
                component.reset()
