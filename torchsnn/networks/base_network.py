import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
