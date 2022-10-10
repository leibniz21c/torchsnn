import torch
import torch.nn as nn
from typing import Optional

from torchsnn.synapses import BaseSynapses
from torchsnn.neurons import BaseNeurons


class LinearSynapses(BaseSynapses):
    def __init__(
        self,
        source_neurons: BaseNeurons,
        target_neurons: BaseNeurons,
        adj_mat: Optional[torch.BoolTensor] = None,
    ):
        super().__init__(source_neurons, target_neurons, adj_mat)
        self.synapses = nn.Linear(
            in_features=source_neurons.num_neurons,
            out_features=target_neurons.num_neurons,
            bias=False,
        )
        for param in self.synapses.parameters():
            param.requires_grad = False

        if self.adj_mat != None:
            self.synapses.weight.data *= self.adj_mat

    @property
    def weight(self):
        return self.synapses.weight.data

    @weight.setter
    def weight(self, value: torch.Tensor):
        self.synapses.weight.data = value

    def forward(self, x):
        return self.synapses(x)

    def reset(self):
        pass
