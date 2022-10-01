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
        self.synapses.weight.data *= self.adj_mat

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, mat: torch.FloatTensor):
        num_source_neurons = self.source_neurons.num_neurons
        num_target_neurons = self.target_neurons.num_neurons

        if len(mat.shape) != 2:
            raise ValueError(f"Length of shape of weight matrix must be {2}.")

        if num_target_neurons != mat.shape[0] or num_source_neurons != mat.shape[1]:
            raise ValueError(
                f"Weight matrix must be ({num_target_neurons}, {num_source_neurons}) shape, but got ({mat.shape[0]}, {mat.shape[1]})."
            )
        self._weight = mat

    def forward(self, x):
        return self.synapses(x)

    def reset(self):
        pass
