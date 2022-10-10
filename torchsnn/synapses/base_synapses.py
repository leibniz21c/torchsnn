import torch
import torch.nn as nn
from typing import Optional

from torchsnn.neurons import BaseNeurons


class BaseSynapses(nn.Module):
    r"""Base synapses type.
    If you want to create and use a custom synapses, you must inherit this class.

    Args:
        source_neurons (BaseNeurons): source neurons.
        target_neurons (BaseNeurons): target neurons.
        adj_mat (torch.BoolTensor, optional): connection topology.
            If ``adj_mat`` is ``None``, synapses have fully connected topology.
        learner (BaseLearner, optional): STDP learner,
            If ``learner`` is ``None``, this layer is freezed.
    """

    def __init__(
        self,
        source_neurons: BaseNeurons,
        target_neurons: BaseNeurons,
        adj_mat: Optional[torch.BoolTensor] = None,
    ):
        super().__init__()
        self.__running_instance = False
        self.source_neurons = source_neurons
        self.target_neurons = target_neurons
        self.adj_mat = adj_mat
        self.reset()
        self.__running_instance = True

    @property
    def source_neurons(self):
        return self._source_neurons

    @source_neurons.setter
    def source_neurons(self, neurons: BaseNeurons):
        self._source_neurons = neurons
        if self.__running_instance:
            self.reset()

    @property
    def target_neurons(self):
        return self._target_neurons

    @target_neurons.setter
    def target_neurons(self, neurons: BaseNeurons):
        self._target_neurons = neurons
        if self.__running_instance:
            self.reset()

    @property
    def adj_mat(self):
        return self._adj_mat

    @adj_mat.setter
    def adj_mat(self, adj_matrix: BaseNeurons):
        if adj_matrix == None:
            self._adj_mat = None
            return

        num_source_neurons = self.source_neurons.num_neurons
        num_target_neurons = self.target_neurons.num_neurons

        if (
            adj_matrix.shape[0] != num_target_neurons
            or adj_matrix.shape[1] != num_source_neurons
        ):
            raise ValueError(
                f"Adj matrix must be ({num_target_neurons}, {num_source_neurons}) shape, but got ({adj_matrix.shape[0]}, {adj_matrix.shape[1]})."
            )
        self.register_buffer("_adj_mat", adj_matrix)

        if self.__running_instance:
            self.reset()

    def forward(self, x):
        if self.adj_mat == None:
            return x
        return self.adj_mat * x

    def reset(self):
        pass
