import torch
import torch.nn as nn
from typing import Union

from torchsnn.learning import BaseLearner
from torchsnn.synapses import BaseSynapses


class PairBasedSTDP(BaseLearner):
    def __init__(
        self,
        synapses: BaseSynapses,
        pre_synaptic_trace: bool = False,
        post_synaptic_trace: bool = False,
        learning_rate_pre: Union[float, torch.Tensor] = 1e-3,
        learning_rate_post: Union[float, torch.Tensor] = 1e-3,
        tau_plus: float = 2.0,
        tau_minus: float = 2.0,
        a_plus: float = 1.0,
        a_minus: float = 1.0,
        A_plus: float = 1.0,
        A_minus: float = 1.0,
        w_min: float = -1.0,
        w_max: float = 1.0,
    ):
        super().__init__(
            synapses,
            pre_synaptic_trace,
            post_synaptic_trace,
            learning_rate_pre,
            learning_rate_post,
            tau_plus,
            tau_minus,
            a_plus,
            a_minus,
        )
        self.__running_instance = False
        self.A_plus_ = A_plus
        self.A_minus_ = A_minus
        self.w_min = w_min
        self.w_max = w_max
        self.reset()
        self.__running_instance = True

    @property
    def A_plus_(self):
        return self._A_plus_

    @A_plus_.setter
    def A_plus_(self, value: float):
        self._A_plus_ = value
        if self.__running_instance:
            self.reset()

    @property
    def A_minus_(self):
        return self._A_minus_

    @A_minus_.setter
    def A_minus_(self, value: float):
        self._A_minus_ = value
        if self.__running_instance:
            self.reset()

    @property
    def A_plus(self):
        return self._A_plus

    @property
    def A_minus(self):
        return self._A_minus

    @property
    def w_min(self):
        return self._w_min

    @w_min.setter
    def w_min(self, value: float):
        self._w_min = value

    @property
    def w_max(self):
        return self._w_max

    @w_max.setter
    def w_max(self, value: float):
        self._w_max = value

    def reset(self):
        super().reset()
        if self.pre_synaptic_trace_:
            self.register_buffer(
                "_A_plus", torch.zeros(self.synapses.source_neurons.num_neurons)
            )
        if self.post_synaptic_trace_:
            self.register_buffer(
                "_A_minus", torch.zeros(self.synapses.target_neurons.num_neurons)
            )

    def forward(self, x: torch.Tensor, mode: str):
        super().forward(x, mode)

        if mode == "pre" and self.pre_synaptic_trace_:
            self.synapses.weight -= self.learning_rate_pre * torch.matmul(
                (self.A_minus * self.post_synaptic_trace).unsqueeze(dim=-1),
                x.unsqueeze(dim=0),
            )
        if mode == "post" and self.post_synaptic_trace_:
            self.synapses.weight += self.learning_rate_post * torch.matmul(
                (self.A_minus * self.post_synaptic_trace).unsqueeze(dim=-1),
                x.unsqueeze(dim=0),
            )
        self.synapses.weight = torch.clamp(self.synapses.weight, self.w_min, self.w_max)
