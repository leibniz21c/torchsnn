import torch
import torch.nn as nn
from typing import Union

from torchsnn.synapses import BaseSynapses


class BaseLearner(nn.Module):
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
    ):
        super().__init__()

        self.__running_instance = False
        self.synapses = synapses
        self.pre_synaptic_trace_ = pre_synaptic_trace
        self.post_synaptic_trace_ = post_synaptic_trace
        self.learning_rate_pre_ = learning_rate_pre
        self.learning_rate_post_ = learning_rate_post
        self.tau_plus_ = tau_plus
        self.tau_minus_ = tau_minus
        self.a_plus_ = a_plus
        self.a_minus_ = a_minus
        self.reset()
        self.__running_instance = True

    @property
    def a_plus_(self):
        return self._a_plus_

    @a_plus_.setter
    def a_plus_(self, value: float):
        self._a_plus_ = value
        if self.__running_instance:
            self.reset()

    @property
    def a_minus_(self):
        return self._a_minus_

    @a_minus_.setter
    def a_minus_(self, value: float):
        self._a_minus_ = value
        if self.__running_instance:
            self.reset()

    @property
    def tau_plus_(self):
        return self._tau_plus_

    @tau_plus_.setter
    def tau_plus_(self, value: float):
        self._tau_plus_ = value
        if self.__running_instance:
            self.reset()

    @property
    def tau_minus_(self):
        return self._tau_minus_

    @tau_minus_.setter
    def tau_minus_(self, value: float):
        self._tau_minus_ = value
        if self.__running_instance:
            self.reset()

    @property
    def synapses(self):
        return self._synapses

    @synapses.setter
    def synapses(self, obj: BaseSynapses):
        self._synapses = obj
        if self.__running_instance:
            self.reset()

    @property
    def pre_synaptic_trace_(self):
        return self._pre_synaptic_trace_

    @pre_synaptic_trace_.setter
    def pre_synaptic_trace_(self, value: bool):
        self._pre_synaptic_trace_ = value
        if self.__running_instance:
            self.reset()

    @property
    def post_synaptic_trace_(self):
        return self._post_synaptic_trace_

    @post_synaptic_trace_.setter
    def post_synaptic_trace_(self, value: bool):
        self._post_synaptic_trace_ = value
        if self.__running_instance:
            self.reset()

    @property
    def learning_rate_pre_(self):
        return self._learning_rate_pre_

    @learning_rate_pre_.setter
    def learning_rate_pre_(self, value: Union[float, torch.Tensor]):
        self._learning_rate_pre_ = value
        if self.__running_instance:
            self.reset()

    @property
    def learning_rate_post_(self):
        return self._learning_rate_post_

    @learning_rate_post_.setter
    def learning_rate_post_(self, value: Union[float, torch.Tensor]):
        self._learning_rate_post_ = value
        if self.__running_instance:
            self.reset()

    @property
    def pre_synaptic_trace(self):
        return self._pre_synaptic_trace

    @pre_synaptic_trace.setter
    def pre_synaptic_trace(self, value: torch.Tensor):
        self._pre_synaptic_trace = value

    @property
    def post_synaptic_trace(self):
        return self._post_synaptic_trace

    @post_synaptic_trace.setter
    def post_synaptic_trace(self, value: torch.Tensor):
        self._post_synaptic_trace = value

    @property
    def learning_rate_pre(self):
        return self._learning_rate_pre

    @property
    def learning_rate_post(self):
        return self._learning_rate_post

    @property
    def tau_plus(self):
        return self._tau_plus

    @property
    def tau_minus(self):
        return self._tau_minus

    @property
    def a_plus(self):
        return self._a_plus

    @property
    def a_minus(self):
        return self._a_minus

    def reset(self):
        # Synaptic traces
        if self.pre_synaptic_trace_:
            self.register_buffer(
                "_pre_synaptic_trace",
                torch.zeros(self.synapses.source_neurons.num_neurons),
            )
            self.register_buffer(
                "_tau_plus",
                torch.zeros(self.synapses.source_neurons.num_neurons) + self.tau_plus_,
            )
            self.register_buffer(
                "_a_plus",
                torch.zeros(self.synapses.source_neurons.num_neurons) + self.a_plus_,
            )
        if self.post_synaptic_trace_:
            self.register_buffer(
                "_post_synaptic_trace",
                torch.zeros(self.synapses.target_neurons.num_neurons),
            )
            self.register_buffer(
                "_tau_minus",
                torch.zeros(self.synapses.target_neurons.num_neurons) + self.tau_minus_,
            )
            self.register_buffer(
                "_a_minus",
                torch.zeros(self.synapses.target_neurons.num_neurons) + self.a_minus_,
            )
        # Learning rates
        if isinstance(self.learning_rate_pre_, float):
            self.register_buffer(
                "_learning_rate_pre",
                torch.zeros_like(self.synapses.weight) + self.learning_rate_pre_,
            )
        elif isinstance(self.learning_rate_pre_, torch.Tensor):
            if self.synapses.weight.shape == self.learning_rate_pre_.shape:
                self.register_buffer("_learning_rate_pre", self.learning_rate_pre_)
            else:
                raise ValueError(
                    f"learning_rate must be float or {self.synapses.weight.shape} shaped tensor."
                )
        if isinstance(self.learning_rate_post_, float):
            self.register_buffer(
                "_learning_rate_post",
                torch.zeros_like(self.synapses.weight) + self.learning_rate_post_,
            )
        elif isinstance(self.learning_rate_post_, torch.Tensor):
            if self.synapses.weight.shape == self.learning_rate_post_.shape:
                self.register_buffer("_learning_rate_post", self.learning_rate_post_)
            else:
                raise ValueError(
                    f"learning_rate must be float or {self.synapses.weight.shape} shaped tensor."
                )

    def forward(self, x: torch.Tensor, mode: str):
        """Learner forward method.
        Compute synaptic traces
        .. math::
            \frac{dx_pre}{dt} = - \frac{x_pre}{tau_pre} + a_plus_pre * \sum(x)
            \frac{dx_post}{dt} = - \frac{x_post}{tau_post} + a_plus_post * \sum(x)

        Args:
            x (torch.Tensor): spike signal.
            mode (str): synaptic mode.
                'pre' or 'post'.

        Returns:
            None
        """
        if mode == "pre" and self.pre_synaptic_trace_:
            self.pre_synaptic_trace += (
                -self.pre_synaptic_trace / self.tau_plus + self.a_plus * sum(x)
            )
        elif mode == "post" and self.post_synaptic_trace_:
            self.post_synaptic_trace += (
                -self.post_synaptic_trace / self.tau_minus + self.a_minus * sum(x)
            )
