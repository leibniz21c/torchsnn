from typing import Union
import torch
from torchsnn.neurons import BaseNeurons


class LIFNeurons(BaseNeurons):
    def __init__(
        self,
        num_neurons: int,
        tau: Union[int, float] = 2.0,
        init_v_thres: float = 1.0,
        init_v_reset: float = 0.0,
        init_refrac: int = 5.0,
        dt: int = 1,
    ):
        super().__init__(num_neurons, init_v_thres, init_v_reset, init_refrac, dt)
        self.tau = tau

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value: Union[int, float]):
        if value > 0:
            self.register_buffer("_tau", torch.zeros(self.num_neurons) + value)
        else:
            raise ValueError(f"tau must be greater than 0, but {value} <= 0.")

    def forward(self, x):
        """
        .. math::
            \tau \frac{dv}{dt} = x - (v - v_\text{reset})
        """
        self.v += (x - (self.v - self.v_reset)) * self.dt / self.tau
        return super().forward(x)
