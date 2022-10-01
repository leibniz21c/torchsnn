from torchsnn.neurons import BaseNeurons
from typing import Union


class LIFNeurons(BaseNeurons):
    def __init__(
        self,
        num_neurons: int,
        tau: Union[int, float] = 2.0,
        init_v_thres: float = 1.0,
        init_v_reset: float = 0.0,
        init_refrac: int = 5.0,
        dt: int = 1,
        spike_scaling: Union[int, float] = 1,
    ):
        super().__init__(
            num_neurons, init_v_thres, init_v_reset, init_refrac, dt, spike_scaling
        )
        self.tau = tau

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value: Union[int, float]):
        if value >= 2:
            self._tau = value
        else:
            raise ValueError(f"tau must be greater than 2, but {value} < 2.")

    def forward(self, x):
        """
        .. math::
            \tau \frac{dv}{dt} = x - (v - v_\text{reset})
        """
        self.v += (x - (self.v - self.v_reset)) * self.dt / self.tau
        return super().forward(x)
