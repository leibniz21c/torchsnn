from torchsnn.neurons import BaseNeurons
from typing import Union


class IFNeurons(BaseNeurons):
    def __init__(
        self,
        num_neurons: int,
        init_v_thres: float = 1.0,
        init_v_reset: float = 0.0,
        init_refrac: int = 5.0,
        dt: int = 1,
        spike_scaling: Union[int, float] = 1,
    ):
        super().__init__(
            num_neurons, init_v_thres, init_v_reset, init_refrac, dt, spike_scaling
        )

    def forward(self, x):
        """
        .. math::
            \frac{dv}{dt} = x
        """
        self.v += x * self.dt
        return super().forward(x)
