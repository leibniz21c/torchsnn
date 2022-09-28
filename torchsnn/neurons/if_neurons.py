from torchsnn.neurons import BaseNeurons
from typing import Union


class IFNeurons(BaseNeurons):
    def __init__(
        self, 
        n: int,
        init_v_thres: float = 1.,
        init_v_reset: float = 0.,
        init_refrac: int = 5.,
        dt: int = 1, 
        spike_scaling: Union[int, float] = 1
    ):
        super().__init__(
            n, 
            init_v_thres, 
            init_v_reset, 
            init_refrac, 
            dt, 
            spike_scaling
        )
        
    def forward(self, x):
        """
        .. math::
            \frac{dv}{dt} = x
        """
        self.v += x
        return super().forward(x)