import torch
import torch.nn as nn
from typing import Union


class BaseNeurons(nn.Module):
    def __init__(
        self, 
        n: int,
        init_v_thres: float = 1.,
        init_v_reset: float = 0.,
        init_refrac: int = 5.,
        dt: int = 1, 
        spike_scaling: Union[int, float] = 1
    ):
        super().__init__()
        self.__running_instance = False

        self.n = n
        self.dt = dt
        self.spike_scaling = spike_scaling

        self.init_v_thres = init_v_thres
        self.init_v_reset = init_v_reset
        self.init_refrac = init_refrac

        self.reset()
        self.__running_instance = True


    @property
    def n(self):
        return self._n


    @n.setter
    def n(self, value: int):
        if value > 0:
            self._n = value
        else:
            raise ValueError(f'n must be a positive integer, not {value}.')

        if self.__running_instance:
            self.reset()


    def dt(self):
        return self._dt


    @dt.setter
    def dt(self, value: int):
        if value > 0:
            self._dt = value
        else:
            raise ValueError(f'dt must be a positive integer, not {value}.')


    @property
    def spike_scaling(self):
        return self._spike_scaling


    @spike_scaling.setter
    def spike_scaling(self, value: Union[int, float]):
        if value > 0:
            self._spike_scaling = value
        else:
            raise ValueError(f'spike_scaling must be a positive number, not {value}.')


    @property
    def init_v_thres(self):
        return self._init_v_thres


    @init_v_thres.setter
    def init_v_thres(self, value: float):
        self._init_v_thres = value
        if self.__running_instance:
            self.reset()


    @property
    def init_v_reset(self):
        return self.init_v_reset


    @init_v_reset.setter
    def init_v_reset(self, value: float):
        self._init_v_reset = value
        if self.__running_instance:
            self.reset()


    @property
    def init_refrac(self):
        return self.init_refrac


    @init_refrac.setter
    def init_refrac(self, value: int):
        self._init_refrac = value
        if self.__running_instance:
            self.reset()


    def forward(self, x):
        # Compute spike firing neurons
        self.timer_ref += self.dt
        ref_mask = self.timer_ref >= self.refrac
        spike_cand_mask = self.v >= self.v_thres
        spikes = ref_mask & spike_cand_mask

        # Reset voltages and refractory timers
        self.v[spikes] = self.v_reset[spikes]
        self.timer_ref[spikes] = 0

        return spikes.to(torch.int32)*self.spike_scaling
    

    def reset(self):
        self.register_buffer(
            'v', torch.zeros(self.n)
        )
        self.register_buffer(
            'v_thres', torch.zeros(self.n) + self.init_v_thres
        )
        self.register_buffer(
            'v_reset', torch.zeros(self.n) + self.init_v_reset
        )
        self.register_buffer(
            'refrac', torch.zeros(self.n).to(torch.int32) + self.init_refrac
            )
        self.register_buffer(
            'timer_ref', torch.zeros(self.n).to(torch.int32) + self.init_refrac
        )

        
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = super().extra_repr()
        
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, buffer in self._buffers.items():
            mod_str = repr(buffer)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str