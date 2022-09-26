import torch
import torch.nn as nn
from typing import Iterable, Optional, Union


class BaseNeurons(nn.Module):
    #
    #
    #   TODO
    #
    #
    def __init__(
        self, 
        n,
        v_thres=1,
        v_reset=0,
        refrac=5,
        dt=1, 
        spike_scaling=1
    ):
        super().__init__()
        self.n = n
        self.dt = dt
        self.spike_scaling = spike_scaling
        self.__v_thres = v_thres
        self.__v_reset = v_reset
        self.__refrac = refrac
        
        self.register_buffer('v', torch.zeros(self.n))
        self.register_buffer('v_thres', torch.zeros(self.n) + self.__v_thres)
        self.register_buffer('v_reset', torch.zeros(self.n) + self.__v_reset)
        self.register_buffer('refrac', torch.zeros(self.n).to(torch.int32) + self.__refrac)
        self.register_buffer('timer_ref', torch.zeros(self.n).to(torch.int32) + self.__refrac)
    
    def forward(self, x):
        self.timer_ref += self.dt
        ref_mask = self.timer_ref >= self.refrac
        spike_cand_mask = self.v >= self.v_thres
        spikes = ref_mask & spike_cand_mask
        
        # Reset
        self.v[spikes] = self.v_reset[spikes]
        self.timer_ref[spikes] = 0
        return spikes.to(torch.int32)*self.spike_scaling
    
    def reset(self):
        self.register_buffer('v', torch.zeros(self.n))
        self.register_buffer('v_thres', torch.zeros(self.n) + self.__v_thres)
        self.register_buffer('v_reset', torch.zeros(self.n) + self.__v_reset)
        self.register_buffer('refrac', torch.zeros(self.n).to(torch.int32) + self.__refrac)
        self.register_buffer('timer_ref', torch.zeros(self.n).to(torch.int32) + self.__refrac)
        
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


class IFNeurons(BaseNeurons):
    #
    #
    #   TODO
    #
    #
    def __init__(
        self,
        n,
        v_thres=1,
        v_reset=0,
        refrac=5,
        dt=1,
        spike_scaling=1
    ):
        super().__init__(n, v_thres, v_reset, refrac, dt, spike_scaling)
        
    def forward(self, x):
        self.v += x
        return super().forward(x)