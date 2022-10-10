import torch
import torch.nn as nn
from typing import Union


class BaseNeurons(nn.Module):
    r"""Base neurons type.
    If you want to create and use a custom neuron, you must inherit this class.

    Args:
        num_neurons (int): number of neurons.
        init_v_thres (float): initial threshold.
            If the membrane voltage exceeds that value, the neuron fires.
            The initial threshold of all neurons is initialized to that value.
        init_v_reset (float): initial reset potential.
            If the neuron fires, the membrane voltage is initialized to the corresponding value.
        init_refrac (int): initial absolute refractory period.
            After a neuron fires, it cannot fire again during that period.
            The units are milliseconds(ms).
        dt (int): default unit of calculation time.
        spike_scaling (int, float): spike voltage.
    """

    def __init__(
        self,
        num_neurons: int,
        init_v_thres: float = 1.0,
        init_v_reset: float = 0.0,
        init_refrac: int = 5,
        dt: int = 1,
    ):
        super().__init__()
        self.__running_instance = False

        # Setup
        self.num_neurons = num_neurons
        self.dt = dt

        self.init_v_thres = init_v_thres
        self.init_v_reset = init_v_reset
        self.init_refrac = init_refrac

        self.reset()
        self.__running_instance = True

    @property
    def num_neurons(self):
        return self._num_neurons

    @num_neurons.setter
    def num_neurons(self, value: int):
        if value > 0:
            self._num_neurons = value
        else:
            raise ValueError(f"n must be a positive integer, not {value}.")

        if self.__running_instance:
            self.reset()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value: int):
        if value > 0:
            self._dt = value
        else:
            raise ValueError(f"dt must be a positive integer, not {value}.")

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
        return self._init_v_reset

    @init_v_reset.setter
    def init_v_reset(self, value: float):
        self._init_v_reset = value
        if self.__running_instance:
            self.reset()

    @property
    def init_refrac(self):
        return self._init_refrac

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

        return spikes.float()

    def reset(self):
        self.register_buffer("v", torch.zeros(self.num_neurons) + self.init_v_reset)
        self.register_buffer(
            "v_thres", torch.zeros(self.num_neurons) + self.init_v_thres
        )
        self.register_buffer(
            "v_reset", torch.zeros(self.num_neurons) + self.init_v_reset
        )
        self.register_buffer(
            "refrac", torch.zeros(self.num_neurons).to(torch.int32) + self.init_refrac
        )
        self.register_buffer(
            "timer_ref",
            torch.zeros(self.num_neurons).to(torch.int32) + self.init_refrac,
        )

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = super().extra_repr()

        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, buffer in self._buffers.items():
            mod_str = repr(buffer.shape)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
