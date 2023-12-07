# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any

from enum import Enum
import math 
import torch
import torch.nn.functional as F
from torch import Tensor

from dsmeasure2.core.dsm_tensor import AbstractTensor
from dsmeasure2.core.dsm_device import AbstractDeviceConfig, AbstractDevice
from dsmeasure2.core.dsm_operator import AbstractOperatorConfig, \
                                         AbstractOperator, \
                                         OperatorComputationalConfig, \
                                         OperatorNonComputationalConfig , \
                                         OperatorCustomConfig , \
                                         OpStaticComputational , \
                                         OpStaticNonComputational, \
                                         OpStaticDerivative

from dsmeasure2.core.dsm_device_mng import DeviceManager
from dsmeasure2.device.device_cuda import DeviceCUDA
from dsmeasure2.device.device_pcie import DevicePCIE4

from dsmeasure2.graph.tensor_define import ActivationTensor, WeightTensor, TensorState

class FlattenOperator(OpStaticComputational):
    def __init__(self, 
                 config: OperatorComputationalConfig, 
                 callback: Callable[..., Any] = None):
        super().__init__(config)
        self.estimate_runtime: int = 0
        self.input: list[ActivationTensor] = []
        self.output: list[ActivationTensor] = []
        self.weight: WeightTensor = None
        self.intermediate_memory: int = 0
        self.device_name: list[str] = ['cuda:0']
        self.callback: Callable[..., Any] = callback
        self._prev_done = self._next = self._prev = None

    def add_next(self, next_op):
        raise Exception("flatten operator does not support add_next")

    def estimate(self, *tensor_in: Tensor) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self):
        _devices: list[AbstractDevice] = \
            [DeviceManager().find_by_name(_dname) for _dname in self.device_name]
        assert None not in _devices, "device not found"
        
        if _devices[0].try_occupy(self.estimate_runtime, \
                                  memory=sum([_tsz.tensor_size for _tsz in self.output]), 
                                  computational=False) == False:
            return False
        if False in [_dev.try_occupy(self.estimate_runtime) for _dev in _devices[1:]]:
            return False
        
        def _apply_cb():
            for _output in self.output:
                _output.materialize()
            for _input in self.input:
                _input.ref_count -= 1
            mem_free = sum(
                [_input.clear_state() for _input in self.input if _input.ref_count == 0])
            if mem_free:
                _devices[0].occupy(10, None, memory=-mem_free, computational=False)
            self.callback() if self.callback else None
        
        # un-computational device occupy 1us won't fail
        _devices[0].occupy(1, None, \
                           memory=sum([_tsz.tensor_size for _tsz in self.output]), \
                           computational=False)
        _devices[0].occupy(self.estimate_runtime, _apply_cb, \
                           memory=self.intermediate_memory, computational=True)
        for _dev in _devices[1:]:
            _dev.occupy(self.estimate_runtime, None)
        return True
    
    def reset(self) -> None:
        super().reset()
        for _input in self.input:
            if _input.state == TensorState.UNKNOWN:
                _input.initiate_state()
            _input.ref_count += 1
        for i in range(len(self.output)):
            if self.output[i] == TensorState.UNKNOWN:
                self.output[i].initiate_state()

class FlattenInitiate(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, callback: Callable[..., Any] = None):
        super().__init__(config)
        self.estimate_runtime: int = 1
        self.inputs: list[ActivationTensor] = []
        self.weight: list[WeightTensor] = []
        self.device_name: str = 'cuda:0'
        self.callback: Callable[..., Any] = callback
        self._prev_done = self._next = self._prev = None

    def add_next(self, next_op):
        raise Exception("flatten operator does not support add_next")

    def estimate(self, *tensor_in: Tensor) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self) -> bool:
        _device = DeviceManager().find_by_name(self.device_name)
        assert _device, "device not found"
        
        def _apply_cb():
            for _input in self.inputs:
                _input.materialize()
            self.callback() if self.callback else None
        # un-computational device occupy 10us won't fail
        _device.occupy(1, _apply_cb, \
                          memory=sum([_tsz.tensor_size for _tsz in self.inputs] + [_tsz.tensor_size for _tsz in self.weight]), 
                          computational=False)
        return True
    
    def reset(self) -> None:
        super().reset()
        assert len(self.inputs) > 0, \
            "{name} invalid computational graph".format(name=str(self))
        for i in range(len(self.inputs)):
            if self.inputs[i] == TensorState.UNKNOWN:
                self.inputs[i].initiate_state()
        
        for i in range(len(self.weight)):
            if self.weight[i] == TensorState.DESTROYED:
                self.weight[i].materialize()