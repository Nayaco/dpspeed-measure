# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any

from enum import Enum
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
                 checkpointing: bool = False, 
                 callback: Callable[..., Any] = None):
        super().__init__(config)
        self._estimate_runtime: int = 0
        self._input: list[ActivationTensor] = []
        self._output: list[ActivationTensor] = []
        self._weight: WeightTensor = None
        self._intermediate_memory: int = 0
        self._device_name: list[str] = ['cuda:0']
        self._callback: Callable[..., Any] = callback
        self._prev_done = self._next = self._prev = None
        self._checkpointing = checkpointing

    def add_next(self, next_op):
        raise Exception("flatten operator does not support add_next")

    def estimate(self, *tensor_in: Tensor) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self):
        _devices: list[AbstractDevice] = \
            [DeviceManager().find_by_name(_dname) for _dname in self._device_name]
        assert None not in _devices, "device not found"
        
        if _devices[0].try_occupy(self._estimate_runtime, \
                                  memory=sum([_ts.tensor_size for _ts in self._output]), 
                                  computational=False) == False:
            return False
        if False in [_dev.try_occupy(self._estimate_runtime) for _dev in _devices[1:]]:
            return False
        
        def _apply_cb():
            for _output in self._output:
                _output.materialize()
            # if not self._checkpointing:
            for _input in self._input:
                _input.ref_count -= 1
            mem_free = sum(
                [_input.clear_state() for _input in self._input if _input.ref_count == 0])
            if mem_free:
                _devices[0].occupy(1, None, memory=-mem_free, computational=False)
            self._callback() if self._callback is not None else None
        
        # un-computational device occupy 1us won't fail
        _devices[0].occupy(1, None, \
                           memory=sum([_ts.tensor_size for _ts in self._output]), \
                           computational=False)
        _devices[0].occupy(self._estimate_runtime, _apply_cb, \
                           memory=self._intermediate_memory, computational=True)
        for _dev in _devices[1:]:
            _dev.occupy(self._estimate_runtime, None)
        return True
    
    def reset(self) -> None:
        super().reset()
        for _input in self._input:
            _input.state != TensorState.UNKNOWN or _input.initiate_state()
            _input.ref_count += 1
        for _output in self._output:
            _output.state != TensorState.UNKNOWN or _output.initiate_state()  
                

class FlattenInitiate(OpStaticComputational):
    def __init__(self, 
                 config: OperatorComputationalConfig, 
                 callback: Callable[..., Any] = None):
        super().__init__(config)
        self._estimate_runtime: int = 1
        self._inputs: list[ActivationTensor] = []
        self._weight: list[WeightTensor] = []
        self._device_name: str = 'cuda:0'
        self._callback: Callable[..., Any] = callback
        self._prev_done = self._next = self._prev = None

    def add_next(self, next_op):
        raise Exception("flatten operator does not support add_next")

    def estimate(self, *tensor_in: Tensor) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self) -> bool:
        _device = DeviceManager().find_by_name(self._device_name)
        assert _device, "device not found"
        
        def _apply_cb():
            for _input in self._inputs:
                _input.materialize()
            self._callback() if self._callback is not None else None
        # un-computational device occupy 10us won't fail
        _device.occupy(1, _apply_cb, \
                          memory=sum(
                              [_ts.tensor_size for _ts in self._inputs] + \
                              [_ts.tensor_size for _ts in self._weight]), 
                          computational=False)
        return True
    
    def reset(self) -> None:
        assert len(self._inputs) > 0, "invalid computational graph"
        super().reset()
        for _input in self._inputs:
            _input.state == TensorState.UNKNOWN or _input.initiate_state()
        
        for _weight in self._weight:
            _weight.state == TensorState.DESTROYED or _weight.materialize()