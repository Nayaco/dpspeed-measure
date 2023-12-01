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

class UnaryOperator(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, callback: Callable[..., Any] = None):
        super().__init__(config)
        self.estimate_runtime: int = 0
        self.input: ActivationTensor = None
        self.output: list[ActivationTensor] = []
        self.weight: WeightTensor = None
        self.intermediate_memory: int = 0
        self.device_name: list[str] = ['cuda:0']
        self.callback: Callable[..., Any] = callback

    def set_parameters(self,
                       estimate_runtime: int,
                       input: ActivationTensor, 
                       output: list[ActivationTensor], 
                       weight: WeightTensor,
                       intermediate_memory: int,
                       device_name: list[str] = ['cuda:0']):
        """
        device_name[0] should be the only one that apply computation
        """
        self.estimate_runtime = estimate_runtime
        self.input = input  
        self.output = output
        self.weight = weight
        self.intermediate_memory = intermediate_memory
        self.device_name = device_name

    def estimate(self, *tensor_in: Tensor) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self):
        assert self.input and self.input.state == TensorState.AVAILABLE, \
            "{name} invalid computational graph".format(name=str(self))

        _devices: list[AbstractDevice] = [DeviceManager().find_by_name(_dname) for _dname in self.device_name]
        assert None not in _devices, "device not found"
        
        if _devices[0].try_occupy(self.estimate_runtime, \
                                  memory=sum([_tsz.tensor_size for _tsz in self.output]), 
                                  computational=False) == False:
            return False
        for _dev in _devices[1:]:
            if _dev.try_occupy(self.estimate_runtime) == False:
                return False
        
        def _apply_cb():
            for i in range(len(self.output)):
                self.output[i].materialize()
            self.default_apply_cb()

            self.input.ref_count -= 1
            if self.input.ref_count == 0:
                _devices[0].occupy(10, None, memory=-self.input.tensor_size, computational=False)
                self.input.clear_state()

            if self.callback is not None:
                self.callback()
        # un-computational device occupy 10us won't fail
        _devices[0].occupy(10, None, memory=sum([_tsz.tensor_size for _tsz in self.output]), 
                           computational=False)
        _devices[0].occupy(self.estimate_runtime, _apply_cb, \
                           memory=self.intermediate_memory, computational=True)
        for _dev in _devices[1:]:
            _dev.occupy(self.estimate_runtime, None)
        return True
    
    def reset(self) -> None:
        super().reset()
        assert self.input is not None, \
            "{name} invalid computational graph".format(name=str(self))
        if self.input.state == TensorState.UNKNOWN:
            self.input.initiate_state()
        self.input.ref_count += 1
        for i in range(len(self.output)):
            if self.output[i] == TensorState.UNKNOWN:
                self.output[i].initiate_state()

class BinaryOperator(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, callback: Callable[..., Any] = None):
        super().__init__(config)
        self.estimate_runtime: int = 0
        self.input_a: ActivationTensor = None
        self.input_b: ActivationTensor = None
        self.output: list[ActivationTensor] = []
        self.weight: WeightTensor = None
        self.intermediate_memory: int = 0
        self.device_name: list[str] = ['cuda:0']
        self.callback: Callable[..., Any] = callback

    def set_parameters(self,
                       estimate_runtime: int,
                       input_a: ActivationTensor, 
                       input_b: ActivationTensor, 
                       output: list[ActivationTensor], 
                       weight: WeightTensor,
                       intermediate_memory: int,
                       device_name: list[str] = ['cuda:0']):
        """
        device_name[0] should be the only one that apply computation
        """
        self.estimate_runtime = estimate_runtime
        self.input_a = input_a
        self.input_b = input_b
        self.output = output
        self.weight = weight
        self.intermediate_memory = intermediate_memory
        self.device_name = device_name

    def estimate(self, *tensor_in: Tensor) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self):
        assert self.input_a and self.input_b and self.input_a.state == TensorState.AVAILABLE and \
            self.input_b.state == TensorState.AVAILABLE, \
                "{name} invalid computational graph".format(name=str(self))

        _devices: list[AbstractDevice] = [DeviceManager().find_by_name(_dname) for _dname in self.device_name]
        assert None not in _devices, "device not found"
        
        if _devices[0].try_occupy(self.estimate_runtime, \
                                  memory=sum([_tsz.tensor_size for _tsz in self.output]), 
                                  computational=False) == False:
            return False
        for _dev in _devices[1:]:
            if _dev.try_occupy(self.estimate_runtime) == False:
                return False
            
        def _apply_cb():
            for i in range(len(self.output)):
                self.output[i].materialize()
            self.default_apply_cb()

            self.input_a.ref_count -= 1
            if self.input_a.ref_count == 0:
                _devices[0].occupy(10, None, memory=-self.input_a.tensor_size, computational=False)
                self.input_a.clear_state()
            self.input_b.ref_count -= 1
            if self.input_b.ref_count == 0:
                _devices[0].occupy(10, None, memory=-self.input_b.tensor_size, computational=False)
                self.input_b.clear_state()

            if self.callback is not None:
                self.callback()
        # un-computational device occupy 10us won't fail
        _devices[0].occupy(10, None, memory=sum([_tsz.tensor_size for _tsz in self.output]), 
                           computational=False)
        _devices[0].occupy(self.estimate_runtime, _apply_cb, \
                           memory=self.intermediate_memory, computational=True)
        for _dev in _devices[1:]:
            _dev.occupy(self.estimate_runtime)
        return True
    
    def reset(self) -> None:
        super().reset()
        assert self.input_a is not None and self.input_b is not None, \
            "{name} invalid computational graph".format(name=str(self))
        if self.input_a.state == TensorState.UNKNOWN:
            self.input_a.initiate_state()
        if self.input_b.state == TensorState.UNKNOWN:
            self.input_b.initiate_state()
        self.input_a.ref_count += 1
        self.input_b.ref_count += 1
        for i in range(len(self.output)):
            if self.output[i] == TensorState.UNKNOWN:
                self.output[i].initiate_state()

class TernaryOperator(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, callback: Callable[..., Any] = None):
        super().__init__(config)
        self.estimate_runtime: int = 0
        self.input_a: ActivationTensor = None
        self.input_b: ActivationTensor = None
        self.input_c: ActivationTensor = None
        self.output: list[ActivationTensor] = []
        self.weight: WeightTensor = None
        self.intermediate_memory: int = 0
        self.device_name: list[str] = ['cuda:0']
        self.callback: Callable[..., Any] = callback

    def set_parameters(self,
                       estimate_runtime: int,
                       input_a: ActivationTensor, 
                       input_b: ActivationTensor,
                       input_c: ActivationTensor, 
                       output: list[ActivationTensor], 
                       weight: WeightTensor,
                       intermediate_memory: int,
                       device_name: list[str] = ['cuda:0']):
        """
        device_name[0] should be the only one that apply computation
        """
        self.estimate_runtime = estimate_runtime
        self.input_a = input_a
        self.input_b = input_b
        self.input_c = input_c
        self.output = output
        self.weight = weight
        self.intermediate_memory = intermediate_memory
        self.device_name = device_name

    def estimate(self, *tensor_in: Tensor) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self):
        assert self.input_a and self.input_b and self.input_c and \
            self.input_a.state == TensorState.AVAILABLE and \
            self.input_b.state == TensorState.AVAILABLE and \
            self.input_c.state == TensorState.AVAILABLE, \
                "{name} invalid computational graph".format(name=str(self))

        _devices: list[AbstractDevice] = [DeviceManager().find_by_name(_dname) for _dname in self.device_name]
        assert None not in _devices, "device not found"
        
        if _devices[0].try_occupy(self.estimate_runtime, \
                                  memory=sum([_tsz.tensor_size for _tsz in self.output]), 
                                  computational=False) == False:
            return False
        for _dev in _devices[1:]:
            if _dev.try_occupy(self.estimate_runtime) == False:
                return False
            
        def _apply_cb():
            for i in range(len(self.output)):
                self.output[i].materialize()
            self.default_apply_cb()

            self.input_a.ref_count -= 1
            if self.input_a.ref_count == 0:
                _devices[0].occupy(10, None, memory=-self.input_a.tensor_size, computational=False)
                self.input_a.clear_state()
            self.input_b.ref_count -= 1
            if self.input_b.ref_count == 0:
                _devices[0].occupy(10, None, memory=-self.input_b.tensor_size, computational=False)
                self.input_b.clear_state()
            self.input_c.ref_count -= 1
            if self.input_c.ref_count == 0:
                _devices[0].occupy(10, None, memory=-self.input_c.tensor_size, computational=False)
                self.input_c.clear_state()

            if self.callback is not None:
                self.callback()
        # un-computational device occupy 10us won't fail
        _devices[0].occupy(10, None, memory=sum([_tsz.tensor_size for _tsz in self.output]), 
                           computational=False)
        _devices[0].occupy(self.estimate_runtime, _apply_cb, \
                           memory=self.intermediate_memory, computational=True)
        for _dev in _devices[1:]:
            _dev.occupy(self.estimate_runtime)
        return True
    
    def reset(self) -> None:
        super().reset()
        assert self.input_a is not None and self.input_b is not None and \
            self.input_c is not None, \
                "{name} invalid computational graph".format(name=str(self))
        if self.input_a.state == TensorState.UNKNOWN:
            self.input_a.initiate_state()
        if self.input_b.state == TensorState.UNKNOWN:
            self.input_b.initiate_state()
        if self.input_c.state == TensorState.UNKNOWN:
            self.input_c.initiate_state()
        self.input_a.ref_count += 1
        self.input_b.ref_count += 1
        self.input_c.ref_count += 1
        for i in range(len(self.output)):
            if self.output[i] == TensorState.UNKNOWN:
                self.output[i].initiate_state()

class InitiateOperator(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, callback: Callable[..., Any] = None):
        super().__init__(config)
        self.estimate_runtime: int = 0
        self.inputs: list[ActivationTensor] = []
        self.weight: list[WeightTensor] = []
        self.device_name: str = ['cuda:0']
        self.callback: Callable[..., Any] = callback

    def set_parameters(self,
                       inputs: list[ActivationTensor],
                       weight: list[WeightTensor],
                       device_name: str = 'cuda:0'):
        self.inputs = inputs
        self.weight = weight
        self.device_name = device_name

    def estimate(self, *tensor_in: Tensor) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self):
        assert sum([_input.state == TensorState.DESTROYED for _input in self.inputs]) == len(self.inputs), \
            "{name} invalid computational graph".format(name=str(self))
        _device = DeviceManager().find_by_name(self.device_name)
        assert _device is not None, "device not found"
        
        def _apply_cb():
            for i in range(len(self.inputs)):
                self.inputs[i].materialize()
            self.default_apply_cb()

            if self.callback is not None:
                self.callback()
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