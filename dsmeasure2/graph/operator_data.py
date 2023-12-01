# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any

from enum import Enum
import math 
import torch
import torch.nn.functional as F
from torch import Tensor

from dsmeasure2.core.dsm_tensor import AbstractTensor
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

class DefaultCopyOperator(OpStaticNonComputational):
    """
    default offload operator
        config: OperatorNonComputationalConfig
        alloc_memory: memory allocated (MB)
        device_name[2]: [cuda device name: str, pcie device name: str]
    """
    def __init__(self, config: OperatorNonComputationalConfig, 
                 alloc_memory: int = 0, 
                 device_name: list[str] = ['cuda:0', 'pcie:0'],
                 callback: Callable[..., Any] = None):
        super().__init__(config)
        
        self.device_name: list[str] = device_name
        self.alloc_memory: int = alloc_memory
        self.callback: Callable[..., Any] = callback

    def estimate(self, *tensor_in: Tensor) -> int:
        device_1: DevicePCIE4 = DeviceManager().find_by_name(self.device_name[1])
        pcie_bandwidth_p2p = device_1.config.pcie_bandwidth_p2p
        return int(self.alloc_memory / pcie_bandwidth_p2p)
    
    def apply(self) -> bool:
        device_0: DeviceCUDA = DeviceManager().find_by_name(self.device_name[0])
        device_1: DevicePCIE4 = DeviceManager().find_by_name(self.device_name[1])
        assert device_0 is not None and device_1 is not None, "device not found"
        def _apply_cb():
            self.default_apply_cb()
            if self.callback is not None:
                self.callback()
        return device_1.occupy(None, _apply_cb, dsize=self.alloc_memory)

class DefaultMallocOperator(OpStaticNonComputational):
    """
    default memory malloc operator
        config: OperatorNonComputationalConfig
        alloc_memory: memory allocated (MB)
        device_name[1]: [cuda device name: str]
    """
    def __init__(self, config: OperatorNonComputationalConfig, 
                 alloc_memory: int = 0, 
                 device_name: list[str] = ['cuda:0'],
                 callback: Callable[..., Any] = None):
        super().__init__(config)
        
        self.device_name: list[str] = device_name
        self.alloc_memory: int = alloc_memory
        self.callback: Callable[..., Any] = callback
        
    def estimate(self, *tensor_in: Tensor) -> int:
        """
        default free memory to torch cuda memory pool (10us cost estimated)
        """
        return 10
    
    def apply(self) -> bool:
        """
        alloc alloc_memory
        """
        device_0: DeviceCUDA = DeviceManager().find_by_name(self.device_name[0])
        assert device_0 is not None, "device not found"
        def _apply_cb():
            self.default_apply_cb()
            if self.callback is not None:
                self.callback()
        return device_0.occupy(self.estimate(), _apply_cb, \
                             memory=self.alloc_memory, computational=False)

class DefaultFreeOperator(OpStaticNonComputational):
    """
    default memory free operator
        config: OperatorNonComputationalConfig
        alloc_memory: memory allocated (MB)
        device_name[1]: [cuda device name: str]
    """
    def __init__(self, config: OperatorNonComputationalConfig, 
                 alloc_memory: int = 0, 
                 device_name: list[str] = ['cuda:0'],
                 callback: Callable[..., Any] = None):
        super().__init__(config)
        
        self.device_name: list[str] = device_name
        self.alloc_memory: int = alloc_memory
        self.callback: Callable[..., Any] = callback
        
    def estimate(self, *tensor_in: Tensor) -> int:
        """
        default free memory to torch cuda memory pool (10us cost estimated)
        """
        return 10
    
    def apply(self) -> bool:
        """
        free alloc_memory
        """
        device_0: DeviceCUDA = DeviceManager().find_by_name(self.device_name[0])
        assert device_0 is not None, "device not found"
        def _apply_cb():
            self.default_apply_cb()
            if self.callback is not None:
                self.callback()
        return device_0.occupy(self.estimate(), _apply_cb, \
                             memory=-self.alloc_memory, computational=False)

class DefaultOffloadOperator(OpStaticDerivative):
    def __init__(self, config: OperatorCustomConfig, 
                 tensor_offload: AbstractTensor = None,
                 device_name: list[str] = ['cuda:0', 'pcie:0']):
        super().__init__(config)
        self.tensor_offload: AbstractTensor = tensor_offload
        self.device_name: list[str] = device_name
        self.memcpy_operator: DefaultCopyOperator = \
                DefaultCopyOperator(config, self.tensor_offload.tensor_size, device_name)
        self.cufree_operator: DefaultFreeOperator = \
                DefaultFreeOperator(config, self.tensor_offload.tensor_size, device_name[0], self.tensor_offload.offload)
        
        self.memcpy_operator.add_next(self.cufree_operator)
        self._subop = [self.memcpy_operator, self.cufree_operator]

class DefaultFetchOperator(OpStaticDerivative):
    def __init__(self, config: OperatorCustomConfig, 
                 tensor_offload: AbstractTensor = None,
                 device_name: list[str] = ['cuda:0', 'pcie:0']):
        super().__init__(config)
        self.tensor_offload: AbstractTensor = tensor_offload
        self.device_name: list[str] = device_name
        self.alloc_operator: DefaultMallocOperator = \
                DefaultFreeOperator(config, self.tensor_offload.tensor_size, device_name[0])
        self.memcpy_operator: DefaultCopyOperator = \
                DefaultCopyOperator(config, self.tensor_offload.tensor_size, device_name, self.tensor_offload.offload)
        
        self.alloc_operator.add_next(self.memcpy_operator)
        self._subop = [self.alloc_operator, self.memcpy_operator]