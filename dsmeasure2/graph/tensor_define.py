# Copyright (c) 2023, ISCS, Wenjie Zhang.
import math

from dataclasses import dataclass
from typing import Callable, Any
from enum import Enum

from dsmeasure2.core.dsm_tensor import AbstractTensor

class TensorState(Enum):
    AVAILABLE = 0
    OFFLOADED = 1
    DESTROYED = 2
    UNKNOWN = 63

class WeightTensor(AbstractTensor):
    """
    """
    def __init__(self, tensor_uid: int = 0, tensor_size: int = 0):
        super().__init__(tensor_uid, tensor_size)
        self.state: TensorState = TensorState.AVAILABLE

    def offload(self):
        self.state = TensorState.OFFLOADED
        
    def materialize(self):
        self.state = TensorState.AVAILABLE

    def destroy(self):
        raise Exception("weight tensor cannot be destroyed")

class ActivationTensor(AbstractTensor):
    """
    """
    def __init__(self, tensor_uid: int = 0, tensor_size: int = 0):
        super().__init__(tensor_uid, tensor_size)
        self.state: TensorState = TensorState.UNKNOWN
        self.ref_count: int = 0

    def offload(self):
        self.state = TensorState.OFFLOADED
    
    def materialize(self):
        self.state = TensorState.AVAILABLE

    def destroy(self):
        self.state =  TensorState.DESTROYED

    def clear_state(self):
        self.state = TensorState.UNKNOWN
        
    def initiate_state(self):
        self.state = TensorState.DESTROYED
        self.ref_count = 0
    
