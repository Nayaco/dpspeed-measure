# Copyright (c) 2023, ISCS, Wenjie Zhang.
from typing import Callable, Any

from dsmeasure2.core.dsm_tensor import AbstractTensor
from dsmeasure2.core.dsm_device import AbstractDeviceConfig, AbstractDevice
from dsmeasure2.core.dsm_operator import AbstractOperatorConfig, \
                                         AbstractOperator, \
                                         OperatorComputationalConfig, \
                                         OperatorNonComputationalConfig, \
                                         OperatorCustomConfig, \
                                         OpStaticComputational, \
                                         OpStaticNonComputational, \
                                         OpStaticDerivative

from dsmeasure2.core.dsm_device_mng import DeviceManager
from dsmeasure2.device.device_cuda import DeviceCUDA
from dsmeasure2.device.device_pcie import DevicePCIE4

from dsmeasure2.graph.tensor_define import ActivationTensor, WeightTensor, TensorState
from dsmeasure2.flatten.flatten_operator import FlattenOperator, FlattenInitiate

class FlattenController(AbstractOperator):
    def __init__(self, config: AbstractOperatorConfig, callback: Callable[..., Any] = None):
        super().__init__(config)
        self._prev_done = self._next = self._prev = None
        self._callback: Callable[..., Any] = callback

    def add_next(self, next_op):
        raise Exception("flatten operator does not support add_next")
        
class FlattenBranch(FlattenController):
    def __init__(self, config: AbstractOperatorConfig,
                 branch_dst: Any, 
                 callback: Callable[..., Any] = None):
        super().__init__(config, callback)
        self._flatten_stream_to: FlattenStream = branch_dst
    
    def apply(self) -> bool:
        self._flatten_stream_to._activate = True
        return True

class FlattenStream:
    def __init__(self, 
                 flat_seq: list[FlattenOperator|FlattenInitiate], 
                 boot_pnt: int=0, 
                 reentrant: bool=False) -> None:
        self._flat_seq = flat_seq
        self._stream_cnt = -boot_pnt
        self._activate = False
        self._idle = True
        self._reentrant = reentrant

    def forward(self) -> FlattenInitiate|FlattenOperator|None:
        if not self._activate:
            return None
        self._stream_cnt += 1
        def _callback():
            self._activate = True
        if 0 <= self._stream_cnt < len(self._flat_seq):
            self._flat_seq[self._stream_cnt].callback = _callback
            return self._flat_seq[self._stream_cnt].apply()
        return None

    def __iter__(self):
        return iter(self._flat_seq)
    
    def __getitem__(self, index: int):
        return self._flat_seq[index]
