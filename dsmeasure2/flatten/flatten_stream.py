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
from dsmeasure2.core.dsm_operator_mng import OperatorManager

from dsmeasure2.graph.tensor_define import ActivationTensor, WeightTensor, TensorState
from dsmeasure2.flatten.flatten_operator import FlattenOperator, FlattenInitiate

class FlattenController(OpStaticDerivative):
    def __init__(self, config: OperatorCustomConfig, callback: Callable[..., Any] = None):
        super().__init__(config)
        self._prev_done = self._next = self._prev = None
        self._callback: Callable[..., Any] = callback
        self._subop = []

    def add_next(self, next_op):
        raise Exception("flatten operator does not support add_next")
    
    def apply(self):
        self._callback() if self._callback else None
        return True
        
class FlattenStream:
    def __init__(self, 
                 flat_seq: list[int]|list[FlattenOperator|FlattenInitiate|FlattenController], 
                 reentrant: bool=False) -> None:
        self._flat_seq: list[FlattenOperator|FlattenInitiate|FlattenController] = \
                [OperatorManager().find(_id) for _id in flat_seq] \
                    if isinstance(flat_seq[0], int) else flat_seq
        self._stream_cnt = 0
        self._activate = False
        self._idle = True
        self._reentrant = reentrant
        self._pause = False

    @property
    def finish(self) -> bool:
        return self._stream_cnt >= len(self._flat_seq)
    
    @property
    def pause(self) -> bool:
        return self._stream_cnt < len(self._flat_seq) and (not self._activate) and self._pause
    
    def forward(self) -> FlattenInitiate|FlattenOperator|FlattenController|None:
        if not self._activate or \
            0 > self._stream_cnt or self._stream_cnt >= len(self._flat_seq):
            return None
        
        def _callback():
            self._activate = True
            self._stream_cnt += 1
        def _callback_pause():
            self._pause = True
            self._stream_cnt += 1
            
        self._flat_seq[self._stream_cnt]._callback = \
            _callback_pause if isinstance(self._flat_seq[self._stream_cnt], FlattenPause) else _callback
        
        self._activate = False
        return self._flat_seq[self._stream_cnt]
    
    def reset(self):
        self._stream_cnt = 0
        self._activate = False
        for _flat_op in self._flat_seq:
            _flat_op._callback = None
            _flat_op.reset() if not isinstance(_flat_op, FlattenController) else None

    def __iter__(self):
        return iter(self._flat_seq)
    
    def __getitem__(self, index: int) -> FlattenInitiate|FlattenOperator|FlattenController:
        return self._flat_seq[index]

class FlattenBranch(FlattenController):
    def __init__(self, config: OperatorCustomConfig,
                 branch_dst: list[FlattenStream],
                 callback: Callable[..., Any] = None):
        super().__init__(config, callback)
        
        self._flatten_stream_to: list[FlattenStream] = branch_dst
        
    def apply(self) -> bool:
        for _to in self._flatten_stream_to:
            _to._activate = True
            _to._pause = False 
        return super().apply()
    
class FlattenMerge(FlattenController):
    def __init__(self, config: OperatorCustomConfig,
                 branch_src: list[FlattenStream],
                 callback: Callable[..., Any] = None):
        super().__init__(config, callback)

        self._flatten_stream_from: list[FlattenStream] = branch_src

    def apply(self) -> bool:
        # print(self, False not in [_from.finish or _from.pause for _from in self._flatten_stream_from])
        return (False not in [_from.finish or _from.pause for _from in self._flatten_stream_from]) \
            and super().apply()
    
class FlattenPause(FlattenController):
    def __init__(self, config: OperatorCustomConfig,
                 callback: Callable[..., Any] = None):
        super().__init__(config, callback)

    def apply(self) -> bool:
        return super().apply()

def stream_synchronize(_main_stream: FlattenStream, _slave_streams: list[FlattenStream], _source_op_index: int):
    _merge_op: FlattenMerge = OperatorManager().register(
        FlattenMerge(OperatorCustomConfig(
            op_name=_main_stream[_source_op_index]._config.op_name+'_sync_merge'),
            _slave_streams) )
    _main_stream._flat_seq.insert(_source_op_index+1, _merge_op)