# Copyright (c) 2023, ISCS, Wenjie Zhang.
from typing import Callable, Any

import numpy as np
import torch

from dsmeasure2.core.dsm_device_mng import DeviceManager
from dsmeasure2.core.dsm_operator_mng import OperatorManager

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

from dsmeasure2.device.device_cuda import DeviceCUDA, DeviceCUDAConfig
from dsmeasure2.device.device_pcie import DevicePCIE4, DevicePCIEConfig

from dsmeasure2.core.dsm_tensor_mng import TensorManager
from dsmeasure2.graph.tensor_define import ActivationTensor, WeightTensor, TensorState
from dsmeasure2.flatten.flatten_operator import FlattenOperator, FlattenInitiate
from dsmeasure2.flatten.flatten_stream import FlattenStream, FlattenBranch, FlattenMerge, FlattenPause

class FlattenDrop(OpStaticNonComputational):
    def __init__(self, 
                 config: OperatorNonComputationalConfig, 
                 callback: Callable[..., Any] = None):
        super().__init__(config)
        self._estimate_runtime: int = 1
        self._tensors: list[ActivationTensor] = []
        self._device_name: list[str] = ['cuda:0']
        self._callback: Callable[..., Any] = callback
        self._prev_done = self._next = self._prev = None

    def add_next(self, next_op):
        raise Exception("flatten operator does not support add_next")

    def estimate(self, *tensor_in: Any) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self) -> bool:
        assert False not in [_ts.state == TensorState.AVAILABLE for _ts in self._tensors], \
            "at least 1 tensor not offloaded"
        _device_0: DeviceCUDA = DeviceManager().find_by_name(self._device_name[0])
        
        assert _device_0 is not None, "device not found"
        
        _tensor_size_tot: int = sum([_ts.tensor_size for _ts in self._tensors])
        def _apply_cb():
            for _tensor in self._tensors:
                _tensor.destroy()
            self._callback() if self._callback is not None else None

        # un-computational device occupy 10us won't fail
        return _device_0.occupy(1, None, memory=-_tensor_size_tot, computational=False)
    
    def reset(self) -> None:
        super().reset()

def make_checkpoint(_main_stream: FlattenStream, _source_op_index: int, _offload_uid: int):
    
    _checkpoint_stream = FlattenStream(
        [])
        
    _branch_op_checkpoint: FlattenBranch = OperatorManager().register(
        FlattenBranch(OperatorCustomConfig(
            op_name=_main_stream[_source_op_index]._config.op_name+'_branch_checkpoint'),
            [_offload_stream]) )
    _main_stream._flat_seq.insert(_source_op_index+1, _branch_op_offload)