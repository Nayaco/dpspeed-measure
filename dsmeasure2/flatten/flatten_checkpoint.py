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
        # un-computational device occupy <10us won't fail
        return _device_0.occupy(1, _apply_cb, memory=-_tensor_size_tot, computational=False)
    
    def reset(self) -> None:
        super().reset()

def make_entire_checkpoint(_main_stream: FlattenStream, 
                    _source_op_index_from: int, _source_op_index_to: int, 
                    _target_op_index_from: int) -> FlattenStream:
    _drop_tensors: list[ActivationTensor] = []
    _flatten_checkpoints: list[FlattenOperator] = []
    for _op in _main_stream._flat_seq[_source_op_index_from:_source_op_index_to]:
        _drop_tensors.extend(_op._output)
        _flatten_checkpoints.append(OperatorManager().register(
            FlattenOperator(OperatorComputationalConfig(
                op_name=_op._config.op_name+'_checkpoint',),
                checkpointing=True) ))
        _flatten_checkpoints[-1]._input = _op._input
        _flatten_checkpoints[-1]._output = _op._output
        _flatten_checkpoints[-1]._weight = _op._weight
        _flatten_checkpoints[-1]._intermediate_memory = _op._intermediate_memory
        _flatten_checkpoints[-1]._estimate_runtime = _op._estimate_runtime
        _flatten_checkpoints[-1]._device_name = _op._device_name
    
    _flatten_drop: FlattenDrop = OperatorManager().register(
        FlattenDrop(OperatorNonComputationalConfig(
            op_name=_main_stream._flat_seq[_source_op_index_from]._config.op_name+'_drop')) )
    _flatten_drop._tensors = _drop_tensors
    _flatten_pause: FlattenPause = OperatorManager().register(
        FlattenPause(OperatorNonComputationalConfig(
            op_name=_main_stream._flat_seq[_source_op_index_from]._config.op_name+'_pause')) )

    _checkpoint_stream = FlattenStream([_flatten_drop, _flatten_pause, *_flatten_checkpoints])
    
    _branch_op_checkpoint: FlattenBranch = OperatorManager().register(
        FlattenBranch(OperatorCustomConfig(
            op_name=_main_stream[_source_op_index_from]._config.op_name + '_' + \
                    _main_stream[_source_op_index_to]._config.op_name + \
                    '_branch_checkpoint'),
            [_checkpoint_stream]) )
    _main_stream._flat_seq.insert(_source_op_index_to+1, _branch_op_checkpoint)
    
    _branch_op_rematerialize: FlattenBranch = OperatorManager().register(
        FlattenBranch(OperatorCustomConfig(
            op_name=_main_stream[_source_op_index_from]._config.op_name + '_' + \
                    _main_stream[_source_op_index_to]._config.op_name + \
                    '_branch_rematerialize'),
            [_checkpoint_stream]) )
    _main_stream._flat_seq.insert(_target_op_index_from+1, _branch_op_rematerialize)
    _merge_op_rematerialize: FlattenMerge = OperatorManager().register(
        FlattenMerge(OperatorCustomConfig(
            op_name=_main_stream[_source_op_index_from]._config.op_name + '_' + \
                    _main_stream[_source_op_index_to]._config.op_name + \
                    '_merge_rematerialize'),
            [_checkpoint_stream]) )
    _main_stream._flat_seq.insert(_target_op_index_from+2, _merge_op_rematerialize)

    return _checkpoint_stream