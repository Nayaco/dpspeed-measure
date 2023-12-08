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
from dsmeasure2.core.dsm_tensor_mng import TensorManager

from dsmeasure2.graph.tensor_define import ActivationTensor, WeightTensor, TensorState
from dsmeasure2.flatten.flatten_operator import FlattenOperator, FlattenInitiate
from dsmeasure2.flatten.flatten_stream import FlattenStream, FlattenBranch, FlattenMerge, FlattenPause

class FlattenOffload(OpStaticNonComputational):
    def __init__(self, 
                 config: OperatorNonComputationalConfig, 
                 callback: Callable[..., Any] = None):
        super().__init__(config)
        self._estimate_runtime: int = 1
        self._tensors: list[ActivationTensor|WeightTensor] = []
        self._device_name: list[str] = ['cuda:0', 'pcie:0']
        self._callback: Callable[..., Any] = callback
        self._prev_done = self._next = self._prev = None

    def add_next(self, next_op):
        raise Exception("flatten operator does not support add_next")

    def estimate(self, *tensor_in: Any) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self) -> bool:
        # if False in [_ts.state == TensorState.AVAILABLE for _ts in self._tensors]:
        #     for _ts in self._tensors:
        #         if _ts.state != TensorState.AVAILABLE:
        #             print(_ts, _ts.state)
        #     print(self)
        assert False not in [_ts.state == TensorState.AVAILABLE for _ts in self._tensors], \
            "at least 1 tensor not available"
        _device_computational: DeviceCUDA = DeviceManager().find_by_name(self._device_name[0])
        _device_transfer: DevicePCIE4 = DeviceManager().find_by_name(self._device_name[1])
        
        assert _device_computational is not None and _device_transfer is not None, \
            "device not found"
        
        _tensor_size_tot: int = sum([_ts.tensor_size for _ts in self._tensors])

        def _apply_cb():
            # un-computational device occupy 10us won't fail
            _device_computational.occupy(1, None, \
                memory=-_tensor_size_tot,
                computational=False)
            for _tensor in self._tensors:
                _tensor.offload()
            self._callback() if self._callback is not None else None
        return _device_transfer.occupy(-1, _apply_cb, dsize=_tensor_size_tot)
    
    def reset(self) -> None:
        super().reset()

class FlattenLoadin(OpStaticNonComputational):
    def __init__(self, 
                 config: OperatorNonComputationalConfig, 
                 callback: Callable[..., Any] = None):
        super().__init__(config)
        self._estimate_runtime: int = 1
        self._tensors: list[ActivationTensor|WeightTensor] = []
        self._device_name: list[str] = ['cuda:0', 'pcie:0']
        self._callback: Callable[..., Any] = callback
        self._prev_done = self._next = self._prev = None

    def add_next(self, next_op):
        raise Exception("flatten operator does not support add_next")

    def estimate(self, *tensor_in: Any) -> int:
        return super().estimate(*tensor_in)
    
    def apply(self) -> bool:
        assert False not in [_ts.state == TensorState.OFFLOADED for _ts in self._tensors], \
            "at least 1 tensor not offloaded"
        _device_computational: DeviceCUDA = DeviceManager().find_by_name(self._device_name[0])
        _device_transfer: DevicePCIE4 = DeviceManager().find_by_name(self._device_name[1])
        
        assert _device_computational is not None and _device_transfer is not None, \
            "device not found"
        
        _tensor_size_tot: int = sum([_ts.tensor_size for _ts in self._tensors])
        def _apply_cb():
            for _tensor in self._tensors:
                _tensor.materialize()
            self._callback() if self._callback is not None else None

        # un-computational device occupy 10us won't fail
            _device_computational.occupy(1, None, \
                memory=_tensor_size_tot, 
                computational=False)
        return _device_transfer.occupy(1, _apply_cb, dsize=_tensor_size_tot)
    
    def reset(self) -> None:
        super().reset()

def make_passive_offload(_main_stream: FlattenStream, _source_op_index: int, _offload_uid: int):
    assert True in [_input.tensor_uid == _offload_uid for _input in _main_stream[_source_op_index]._input], \
        "offload tensor not found"
    
    _target_op_index = 0
    for _i, _op in enumerate(_main_stream):
        if _i > _source_op_index and isinstance(_op, FlattenOperator) and \
            _offload_uid in [_input.tensor_uid for _input in _op._input]:
            _target_op_index = _i
            break
    assert _target_op_index != 0, "intermediate tensor cannot be offloaded"

    _offload_op: FlattenOffload = OperatorManager().register(
        FlattenOffload(OperatorNonComputationalConfig(
            op_name=_main_stream[_source_op_index]._config.op_name+'_offload')) )
    _offload_op._tensors = [TensorManager().find(_offload_uid)]
    
    _loadin_op: FlattenLoadin = OperatorManager().register(
        FlattenLoadin(OperatorNonComputationalConfig(
            op_name=_main_stream[_source_op_index]._config.op_name+'_loadin')) )
    _loadin_op._tensors = [TensorManager().find(_offload_uid)]

    _offload_loadin_pause: FlattenPause = OperatorManager().register(
        FlattenPause(OperatorCustomConfig(
            op_name=_main_stream[_source_op_index]._config.op_name+'_offload_loadin_pause')) )
    
    _offload_stream = FlattenStream(
        [_offload_op, _offload_loadin_pause, _loadin_op])

    _branch_op_offload: FlattenBranch = OperatorManager().register(
        FlattenBranch(OperatorCustomConfig(
            op_name=_main_stream[_source_op_index]._config.op_name+'_branch_offload'),
            [_offload_stream]) )
    _main_stream._flat_seq.insert(_source_op_index+1, _branch_op_offload)
    _target_op_index += 1
    
    _branch_op_loadin: FlattenBranch = OperatorManager().register(
        FlattenBranch(OperatorCustomConfig(
            op_name=_main_stream[_source_op_index]._config.op_name+'_branch_loadin'),
            [_offload_stream]) )
    _main_stream._flat_seq.insert(_target_op_index, _branch_op_loadin)
    _target_op_index += 1

    _merge_op: FlattenMerge = OperatorManager().register(
        FlattenMerge(OperatorCustomConfig(
            op_name=_main_stream[_source_op_index]._config.op_name+'_merge'),
            [_offload_stream]) )
    _main_stream._flat_seq.insert(_target_op_index, _merge_op)

    return _offload_stream