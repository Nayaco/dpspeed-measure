# Copyright (c) 2023, ISCS, Wenjie Zhang.

from functools import cache

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
from dsmeasure2.flatten.flatten_operator import FlattenOperator, FlattenInitiate
from dsmeasure2.flatten.flatten_stream import FlattenStream, FlattenController, FlattenMerge, FlattenBranch

@cache
class FlattenEngine:
    def __init__(self):
        self._cuda_mem_trace: list[int] = []
        self._tensor_ref_cnt: dict = {}
        self._interval_tot: int = 0
    
    def reset(self):
        self._cuda_mem_trace = []
        self._tensor_ref_cnt = {}
        self._interval_tot = 0

    def evaluation(self, flat_streams: list[FlattenStream], 
                   interval: int = 10, 
                   devices: str = ['cuda:0', 'pcie:0'], 
                   verbose: bool = False):
        self.reset()
        for _device in devices:
            DeviceManager().find_by_name(_device).reset()
        for _flat_stream in flat_streams:
            _flat_stream.reset()
        flat_streams[0]._activate = True
        _ready_queue = []
        # tensor ref conunt to see if it's intermediate
        self._tensor_ref_cnt = {}

        while False in [_flat_stream.finish for _flat_stream in flat_streams]:
            # Collect OPs that are ready to be executed
            for _flat_stream in flat_streams:
                _op = _flat_stream.forward()
                _op is not None and _ready_queue.append(_op)
            # if _flag:
            #     print(_ready_queue)
            #     print([_flat_stream._stream_cnt for _flat_stream in flat_streams])
            #     exit(0)
            # Execute OPs

            _ready_queue_new = []
            # merge will stop the src stream until required dst stream is done 
            # btw. merge control done will cost 1 interval
            _exec_control = [isinstance(_op, FlattenController) and not isinstance(_op, FlattenMerge) \
                                for _op in _ready_queue]
            _exec_mask = _exec_control if True in _exec_control else [True]*len(_ready_queue)
            
            for i, _op in enumerate(_ready_queue):
                _ret = _op.apply() if _exec_mask[i] else False
                # DEBUG
                if _ret and verbose:
                    print(_op)
                # intermediate tensor count
                if _ret and isinstance(_op, FlattenOperator):
                    for _input in _op._input:
                        self._tensor_ref_cnt[_input.tensor_uid] = \
                            [*self._tensor_ref_cnt[_input.tensor_uid], self._interval_tot] \
                                if _input.tensor_uid in self._tensor_ref_cnt else [self._interval_tot]
                
                _ready_queue_new.append(_op) if not _ret else None
            _ready_queue = _ready_queue_new
            # Computation/Transfer Execution
            if not (True in _exec_control):
                for _device in devices:
                    DeviceManager().find_by_name(_device).run(interval)
                self._interval_tot += 1
                # Profiling
                self._cuda_mem_trace.append(DeviceManager().find_by_name('cuda:0').memory_used)