# Copyright (c) 2023, ISCS, Wenjie Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import builtins
import datetime
import json
import sys
import time

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
from dsmeasure2.graph.operator_graph import UnaryOperator, BinaryOperator, TernaryOperator, InitiateOperator
from dsmeasure2.graph.unary_operator import make_linear, make_layernorm, make_dropout, make_gelu
from dsmeasure2.graph.binary_operator import make_add, make_matmul

from functools import cache

@cache
class CostEngine:
    def __init__(self):
        self.max_cuda_memory = 0
        self.current_cuda_memory = 0
        
        self.total_cuda_util = 0.0
        self.total_pcie_util = 0.0
        
        self.cuda_util_trace = []
        self.pcie_util_trace = []
        self.cuda_memory_trace = []
    
    def reset(self):
        self.max_cuda_memory = 0
        self.current_cuda_memory = 0
        self.total_cuda_util = 0.0
        self.total_pcie_util = 0.0
        self.cuda_util_trace = []
        self.pcie_util_trace = []
        self.cuda_memory_trace = []

    def evaluation(self, interval: int, operators: list[int], graph_head: list[int] = [0], time_limit: int = None):
        """
        evaluation the operators
            interval: us
            operators: list(uid of AbstractOperator)
                list of independent training pipeline
        """
        time_limit = time_limit if time_limit is not None else 1e10
        op_manager = OperatorManager()
        t_manager = TensorManager()
        dev_manager = DeviceManager()
        self.reset()

        wait_queue: list[int] = []
        ready_queue: list[int] = []
        for op_uid in operators:
            op_manager.operators[op_uid].reset()

        ready_queue.extend([operators[_head] for _head in graph_head])
        
        eval_done = False
        
        while (len(ready_queue) > 0 or len(wait_queue) > 0 or not eval_done) and time_limit > 0:
            # check if task in ready queue is can be executed
            new_ready_queue: list[int] = []
            for r_op_uid in ready_queue:
                # while not op_manager.operators[r_op_uid]._config.is_prime:
                #     r_op_uid = \
                #         op_manager.operators[r_op_uid].subop()._config.op_uid
                ret = op_manager.operators[r_op_uid].apply()

                if ret == True:
                    print(op_manager.operators[r_op_uid])
                    for n_op in op_manager.operators[r_op_uid]._next:
                        if n_op._config.op_uid not in wait_queue:
                            wait_queue.append(n_op._config.op_uid)
                else:
                    new_ready_queue.append(r_op_uid)
            ready_queue = new_ready_queue
            # execution
            eval_done = True
            for d_uid in dev_manager.devices.keys():
                dev_manager.devices[d_uid].run(interval)
                eval_done = eval_done and dev_manager.devices[d_uid].is_idle()
            # profile
            self.current_cuda_memory = dev_manager.find_by_name('cuda:0').memory_used
            self.cuda_memory_trace.append(self.current_cuda_memory)
            self.max_cuda_memory = max(self.max_cuda_memory, self.current_cuda_memory)

            # check if task in wait queue can be inserted into ready queue
            new_wait_queue: list[int] = []
            # print([str(op_manager.operators[w_op_uid]) for w_op_uid in wait_queue])
            for w_op_uid in wait_queue:
                if op_manager.operators[w_op_uid]._prev_done == len(op_manager.operators[w_op_uid]._prev):
                    ready_queue.append(w_op_uid)
                else:
                    new_wait_queue.append(w_op_uid)
            wait_queue = new_wait_queue

if __name__ == "__main__":
    DeviceManager().register(DeviceCUDAConfig(memory_max_capacity=40000, memory_limit_capacity=40000))
    DeviceManager().register(DevicePCIEConfig())

    l1_fwd, l1_bwd = make_linear(ActivationTensor(tensor_size=128 * 4), 
                                 ActivationTensor(tensor_size=128 * 4), 
                                 WeightTensor(tensor_size=128 * 4),
                                 rt_fwd=50, rt_bwd=50)
    TensorManager().register(l1_fwd.input)
    TensorManager().register(l1_fwd.weight)
    TensorManager().register(l1_fwd.output[0])
    TensorManager().register(l1_bwd.input_a)
    TensorManager().register(l1_bwd.weight)
    TensorManager().register(l1_bwd.output[0])

    def loss_done():
        print("loss done")
    loss_fn = UnaryOperator(OperatorComputationalConfig(op_name="loss"), loss_done)
    loss_fn.input = l1_fwd.output[0]
    loss_fn.output = [ActivationTensor(tensor_size=128)]
    loss_fn.estimate_runtime = 50
    TensorManager().register(loss_fn.input)
    TensorManager().register(loss_fn.output[0])
    
    l1_bwd.input_b = loss_fn.output[0]
    
    initiate_fn = InitiateOperator(OperatorComputationalConfig(op_name="initiate"))
    initiate_fn.set_parameters(
        [l1_fwd.input],
        [l1_fwd.weight],
        'cuda:0'
    )

    initiate_fn.add_next(l1_fwd)
    l1_fwd.add_next(loss_fn)
    loss_fn.add_next(l1_bwd)

    OperatorManager().register(initiate_fn)
    OperatorManager().register(l1_fwd)
    OperatorManager().register(l1_bwd)
    OperatorManager().register(loss_fn)
    
    CostEngine().evaluation(10, [_op._config.op_uid for _op in [initiate_fn, l1_fwd, loss_fn, l1_bwd]])
    print(CostEngine().cuda_memory_trace)
    
