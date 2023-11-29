# Copyright (c) 2023, ISCS, Wenjie Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler

from dsmeasure.core.abstract_operator import AbstractOperatorConfig, AbstractOperator
from dsmeasure.core.device_manager import DeviceManager
from dsmeasure.core.operator_manager import OperatorManager

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

    def evaluation(self, interval: int, operators: list[int], time_limit: int = None):
        """
        evaluation the operators
            interval: us
            operators: list(uid of AbstractOperator)
                list of independent training pipeline
        """
        time_limit = time_limit if time_limit is not None else 1e10
        op_manager = OperatorManager()
        dev_manager = DeviceManager()
        self.reset()

        wait_queue: list[int] = []
        ready_queue: list[int] = []
        for op_uid in operators:
            op_manager.operators[op_uid].reset()
            ready_queue.append(op_uid)
        while (len(ready_queue) > 0 or len(wait_queue) > 0) and time_limit > 0:
            # check if task in ready queue is can be executed
            new_ready_queue: list[int] = []
            for r_op_uid in ready_queue:
                while not op_manager.operators[r_op_uid]._config.is_prime:
                    r_op_uid = \
                        op_manager.operators[r_op_uid].subop()._config.op_uid
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
            for d_uid in dev_manager.devices.keys():
                dev_manager.devices[d_uid].run(interval)

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
            
