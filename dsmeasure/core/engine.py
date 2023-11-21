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

    def evaluation(self, interval: int, operators: list[int]):
        """
        evaluation the operators
            interval: us
            operators: list(uid of AbstractOperator)
                list of independent training pipeline
        """
        op_manager = OperatorManager()
        dev_manager = DeviceManager()
        self.reset()
        wait_queue: list[int] = []
        ready_queue: list[int] = []
        for op_uid in operators:
            op_manager.operators[op_uid].reset()
            ready_queue.append(op_uid)
        while len(ready_queue) > 0 or len(wait_queue) > 0:
            new_ready_queue: list[int] = []
            for r_op_uid in ready_queue:
                while op_manager.operators[ready_queue[0]].subop()._config.is_prime:
                    r_op_uid = \
                        op_manager.operators[ready_queue[0]].subop()._config.op_uid
                ret = op_manager.operators[ready_queue[0]].apply()
            
                if ret[0] == True:
                    if op_manager.operators[ready_queue[0]]._config.is_prime:
                        for op in op_manager.operators[ready_queue[0]]._next:
                            if op._prev_done == len(op._prev):
                                wait_queue.append(op._config.op_uid)
                    else:
                else:
                    new_ready_queue.append(ready_queue[r_op_uid])
    
            ready_queue = new_ready_queue
            for d_uid in dev_manager.devices.keys():
                dev_manager.devices[d_uid].run(interval)
            for op in wait_queue:
                if op._prev_done == len(op._prev):
                    ready_queue.append(op)
                    wait_queue.remove(op)
            

            

            


        
