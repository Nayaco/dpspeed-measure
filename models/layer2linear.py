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
from typing import Tuple

import warnings

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
from torch import Tensor

from dsmeasure.core.abstract_operator import AbstractOperatorConfig, AbstractOperator
from dsmeasure.common_operators.op_common import OpStaticComputational, OpStaticNonComputational
from dsmeasure.common_operators.op_config import OperatorComputationalConfig, OperatorNonComputationalConfig, OperatorCustomConfig
from dsmeasure.device.gpu import DeviceCUDAConfig, DeviceCUDA

from dsmeasure.core.device_manager import DeviceManager
from dsmeasure.core.operator_manager import OperatorManager

"""
This file is used to test the correctness of the dsmeasure framework.
CudaMalloc_InitModel(0us, 100MB)
-> 
CudaMalloc_Linear1(10us, 10MB) 
->
CudaMalloc_Linear2(10us, 10MB) | Linear1(700us, 0MB)
->
Linear2(700us, 0MB)
->
RelU(400us, 10MB)
->
CudaMalloc_RelUBackward(10us, 10MB)
->
RelUBackward(450us, 10MB)
->
CudaFree_Linear2(10us, 10MB) | CudaMalloc_LinearBackward2(10us, 10MB) 
->
LinearBackward2(800us, 0MB)
->
CudaFree_Linear1(10us, 10MB) | CudaFree_ReluBackward(10us, 10MB) | CudaMalloc_LinearBackward1(10us, 10MB)
->
LinearBackward1(800us, 0MB)
->
CudaFree_LinearBackward2(10us, 10MB) | CudaFree_LinearBackward1(10us, 10MB)

(linear2[
    (linear)
    (linear)
    (relu)
    (relu_backward)
    (linear_backward)
    (linear_backward)
])
"""

class CudaMalloc(OpStaticNonComputational):
    def __init__(self, config: OperatorNonComputationalConfig, alloc_memory: int):
        """
        config: OperatorComputationalConfig
        alloc_memory: int
        """
        super().__init__(config)
        self.estimate_runtime: int = int(10)
        self.alloc_memory = alloc_memory

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)

    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        if cuda is None:
            raise Exception("device not found")
        return cuda.occupy(self.estimate_runtime, self.default_apply_cb, \
                           memory=self.alloc_memory, computational=False)
    
class Linear(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, compute_time: int):
        super().__init__(config)
        self.estimate_runtime: int = compute_time

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)
    
    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        if cuda is None:
            raise Exception("device not found")
        return cuda.occupy(self.estimate_runtime, self.default_apply_cb, \
                           memory=0, computational=True)

class RelU(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, compute_time: int):
        super().__init__(config)
        self.estimate_runtime: int = compute_time

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)
    
    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        if cuda is None:
            raise Exception("device not found")
        return cuda.occupy(self.estimate_runtime, self.default_apply_cb, \
                           memory=0, computational=True)

class Linear2Network(AbstractOperator):
    def __init__(self, config: OperatorCustomConfig):
        super().__init__(config)

        self.linear1 = Linear(OperatorComputationalConfig(
            op_uid=0, op_name="linear1"), 70)
        self.linear2 = Linear(OperatorComputationalConfig(
            op_uid=0, op_name="linear2"), 70)
        self.relu = RelU(OperatorComputationalConfig(
            op_uid=0, op_name="relu"), 40)
        self.linear1_backward = Linear(OperatorComputationalConfig(
            op_uid=0, op_name="linear1_backward"), 80)
        self.linear2_backward = Linear(OperatorComputationalConfig(
            op_uid=0, op_name="linear2_backward"), 80)
        self.relu_backward = RelU(OperatorComputationalConfig(
            op_uid=0, op_name="relu_backward"), 50)

        self.cuda_malloc0 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_malloc_init"), 100)
        self.cuda_malloc1 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_malloc_linear1"), 10)
        self.cuda_malloc2 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_malloc_linear2"), 10)
        self.cuda_malloc3 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_malloc_relu_gradin"), 10)
        self.cuda_malloc4 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_malloc_linear2_gradin"), 10)
        self.cuda_malloc5 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_malloc_linear1_gradin"), 10)

        self.cuda_free1 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_free_linear1"), -10)
        self.cuda_free2 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_free_linear2"), -10)
        self.cuda_free3 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_free_relu_gradin"), -10)
        self.cuda_free4 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_free_linear1_gradin"), -10)
        self.cuda_free5 = CudaMalloc(OperatorNonComputationalConfig(
            op_uid=0, op_name="cuda_free_linear2_gradin"), -10)

        self.cuda_malloc0.add_next(self.cuda_malloc1)
        
        self.cuda_malloc1.add_next(self.linear1)
        self.cuda_malloc1.add_next(self.cuda_malloc2)
        
        self.cuda_malloc2.add_next(self.linear2)
        
        self.linear2.add_next(self.relu)
        
        self.relu.add_next(self.cuda_malloc3)
        
        self.cuda_malloc3.add_next(self.relu_backward)
        
        self.relu_backward.add_next(self.cuda_malloc4)
        self.relu_backward.add_next(self.cuda_free2)
        

        self.cuda_malloc4.add_next(self.linear2_backward)
        
        self.linear2_backward.add_next(self.cuda_free1)
        self.linear2_backward.add_next(self.cuda_free3)
        self.linear2_backward.add_next(self.cuda_malloc5)

        self.cuda_malloc5.add_next(self.linear1_backward)

        self.linear1_backward.add_next(self.cuda_free4)
        self.linear1_backward.add_next(self.cuda_free5)

        # op_manager = OperatorManager()
        # self.linear1_uid, _ = op_manager.register(self.linear1)
        # self.linear2_uid, _ = op_manager.register(self.linear2)
        # self.relu_uid, _ = op_manager.register(self.relu)
        # self.linear1_backward_uid, _ = op_manager.register(self.linear1_backward)
        # self.linear2_backward_uid, _ = op_manager.register(self.linear2_backward)
        # self.relu_backward_uid, _ = op_manager.register(self.relu_backward)
        # self.cuda_malloc0_uid, _ = op_manager.register(self.cuda_malloc0)
        # self.cuda_malloc1_uid, _ = op_manager.register(self.cuda_malloc1)
        # self.cuda_malloc2_uid, _ = op_manager.register(self.cuda_malloc2)
        # self.cuda_malloc3_uid, _ = op_manager.register(self.cuda_malloc3)
        # self.cuda_malloc4_uid, _ = op_manager.register(self.cuda_malloc4)
        # self.cuda_malloc5_uid, _ = op_manager.register(self.cuda_malloc5)
        # self.cuda_free1_uid, _ = op_manager.register(self.cuda_free1)
        # self.cuda_free2_uid, _ = op_manager.register(self.cuda_free2)
        # self.cuda_free3_uid, _ = op_manager.register(self.cuda_free3)
        # self.cuda_free4_uid, _ = op_manager.register(self.cuda_free4)
        # self.cuda_free5_uid, _ = op_manager.register(self.cuda_free5)

        self._subop = self.cuda_malloc0

    def __repr__(self) -> str:
        ret = ""
        _queue = [(self._subop._config.op_uid, 0)]
        c_depth = 0
        while len(_queue) > 0:
            _uid, _depth = _queue.pop(0)
            if _depth > c_depth:
                ret += "\n"
                c_depth = _depth
            ret += OperatorManager().find(_uid).__repr__() + ' '
            _queue.extend([(_op._config.op_uid, _depth + 1) for _op in OperatorManager().find(_uid)._next])
        return ret

        
        


        

        
