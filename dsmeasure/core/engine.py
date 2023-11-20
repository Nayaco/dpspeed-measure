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

    def evaluation(self, operators: list[tuple[int, AbstractOperator]]):
        """
        evaluation the operators
            operators: list(tuple(int, AbstractOperator))
                arrive time, operator
        """
        self.reset()
        


        
