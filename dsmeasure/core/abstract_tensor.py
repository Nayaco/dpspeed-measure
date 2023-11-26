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

from dataclasses import dataclass
from typing import Callable, Tuple
from abc import abstractmethod


class AbstractTensor:
    """
    Define the tensors for Graph G<V,E>, where indicate whether the memory should be free
    size: int
    """
    def __init__(self, size: int):
        self.size: int = size
        self.denpend_count: int = 0
        self.denpend_done: int = 0

    def reset(self):
        self.denpend_done = 0

    def required(self):
        self.denpend_count += 1
    
    def done(self):
        self.denpend_done += 1

    def no_required(self):
        return self.denpend_done == self.denpend_count