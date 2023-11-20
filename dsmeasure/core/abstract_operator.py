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

@dataclass
class AbstractOperatorConfig:
    """
    
    """
    op_uid: int = 0
    op_name: str = None

    is_computational = False
    is_prime = False

    def __post_init__(self):
        pass

class AbstractOperator:
    """
    Define the operators as Graph G<V,E> where V are operators, E define 
    dependency between contiguous operator excutions. Sibling operations
    can be well paralleled.
    """
    def __init__(self, config: AbstractOperatorConfig):
        self._config = config
        self._next = []
        self._prev = []

    def add_next(self, next_op):
        self._next.append(next_op)
        next_op._prev.append(self)
    
    """
    *tensor_in: tensor shapes
    return operator excution time and output tensor shape 
    """
    @abstractmethod
    def estimate(self, *tensor_in: torch.Tensor) -> Tuple[int, torch.Tensor]:
        pass
    
    def __repr__(self) -> str:
        return "<operator-%3d, %s>" % (self._config.op_uid, '{0: <5}'.format(self._config.op_name or 'None'))
    
    def __len__(self) -> int:
        return 0 if self._next is None else len(self._next)