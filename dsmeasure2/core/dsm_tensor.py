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
from abc import abstractmethod, ABC

class AbstractTensor(ABC):
    """
    Define the tensors for Graph G<V,E>, where indicate whether the memory should be free
    size: int
    """
    tensor_uid: int = 0
    tensor_size: int = 0

    def __init__(self, tensor_uid: int = 0, tensor_size: int = 0):
        self.tensor_uid = tensor_uid
        self.tensor_size = tensor_size
        

    @abstractmethod
    def offload(self):
        pass
    
    @abstractmethod
    def materialize(self):
        pass

    @abstractmethod
    def destroy(self):
        pass
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return "Tensor(uid={},size={})".format(self.tensor_uid, self.tensor_size)
