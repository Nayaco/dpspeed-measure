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
from typing import Callable, Any
from abc import abstractmethod, ABC

@dataclass
class AbstractDeviceConfig:
    """
    basic device config, inlcude:
        device uid
        device name
    """
    device_uid: int = 0
    device_name: str = None

    is_computational: bool = False
    is_transferatble: bool = False

    def __post_init__(self):
        pass

class AbstractDevice(ABC):
    """
    device interface, inlcude: try_occupy(), occupy(), run()
    """
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def occupy(self, run_time: int, callback: Callable[..., Any] | None, **kwargs):
        pass
    @abstractmethod
    def try_occupy(self, run_time: int, **kwargs):
        pass
    @abstractmethod
    def run(self, interval: int):
        pass
    @abstractmethod
    def is_idle(self):
        pass
    @abstractmethod
    def reset(self):
        pass