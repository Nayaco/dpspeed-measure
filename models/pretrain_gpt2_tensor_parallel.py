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

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
from torch import Tensor
import torch.nn as nn

from dsmeasure.core.abstract_operator import AbstractOperatorConfig, AbstractOperator
from dsmeasure.common_operators.op_common import OpStaticComputational, OpStaticNonComputational
from dsmeasure.common_operators.op_config import OperatorComputationalConfig, OperatorNonComputationalConfig, OperatorCustomConfig
from dsmeasure.device.gpu import DeviceCUDAConfig, DeviceCUDA

from dsmeasure.core.device_manager import DeviceManager

class CudaMelloc(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig, alloc_memory: int):
        """
        config: OperatorComputationalConfig
        alloc_memory: int
        """
        super().__init__(config)
        self.estimate_runtime: int = int(0)
        self.alloc_memory = alloc_memory

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)

    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        return cuda.occupy(self.estimate_runtime, self.default_apply_cb, \
                           memory=self.alloc_memory, computational=False)

class Embedding(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig):
        super().__init__(config)
        self.estimate_runtime: int = int(4000)

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)
    
    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        return cuda.occupy(self.estimate_runtime, None, memory=0, computational=True)
    
class Linear(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig):
        super().__init__(config)
        self.estimate_runtime: int = int(700)

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)
    
    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        return cuda.occupy(self.estimate_runtime, None, memory=0, computational=True)

class LayerNorm(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig):
        super().__init__(config)

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)
    
    def apply(self):
        cuda: DeviceCUDA = DeviceManager().find_by_name('cuda:0')
        cuda.occupy(self.estimate_runtime, None, memory=0, computational=True)

class SelfAttention(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig):
        super().__init__(config)

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)
    
class SelfAttention(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig):
        super().__init__(config)

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)