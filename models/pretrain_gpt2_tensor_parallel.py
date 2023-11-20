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

from dsmeasure.core.device_manager import GetDeviceManager

class Embedding(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig):
        super().__init__(config)

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)
    
class Linear(OpStaticComputational):
    def __init__(self, config: OperatorComputationalConfig):
        super().__init__(config)

    def estimate(self, *tensor_in: Tensor) -> Tuple[int, Tensor]:
        return super().estimate(*tensor_in)

class LayerNorm(OpStaticComputational):